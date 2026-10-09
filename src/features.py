# ===========================================================================
# Построение признаков для глобальной (панельной) модели прогнозирования.
#
# В проверочном эксперименте обучается модель на один месяц вперёд.
# Для дальних горизонтов прогноз подставляется вместо будущего наблюдения.
# Функция признаков также принимает произвольный горизонт, но прямые
# многошаговые модели в этом эксперименте не обучались.
#
# Технически для скорости лаги берём из "широкой" матрицы (строки — ряды,
# столбцы — месяцы). Это эквивалентно сдвигам, но работает быстро на всей
# панели из ~12 тысяч рядов. Маленькие перечисления (список лагов, окон)
# проходим обычными циклами для наглядности.
# ===========================================================================

import numpy as np
import pandas as pd

from src import calendar_features


def prepare_wide_value_matrix(consumption):
    """
    Превратить длинную таблицу расходов в "широкую" матрицу значений.

    Строки матрицы — ряды (пара территория-категория), заданные
    мультииндексом (territory_id, category).
    Столбцы — месяцы (Period[M]).
    Значения — расходы value.

    Такая матрица удобна для быстрого взятия лагов: лаг на k месяцев — это
    просто соседний столбец.
    """
    wide_matrix = consumption.pivot_table(
        index=[
            "territory_id",
            "category",
        ],
        columns="month",
        values="value",
        aggfunc="mean",
    )
    # Сортируем столбцы по возрастанию месяца, чтобы сдвиги были корректны.
    wide_matrix = wide_matrix.sort_index(axis=1)
    return wide_matrix


def build_static_features(wide_matrix, market_access, territory_dictionary,
                          forecast_config):
    """
    Построить статические признаки ряда, не зависящие от месяца:
        - market_access (индекс доступности рынков),
        - region_code (код региона как категориальный признак),
        - координаты центра МО (широта/долгота),
        - категория расходов (как категориальный признак).

    Возвращает таблицу, индексированную так же, как широкая матрица
    (territory_id, category).
    """
    feature_flags = forecast_config["features"]

    # Основа — ключи рядов из широкой матрицы.
    series_keys = wide_matrix.index.to_frame(index=False)
    static_features = series_keys.copy()

    # Индекс доступности рынков (по территории).
    if feature_flags["use_market_access"]:
        static_features = static_features.merge(
            market_access, on="territory_id", how="left"
        )

    # Региональные и географические признаки из справочника.
    columns_from_dictionary = [
        "territory_id",
    ]
    if feature_flags["use_region"]:
        columns_from_dictionary.append("region_code")
    if feature_flags["use_coordinates"]:
        columns_from_dictionary.append("center_latitude")
        columns_from_dictionary.append("center_longitude")

    if len(columns_from_dictionary) > 1:
        dictionary_subset = territory_dictionary[columns_from_dictionary]
        static_features = static_features.merge(
            dictionary_subset, on="territory_id", how="left"
        )

    # Категорию расходов кодируем числом (нужно для LightGBM).
    list_of_categories = sorted(static_features["category"].unique().tolist())
    category_to_code = {}
    for code_number, category_name in enumerate(list_of_categories):
        category_to_code[category_name] = code_number
    static_features["category_code"] = static_features["category"].map(category_to_code)

    # Устанавливаем тот же индекс, что и у широкой матрицы.
    static_features = static_features.set_index([
        "territory_id",
        "category",
    ])
    static_features = static_features.reindex(wide_matrix.index)

    return static_features, category_to_code


def build_features_at_origin(wide_matrix, static_features, calendar_table,
                             origin_month, horizon, forecast_config,
                             news_features_table=None):
    """
    Собрать матрицу признаков для всех рядов на момент отсечки origin_month
    для заданного горизонта horizon.

    Признаки считаются ТОЛЬКО по данным до origin_month включительно.
    Цель (target) — значение ряда в месяце origin_month + horizon, если оно
    известно; иначе NaN (такие строки пойдут только на предсказание).

    Возвращает таблицу, где каждая строка — один ряд, с колонками признаков
    и колонкой target.
    """
    feature_flags = forecast_config["features"]
    use_log1p = forecast_config["target"]["use_log1p"]

    list_of_lags = feature_flags["lags"]
    list_of_rolling_windows = feature_flags["rolling_means"]

    # Список всех столбцов-месяцев матрицы.
    all_month_columns = list(wide_matrix.columns)

    # Проверяем, что origin_month есть в матрице.
    if origin_month not in all_month_columns:
        raise ValueError("Месяц отсечки отсутствует в данных: " + str(origin_month))

    index_of_origin = all_month_columns.index(origin_month)

    # ---- Начинаем собирать таблицу признаков --------------------------------
    features_table = pd.DataFrame(index=wide_matrix.index)

    # (1) Лаговые признаки: значение ряда k месяцев назад относительно origin.
    for lag_in_months in list_of_lags:
        index_of_lagged_month = index_of_origin - (lag_in_months - 1)
        column_name = "lag_" + str(lag_in_months)
        if index_of_lagged_month >= 0:
            lagged_month = all_month_columns[index_of_lagged_month]
            lagged_values = wide_matrix[lagged_month]
        else:
            # Если месяца нет (ушли за начало истории) — ставим пропуск.
            lagged_values = pd.Series(np.nan, index=wide_matrix.index)

        if use_log1p:
            features_table[column_name] = np.log1p(lagged_values)
        else:
            features_table[column_name] = lagged_values

    # (2) Скользящие средние по последним W месяцам (заканчивая на origin).
    for window_in_months in list_of_rolling_windows:
        start_index = index_of_origin - window_in_months + 1
        if start_index < 0:
            start_index = 0
        months_in_window = all_month_columns[start_index: index_of_origin + 1]
        values_in_window = wide_matrix[months_in_window]

        mean_over_window = values_in_window.mean(axis=1)
        column_name = "rollmean_" + str(window_in_months)
        if use_log1p:
            features_table[column_name] = np.log1p(mean_over_window)
        else:
            features_table[column_name] = mean_over_window

    # (3) Признак год-к-году: отношение последнего значения к значению 12 мес
    #     назад. Показывает текущий темп роста ряда.
    index_of_year_ago = index_of_origin - 12
    last_value = wide_matrix[all_month_columns[index_of_origin]]
    if index_of_year_ago >= 0:
        year_ago_value = wide_matrix[all_month_columns[index_of_year_ago]]
        # Защита от деления на ноль: заменяем нули маленьким числом.
        safe_year_ago = year_ago_value.replace(0.0, np.nan)
        features_table["yoy_ratio"] = last_value / safe_year_ago
    else:
        features_table["yoy_ratio"] = np.nan

    # (4) Календарные признаки ЦЕЛЕВОГО месяца (они известны заранее).
    target_month = origin_month + horizon
    calendar_row = calendar_table[calendar_table["month"] == target_month]
    if len(calendar_row) == 1:
        features_table["target_month_number"] = int(calendar_row["month_number"].iloc[0])
        if feature_flags["use_holiday_features"]:
            features_table["target_working_days"] = int(calendar_row["working_days"].iloc[0])
            features_table["target_non_working_days"] = int(calendar_row["non_working_days"].iloc[0])
            features_table["target_share_working"] = float(calendar_row["share_working_days"].iloc[0])
    else:
        # Если календаря на целевой месяц нет — ставим нейтральные значения.
        features_table["target_month_number"] = target_month.month
        if feature_flags["use_holiday_features"]:
            features_table["target_working_days"] = np.nan
            features_table["target_non_working_days"] = np.nan
            features_table["target_share_working"] = np.nan

    # (5) Горизонт как признак (одна модель на горизонт, но признак не мешает).
    features_table["horizon"] = horizon

    # (6) Статические признаки ряда (market_access, регион, координаты, категория).
    for static_column_name in static_features.columns:
        if static_column_name == "category":
            # Текстовую категорию не добавляем, используем её числовой код.
            continue
        features_table[static_column_name] = static_features[static_column_name]

    # (7) Новостной признак (если включён и подготовлен).
    if feature_flags.get("use_news_features", False) and news_features_table is not None:
        features_table = attach_news_feature(
            features_table, news_features_table, origin_month, static_features
        )

    # (8) Макропризнак ИПЦ (если включён и колонка cpi_signal присутствует).
    #     Доступность во времени уже учтена при построении таблицы макро
    #     (в строке месяца t лежит ИПЦ за t-1), а в рекурсии прогноза момент
    #     знания заморожен и последнее значение переносится вперёд.
    has_cpi_column = (news_features_table is not None
                      and "cpi_signal" in news_features_table.columns)
    if feature_flags.get("use_cpi_feature", False) and has_cpi_column:
        features_table = attach_cpi_feature(
            features_table, news_features_table, origin_month, static_features
        )

    # ---- Цель (target) ------------------------------------------------------
    if target_month in all_month_columns:
        target_values = wide_matrix[target_month]
    else:
        target_values = pd.Series(np.nan, index=wide_matrix.index)

    if use_log1p:
        features_table["target"] = np.log1p(target_values)
    else:
        features_table["target"] = target_values

    # Запоминаем служебную информацию, пригодится при сборе прогнозов.
    features_table["origin_month"] = str(origin_month)
    features_table["target_month"] = str(target_month)

    return features_table


def attach_news_feature(features_table, news_features_table, origin_month,
                        static_features):
    """
    Присоединить новостной признак, доступный НА момент отсечки origin_month.

    Чтобы не заглядывать в будущее, берём новостной сигнал последнего
    известного месяца (origin_month) по региону ряда.
    news_features_table ожидается с колонками: month, region_code, news_signal.
    """
    # Берём срез новостей за месяц отсечки.
    news_this_month = news_features_table[
        news_features_table["month"].astype(str) == str(origin_month)
    ]

    # Словарь регион -> новостной сигнал за месяц отсечки.
    region_to_signal = {}
    for _, news_row in news_this_month.iterrows():
        region_to_signal[int(news_row["region_code"])] = float(news_row["news_signal"])

    # Для каждого ряда подставляем сигнал его региона.
    region_codes_of_series = static_features["region_code"].values
    news_values = []
    for region_code_value in region_codes_of_series:
        if pd.notna(region_code_value) and int(region_code_value) in region_to_signal:
            news_values.append(region_to_signal[int(region_code_value)])
        else:
            news_values.append(0.0)

    features_table["news_signal"] = news_values
    return features_table


def attach_cpi_feature(features_table, macro_features_table, origin_month,
                       static_features):
    """
    Присоединить макропризнак ИПЦ, доступный НА момент отсечки origin_month.

    В строке месяца origin_month уже лежит последнее опубликованное значение
    ИПЦ (за предыдущий месяц), поэтому заглядывания в будущее нет. ИПЦ —
    федеральный показатель, одинаков для всех рядов на данный месяц.
    macro_features_table ожидается с колонками: month, cpi_signal.
    """
    macro_this_month = macro_features_table[
        macro_features_table["month"].astype(str) == str(origin_month)
    ]

    if len(macro_this_month) > 0:
        cpi_value_this_month = float(macro_this_month["cpi_signal"].iloc[0])
    else:
        cpi_value_this_month = 0.0

    number_of_series = len(features_table)
    cpi_values = []
    for series_number in range(number_of_series):
        cpi_values.append(cpi_value_this_month)

    features_table["cpi_signal"] = cpi_values
    return features_table


def get_feature_column_names(features_table):
    """
    Вернуть список колонок-признаков (без служебных колонок и цели).
    """
    columns_to_exclude = [
        "target",
        "origin_month",
        "target_month",
    ]
    feature_column_names = []
    for column_name in features_table.columns:
        if column_name not in columns_to_exclude:
            feature_column_names.append(column_name)
    return feature_column_names

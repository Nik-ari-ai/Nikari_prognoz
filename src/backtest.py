# ===========================================================================
# Бэктест прогнозных моделей методом скользящего начала (rolling-origin).
#
# Логика честной проверки:
#   - Берём момент отсечки (origin) = последний "наблюдаемый" месяц.
#   - Модель видит только историю до origin включительно.
#   - Делаем прогноз на горизонты h = 1, 3, 6, 12 месяцев.
#   - Сравниваем прогноз с фактом в месяце origin + h (если факт известен).
#   - Передвигаем отсечку на месяц вперёд и повторяем.
#
# Так для каждой модели накапливается набор пар (факт, прогноз) по каждому
# горизонту, по которым считаются MAE и R^2.
#
# Пообъектные модели (SeasonalNaive, Prophet, ETS) обучаются на каждый ряд.
# Глобальная модель LightGBM обучается один раз на каждую отсечку сразу на
# всей панели и предсказывает все ряды.
# ===========================================================================

import time

import numpy as np
import pandas as pd

from src import features as features_module
from src import models_forecast
from src import utils


def select_stratified_territory_sample(consumption, number_of_territories,
                                       territory_dictionary, random_seed):
    """
    Выбрать представительную выборку территорий для "турнира моделей".

    Стратификация идёт по двум осям:
        1) регион (чтобы в выборке были разные части страны),
        2) уровень расходов (чтобы были и крупные, и небольшие МО).

    Так выборка не смещена в сторону только крупных городов или одного региона.

    Возвращает отсортированный список territory_id.
    """
    import numpy as np

    random_generator = np.random.default_rng(random_seed)

    # Берём средние расходы по категории "Все категории" для каждой территории
    # как меру "размера" МО.
    # Для выбора выборки доступны только первые шесть месяцев.
    # Будущие расходы не должны определять состав проверяемых МО.
    sample_end = consumption["month"].min() + 5
    known_consumption = consumption[consumption["month"] <= sample_end]
    total_category_rows = known_consumption[
        known_consumption["category"] == "Все категории"
    ]
    mean_spending_per_territory = (
        total_category_rows.groupby("territory_id")["value"]
        .mean()
        .reset_index(name="mean_spending")
    )

    # Присоединяем регион территории.
    region_of_territory = territory_dictionary[[
        "territory_id",
        "region_code",
    ]]
    territory_table = mean_spending_per_territory.merge(
        region_of_territory, on="territory_id", how="left"
    )

    # Делим территории на три группы по уровню расходов (низкий/средний/высокий).
    territory_table["spending_group"] = pd.qcut(
        territory_table["mean_spending"], q=3, labels=[
            "низкий",
            "средний",
            "высокий",
        ]
    )

    # Сколько территорий брать из каждой комбинации (регион x группа расходов).
    # Чтобы покрыть разные регионы, берём пропорционально, но не меньше базовой доли.
    all_territory_ids = territory_table["territory_id"].tolist()
    total_available = len(all_territory_ids)

    if number_of_territories >= total_available:
        # Если запрошено больше, чем есть — берём все.
        selected_ids = sorted(all_territory_ids)
        return selected_ids

    # Формируем страты и набираем из них примерно поровну.
    selected_ids = []
    list_of_strata = []
    for region_value, region_subset in territory_table.groupby("region_code"):
        for group_value, group_subset in region_subset.groupby("spending_group", observed=True):
            if len(group_subset) > 0:
                stratum_ids = group_subset["territory_id"].tolist()
                random_generator.shuffle(stratum_ids)
                list_of_strata.append(stratum_ids)

    # Перемешиваем порядок страт для честного распределения квоты.
    random_generator.shuffle(list_of_strata)

    # Набираем по одному id из каждой страты по кругу, пока не наберём нужное число.
    position_in_each_stratum = {}
    for stratum_index in range(len(list_of_strata)):
        position_in_each_stratum[stratum_index] = 0

    keep_going = True
    while keep_going and len(selected_ids) < number_of_territories:
        added_in_this_round = 0
        for stratum_index in range(len(list_of_strata)):
            if len(selected_ids) >= number_of_territories:
                break
            stratum_ids = list_of_strata[stratum_index]
            position = position_in_each_stratum[stratum_index]
            if position < len(stratum_ids):
                selected_ids.append(stratum_ids[position])
                position_in_each_stratum[stratum_index] = position + 1
                added_in_this_round = added_in_this_round + 1
        if added_in_this_round == 0:
            keep_going = False

    selected_ids = sorted(selected_ids)
    return selected_ids


def build_list_of_cutoffs(all_months, min_train_months, origin_step):
    """
    Построить список месяцев-отсечек для бэктеста.

    all_months : список всех месяцев (в порядке возрастания).
    min_train_months : минимальная длина обучающей выборки.
    origin_step : шаг между отсечками.

    Отсечка допустима, если до неё включительно есть не меньше
    min_train_months месяцев и после неё есть хотя бы один месяц (для
    проверки прогноза хотя бы на горизонт 1).
    """
    list_of_months = list(all_months)
    total_number_of_months = len(list_of_months)

    list_of_cutoffs = []
    first_cutoff_index = min_train_months - 1
    last_cutoff_index = total_number_of_months - 2  # нужен хотя бы один месяц после

    cutoff_index = first_cutoff_index
    while cutoff_index <= last_cutoff_index:
        list_of_cutoffs.append(list_of_months[cutoff_index])
        cutoff_index = cutoff_index + origin_step

    return list_of_cutoffs


def valid_horizons_for_cutoff(cutoff_month, all_months, list_of_horizons):
    """
    Вернуть список горизонтов, для которых существует фактическое значение
    после данной отсечки (то есть прогноз можно проверить).
    """
    list_of_months = list(all_months)
    index_of_cutoff = list_of_months.index(cutoff_month)
    last_index = len(list_of_months) - 1

    horizons_that_fit = []
    for horizon in list_of_horizons:
        if index_of_cutoff + horizon <= last_index:
            horizons_that_fit.append(horizon)
    return horizons_that_fit


# ---------------------------------------------------------------------------
# Бэктест пообъектных моделей на выборке рядов
# ---------------------------------------------------------------------------

def run_per_series_backtest(sample_wide_matrix, all_months, list_of_cutoffs,
                            list_of_horizons, models_to_run, forecast_config,
                            progress_log_path=None):
    """
    Прогнать пообъектные модели (SeasonalNaive, Prophet, ETS) по выборке рядов.

    sample_wide_matrix : широкая матрица значений только для рядов выборки
        (индекс territory_id, category; столбцы — месяцы).
    list_of_cutoffs : список месяцев-отсечек.
    models_to_run : словарь флагов из конфига (какие модели запускать).
    Возвращает длинную таблицу результатов с колонками:
        territory_id, category, model, cutoff_month, horizon, actual, predicted.
    """
    list_of_months = list(all_months)
    prophet_parameters = forecast_config["prophet_params"]
    ets_parameters = forecast_config["ets_params"]

    collected_result_rows = []
    number_of_series = len(sample_wide_matrix)

    series_counter = 0
    for series_key, row_of_values in sample_wide_matrix.iterrows():
        series_counter = series_counter + 1
        territory_id_value = series_key[0]
        category_value = series_key[1]

        # Периодически пишем прогресс в лог-файл, чтобы следить за ходом.
        if progress_log_path is not None and series_counter % 20 == 0:
            _append_progress(
                progress_log_path,
                "per-series: обработано рядов "
                + str(series_counter) + " из " + str(number_of_series),
            )

        for cutoff_month in list_of_cutoffs:
            index_of_cutoff = list_of_months.index(cutoff_month)

            # История ряда до отсечки включительно.
            months_up_to_cutoff = list_of_months[0: index_of_cutoff + 1]
            history_values = row_of_values[months_up_to_cutoff].values

            # Пропускаем, если в истории есть пропуски (неполный ряд).
            if np.any(pd.isna(history_values)):
                continue

            horizons_here = valid_horizons_for_cutoff(
                cutoff_month, all_months, list_of_horizons
            )
            if len(horizons_here) == 0:
                continue

            # ---- SeasonalNaive -------------------------------------------
            if models_to_run["seasonal_naive"]:
                predictions = models_forecast.forecast_seasonal_naive(
                    history_values, horizons_here
                )
                _record_predictions(
                    collected_result_rows, territory_id_value, category_value,
                    "SeasonalNaive", cutoff_month, predictions,
                    row_of_values, list_of_months, index_of_cutoff,
                )

            # ---- Prophet -------------------------------------------------
            if models_to_run["prophet"]:
                predictions = models_forecast.forecast_prophet(
                    months_up_to_cutoff, history_values, horizons_here,
                    prophet_parameters,
                )
                _record_predictions(
                    collected_result_rows, territory_id_value, category_value,
                    "Prophet", cutoff_month, predictions,
                    row_of_values, list_of_months, index_of_cutoff,
                )

            # ---- ETS -----------------------------------------------------
            if models_to_run["ets"]:
                predictions = models_forecast.forecast_ets(
                    history_values, horizons_here, ets_parameters
                )
                _record_predictions(
                    collected_result_rows, territory_id_value, category_value,
                    "ETS", cutoff_month, predictions,
                    row_of_values, list_of_months, index_of_cutoff,
                )

    results_table = pd.DataFrame(collected_result_rows)
    return results_table


def _record_predictions(collected_result_rows, territory_id_value,
                        category_value, model_name, cutoff_month,
                        predictions_by_horizon, row_of_values,
                        list_of_months, index_of_cutoff):
    """
    Вспомогательная функция: сопоставить прогнозы с фактами и добавить строки
    результата в общий список.
    """
    for horizon, predicted_value in predictions_by_horizon.items():
        index_of_target = index_of_cutoff + horizon
        if index_of_target > len(list_of_months) - 1:
            continue
        target_month = list_of_months[index_of_target]
        actual_value = row_of_values[target_month]

        if pd.isna(actual_value):
            continue

        one_row = {
            "territory_id": territory_id_value,
            "category": category_value,
            "model": model_name,
            "cutoff_month": str(cutoff_month),
            "horizon": horizon,
            "actual": float(actual_value),
            "predicted": float(predicted_value),
        }
        collected_result_rows.append(one_row)


# ---------------------------------------------------------------------------
# Бэктест глобальной модели LightGBM
# ---------------------------------------------------------------------------

def train_one_step_global_model(full_wide_matrix, static_features,
                                calendar_table, all_months, last_train_month,
                                forecast_config, news_features_table=None,
                                forecaster_factory=None):
    """
    Обучить глобальную одношаговую модель LightGBM (прогноз на 1 месяц вперёд)
    на всех рядах панели, используя только данные до месяца last_train_month
    включительно (без заглядывания в будущее).

    Обучающие примеры: для каждого допустимого месяца-отсечки origin_train
    признаки на origin_train и цель — значение в origin_train + 1.
    Берутся только те origin_train, для которых цель известна (origin+1 не
    позже last_train_month).

    Возвращает обученный объект GlobalLightGbmForecaster (ключ горизонта = 1).
    """
    list_of_months = list(all_months)
    lightgbm_parameters = forecast_config["lightgbm_params"]
    use_log1p = forecast_config["target"]["use_log1p"]
    minimum_train_origin_index = forecast_config["backtest"]["min_train_months"] - 1

    index_of_last_train_month = list_of_months.index(last_train_month)
    # Последняя отсечка обучения такая, что цель (origin+1) не позже обучающего
    # горизонта данных.
    last_train_origin_index = index_of_last_train_month - 1

    training_tables = []
    train_origin_index = minimum_train_origin_index
    while train_origin_index <= last_train_origin_index:
        origin_train_month = list_of_months[train_origin_index]
        features_for_this_origin = features_module.build_features_at_origin(
            full_wide_matrix, static_features, calendar_table,
            origin_train_month, 1, forecast_config, news_features_table,
        )
        training_tables.append(features_for_this_origin)
        train_origin_index = train_origin_index + 1

    if len(training_tables) == 0:
        return None, None

    training_features_table = pd.concat(training_tables, axis=0)
    feature_column_names = features_module.get_feature_column_names(
        training_features_table
    )

    # По умолчанию — LightGBM (поведение не меняется). Фабрика позволяет
    # подключить другую глобальную модель (например CatBoost) тем же протоколом.
    if forecaster_factory is None:
        forecaster = models_forecast.GlobalLightGbmForecaster(
            lightgbm_parameters, use_log1p
        )
    else:
        forecaster = forecaster_factory(use_log1p)
    forecaster.fit_for_horizon(training_features_table, feature_column_names, 1)
    return forecaster, feature_column_names


def recursive_global_forecast(forecaster, working_wide_matrix, static_features,
                              calendar_table, origin_month, maximum_horizon,
                              forecast_config, news_features_table=None):
    """
    Рекурсивно развернуть одношаговую глобальную модель на maximum_horizon
    месяцев вперёд от origin_month.

    Идея: модель умеет прогнозировать только на 1 месяц вперёд. Чтобы получить
    прогноз на 2 месяца, мы берём предсказанное значение за 1-й месяц, дописываем
    его в рабочую матрицу как будто это факт, и снова прогнозируем на 1 месяц.
    Повторяем нужное число раз. Так накапливается прогноз на все горизонты.

    working_wide_matrix : рабочая КОПИЯ широкой матрицы; будет дополняться
        предсказанными столбцами будущих месяцев. Должна содержать столбцы
        вплоть до origin_month.
    Возвращает словарь: горизонт -> Series прогнозов по всем рядам матрицы.
    """
    # Работаем на копии, чтобы не портить исходную матрицу.
    working_matrix = working_wide_matrix.loc[:, :origin_month].copy()

    # В рекурсии момент знания новостей не сдвигается в будущее.
    # Для будущих шагов продолжаем последний известный региональный сигнал.
    available_news = None
    if news_features_table is not None:
        news_months = pd.PeriodIndex(news_features_table["month"].astype(str), freq="M")
        available_news = news_features_table[news_months <= origin_month].copy()
        future_news = []
        for region, history in available_news.groupby("region_code"):
            last_news = history.sort_values("month").iloc[-1]
            for step in range(1, maximum_horizon):
                row = last_news.copy()
                row["month"] = origin_month + step
                future_news.append(row)
        if len(future_news) > 0:
            available_news = pd.concat(
                [
                    available_news,
                    pd.DataFrame(future_news),
                ],
                ignore_index=True,
            )

    predictions_by_horizon = {}

    current_origin_month = origin_month
    for step_number in range(1, maximum_horizon + 1):
        # Строим признаки на текущую отсечку для горизонта 1.
        features_at_origin = features_module.build_features_at_origin(
            working_matrix, static_features, calendar_table,
            current_origin_month, 1, forecast_config, available_news,
        )
        one_step_predictions = forecaster.predict_for_horizon(features_at_origin, 1)

        # Месяц, на который сделан прогноз.
        forecast_month = current_origin_month + 1

        # Дописываем прогноз как новый столбец (как будто это наблюдение),
        # чтобы следующий шаг мог взять его в качестве лага.
        working_matrix[forecast_month] = one_step_predictions
        working_matrix = working_matrix.sort_index(axis=1)

        # Сохраняем прогноз этого шага (горизонт = номер шага).
        predictions_series = pd.Series(
            one_step_predictions, index=working_matrix.index
        )
        predictions_by_horizon[step_number] = predictions_series

        # Сдвигаем отсечку на шаг вперёд.
        current_origin_month = forecast_month

    return predictions_by_horizon


def run_global_model_backtest(full_wide_matrix, sample_index, static_features,
                              calendar_table, all_months, list_of_cutoffs,
                              list_of_horizons, forecast_config,
                              news_features_table=None, progress_log_path=None):
    """
    Прогнать глобальную модель LightGBM в режиме скользящего начала с
    рекурсивным одношаговым прогнозом.

    Для каждой отсечки:
        1) обучаем одношаговую модель на данных строго до отсечки;
        2) рекурсивно разворачиваем прогноз на максимальный горизонт;
        3) сравниваем прогнозы на нужных горизонтах с фактом по рядам выборки.

    Возвращает длинную таблицу результатов того же формата, что и пообъектный
    бэктест.
    """
    list_of_months = list(all_months)
    maximum_horizon = max(list_of_horizons)
    collected_result_rows = []

    for cutoff_month in list_of_cutoffs:
        horizons_here = valid_horizons_for_cutoff(
            cutoff_month, all_months, list_of_horizons
        )
        if len(horizons_here) == 0:
            continue

        if progress_log_path is not None:
            _append_progress(
                progress_log_path,
                "global LightGBM: отсечка " + str(cutoff_month),
            )

        # (1) Обучаем одношаговую модель на данных до отсечки.
        forecaster, feature_column_names = train_one_step_global_model(
            full_wide_matrix, static_features, calendar_table, all_months,
            cutoff_month, forecast_config, news_features_table,
        )
        if forecaster is None:
            continue

        # (2) Готовим рабочую матрицу: только столбцы до отсечки включительно,
        #     чтобы рекурсия не "подсмотрела" будущие факты.
        index_of_cutoff = list_of_months.index(cutoff_month)
        months_up_to_cutoff = list_of_months[0: index_of_cutoff + 1]
        working_matrix = full_wide_matrix[months_up_to_cutoff].copy()

        # (3) Рекурсивно разворачиваем прогноз на максимальный горизонт.
        #     Ограничиваем горизонт тем, что есть факт для проверки.
        max_horizon_here = max(horizons_here)
        predictions_by_horizon = recursive_global_forecast(
            forecaster, working_matrix, static_features, calendar_table,
            cutoff_month, max_horizon_here, forecast_config, news_features_table,
        )

        # (4) Записываем результаты по нужным горизонтам и рядам выборки.
        for horizon in horizons_here:
            if horizon not in predictions_by_horizon:
                continue
            predictions_series = predictions_by_horizon[horizon]
            target_month = cutoff_month + horizon

            for series_key in sample_index:
                predicted_value = predictions_series.loc[series_key]
                if target_month in full_wide_matrix.columns:
                    actual_value = full_wide_matrix.loc[series_key, target_month]
                else:
                    actual_value = np.nan
                if pd.isna(actual_value):
                    continue

                one_row = {
                    "territory_id": series_key[0],
                    "category": series_key[1],
                    "model": "LightGBM_global",
                    "cutoff_month": str(cutoff_month),
                    "horizon": horizon,
                    "actual": float(actual_value),
                    "predicted": float(predicted_value),
                }
                collected_result_rows.append(one_row)

    results_table = pd.DataFrame(collected_result_rows)
    return results_table


# ---------------------------------------------------------------------------
# Подсчёт метрик по результатам бэктеста
# ---------------------------------------------------------------------------

def compute_metrics_by_model_and_horizon(results_table):
    """
    По длинной таблице результатов (факт/прогноз) посчитать MAE, R^2 и sMAPE
    для каждой пары (модель, горизонт).

    Возвращает таблицу с колонками:
        model, horizon, MAE, R2, sMAPE, n_points.
    """
    collected_metric_rows = []

    list_of_models = sorted(results_table["model"].unique().tolist())
    list_of_horizons = sorted(results_table["horizon"].unique().tolist())

    for model_name in list_of_models:
        for horizon in list_of_horizons:
            subset = results_table[
                (results_table["model"] == model_name)
                & (results_table["horizon"] == horizon)
            ]
            if len(subset) == 0:
                continue

            metrics = utils.compute_all_metrics(
                subset["actual"].values, subset["predicted"].values
            )
            one_row = {
                "model": model_name,
                "horizon": horizon,
                "MAE": metrics["MAE"],
                "R2": metrics["R2"],
                "sMAPE": metrics["sMAPE"],
                "n_points": metrics["n_points"],
            }
            collected_metric_rows.append(one_row)

    metrics_table = pd.DataFrame(collected_metric_rows)
    metrics_table = metrics_table.sort_values(
        by=[
            "horizon",
            "MAE",
        ]
    ).reset_index(drop=True)
    return metrics_table


def filter_to_common_prediction_keys(results_table):
    """
    Оставить только те точки прогноза (территория, категория, отсечка,
    горизонт), которые есть у ВСЕХ моделей. Это нужно для честного сравнения:
    все модели оцениваются на одном и том же множестве прогнозов.

    Возвращает отфильтрованную таблицу.
    """
    list_of_models = sorted(results_table["model"].unique().tolist())
    number_of_models = len(list_of_models)

    # Ключ точки прогноза.
    key_columns = [
        "territory_id",
        "category",
        "cutoff_month",
        "horizon",
    ]
    if results_table.duplicated(key_columns + [
        "model",
    ]).any():
        raise ValueError("Одна модель записала несколько прогнозов для одного ключа.")
    if not np.isfinite(results_table[[
        "actual",
        "predicted",
    ]]).all().all():
        raise ValueError("Для сравнения нужны конечные факты и прогнозы.")
    facts_per_key = results_table.groupby(key_columns)["actual"].nunique()
    if (facts_per_key != 1).any():
        raise ValueError("Модели сравниваются с разными фактами.")

    # Считаем, сколькими моделями покрыта каждая точка.
    coverage_count = (
        results_table.groupby(key_columns)["model"].nunique().reset_index(
            name="models_covering"
        )
    )
    # Оставляем точки, покрытые всеми моделями.
    fully_covered_keys = coverage_count[
        coverage_count["models_covering"] == number_of_models
    ][key_columns]

    aligned_results = results_table.merge(
        fully_covered_keys, on=key_columns, how="inner"
    )
    return aligned_results


def _append_progress(progress_log_path, message_text):
    """
    Дописать строку с отметкой времени в лог-файл прогресса.
    """
    current_time_text = time.strftime("%H:%M:%S")
    line_to_write = "[" + current_time_text + "] " + message_text + "\n"
    with open(progress_log_path, "a", encoding="utf-8") as opened_file:
        opened_file.write(line_to_write)

# ===========================================================================
# Согласование новостей и датированных событий с данными СберИндекса.
#
# Это ответ на критерий ТЗ "интеграция новостей и корректный способ
# согласования новостных данных с данными СберИндекса".
#
# Три принципа корректного согласования:
#
#   1) ВРЕМЕННАЯ привязка.
#      Данные расходов — месячные. Поэтому новости/события агрегируются до
#      месяца. Каждое событие относится к месяцу своей даты. При желании
#      можно учесть лаг влияния (эффект проявляется в том же или следующем
#      месяце) — параметр effect_lag_months.
#
#   2) ПРОСТРАНСТВЕННАЯ привязка.
#      Федеральное событие (ставка ЦБ, курс рубля) влияет на все МО страны.
#      Региональное событие привязывается к конкретному region_code и влияет
#      только на территории этого региона. Поле scope в таблице событий задаёт
#      уровень.
#
#   3) ЗАЩИТА ОТ ЗАГЛЯДЫВАНИЯ В БУДУЩЕЕ (anti-look-ahead).
#      При прогнозе на момент отсечки t используем только те новости, которые
#      уже произошли к месяцу t включительно. Будущие новости модели недоступны
#      (иначе это была бы утечка информации из будущего).
#
# Числовой сигнал из события: знаковая интенсивность = direction * intensity.
#   direction = +1 (рост потребления), -1 (сдерживание), 0 (нейтрально);
#   intensity = сила события от 1 до 3.
# Сигнал по месяцу и региону — сумма знаковых интенсивностей применимых
# событий, затем экспоненциальное сглаживание для устойчивости.
# ===========================================================================

import numpy as np
import pandas as pd


def load_events_table(path_to_events_csv):
    """
    Загрузить таблицу событий из CSV (её готовит events_calendar).
    Возвращает таблицу с колонкой month (месячный период).
    """
    events_table = pd.read_csv(path_to_events_csv, encoding="utf-8-sig")
    events_table["month"] = pd.PeriodIndex(events_table["month"].astype("string"), freq="M")
    return events_table


def build_monthly_news_signal(events_table, list_of_month_periods,
                              list_of_region_codes, news_config):
    """
    Построить месячный новостной сигнал по каждому региону.

    Для каждой пары (месяц, регион) суммируем знаковые интенсивности
    применимых событий:
        - федеральные события применяются ко всем регионам;
        - региональные — только к своему региону.
    Затем сигнал сглаживается экспоненциально внутри каждого региона по времени.

    Возвращает таблицу с колонками: month (период), region_code, news_signal.
    """
    effect_lag_months = news_config["alignment"]["effect_lag_months"]
    exponential_halflife = news_config["signal"]["ewm_halflife_months"]

    # Готовим пустую сетку (месяц x регион), заполненную нулями.
    collected_rows = []
    for month_period in list_of_month_periods:
        for region_code_value in list_of_region_codes:
            one_row = {
                "month": month_period,
                "region_code": int(region_code_value),
                "raw_signal": 0.0,
            }
            collected_rows.append(one_row)
    grid_table = pd.DataFrame(collected_rows)

    # Индекс для быстрого доступа к строкам сетки по (месяц, регион).
    grid_table = grid_table.set_index(["month", "region_code"])

    # Проходим по всем событиям и добавляем их вклад в сетку.
    for _, event_row in events_table.iterrows():
        event_month = event_row["month"]
        # Учитываем лаг влияния: сдвигаем месяц действия события.
        effective_month = event_month + effect_lag_months

        if effective_month not in list_of_month_periods:
            # Если месяц действия вышел за пределы рассматриваемого диапазона —
            # пропускаем это событие.
            continue

        signed_intensity = float(event_row["direction"]) * float(event_row["intensity"])
        event_scope = event_row["scope"]

        if event_scope == "federal":
            # Федеральное событие добавляем всем регионам.
            for region_code_value in list_of_region_codes:
                key = (effective_month, int(region_code_value))
                grid_table.loc[key, "raw_signal"] = (
                    grid_table.loc[key, "raw_signal"] + signed_intensity
                )
        else:
            # Региональное событие добавляем только своему региону.
            region_code_value = int(event_row["region_code"])
            key = (effective_month, region_code_value)
            if key in grid_table.index:
                grid_table.loc[key, "raw_signal"] = (
                    grid_table.loc[key, "raw_signal"] + signed_intensity
                )

    grid_table = grid_table.reset_index()

    # Экспоненциальное сглаживание сигнала внутри каждого региона по времени.
    smoothed_rows = []
    for region_code_value, region_subset in grid_table.groupby("region_code"):
        region_subset = region_subset.sort_values(by="month").copy()
        smoothed_values = (
            region_subset["raw_signal"]
            .ewm(halflife=exponential_halflife)
            .mean()
        )
        region_subset["news_signal"] = smoothed_values.values
        smoothed_rows.append(region_subset)

    news_features_table = pd.concat(smoothed_rows, axis=0)
    news_features_table = news_features_table[["month", "region_code", "news_signal"]]
    news_features_table = news_features_table.reset_index(drop=True)
    return news_features_table


def verify_no_look_ahead(news_features_table, origin_month):
    """
    Проверка-страховка от заглядывания в будущее: для прогноза на момент
    origin_month оставляем только новостной сигнал месяцев не позже origin_month.

    Возвращает отфильтрованную таблицу новостного сигнала.
    """
    allowed_rows = news_features_table[
        news_features_table["month"] <= origin_month
    ]
    return allowed_rows


def save_news_features(news_features_table, path_to_output_csv):
    """
    Сохранить новостной признак в CSV (месяц как строка).
    """
    table_to_save = news_features_table.copy()
    table_to_save["month"] = table_to_save["month"].astype(str)
    table_to_save.to_csv(path_to_output_csv, index=False, encoding="utf-8-sig")

"""Макропризнаки из внешних официальных источников (Росстат ИПЦ).

Главный принцип — доступность во времени (anti-look-ahead):
    Мы используем ДОПУЩЕНИЕ о доступности с месячным лагом, а не проверенную
    историю реальных дат публикации. Месячный ИПЦ за месяц M публикуется
    Росстатом в начале месяца M+1, поэтому мы консервативно кладём в строку
    месяца t значение ИПЦ за месяц t-1. Историю фактических публикаций и
    ревизий (published_at по каждому выпуску) этот файл НЕ содержит; это
    допущение, а не воспроизведённый календарь публикаций.

Это согласовано с допущением проекта «данные месяца доступны на его конец»:
расходы месяца t доступны на конец t, а ИПЦ месяца t ещё не опубликован,
поэтому берём последнюю опубликованную величину (за t-1).
"""

import pandas as pd


def load_cpi_table(path_to_cpi_csv):
    """Прочитать месячный ИПЦ. Возвращает колонки month (Period[M]),
    cpi_mom_change_pct (прирост к предыдущему месяцу, %)."""
    cpi_table = pd.read_csv(path_to_cpi_csv, encoding="utf-8-sig")
    cpi_table["month"] = pd.PeriodIndex(cpi_table["month"].astype("string"), freq="M")
    cpi_table["cpi_mom_change_pct"] = cpi_table["cpi_mom_change_pct"].astype("float64")
    cpi_table = cpi_table.sort_values("month").reset_index(drop=True)
    return cpi_table


def build_cpi_signal_by_month(cpi_table, list_of_month_periods):
    """Построить доступный на каждый месяц сигнал ИПЦ со сдвигом публикации.

    Для месяца t значение сигнала = ИПЦ за месяц t-1 (последнее опубликованное
    к концу месяца t). Если данных за t-1 нет, сигнал = 0 (нейтрально).

    Возвращает словарь month(Period) -> cpi_signal(float).
    """
    value_by_month = {}
    for _, row in cpi_table.iterrows():
        value_by_month[row["month"]] = float(row["cpi_mom_change_pct"])

    signal_by_month = {}
    for month_period in list_of_month_periods:
        previous_month = month_period - 1
        if previous_month in value_by_month:
            signal_by_month[month_period] = value_by_month[previous_month]
        else:
            signal_by_month[month_period] = 0.0
    return signal_by_month


def build_macro_table(cpi_table, news_table, list_of_month_periods, regions):
    """Собрать таблицу макропризнаков в формате, который понимает модель.

    На вход: таблица ИПЦ и (необязательно) таблица новостей ЦБ с колонкой
    news_signal. На выход — таблица с колонками month, region_code,
    news_signal, cpi_signal.

    news_signal берётся из news_table (сумма изменений ставки за месяц).
    Если news_table не передан, news_signal = 0 (признак нейтрален).
    cpi_signal одинаков для всех регионов (федеральный показатель).
    """
    cpi_signal_by_month = build_cpi_signal_by_month(cpi_table, list_of_month_periods)

    # Быстрый доступ к новостному сигналу по (month, region_code).
    news_signal_lookup = {}
    if news_table is not None:
        for _, news_row in news_table.iterrows():
            key = (news_row["month"], int(news_row["region_code"]))
            news_signal_lookup[key] = float(news_row["news_signal"])

    collected_rows = []
    for month_period in list_of_month_periods:
        for region_code_value in regions:
            key = (month_period, int(region_code_value))
            if key in news_signal_lookup:
                news_value = news_signal_lookup[key]
            else:
                news_value = 0.0
            one_row = {
                "month": month_period,
                "region_code": int(region_code_value),
                "news_signal": news_value,
                "cpi_signal": cpi_signal_by_month[month_period],
            }
            collected_rows.append(one_row)

    macro_table = pd.DataFrame(collected_rows)
    return macro_table

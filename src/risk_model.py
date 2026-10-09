"""Модель риска события: вероятность шока в ближайшие 1-3 месяца.

Это ОТДЕЛЬНАЯ от детекции задача: детектор срабатывает после факта, а модель
риска оценивает вероятность будущего шока по информации, доступной на момент t.

Строго без заглядывания в будущее:
  - признаки на месяц t считаются только по данным до t включительно;
  - метка «будет шок в t+1..t+3» на обучении используется лишь там, где весь
    будущий интервал уже наступил к концу обучающего периода.

Признаки на месяц t для МО:
  - годовая лог-разность yoy_t и её изменение к t-1;
  - месячная лог-разность mom_t и mom_{t-1};
  - волатильность последних трёх месячных разностей;
  - изменение ключевой ставки ЦБ, опубликованное к месяцу t;
  - доступный на t ИПЦ (инфляция за t-1).
"""

import numpy as np
import pandas as pd


def build_risk_dataset(wide_total, shock_labels, cbr_month_change, cpi_signal_by_month,
                       horizon_months=3, onset_set=None, min_duration=2):
    """Собрать таблицу (МО, месяц) с признаками на t и меткой шока в t+1..t+h.

    wide_total: широкая матрица «Все категории» (строки МО, столбцы месяцы).
    shock_labels: таблица из shock_labeling.label_shocks (колонки
        territory_id, month, is_shock).
    cbr_month_change: dict month(str) -> изменение ставки в п.п. за месяц.
    cpi_signal_by_month: dict month(Period) -> доступный ИПЦ.
    Возвращает таблицу признаков с колонками признаков, 'label', 'territory_id',
    'month'.
    """
    month_columns = list(wide_total.columns)

    # Быстрый доступ к меткам шока.
    shock_lookup = {}
    for _, row in shock_labels.iterrows():
        shock_lookup[(int(row["territory_id"]), str(row["month"]))] = bool(row["is_shock"])

    collected_rows = []
    for territory_id in wide_total.index:
        series = wide_total.loc[territory_id]
        for position in range(len(month_columns)):
            current_month = month_columns[position]

            # Нужна годовая разность -> нужен месяц 12 назад.
            if position < 12:
                continue
            # Нужны месяцы для месячных разностей.
            if position < 1:
                continue

            current_value = series.iloc[position]
            year_ago_value = series.iloc[position - 12]
            previous_value = series.iloc[position - 1]
            if (pd.isna(current_value) or pd.isna(year_ago_value)
                    or pd.isna(previous_value)):
                continue
            if current_value <= 0 or year_ago_value <= 0 or previous_value <= 0:
                continue

            yoy_t = float(np.log(current_value) - np.log(year_ago_value))
            mom_t = float(np.log(current_value) - np.log(previous_value))

            # yoy_{t-1} и mom_{t-1}.
            if position - 13 >= 0 and not pd.isna(series.iloc[position - 13]) and series.iloc[position - 13] > 0:
                yoy_prev = float(np.log(previous_value) - np.log(series.iloc[position - 13]))
            else:
                yoy_prev = np.nan
            if position - 2 >= 0 and not pd.isna(series.iloc[position - 2]) and series.iloc[position - 2] > 0:
                mom_prev = float(np.log(previous_value) - np.log(series.iloc[position - 2]))
            else:
                mom_prev = np.nan

            # Волатильность последних трёх месячных разностей.
            recent_moms = []
            for back in range(0, 3):
                p = position - back
                if p - 1 >= 0 and not pd.isna(series.iloc[p]) and not pd.isna(series.iloc[p - 1]):
                    if series.iloc[p] > 0 and series.iloc[p - 1] > 0:
                        recent_moms.append(float(np.log(series.iloc[p]) - np.log(series.iloc[p - 1])))
            if len(recent_moms) >= 2:
                recent_volatility = float(np.std(recent_moms))
            else:
                recent_volatility = np.nan

            # Внешние признаки, доступные на t.
            rate_change = float(cbr_month_change.get(str(current_month), 0.0))
            cpi_value = float(cpi_signal_by_month.get(current_month, 0.0))

            # Метка СОСТОЯНИЯ: есть ли шок в t+1..t+h (вспомогательная).
            state_label = 0
            # Метка НАЧАЛА (onset): начинается ли НОВЫЙ эпизод в t+1..t+h.
            onset_label = 0
            for step in range(1, horizon_months + 1):
                future_position = position + step
                if future_position >= len(month_columns):
                    continue
                future_month = month_columns[future_position]
                if shock_lookup.get((int(territory_id), str(future_month)), False):
                    state_label = 1
                if onset_set is not None and (int(territory_id), str(future_month)) in onset_set:
                    onset_label = 1

            # Доступен ли полный будущий интервал в данных.
            full_future_available = (position + horizon_months) < len(month_columns)

            # Когда метка СОЗРЕВАЕТ (подтверждается): эпизод объявляется только
            # после min_duration подряд экстремальных месяцев, поэтому onset в
            # месяце s подтверждается в s+(min_duration-1). Самый поздний
            # возможный onset окна — t+h, он подтверждается в t+h+(min_duration-1).
            label_available_position = position + horizon_months + (min_duration - 1)
            if label_available_position < len(month_columns):
                label_available_at = str(month_columns[label_available_position])
                label_matured = True
            else:
                # За границей данных метку нельзя подтвердить.
                label_available_at = None
                label_matured = False

            collected_rows.append({
                "territory_id": int(territory_id),
                "month": str(current_month),
                "yoy_t": yoy_t,
                "yoy_change": (yoy_t - yoy_prev) if not np.isnan(yoy_prev) else 0.0,
                "mom_t": mom_t,
                "mom_prev": mom_prev if not np.isnan(mom_prev) else 0.0,
                "recent_volatility": recent_volatility if not np.isnan(recent_volatility) else 0.0,
                "rate_change": rate_change,
                "cpi_signal": cpi_value,
                "label": int(state_label),
                "onset_label": int(onset_label),
                "full_future_available": bool(full_future_available),
                "label_available_at": label_available_at,
                "label_matured": bool(label_matured),
            })

    return pd.DataFrame(collected_rows)

"""Постановка риска с МЕНЬШИМ необходимым прогревом (заранее зафиксирована).

Зачем: прежняя разметка опиралась на годовую лог-разность (прогрев 12 месяцев),
из-за чего сигнал существовал только за 2024 год, а созревших обучающих срезов
оставался один (январь 2024). Эта постановка сокращает прогрев до окна W (≈6
месяцев), поэтому сигнал существует с конца 2023 года и обучающих временных
срезов становится много. Определение шока фиксируется заранее и НЕ подбирается
под желаемый результат.

Определение (пред-регистрируется):
  - месячная лог-разность r_t = log(y_t) - log(y_{t-1});
  - скользящая медиана baseline_t и скользящий MAD по окну [t-W, t-1] (по МО);
  - масштаб scale_t = max(1.4826 * MAD, pooled_floor), где pooled_floor —
    устойчивый нижний порог масштаба, оценённый по доступным (обучающим)
    данным, чтобы избежать вырождения;
  - отклонение z_t = (r_t - baseline_t) / scale_t;
  - месяц экстремален, если |z_t| > k (k задаётся, по умолчанию 3);
  - эпизод шока = серия из не менее MIN_DURATION подряд экстремальных месяцев;
  - onset = первый месяц такой серии.

Все величины на момент t используют только прошлое (строго до t для baseline и
масштаба), поэтому сигнал каузален.
"""

import numpy as np
import pandas as pd


def build_month_over_month_table(wide_total):
    """Длинная таблица месячной лог-разности r_t по каждому МО.

    wide_total : широкая матрица «Все категории» (строки МО, столбцы месяцы).
    Возвращает таблицу (territory_id, month(Period), r), где r определено
    (есть положительные y_t и y_{t-1}).
    """
    month_columns = list(wide_total.columns)
    collected_rows = []
    for position in range(1, len(month_columns)):
        current_month = month_columns[position]
        previous_month = month_columns[position - 1]
        current_values = wide_total[current_month]
        previous_values = wide_total[previous_month]
        for territory_id in wide_total.index:
            current_value = current_values.loc[territory_id]
            previous_value = previous_values.loc[territory_id]
            if pd.isna(current_value) or pd.isna(previous_value):
                continue
            if current_value <= 0 or previous_value <= 0:
                continue
            r_value = float(np.log(current_value) - np.log(previous_value))
            collected_rows.append({
                "territory_id": int(territory_id),
                "month": current_month,
                "r": r_value,
            })
    return pd.DataFrame(collected_rows)


def add_trailing_baseline_and_scale(mom_table, window):
    """Добавить скользящую медиану, MAD и волатильность по окну [t-W, t-1].

    Для каждого МО значения считаются по строго предыдущим W месяцам (сдвиг на 1,
    чтобы текущий месяц не влиял на собственный базовый уровень).
    Возвращает таблицу с колонками trailing_median, trailing_mad, trailing_vol;
    строки с неполным окном (меньше W прошлых точек) отбрасываются.
    """
    collected_frames = []
    for territory_id, group in mom_table.groupby("territory_id"):
        group = group.sort_values("month").reset_index(drop=True)
        r_series = group["r"]

        shifted = r_series.shift(1)
        trailing_median = shifted.rolling(window=window, min_periods=window).median()
        trailing_vol = shifted.rolling(window=window, min_periods=window).std()

        # Скользящий MAD: median(|x - median|) по окну.
        mad_values = []
        for end_position in range(len(group)):
            start_position = end_position - window
            if start_position < 0 or end_position < 1:
                mad_values.append(np.nan)
                continue
            window_values = r_series.iloc[start_position:end_position].to_numpy(dtype=float)
            if len(window_values) < window:
                mad_values.append(np.nan)
                continue
            window_median = np.median(window_values)
            mad = np.median(np.abs(window_values - window_median))
            mad_values.append(mad)

        group = group.copy()
        group["trailing_median"] = trailing_median.to_numpy()
        group["trailing_mad"] = np.asarray(mad_values, dtype=float)
        group["trailing_vol"] = trailing_vol.to_numpy()
        collected_frames.append(group)

    result = pd.concat(collected_frames, ignore_index=True)
    result = result.dropna(subset=["trailing_median", "trailing_mad"]).reset_index(drop=True)
    return result


def estimate_pooled_floor(signal_table, allowed_months):
    """Оценить нижний порог масштаба по доступным (обучающим) месяцам.

    Берём медиану робастного масштаба (1.4826 * MAD) по всем (МО, месяц) из
    allowed_months. Это устойчивая типичная величина разброса, которая не даёт
    отдельным почти-постоянным рядам раздувать z.
    """
    allowed = set()
    for one_month in allowed_months:
        allowed.add(str(one_month))
    subset = signal_table[signal_table["month"].astype(str).isin(allowed)]
    robust_scales = 1.4826 * subset["trailing_mad"].to_numpy(dtype=float)
    robust_scales = robust_scales[robust_scales > 0]
    if len(robust_scales) == 0:
        return 1e-6
    return float(np.median(robust_scales))


def add_deviation_z(signal_table, pooled_floor):
    """Добавить отклонение z_t = (r - baseline) / max(1.4826*MAD, floor)."""
    table = signal_table.copy()
    robust_scale = 1.4826 * table["trailing_mad"].to_numpy(dtype=float)
    floored_scale = np.maximum(robust_scale, pooled_floor)
    table["scale"] = floored_scale
    table["z"] = (table["r"].to_numpy(dtype=float) - table["trailing_median"].to_numpy(dtype=float)) / floored_scale
    return table


def label_onsets(signal_with_z, threshold_k, min_duration):
    """Разметить экстремальные месяцы, эпизоды и onset-месяцы.

    Возвращает (таблицу с колонками is_extreme/is_shock, множество onset-пар
    (territory_id, month_str)).
    """
    table = signal_with_z.copy()
    table["is_extreme"] = np.abs(table["z"].to_numpy(dtype=float)) > threshold_k
    table = table.sort_values(["territory_id", "month"]).reset_index(drop=True)
    table["is_shock"] = False

    onset_set = set()
    for territory_id, group in table.groupby("territory_id"):
        indices = group.index.tolist()
        extreme_flags = group["is_extreme"].tolist()
        month_list = group["month"].tolist()
        run_length = 0
        run_start = 0
        for position in range(len(extreme_flags)):
            if extreme_flags[position]:
                if run_length == 0:
                    run_start = position
                run_length += 1
            else:
                if run_length >= min_duration:
                    for mark in range(run_start, position):
                        table.loc[indices[mark], "is_shock"] = True
                    onset_set.add((int(territory_id), str(month_list[run_start])))
                run_length = 0
        if run_length >= min_duration:
            for mark in range(run_start, len(extreme_flags)):
                table.loc[indices[mark], "is_shock"] = True
            onset_set.add((int(territory_id), str(month_list[run_start])))

    return table, onset_set


def build_features_and_labels(signal_with_z, onset_set, cbr_month_change,
                              cpi_signal_by_month, horizon, min_duration):
    """Собрать признаки на месяц t и метку начала шока в t+1..t+horizon.

    Признаки на t: z_t, r_t, trailing_vol_t, изменение ставки ЦБ, ИПЦ.
    label_available_at = t + horizon + (min_duration-1) — месяц подтверждения.
    is_onset_month помечает сам месяц начала эпизода (для событийной оценки).
    """
    onset_lookup = set(onset_set)
    table = signal_with_z.sort_values(["territory_id", "month"]).reset_index(drop=True)

    collected_rows = []
    for territory_id, group in table.groupby("territory_id"):
        group = group.sort_values("month").reset_index(drop=True)
        months_here = list(group["month"])
        month_to_index = {}
        for index_value in range(len(months_here)):
            month_to_index[str(months_here[index_value])] = index_value

        for position in range(len(group)):
            current_month = months_here[position]

            onset_label = 0
            for step in range(1, horizon + 1):
                future_month = current_month + step
                if (int(territory_id), str(future_month)) in onset_lookup:
                    onset_label = 1
                    break

            label_available_month = current_month + horizon + (min_duration - 1)
            label_matured = True  # созревание проверяется относительно данных ниже

            cbr_value = float(cbr_month_change.get(str(current_month), 0.0))
            cpi_value = float(cpi_signal_by_month.get(current_month, 0.0))

            collected_rows.append({
                "territory_id": int(territory_id),
                "month": str(current_month),
                "z": float(group.loc[position, "z"]),
                "r": float(group.loc[position, "r"]),
                "trailing_vol": float(group.loc[position, "trailing_vol"]) if pd.notna(group.loc[position, "trailing_vol"]) else 0.0,
                "rate_change": cbr_value,
                "cpi_signal": cpi_value,
                "onset_label": int(onset_label),
                "is_onset_month": (int(territory_id), str(current_month)) in onset_lookup,
                "label_available_at": str(label_available_month),
            })

    return pd.DataFrame(collected_rows)

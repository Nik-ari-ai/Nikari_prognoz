"""Прикладное целевое событие: шок как отклонение МО от собственного сезонного
тренда И общего движения (сезонно-нормализованный шок). Каузальная разметка.

Мотивация (приоритет аудита): прежние целевые события либо смещены к декабрю
(месячная разметка), либо редки (level-shift). Здесь целевое событие ближе к
прикладному смыслу: шок = месяц, в котором расходы МО ведут себя НЕОБЫЧНО
относительно (а) общего движения всех МО в этот месяц и (б) собственного
типичного уровня МО. Декабрьские календарные пики исключаются автоматически,
потому что общий декабрьский скачок входит в «общее движение» и вычитается.

Определение (каузальное, пред-регистрируется параметрами из reference period):
  - месячная лог-разность r_mo(t) = log y_mo(t) - log y_mo(t-1);
  - ОБЩЕЕ движение c(t) = медиана r по всем МО в месяц t (контемпоральная
    кросс-секционная нормализация: убирает общий для всех МО сезонный скачок);
  - идиосинкратическое отклонение d_mo(t) = r_mo(t) - c(t);
  - центрирование и масштаб по МО оцениваются ТОЛЬКО по reference period
    (строго до тестового периода): mean_ref, scale_ref = max(1.4826*MAD, floor);
  - нормированное отклонение z_mo(t) = (d_mo(t) - mean_ref) / scale_ref;
  - пороги-квантили берутся из reference-распределения |z| (пред-регистрируются
    уровни квантилей): q_persist (ниже) и q_single (выше);
  - ПОЛОЖИТЕЛЬНЫЙ шок: z > q_persist подряд >= persistence месяцев, ИЛИ
    z > q_single в один месяц; ОТРИЦАТЕЛЬНЫЙ — симметрично по z < -q;
  - onset = первый месяц такого шока (раздельно для + и -).

Каузальность: все ПАРАМЕТРЫ (mean, scale, квантильные пороги) оценены по данным
строго до теста. c(t) — контемпоральный агрегат месяца t, часть определения
события (как факт используется для ground truth), а не настраиваемый по тесту
параметр.
"""

import numpy as np
import pandas as pd


def build_month_over_month_table(wide_total):
    """Длинная таблица месячной лог-разности r_mo(t) по каждому МО."""
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


def compute_common_movement(mom_table):
    """Общее движение c(t) = медиана r по всем МО в каждый месяц t."""
    common_by_month = {}
    for month_value, group in mom_table.groupby("month"):
        common_by_month[str(month_value)] = float(group["r"].median())
    return common_by_month


def build_seasonal_signal(mom_table, common_by_month, reference_cutoff):
    """Построить нормированное отклонение z_mo(t) с каузальными параметрами.

    reference_cutoff : месяц (строка), по который включительно оцениваются
                       mean и scale каждого МО (строго до тестового периода).
    Возвращает таблицу (territory_id, month, r, d, z) и pooled_floor.
    """
    cutoff_period = pd.Period(str(reference_cutoff), freq="M")

    table = mom_table.copy()
    deviations = []
    for _, row in table.iterrows():
        deviations.append(row["r"] - common_by_month.get(str(row["month"]), 0.0))
    table["d"] = np.asarray(deviations, dtype=float)

    # Pooled floor масштаба: медиана по МО робастного масштаба d на reference.
    robust_scales_reference = []
    reference_mask = []
    for _, row in table.iterrows():
        reference_mask.append(pd.Period(str(row["month"]), freq="M") <= cutoff_period)
    table["_is_reference"] = np.asarray(reference_mask, dtype=bool)

    for territory_id, group in table.groupby("territory_id"):
        reference_group = group[group["_is_reference"]]
        if len(reference_group) >= 2:
            median_d = float(np.median(reference_group["d"].to_numpy(dtype=float)))
            mad = float(np.median(np.abs(reference_group["d"].to_numpy(dtype=float) - median_d)))
            robust_scale = 1.4826 * mad
            if robust_scale > 0:
                robust_scales_reference.append(robust_scale)
    if len(robust_scales_reference) > 0:
        pooled_floor = float(np.median(robust_scales_reference))
    else:
        pooled_floor = 1e-6
    if pooled_floor <= 0:
        pooled_floor = 1e-6

    # Центрирование и масштаб по МО из reference period.
    collected_frames = []
    for territory_id, group in table.groupby("territory_id"):
        group = group.sort_values("month").reset_index(drop=True)
        reference_group = group[group["_is_reference"]]
        if len(reference_group) >= 2:
            mean_reference = float(np.median(reference_group["d"].to_numpy(dtype=float)))
            median_d = mean_reference
            mad = float(np.median(np.abs(reference_group["d"].to_numpy(dtype=float) - median_d)))
            scale_reference = max(1.4826 * mad, pooled_floor)
        else:
            mean_reference = 0.0
            scale_reference = pooled_floor
        group = group.copy()
        group["z"] = (group["d"].to_numpy(dtype=float) - mean_reference) / scale_reference
        collected_frames.append(group)

    result = pd.concat(collected_frames, ignore_index=True)
    result = result.drop(columns=["_is_reference"])
    return result, pooled_floor


def estimate_quantile_thresholds(signal_table, reference_cutoff, persist_quantile, single_quantile):
    """Пороги-квантили по reference-распределению |z| (каузально, до теста)."""
    cutoff_period = pd.Period(str(reference_cutoff), freq="M")
    reference_values = []
    for _, row in signal_table.iterrows():
        if pd.Period(str(row["month"]), freq="M") <= cutoff_period:
            reference_values.append(abs(float(row["z"])))
    reference_values = np.asarray(reference_values, dtype=float)
    if len(reference_values) == 0:
        return 0.0, 0.0
    q_persist = float(np.quantile(reference_values, persist_quantile))
    q_single = float(np.quantile(reference_values, single_quantile))
    return q_persist, q_single


def label_seasonal_shock_onsets(signal_table, q_persist, q_single, persistence):
    """Разметить onset положительных и отрицательных сезонных шоков.

    Возвращает (onset_set_positive, onset_set_negative, onset_set_any).
    Триггеры: z > q_persist подряд >= persistence месяцев, ИЛИ z > q_single в
    один месяц (положительный шок); отрицательный — симметрично по -z.
    """
    table = signal_table.sort_values(["territory_id", "month"]).reset_index(drop=True)

    onset_positive = set()
    onset_negative = set()

    for territory_id, group in table.groupby("territory_id"):
        group = group.sort_values("month").reset_index(drop=True)
        z_values = group["z"].to_numpy(dtype=float)
        month_list = list(group["month"])

        _collect_onsets_one_direction(
            z_values, month_list, territory_id, q_persist, q_single, persistence,
            positive_direction=True, onset_container=onset_positive
        )
        _collect_onsets_one_direction(
            z_values, month_list, territory_id, q_persist, q_single, persistence,
            positive_direction=False, onset_container=onset_negative
        )

    onset_any = set()
    for one_pair in onset_positive:
        onset_any.add(one_pair)
    for one_pair in onset_negative:
        onset_any.add(one_pair)
    return onset_positive, onset_negative, onset_any


def _collect_onsets_one_direction(z_values, month_list, territory_id, q_persist,
                                  q_single, persistence, positive_direction, onset_container):
    """Отметить onset шоков одного знака для одного МО."""
    if positive_direction:
        is_persist_month = z_values > q_persist
        is_single_month = z_values > q_single
    else:
        is_persist_month = z_values < -q_persist
        is_single_month = z_values < -q_single

    is_shock_month = [False] * len(z_values)

    # Персистентный триггер: серия >= persistence подряд превышений q_persist.
    run_length = 0
    run_start = 0
    for position in range(len(z_values)):
        if is_persist_month[position]:
            if run_length == 0:
                run_start = position
            run_length += 1
        else:
            if run_length >= persistence:
                for mark in range(run_start, position):
                    is_shock_month[mark] = True
            run_length = 0
    if run_length >= persistence:
        for mark in range(run_start, len(z_values)):
            is_shock_month[mark] = True

    # Одномоментный триггер: превышение более высокого квантиля q_single.
    for position in range(len(z_values)):
        if is_single_month[position]:
            is_shock_month[position] = True

    # onset = первый месяц шока (предыдущий месяц не был шоком).
    for position in range(len(z_values)):
        if not is_shock_month[position]:
            continue
        previous_is_shock = position > 0 and is_shock_month[position - 1]
        if not previous_is_shock:
            onset_container.add((int(territory_id), str(month_list[position])))

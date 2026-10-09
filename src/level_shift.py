"""Проверяемое определение целевого изменения: УСТОЙЧИВЫЙ СДВИГ УРОВНЯ.

Зачем: месячная разметка scripts/15 ловит импульс-возврат (всплеск с откатом),
во многом декабрьский календарный эффект. Аудит справедливо указал, что это не
то же самое, что устойчивое структурное изменение уровня расходов. Здесь целевое
событие задано ЯВНО и проверяется на синтетике:

  - ПОСТОЯННЫЙ скачок уровня (1000 -> 2000 и далее 2000) ДОЛЖЕН размечаться;
  - одноразовый всплеск с возвратом (1000 -> 2000 -> 1000) НЕ должен.

Определение (каузальная разметка, созревание явное):
  - trailing_level(o) = медиана y по [o-W, o-1] (строго прошлое);
  - месяц o считается «сдвинутым», если ВСЕ P месяцев окна [o, o+P-1] отклонены
    от trailing_level в ОДНУ сторону больше порога: для всех j в [o, o+P-1]
    log(y[j]/trailing_level) > k*scale (сдвиг вверх) ИЛИ < -k*scale (вниз);
  - onset = первый месяц устойчивого сдвига (месяц o «сдвинут», а o-1 — нет).

Требование «все P месяцев в одну сторону» отбрасывает импульс-возврат: во второй
месяц ряд уже вернулся, поэтому окно не единонаправлено.

Метка onset в o наблюдается полностью при o+P-1 <= последний месяц данных.
"""

import numpy as np
import pandas as pd


def estimate_level_scale_floor(wide_total, reference_cutoff=None):
    """Устойчивый нижний порог масштаба в лог-шкале по типичной месячной дисперсии.

    Берём медиану по МО от стандартного отклонения месячных лог-разностей. Это
    не даёт почти-постоянным рядам раздувать сдвиг и задаёт масштаб там, где
    собственная дисперсия уровня близка к нулю.

    ПОПРАВКА АУДИТА v9 (критичная для каузальности): масштаб — это параметр самого
    определения целевого события, поэтому он НЕ должен зависеть от тестового
    будущего. `reference_cutoff` (месяц, например "2023-12") ограничивает оценку
    масштаба данными строго ДО тестового периода. Если он не задан, используется
    вся история — это лишь описательная (transductive) разметка, НЕ каузальная.
    """
    month_columns = list(wide_total.columns)
    if reference_cutoff is not None:
        cutoff_period = pd.Period(str(reference_cutoff), freq="M")
        allowed_columns = []
        for month_column in month_columns:
            if pd.Period(str(month_column), freq="M") <= cutoff_period:
                allowed_columns.append(month_column)
    else:
        allowed_columns = month_columns

    reference_frame = wide_total[allowed_columns]
    per_territory_dispersion = []
    for territory_id in reference_frame.index:
        series_values = reference_frame.loc[territory_id].to_numpy(dtype=float)
        log_changes = []
        for position in range(1, len(allowed_columns)):
            current_value = series_values[position]
            previous_value = series_values[position - 1]
            if current_value > 0 and previous_value > 0:
                log_changes.append(np.log(current_value) - np.log(previous_value))
        if len(log_changes) >= 2:
            per_territory_dispersion.append(float(np.std(log_changes)))
    if len(per_territory_dispersion) == 0:
        return 1e-6
    floor_value = float(np.median(per_territory_dispersion))
    if floor_value <= 0:
        return 1e-6
    return floor_value


def label_level_shift_onsets(wide_total, window, persistence, threshold_k, scale_floor):
    """Разметить onset устойчивых сдвигов уровня по каждому МО.

    wide_total  : широкая матрица «Все категории» (строки МО, столбцы месяцы).
    window      : W — длина прошлого окна для базового уровня.
    persistence : P — сколько месяцев подряд сдвиг должен держаться в одну сторону.
    threshold_k : порог в единицах масштаба.
    scale_floor : нижний порог масштаба (из estimate_level_scale_floor).

    Возвращает (onset_set, last_observed_month). onset_set — множество пар
    (territory_id, month_str) начала устойчивого сдвига.
    """
    month_columns = list(wide_total.columns)
    last_observed_month = month_columns[-1]
    onset_set = set()

    for territory_id in wide_total.index:
        series_values = wide_total.loc[territory_id].to_numpy(dtype=float)

        # Для каждого месяца o считаем, «сдвинут» ли он (устойчиво, в одну сторону).
        is_shift_month = [False] * len(month_columns)
        for origin_position in range(len(month_columns)):
            trailing_start = origin_position - window
            if trailing_start < 0:
                continue
            if origin_position + persistence > len(month_columns):
                continue  # окно сдвига выходит за данные — не наблюдается

            trailing_values = series_values[trailing_start:origin_position]
            if np.any(trailing_values <= 0):
                continue
            trailing_level = float(np.median(trailing_values))
            if trailing_level <= 0:
                continue

            all_up = True
            all_down = True
            for forward_step in range(persistence):
                forward_value = series_values[origin_position + forward_step]
                if forward_value <= 0:
                    all_up = False
                    all_down = False
                    break
                deviation = np.log(forward_value) - np.log(trailing_level)
                if not (deviation > threshold_k * scale_floor):
                    all_up = False
                if not (deviation < -threshold_k * scale_floor):
                    all_down = False
            is_shift_month[origin_position] = bool(all_up or all_down)

        # onset — первый месяц устойчивого сдвига (предыдущий месяц не сдвинут).
        for origin_position in range(len(month_columns)):
            if not is_shift_month[origin_position]:
                continue
            previous_is_shift = origin_position > 0 and is_shift_month[origin_position - 1]
            if not previous_is_shift:
                onset_set.add((int(territory_id), str(month_columns[origin_position])))

    return onset_set, last_observed_month

"""Событийная оценка раннего предупреждения (общий модуль).

Вынесено из scripts/13, чтобы один и тот же проверенный оценщик использовали
и onset-модель (scripts/13), и эксперимент с базовыми правилами и rolling-origin
(scripts/15). Это убирает дублирование и риск рассинхронизации.

Поправка аудита v3 зашита здесь: полный календарь onset передаётся ОТДЕЛЬНО от
месяцев выпуска прогнозов, иначе терялись бы будущие события и ошибочно
учитывались события до первого выпуска.
"""

import numpy as np
import pandas as pd


def event_level_evaluation(release_frame, scores, threshold, horizon, onset_set,
                           alarm_mask=None):
    """Оценить предупреждения по СОБЫТИЯМ с согласованным окном.

    release_frame : таблица выпусков прогнозов (колонки territory_id, month).
    scores        : массив риск-оценок той же длины, что release_frame.
    threshold     : порог тревоги (используется, если alarm_mask не задан).
    horizon       : горизонт предупреждения (мес).
    onset_set     : множество (territory_id, month_str) — ПОЛНЫЙ календарь начал
                    эпизодов по всей истории.
    alarm_mask    : (опционально) булев массив длины scores. Если задан, тревогой
                    считается строка с True (а не score>=threshold). Нужен для
                    ДЕТЕРМИНИРОВАННОГО ровно-K отбора top-K с явным разрешением
                    совпадений оценок, чтобы бюджет тревог не раздувался.

    Правила (фиксированы заранее):
      - предупреждение = выпуск t, помеченный тревогой (alarm_mask либо score>=threshold);
      - событие o доступно для recall, если достижимо хотя бы из одного выпуска
        того же МО: 1 <= o - t <= horizon;
      - предупреждение ловит o, если 1 <= o - t <= horizon (строго до начала);
      - сопоставление один-к-одному, за событием — самое раннее предупреждение;
      - несопоставленные (в т.ч. повторные) предупреждения — ложные тревоги.

    Возвращает словарь метрик; при пустых множествах precision/recall = None.
    """
    frame = release_frame.copy().reset_index(drop=True)
    frame["score"] = np.asarray(scores, dtype="float64")
    if alarm_mask is not None:
        frame["is_alarm"] = np.asarray(alarm_mask, dtype=bool)
    else:
        frame["is_alarm"] = frame["score"].to_numpy(dtype="float64") >= threshold
    frame["month_period"] = pd.PeriodIndex(frame["month"].astype(str), freq="M")

    onset_months_by_territory = {}
    for territory_id, month_str in onset_set:
        onset_months_by_territory.setdefault(int(territory_id), []).append(
            pd.Period(month_str, freq="M")
        )

    total_hits = 0
    total_false = 0
    total_events = 0
    lead_times = []

    for territory_id, group in frame.groupby("territory_id"):
        release_months = sorted(group["month_period"].tolist())
        warning_months = sorted(
            group[group["is_alarm"]]["month_period"].tolist()
        )

        all_onsets = sorted(onset_months_by_territory.get(int(territory_id), []))
        eligible_onsets = []
        for onset_month in all_onsets:
            reachable = False
            for release_month in release_months:
                lead = (onset_month - release_month).n
                if 1 <= lead <= horizon:
                    reachable = True
                    break
            if reachable:
                eligible_onsets.append(onset_month)
        total_events += len(eligible_onsets)

        remaining_warnings = list(warning_months)
        for onset_month in eligible_onsets:
            matched = None
            for warning_month in remaining_warnings:
                lead = (onset_month - warning_month).n
                if 1 <= lead <= horizon:
                    matched = warning_month
                    break
            if matched is not None:
                total_hits += 1
                lead_times.append((onset_month - matched).n)
                remaining_warnings.remove(matched)
        total_false += len(remaining_warnings)

    if (total_hits + total_false) > 0:
        event_precision = total_hits / (total_hits + total_false)
    else:
        event_precision = None
    event_recall = total_hits / total_events if total_events > 0 else None
    false_per_100 = 100.0 * total_false / len(frame) if len(frame) > 0 else None
    mean_lead = float(np.mean(lead_times)) if len(lead_times) > 0 else None

    return {
        "event_hits": int(total_hits),
        "event_false_alarms": int(total_false),
        "onset_events_eligible": int(total_events),
        "event_precision": event_precision,
        "event_recall": event_recall,
        "false_alarms_per_100_mo_months": false_per_100,
        "mean_lead_time_months": mean_lead,
    }


def top_k_alarm_mask(scores, k, tie_break_ids):
    """Детерминированно отобрать РОВНО k тревог (top-K по оценке).

    Совпадения оценок разрешаются по возрастанию tie_break_ids (например,
    territory_id), поэтому результат воспроизводим и число тревог равно ровно
    min(k, len(scores)). Это устраняет раздувание бюджета при равных оценках
    (иначе score>=порог могло бы дать больше k тревог).

    scores        : массив риск-оценок.
    k             : требуемое число тревог.
    tie_break_ids : массив идентификаторов той же длины для разрешения равенств.

    Возвращает булев массив длины scores (True — тревога).
    """
    scores_array = np.asarray(scores, dtype="float64")
    ids_array = np.asarray(tie_break_ids)
    count = len(scores_array)
    mask = np.zeros(count, dtype=bool)

    effective_k = int(k)
    if effective_k <= 0:
        return mask
    if effective_k >= count:
        mask[:] = True
        return mask

    # Сортируем по убыванию оценки, при равенстве — по возрастанию id.
    order_positions = list(range(count))
    def sort_key(position):
        return (-scores_array[position], ids_array[position])
    order_positions.sort(key=sort_key)

    for rank in range(effective_k):
        chosen_position = order_positions[rank]
        mask[chosen_position] = True
    return mask


def threshold_for_alarm_budget(reference_scores, target_alarm_rate):
    """Подобрать порог по РАННИМ оценкам так, чтобы доля тревог была примерно
    равна target_alarm_rate. Порог выбирается на обучающих/ранних данных, не на
    тестовых метках.
    """
    reference = np.asarray(reference_scores, dtype="float64")
    if len(reference) == 0:
        return float("inf")
    quantile_level = 1.0 - target_alarm_rate
    quantile_level = min(max(quantile_level, 0.0), 1.0)
    return float(np.quantile(reference, quantile_level))

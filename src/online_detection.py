"""Три последовательных детектора. Сигнал доступен после получения факта.

Первые calibration_points наблюдений задают нормальный уровень и масштаб.
Они никогда не пересчитываются по будущим точкам. Для рядов потребления
входом должны служить ошибки прогнозов, сделанных до получения факта.
Детектор не предсказывает ещё не произошедший шок.
"""

import numpy as np
import pandas as pd


def detect_online(values, settings):
    """Вернуть моменты тревог, не изменяющиеся при добавлении будущего."""
    values = np.asarray(values, dtype=float)
    warmup = settings["calibration_points"]
    rows = []
    if len(values) <= warmup:
        return pd.DataFrame(columns=[
            "position",
            "method",
            "score",
            "alarm",
        ])
    calibration = values[:warmup]
    if not np.isfinite(calibration).all():
        raise ValueError("В калибровочном отрезке есть пропуски.")
    center = float(np.mean(calibration))
    scale = float(np.std(calibration, ddof=1))
    scale = max(scale, 1e-8)
    upper_sum = 0.0
    lower_sum = 0.0
    average = 0.0
    weight = settings["ewma_alpha"]
    drift = settings["cusum_drift"]

    for position in range(warmup, len(values)):
        if not np.isfinite(values[position]):
            raise ValueError("В последовательности есть пропуски.")
        deviation = (values[position] - center) / scale
        upper_sum = max(0.0, upper_sum + deviation - drift)
        lower_sum = min(0.0, lower_sum + deviation + drift)
        average = weight * deviation + (1.0 - weight) * average
        elapsed = position - warmup + 1
        average_scale = np.sqrt(
            weight / (2.0 - weight) * (1.0 - (1.0 - weight) ** (2 * elapsed))
        )
        scores = {
            "Threshold": abs(deviation),
            "CUSUM": max(upper_sum, abs(lower_sum)),
            "EWMA": abs(average) / average_scale,
        }
        for method, score in scores.items():
            threshold = settings["threshold"]
            if method == "CUSUM":
                threshold = settings["cusum_threshold"]
            alarm = bool(score > threshold)
            rows.append({
                "position": position,
                "method": method,
                "score": float(score),
                "alarm": alarm,
            })
            if alarm and method == "CUSUM":
                upper_sum = 0.0
                lower_sum = 0.0
            if alarm and method == "EWMA":
                average = 0.0
    return pd.DataFrame(rows)


def match_alarms(alarms, events, tolerance):
    """Сопоставить тревоги с событиями один к одному, только после события.

Тревога до события здесь является ложной. Предсказание будущего события
нужно оценивать отдельно, с явно заданным интервалом предупреждения.
"""
    unused_events = sorted(events)
    delays = []
    false_alarms = 0
    for alarm in sorted(alarms):
        matched = None
        for event in unused_events:
            if event <= alarm <= event + tolerance:
                matched = event
                break
        if matched is None:
            false_alarms = false_alarms + 1
        else:
            delays.append(alarm - matched)
            unused_events.remove(matched)
    return len(delays), false_alarms, delays

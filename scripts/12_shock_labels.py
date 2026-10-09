"""Слепая разметка реальных шоков и оценка детекторов на этих метках.

Шаги:
  1. Берём ряды «Все категории» по полным МО (есть все 24 месяца).
  2. Считаем годовую лог-разность и размечаем шоки заранее заданным правилом
     (src/shock_labeling.py) — независимо от детекторов.
  3. Прогоняем последовательные детекторы (src/online_detection.py) по той же
     годовой лог-разности с калибровкой 6 точек (как в real_monitoring) и
     считаем РЕАЛЬНЫЕ precision/recall/F1 и задержку относительно слепых меток.

Честные оговорки:
  - Метки — это прокси-определение шока, а не абсолютная истина.
  - Задержка здесь неотрицательна: последовательный детектор срабатывает
    ПОСЛЕ факта. Предупреждение ДО факта решается отдельной моделью риска
    (scripts/13_risk_model.py).
"""

import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import utils
from src import data_loading
from src import features
from src import shock_labeling
from src import online_detection


def main():
    data_config = utils.load_yaml_config(ROOT / "configs" / "data.yaml")
    settings = utils.load_yaml_config(ROOT / "configs" / "checked.yaml")

    destination = ROOT / "outputs" / "shocks"
    destination.mkdir(parents=True, exist_ok=True)

    print("Загружаю данные...", flush=True)
    consumption = data_loading.load_consumption_dataframe(data_config)
    total = consumption[consumption["category"] == "Все категории"]
    wide_total = total.pivot_table(index="territory_id", columns="month", values="value")
    wide_total = wide_total.sort_index(axis=1)
    # Оставляем МО с полной историей (нужно значение 12 месяцев назад).
    wide_total = wide_total.dropna(axis=0, how="any")
    print("Полных МО для разметки: " + str(len(wide_total)), flush=True)

    # --- Слепая разметка ---------------------------------------------------
    yoy_table = shock_labeling.compute_year_over_year_log_change(wide_total)
    labeled, robust_sigma = shock_labeling.label_shocks(
        yoy_table, threshold_sigma=3.0, min_duration=2
    )
    labeled.to_csv(destination / "shock_labels.csv", index=False)

    shock_count = int(labeled["is_shock"].sum())
    extreme_count = int(labeled["is_extreme"].sum())
    mo_with_shock = int(labeled[labeled["is_shock"]]["territory_id"].nunique())
    total_mo_months = int(len(labeled))

    print("Экстремальных месяцев (разовых): " + str(extreme_count), flush=True)
    print("Шоков (устойчивых, длительность>=2): " + str(shock_count), flush=True)
    print("МО с хотя бы одним шоком: " + str(mo_with_shock), flush=True)

    # Распределение шоков по месяцам.
    shock_by_month = (
        labeled[labeled["is_shock"]].groupby("month").size().reset_index(name="shocks")
    )
    shock_by_month.to_csv(destination / "shock_by_month.csv", index=False)

    # --- Оценка детекторов на реальных метках ------------------------------
    # Готовим настройки детектора: калибровка 6 точек (как в real_monitoring).
    detector_settings = dict(settings["detectors"])
    detector_settings["calibration_points"] = 6
    tolerance = 1  # допускаем совпадение тревоги с шоком в пределах 1 месяца

    # Для каждого МО: позиции месяцев в ряде годовой лог-разности.
    months_sorted = sorted(labeled["month"].unique())
    month_to_position = {}
    for position in range(len(months_sorted)):
        month_to_position[months_sorted[position]] = position

    methods = ["Threshold", "CUSUM", "EWMA"]
    hits_by_method = {}
    false_by_method = {}
    delays_by_method = {}
    for method in methods:
        hits_by_method[method] = 0
        false_by_method[method] = 0
        delays_by_method[method] = []

    calibration_points = detector_settings["calibration_points"]
    # Считаем именно ЭПИЗОДЫ шоков (непрерывные серии), а не отдельные месяцы.
    # И только те эпизоды, что начинаются в зоне наблюдения детектора
    # (позиция старта >= калибровки). Эпизоды, начавшиеся во время калибровки,
    # детектор физически не мог поймать, поэтому в знаменатель recall не идут.
    total_episodes_all = 0
    total_inscope_episodes = 0
    monitored_mo = 0

    for territory_id, group in labeled.groupby("territory_id"):
        group = group.sort_values("month").reset_index(drop=True)
        yoy_values = group["yoy_log_change"].to_numpy(dtype=float)
        if len(yoy_values) <= calibration_points:
            continue
        monitored_mo = monitored_mo + 1

        # Разбиваем is_shock на эпизоды (непрерывные серии True).
        shock_flags = group["is_shock"].tolist()
        episode_start_positions = []
        inside_episode = False
        for local_position in range(len(shock_flags)):
            if shock_flags[local_position] and not inside_episode:
                month_value = group.loc[local_position, "month"]
                episode_start_positions.append(month_to_position[month_value])
                inside_episode = True
            elif not shock_flags[local_position]:
                inside_episode = False

        total_episodes_all = total_episodes_all + len(episode_start_positions)

        # Только эпизоды в зоне наблюдения детектора.
        inscope_starts = []
        for start_position in episode_start_positions:
            if start_position >= calibration_points:
                inscope_starts.append(start_position)
        total_inscope_episodes = total_inscope_episodes + len(inscope_starts)

        alarms_frame = online_detection.detect_online(yoy_values, detector_settings)
        for method in methods:
            method_alarms = alarms_frame[
                (alarms_frame["method"] == method) & (alarms_frame["alarm"])
            ]["position"].tolist()
            hits, false_alarms, delays = online_detection.match_alarms(
                method_alarms, inscope_starts, tolerance
            )
            hits_by_method[method] = hits_by_method[method] + hits
            false_by_method[method] = false_by_method[method] + false_alarms
            for delay_value in delays:
                delays_by_method[method].append(delay_value)

    total_shock_events = total_inscope_episodes

    # Считаем precision/recall/F1 по каждому методу.
    evaluation_rows = []
    for method in methods:
        hits = hits_by_method[method]
        false_alarms = false_by_method[method]
        detections = hits + false_alarms
        precision = hits / detections if detections > 0 else 0.0
        recall = hits / total_shock_events if total_shock_events > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if len(delays_by_method[method]) > 0:
            mean_delay = float(np.mean(delays_by_method[method]))
        else:
            mean_delay = float("nan")
        evaluation_rows.append({
            "method": method,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_detection_delay": mean_delay,
            "hits": hits,
            "false_alarms": false_alarms,
            "true_events": total_shock_events,
        })

    evaluation = pd.DataFrame(evaluation_rows).sort_values("f1", ascending=False)
    evaluation.to_csv(destination / "real_detector_eval.csv", index=False)

    summary = {
        "full_history_mo": int(len(wide_total)),
        "labeling_rule": "YoY лог-разность, центрирование по медиане МО, порог 3*робаст.сигма, длительность>=2",
        "robust_sigma": robust_sigma,
        "extreme_months": extreme_count,
        "shock_months": shock_count,
        "shock_episodes_all": int(total_episodes_all),
        "shock_episodes_in_detector_window": int(total_inscope_episodes),
        "mo_with_shock": mo_with_shock,
        "total_mo_months": total_mo_months,
        "monitored_mo": monitored_mo,
        "recall_denominator": "эпизоды в зоне наблюдения детектора (старт >= калибровки)",
        "note": "Метки — прокси, не истина. Recall считается по эпизодам в окне наблюдения, а не по всем месяцам года. Задержка >=0: детектор срабатывает после факта.",
    }
    with open(destination / "shocks_summary.json", "w", encoding="utf-8") as target:
        json.dump(summary, target, ensure_ascii=False, indent=2)

    print("\n=== РЕАЛЬНАЯ ОЦЕНКА ДЕТЕКТОРОВ (против слепых меток) ===", flush=True)
    print(evaluation.to_string(index=False), flush=True)
    print("\nШоков-месяцев всего: " + str(shock_count)
          + " | МО с шоком: " + str(mo_with_shock), flush=True)


if __name__ == "__main__":
    main()

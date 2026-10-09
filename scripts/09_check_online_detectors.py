"""Отдельный тест причинных детекторов на явно обозначенной синтетике.

Синтетика не заменяет разметку реальных экономических шоков. Первые 12 точек
не содержат изменений по построению. Есть сдвиг уровня, дрейф, изменение
дисперсии и контрольные ряды без изменения. Параметры не подбираются на тесте.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.online_detection import detect_online, match_alarms
from src import utils


def main():
    config = utils.load_yaml_config(ROOT / "configs" / "checked.yaml")
    settings = config["detectors"]
    generator = np.random.default_rng(settings["test_seed"])
    destination = ROOT / "outputs" / "detectors"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    series_rows = []
    details = []
    methods = [
        "Threshold",
        "CUSUM",
        "EWMA",
    ]
    cases = [
        "no_change",
        "level_shift",
        "drift",
        "variance",
    ]
    for number in range(settings["test_series"]):
        values = generator.normal(0.0, 1.0, 60)
        case = cases[number % len(cases)]
        event = int(generator.integers(22, 40))
        events = []
        if case != "no_change":
            events.append(event)
        if case == "level_shift":
            values[event:] = values[event:] + 3.0
        if case == "drift":
            values[event:] = values[event:] + 0.3 * np.arange(1, len(values) - event + 1)
        if case == "variance":
            values[event:] = values[event:] * 3.0
        detected = detect_online(values, settings)
        for position, value in enumerate(values):
            series_rows.append({
                "series": number,
                "case": case,
                "position": position,
                "value": float(value),
                "true_event": position in events,
            })
        for method in methods:
            method_rows = detected[detected["method"] == method]
            alarms = method_rows.loc[method_rows["alarm"], "position"].tolist()
            hits, false_alarms, delays = match_alarms(alarms, events, settings["match_window"])
            records.append({
                "series": number,
                "case": case,
                "method": method,
                "hits": hits,
                "events": len(events),
                "false_alarms": false_alarms,
                "delay_sum": sum(delays),
                "months": len(values) - settings["calibration_points"],
            })
            method_rows = method_rows.copy()
            method_rows["series"] = number
            method_rows["case"] = case
            details.append(method_rows)
    scores = []
    records = pd.DataFrame(records)
    for method, rows in records.groupby("method"):
        hits = int(rows["hits"].sum())
        count = int(rows["events"].sum())
        false_alarms = int(rows["false_alarms"].sum())
        precision = hits / max(1, hits + false_alarms)
        recall = hits / max(1, count)
        delay = None
        if hits > 0:
            delay = float(rows["delay_sum"].sum() / hits)
        scores.append({
            "method": method,
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
            "mean_detection_delay": delay,
            "false_alarms_per_100_points": false_alarms * 100.0 / rows["months"].sum(),
            "hits": hits,
            "true_events": count,
            "false_alarms": false_alarms,
        })
    metrics = pd.DataFrame(scores)
    metrics.to_csv(destination / "synthetic_test_metrics.csv", index=False)
    records.to_csv(destination / "synthetic_per_series.csv", index=False)
    pd.DataFrame(series_rows).to_csv(destination / "synthetic_series.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(destination / "synthetic_alarms.csv", index=False)
    print(metrics.to_string(index=False))

    # Реальные ряды: годовые лог-разности по МО. Они устраняют часть
    # сезонности. После 6 точек калибровки остаётся только 6 месяцев.
    # Это иллюстрация сигналов; эталонной разметки реальных шоков нет.
    raw = pd.read_parquet(ROOT / "data" / "raw" / "consumption.parquet")
    raw = raw[raw["category"] == "Все категории"]
    wide = raw.pivot(index="territory_id", columns="date", values="value")
    wide = wide.sort_index(axis=1).dropna()
    real_settings = settings.copy()
    real_settings["calibration_points"] = 6
    actual_alerts = []
    for territory, row in wide.iterrows():
        values = np.log1p(row.to_numpy(dtype=float))
        yearly_changes = values[12:] - values[:-12]
        detected = detect_online(yearly_changes, real_settings)
        detected["territory_id"] = territory
        dates = []
        for position in detected["position"]:
            dates.append(wide.columns[position + 12])
        detected["observation_month"] = dates
        actual_alerts.append(detected)
    real = pd.concat(actual_alerts, ignore_index=True)
    real.to_csv(destination / "real_monitoring.csv", index=False)
    metadata = {
        "test_data_type": "synthetic",
        "test_seed": settings["test_seed"],
        "real_labels": "unavailable",
        "real_precision_recall": "not calculated",
        "warning_before_shock": "not demonstrated",
        "matching_rule": "Every alarm may match at most one event; only event <= alarm <= event + 6.",
        "duplicate_alarms": "Repeated alarms for one event count as false alarms; no favourable deduplication.",
    }
    with open(destination / "evaluation.json", "w", encoding="utf-8") as target:
        json.dump(metadata, target, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

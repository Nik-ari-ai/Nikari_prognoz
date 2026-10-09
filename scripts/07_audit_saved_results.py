"""Пересчитать старые результаты и сопоставить факты с исходным Parquet.

Запуск: python scripts/07_audit_saved_results.py
Скрипт не обучает модели и не доказывает происхождение сохранённых прогнозов.
Он проверяет арифметику, ключи, покрытие и соответствие фактов датасету.
"""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backtest import compute_metrics_by_model_and_horizon
from src.backtest import filter_to_common_prediction_keys


def file_hash(path):
    """Посчитать SHA-256 файла, читая небольшими блоками."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main():
    destination = ROOT / "outputs" / "audit"
    destination.mkdir(parents=True, exist_ok=True)
    source = ROOT / "evidence" / "original"
    predictions = pd.read_csv(source / "forecasts" / "backtest_all_predictions.csv")
    raw = pd.read_parquet(ROOT / "data" / "raw" / "consumption.parquet")
    keys = [
        "territory_id",
        "category",
        "cutoff_month",
        "horizon",
    ]
    full_keys = keys + [
        "model",
    ]
    if predictions.duplicated(full_keys).any():
        raise ValueError("В прогнозах повторяются ключи.")
    if not np.isfinite(predictions[[
        "actual",
        "predicted",
    ]]).all().all():
        raise ValueError("Есть пропущенные или бесконечные прогнозы.")

    target_months = []
    for row in predictions.itertuples():
        target_month = pd.Period(row.cutoff_month, freq="M") + row.horizon
        target_months.append(str(target_month))
    predictions["date"] = target_months
    compared = predictions.merge(
        raw,
        on=[
            "territory_id",
            "category",
            "date",
        ],
        how="left",
        validate="many_to_one",
    )
    if compared["value"].isna().any():
        raise ValueError("Для сохранённого факта нет строки в исходных данных.")
    maximum_actual_error = float((compared["actual"] - compared["value"]).abs().max())
    if maximum_actual_error != 0.0:
        raise ValueError("Сохранённые факты не совпали с исходными данными.")

    common = filter_to_common_prediction_keys(predictions)
    metrics = compute_metrics_by_model_and_horizon(common)
    saved = pd.read_csv(source / "metrics" / "forecast_metrics_by_horizon_aligned.csv")
    compared_metrics = metrics.merge(saved, on=[
        "model",
        "horizon",
    ], suffixes=[
        "_new",
        "_old",
    ])
    maximum_mae_error = float((compared_metrics["MAE_new"] - compared_metrics["MAE_old"]).abs().max())
    if maximum_mae_error > 1e-8:
        raise ValueError("MAE не совпала с сохранённой таблицей.")
    metrics.to_csv(destination / "recomputed_metrics.csv", index=False)

    mature = common[common["cutoff_month"] >= "2023-12"]
    mature_metrics = compute_metrics_by_model_and_horizon(mature)
    mature_metrics.to_csv(destination / "metrics_after_12_months.csv", index=False)
    parts = []
    for category, rows in common.groupby("category"):
        table = compute_metrics_by_model_and_horizon(rows)
        table["category"] = category
        parts.append(table)
    pd.concat(parts, ignore_index=True).to_csv(destination / "metrics_by_category.csv", index=False)

    negative_counts = {}
    for model, rows in predictions.groupby("model"):
        negative_counts[model] = int((rows["predicted"] < 0).sum())
    summary = {
        "raw_rows": int(len(raw)),
        "raw_territories": int(raw["territory_id"].nunique()),
        "months": int(raw["date"].nunique()),
        "saved_predictions": int(len(predictions)),
        "duplicate_prediction_keys": 0,
        "maximum_actual_difference": maximum_actual_error,
        "maximum_mae_difference": maximum_mae_error,
        "negative_predictions": negative_counts,
        "consumption_sha256": file_hash(ROOT / "data" / "raw" / "consumption.parquet"),
        "market_access_sha256": file_hash(ROOT / "data" / "raw" / "market_access.parquet"),
        "interpretation": "Проверена арифметика сохранённых результатов, а не повторено старое обучение.",
    }
    with open(destination / "audit.json", "w", encoding="utf-8") as target:
        json.dump(summary, target, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

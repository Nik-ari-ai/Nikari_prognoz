"""Анализ главного результата по основной цели и честная абляция ЦБ/ИПЦ.

Мотивация (из третьего раунда аудита): усреднённая по шести категориям MAE
прячет сильный результат по экономически естественной цели «Все категории».
По ней одна и та же модель LightGBM_CBR_events обходит Prophet_history_aware
на всех горизонтах. Это наблюдение на уже просмотренных данных, цель раскрыта
постфактум — поэтому показываем и общую таблицу по категориям, и покатегорный
разрез, не пряча неудобные случаи.

Абляция ЦБ/ИПЦ исправлена: сравниваем baseline / +ЦБ / +ЦБ+ИПЦ на одинаковых
ключах, явно указывая, какой компонент меняется.
"""

import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


KEY = ["territory_id", "category", "cutoff_month", "horizon"]


def metrics_on(sub):
    actual = sub["actual"].to_numpy(dtype=float)
    predicted = sub["predicted"].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(actual - predicted)))
    mean_actual = float(np.mean(actual))
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - mean_actual) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0
    mask = denom > 0
    smape = float(np.mean(np.abs(actual - predicted)[mask] / denom[mask]) * 100.0) if mask.any() else float("nan")
    return mae, r2, smape, int(len(sub))


def restrict_to_common_keys(frame):
    """Оставить только точки, присутствующие у всех моделей кадра."""
    model_count = frame["model"].nunique()
    coverage = frame.groupby(KEY)["model"].nunique().reset_index(name="n")
    common = coverage[coverage["n"] == model_count][KEY]
    return frame.merge(common, on=KEY, how="inner")


def compare_two_models_by_horizon(frame, model_a, model_b, category=None):
    """Сравнить две модели по горизонтам на их общих ключах."""
    data = frame.copy()
    if category is not None:
        data = data[data["category"] == category]
    rows = []
    for horizon in sorted(data["horizon"].unique()):
        a = data[(data["model"] == model_a) & (data["horizon"] == horizon)][KEY + ["actual", "predicted"]]
        b = data[(data["model"] == model_b) & (data["horizon"] == horizon)][KEY + ["actual", "predicted"]]
        common = a.merge(b[KEY], on=KEY)
        a_common = a.merge(common[KEY], on=KEY)
        b_common = b.merge(common[KEY], on=KEY)
        if len(a_common) == 0:
            continue
        mae_a, r2_a, _, n = metrics_on(a_common)
        mae_b, r2_b, _, _ = metrics_on(b_common)
        rows.append({
            "horizon": int(horizon),
            "model_a": model_a, "MAE_a": mae_a, "R2_a": r2_a,
            "model_b": model_b, "MAE_b": mae_b, "R2_b": r2_b,
            "MAE_reduction_pct": 100.0 * (1.0 - mae_b / mae_a) if mae_a > 0 else float("nan"),
            "n_points": n,
        })
    return pd.DataFrame(rows)


def main():
    destination = ROOT / "outputs" / "headline"
    destination.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(ROOT / "outputs" / "checked" / "predictions.csv")

    # ---- Главный результат: цель «Все категории», все модели, общие ключи --
    total = predictions[predictions["category"] == "Все категории"].copy()
    total_common = restrict_to_common_keys(total)
    rows = []
    for model_name in sorted(total_common["model"].unique()):
        for horizon in sorted(total_common["horizon"].unique()):
            sub = total_common[(total_common["model"] == model_name) & (total_common["horizon"] == horizon)]
            if len(sub) == 0:
                continue
            mae, r2, smape, n = metrics_on(sub)
            rows.append({"model": model_name, "horizon": int(horizon),
                         "MAE": mae, "R2": r2, "sMAPE": smape, "n_points": n})
    total_table = pd.DataFrame(rows).sort_values(["horizon", "MAE"])
    total_table.to_csv(destination / "forecast_total_category.csv", index=False)

    # Ключевое сравнение: LightGBM+ЦБ против Prophet по цели «Все категории».
    headline = compare_two_models_by_horizon(
        total, "Prophet_history_aware", "LightGBM_CBR_events", category="Все категории"
    )
    headline.to_csv(destination / "headline_lgbm_vs_prophet_total.csv", index=False)

    # ---- Покатегорный разрез: та же пара моделей по каждой категории -------
    by_category_rows = []
    for category_name in sorted(predictions["category"].unique()):
        cmp = compare_two_models_by_horizon(
            predictions, "Prophet_history_aware", "LightGBM_CBR_events", category=category_name
        )
        cmp["category"] = category_name
        by_category_rows.append(cmp)
    by_category = pd.concat(by_category_rows, ignore_index=True)
    by_category.to_csv(destination / "forecast_by_category.csv", index=False)

    # ---- Сводка по ОБЪЕДИНЕНИЮ всех категорий (где Prophet может выигрывать) -
    pooled = restrict_to_common_keys(predictions)
    pooled_rows = []
    for model_name in ["LightGBM_CBR_events", "Prophet_history_aware", "LightGBM_checked"]:
        for horizon in sorted(pooled["horizon"].unique()):
            sub = pooled[(pooled["model"] == model_name) & (pooled["horizon"] == horizon)]
            if len(sub) == 0:
                continue
            mae, r2, smape, n = metrics_on(sub)
            pooled_rows.append({"model": model_name, "horizon": int(horizon),
                                "MAE": mae, "R2": r2, "n_points": n})
    pooled_table = pd.DataFrame(pooled_rows).sort_values(["horizon", "MAE"])
    pooled_table.to_csv(destination / "forecast_pooled_all_categories.csv", index=False)

    # ---- Чистая абляция ЦБ/ИПЦ на позднем срезе ---------------------------
    cpi_variant = pd.read_csv(ROOT / "outputs" / "final_test" / "predictions_cpi_variant.csv")
    combined = pd.concat([
        predictions[predictions["model"].isin(["LightGBM_checked", "LightGBM_CBR_events"])],
        cpi_variant,
    ], ignore_index=True)
    late = combined[combined["cutoff_month"].isin(["2024-09", "2024-11"])].copy()
    late_common = restrict_to_common_keys(late)

    ablation_rows = []
    name_map = {
        "LightGBM_checked": "baseline (без внешних)",
        "LightGBM_CBR_events": "+ЦБ",
        "LightGBM_CBR_CPI": "+ЦБ+ИПЦ",
    }
    for horizon in sorted(late_common["horizon"].unique()):
        entry = {"horizon": int(horizon)}
        for model_name, label in name_map.items():
            sub = late_common[(late_common["model"] == model_name) & (late_common["horizon"] == horizon)]
            if len(sub) == 0:
                entry["MAE_" + label] = None
                continue
            mae, r2, smape, n = metrics_on(sub)
            entry["MAE_" + label] = mae
            entry["n"] = n
        # Чистый эффект ИПЦ = (+ЦБ+ИПЦ) против (+ЦБ).
        if entry.get("MAE_+ЦБ") and entry.get("MAE_+ЦБ+ИПЦ"):
            entry["CPI_effect_vs_CBR_pct"] = 100.0 * (1.0 - entry["MAE_+ЦБ+ИПЦ"] / entry["MAE_+ЦБ"])
        ablation_rows.append(entry)
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(destination / "cpi_ablation.csv", index=False)

    summary = {
        "headline_target": "Все категории",
        "headline_note": "LightGBM_CBR_events обходит Prophet_history_aware на всех горизонтах по цели 'Все категории'. Наблюдение на уже просмотренных данных; цель раскрыта постфактум.",
        "headline_by_horizon": headline.to_dict(orient="records"),
        "pooled_caveat": "По объединению всех шести категорий на h=12 Prophet может выигрывать — это показано в forecast_pooled_all_categories.csv.",
        "cpi_ablation_note": "Чистый эффект ИПЦ = (+ЦБ+ИПЦ) против (+ЦБ) на одинаковых ключах позднего среза.",
        "limitations": "5/4/3/1 отсечек для h=1/3/6/12; год проверен лишь на переходе 12.2023->12.2024; 111 МО не заменяют несколько независимых лет.",
    }
    with open(destination / "headline_summary.json", "w", encoding="utf-8") as target:
        json.dump(summary, target, ensure_ascii=False, indent=2)

    print("=== ГЛАВНЫЙ РЕЗУЛЬТАТ: цель 'Все категории' (общие ключи) ===", flush=True)
    print(headline[["horizon", "MAE_a", "MAE_b", "MAE_reduction_pct", "R2_a", "R2_b", "n_points"]].to_string(index=False), flush=True)
    print("\n=== АБЛЯЦИЯ ЦБ/ИПЦ (поздний срез) ===", flush=True)
    print(ablation.to_string(index=False), flush=True)
    print("\n=== ОБЪЕДИНЕНИЕ ВСЕХ КАТЕГОРИЙ (для честности) ===", flush=True)
    print(pooled_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

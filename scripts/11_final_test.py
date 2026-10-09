"""Ретроспективное разбиение прогнозов на валидацию и поздний срез.

ЧЕСТНОЕ НАЗВАНИЕ: это НЕ нетронутый независимый финальный тест. Отсечки
сентября и ноября 2024 уже присутствовали в прогнозах проверенного запуска,
то есть эти результаты уже наблюдались ранее. Здесь мы лишь задним числом
делим уже посчитанные прогнозы на ранний (валидация) и поздний (поздний срез)
периоды и смотрим, переносится ли выбор модели. Настоящий независимый тест
требует ещё одного года данных, которого в выборке нет.

Процедура:
  1. Отсечки делятся на ранние и поздние.
       ранние (валидация): 2023-12, 2024-03, 2024-06
       поздние (поздний срез): 2024-09, 2024-11
  2. Правило выбора: модель с наименьшим средним MAE на ранних отсечках.
  3. Та же модель смотрится на позднем срезе; показываются и все остальные.

Все модели сравниваются на ОБЩИХ КЛЮЧАХ (территория, категория, отсечка,
горизонт), иначе сравнение было бы на разных данных.

Ограничение данных: история заканчивается 2024-12, поэтому на позднем срезе
проверяемы только h=1 (2024-09, 2024-11) и h=3 (2024-09). Горизонты 6 и 12
на позднем срезе непроверяемы.

Дополнительно добавляется вариант LightGBM с инфляцией Росстата (ИПЦ) и
сообщениями ЦБ.
"""

import os
import sys
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import utils
from src import data_loading
from src import calendar_features
from src import features
from src import news
from src import macro
from src import backtest


VALIDATION_CUTOFFS = ["2023-12", "2024-03", "2024-06"]
FINAL_CUTOFFS = ["2024-09", "2024-11"]


def add_forecasts(rows, model_name, cutoff, horizons, index, forecasts, wide):
    """Записать прогнозы, для которых есть проверяемый факт (как в 08)."""
    for position in range(len(index)):
        key = index[position]
        for horizon in horizons:
            target = cutoff + horizon
            if target not in wide.columns:
                continue
            actual = wide.loc[key, target]
            if pd.isna(actual):
                continue
            prediction = float(np.asarray(forecasts[horizon])[position])
            if not np.isfinite(prediction):
                raise ValueError("Нечисловой прогноз у модели " + model_name)
            rows.append({
                "territory_id": int(key[0]),
                "category": key[1],
                "model": model_name,
                "cutoff_month": str(cutoff),
                "horizon": int(horizon),
                "actual": float(actual),
                "predicted": max(0.0, prediction),
            })


def compute_metrics(frame):
    """MAE, R2, sMAPE и число точек по (model, horizon, split)."""
    collected = []
    for (model_name, horizon), group in frame.groupby(["model", "horizon"]):
        actual = group["actual"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(actual - predicted)))
        mean_actual = float(np.mean(actual))
        ss_res = float(np.sum((actual - predicted) ** 2))
        ss_tot = float(np.sum((actual - mean_actual) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        collected.append({
            "model": model_name,
            "horizon": int(horizon),
            "MAE": mae,
            "R2": r2,
            "n_points": int(len(group)),
        })
    return pd.DataFrame(collected)


def main():
    settings = utils.load_yaml_config(ROOT / "configs" / "checked.yaml")
    config = utils.load_yaml_config(ROOT / "configs" / "forecast.yaml")
    data_config = utils.load_yaml_config(ROOT / "configs" / "data.yaml")

    destination = ROOT / "outputs" / "final_test"
    destination.mkdir(parents=True, exist_ok=True)

    print("Загружаю данные...", flush=True)
    # Как и в проверенном запуске: берём ВСЕ МО, без отбора по полноте
    # (это исключает смещение выжившего при обучении).
    consumption = data_loading.load_consumption_dataframe(data_config)
    dictionary = data_loading.load_territory_dictionary(data_config)

    wide = features.prepare_wide_value_matrix(consumption)
    months = list(wide.columns)

    # Те же отключения признаков, что и в проверенном запуске.
    config["features"]["use_market_access"] = False
    config["features"]["use_region"] = False
    config["features"]["use_coordinates"] = False
    config["features"]["use_holiday_features"] = False
    config["features"]["use_news_features"] = False
    config["features"]["use_cpi_feature"] = False
    config["lightgbm_params"].update(settings["lightgbm"])
    config["prophet_params"]["minimum_seasonal_history"] = 24
    config["backtest"]["min_train_months"] = 6

    market = pd.DataFrame(columns=["territory_id", "market_access"])
    static, codes = features.build_static_features(wide, market, dictionary, config)
    calendar = calendar_features.build_calendar_features_table(months)

    # Та же выборка территорий для оценки, что и в проверенном запуске.
    # Авторитетный источник — оценочные территории реального прогона (111 МО),
    # а НЕ текущий набор кандидатов сэмплера (ревизия кода сэмплера могла
    # измениться после прогона, см. README).
    evaluated_path = ROOT / "outputs" / "checked" / "evaluated_territories.csv"
    if not evaluated_path.exists():
        evaluated_path = ROOT / "outputs" / "checked" / "sample.csv"
    sample_ids = pd.read_csv(evaluated_path)["territory_id"].tolist()

    # Макротаблица: сообщения ЦБ + инфляция Росстата (ИПЦ).
    events = pd.read_csv(ROOT / "data" / "external" / "cbr_rate_events.csv")
    news_table = news.monthly_rate_changes(events, months, [0])
    cpi_table = macro.load_cpi_table(ROOT / "data" / "external" / "rosstat_cpi.csv")
    macro_table = macro.build_macro_table(cpi_table, news_table, months, [0])
    macro_table.to_csv(destination / "macro_features.csv", index=False)

    macro_static = static.copy()
    macro_static["region_code"] = 0
    macro_config = copy.deepcopy(config)
    macro_config["features"]["use_news_features"] = True
    macro_config["features"]["use_cpi_feature"] = True

    all_origins = VALIDATION_CUTOFFS + FINAL_CUTOFFS

    # Считаем прогнозы НОВОГО варианта (LightGBM + ЦБ + ИПЦ) на всех отсечках.
    macro_rows = []
    for cutoff_text in all_origins:
        cutoff = pd.Period(cutoff_text, freq="M")
        print("Вариант с ИПЦ, отсечка " + cutoff_text, flush=True)
        horizons = backtest.valid_horizons_for_cutoff(cutoff, months, settings["horizons"])
        history = wide.loc[:, :cutoff]
        selected = history.index.get_level_values("territory_id").isin(sample_ids)
        sample = history.loc[selected].dropna()

        macro_model, columns = backtest.train_one_step_global_model(
            wide, macro_static, calendar, months, cutoff, macro_config, macro_table
        )
        macro_forecasts = backtest.recursive_global_forecast(
            macro_model, sample, macro_static.reindex(sample.index), calendar,
            cutoff, max(horizons), macro_config, macro_table
        )
        sample_index = list(sample.index)
        add_forecasts(macro_rows, "LightGBM_CBR_CPI", cutoff, horizons,
                      sample_index, macro_forecasts, wide)

    macro_predictions = pd.DataFrame(macro_rows)
    macro_predictions.to_csv(destination / "predictions_cpi_variant.csv", index=False)

    # Берём прогнозы остальных моделей из проверенного запуска. ВАЖНО:
    # predictions.csv уже содержит Chronos_Bolt_tiny, поэтому отдельный файл
    # foundation_predictions.csv НЕ добавляем (иначе Chronos задвоится).
    existing = pd.read_csv(ROOT / "outputs" / "checked" / "predictions.csv")
    combined = pd.concat([existing, macro_predictions], ignore_index=True)

    # Защита от случайного задвоения: убираем полные дубли ключ+модель.
    key_full = ["territory_id", "category", "cutoff_month", "horizon", "model"]
    combined = combined.drop_duplicates(subset=key_full, keep="first")

    # --- Сравнение строго на ОБЩИХ ключах всех моделей ---------------------
    # Иначе вариант с ИПЦ мог бы оцениваться на чуть другом наборе МО.
    key_point = ["territory_id", "category", "cutoff_month", "horizon"]
    model_count = combined["model"].nunique()
    coverage = combined.groupby(key_point)["model"].nunique().reset_index(name="n_models")
    common_keys = coverage[coverage["n_models"] == model_count][key_point]
    combined = combined.merge(common_keys, on=key_point, how="inner")
    print("Моделей: " + str(model_count)
          + "; общих точек прогноза: " + str(len(common_keys)), flush=True)

    # --- Разделение на ранний (валидация) и поздний срез -------------------
    validation = combined[combined["cutoff_month"].isin(VALIDATION_CUTOFFS)].copy()
    final = combined[combined["cutoff_month"].isin(FINAL_CUTOFFS)].copy()

    validation_metrics = compute_metrics(validation)
    final_metrics = compute_metrics(final)
    validation_metrics.to_csv(destination / "validation_metrics.csv", index=False)
    final_metrics.to_csv(destination / "final_test_metrics.csv", index=False)

    # --- Заранее заданное правило выбора модели по валидации ---------------
    selection = {}
    for horizon in sorted(validation_metrics["horizon"].unique()):
        subset = validation_metrics[validation_metrics["horizon"] == horizon]
        subset = subset.sort_values("MAE")
        best_model = subset.iloc[0]["model"]
        selection[int(horizon)] = {
            "chosen_by_validation": str(best_model),
            "validation_MAE": float(subset.iloc[0]["MAE"]),
        }

    # --- Результат выбранной модели на закрытом финале ---------------------
    for horizon in list(selection.keys()):
        chosen = selection[horizon]["chosen_by_validation"]
        final_row = final_metrics[
            (final_metrics["horizon"] == horizon) & (final_metrics["model"] == chosen)
        ]
        if len(final_row) > 0:
            selection[horizon]["final_test_MAE"] = float(final_row.iloc[0]["MAE"])
            selection[horizon]["final_test_n"] = int(final_row.iloc[0]["n_points"])
            selection[horizon]["final_test_available"] = True
        else:
            selection[horizon]["final_test_MAE"] = None
            selection[horizon]["final_test_available"] = False
            selection[horizon]["note"] = (
                "На закрытом финале нет проверяемых фактов для этого горизонта "
                "(история заканчивается 2024-12)."
            )

    protocol = {
        "nature": "Ретроспективное разбиение уже посчитанных прогнозов, НЕ нетронутый независимый тест.",
        "why_not_independent": "Отсечки 2024-09 и 2024-11 уже были в прогнозах проверенного запуска и наблюдались ранее.",
        "validation_cutoffs": VALIDATION_CUTOFFS,
        "late_slice_cutoffs": FINAL_CUTOFFS,
        "compared_on": "общие ключи всех моделей (территория, категория, отсечка, горизонт)",
        "selection_rule": "Минимальный средний MAE на ранних отсечках для каждого горизонта.",
        "data_limitation": "История до 2024-12: на позднем срезе проверяемы h=1 (2024-09, 2024-11) и h=3 (2024-09).",
        "needs_for_true_test": "Ещё один независимый год данных.",
        "selection": selection,
    }
    with open(destination / "final_test.json", "w", encoding="utf-8") as target:
        json.dump(protocol, target, ensure_ascii=False, indent=2)

    print("\n=== ВАЛИДАЦИЯ (выбор модели) ===", flush=True)
    print(validation_metrics.sort_values(["horizon", "MAE"]).to_string(index=False), flush=True)
    print("\n=== ЗАКРЫТЫЙ ФИНАЛ (однократная оценка) ===", flush=True)
    print(final_metrics.sort_values(["horizon", "MAE"]).to_string(index=False), flush=True)
    print("\n=== ВЫБОР И ФИНАЛЬНЫЙ РЕЗУЛЬТАТ ===", flush=True)
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

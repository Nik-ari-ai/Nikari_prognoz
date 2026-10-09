"""Новый проверочный эксперимент на реальных данных, без подмены метрик.

Все параметры читаются из configs/checked.yaml до обучения.
Маленькая выборка ускоряет аудит. Это исследование, а не независимый финальный
тест: прежний проект уже использовал весь доступный период для выбора моделей.
"""

from pathlib import Path
import copy
import importlib.metadata
import json
import os
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "models_cache" / "matplotlib"))

import numpy as np
import pandas as pd
import yaml

from src import backtest
from src import calendar_features
from src import data_loading
from src import features
from src import models_forecast
from src import news
from src import utils


def add_forecasts(rows, model, cutoff, horizons, index, forecasts, wide):
    """Записать только прогнозы, для которых существует проверяемый факт."""
    for position, key in enumerate(index):
        for horizon in horizons:
            target = cutoff + horizon
            actual = wide.loc[key, target]
            prediction = float(np.asarray(forecasts[horizon])[position])
            if pd.isna(actual):
                continue
            if not np.isfinite(prediction):
                raise ValueError("Модель вернула нечисловой прогноз: " + model)
            rows.append({
                "territory_id": int(key[0]),
                "category": key[1],
                "model": model,
                "cutoff_month": str(cutoff),
                "horizon": horizon,
                "actual": float(actual),
                "predicted": max(0.0, prediction),
            })


def main():
    start = time.time()
    settings = utils.load_yaml_config(ROOT / "configs" / "checked.yaml")
    config = utils.load_yaml_config(ROOT / "configs" / "forecast.yaml")
    data_config = utils.load_yaml_config(ROOT / "configs" / "data.yaml")
    destination = ROOT / "outputs" / "checked"
    destination.mkdir(parents=True, exist_ok=True)
    # Старые CSV могут существовать после предыдущего запуска. До успешного
    # завершения текущего опыта их нельзя объявлять его результатом.
    with open(destination / "run.json", "w", encoding="utf-8") as target:
        json.dump({"status": "running", "started_unix": start}, target, indent=2)
    with open(destination / "settings.yaml", "w", encoding="utf-8") as target:
        yaml.safe_dump(settings, target, allow_unicode=True, sort_keys=False)

    # Не отбираем МО по полноте будущих 24 месяцев.
    consumption = data_loading.load_consumption_dataframe(data_config)
    dictionary = data_loading.load_territory_dictionary(data_config)
    first_origin = pd.Period(settings["origins"][0], freq="M")
    known = consumption[consumption["month"] <= first_origin]
    sample_ids = backtest.select_stratified_territory_sample(
        known, settings["sample_size"], dictionary, settings["random_seed"]
    )
    pd.DataFrame({"territory_id": sample_ids}).to_csv(destination / "sample.csv", index=False)
    wide = features.prepare_wide_value_matrix(consumption)
    months = list(wide.columns)
    config["features"]["use_market_access"] = False
    config["features"]["use_region"] = False
    config["features"]["use_coordinates"] = False
    config["features"]["use_holiday_features"] = False
    config["features"]["use_news_features"] = False
    config["lightgbm_params"].update(settings["lightgbm"])
    config["prophet_params"]["minimum_seasonal_history"] = 24
    config["backtest"]["min_train_months"] = 6
    with open(destination / "forecast_settings.yaml", "w", encoding="utf-8") as target:
        yaml.safe_dump(config, target, allow_unicode=True, sort_keys=False)
    market = pd.DataFrame(columns=[
        "territory_id",
        "market_access",
    ])
    static, codes = features.build_static_features(wide, market, dictionary, config)
    calendar = calendar_features.build_calendar_features_table(months)
    news_table = None
    news_static = static.copy()
    if settings.get("news_ablation", False):
        events = pd.read_csv(ROOT / "data" / "external" / "cbr_rate_events.csv")
        news_table = news.monthly_rate_changes(events, months, [
            0,
        ])
        news_static["region_code"] = 0
        news_config = copy.deepcopy(config)
        news_config["features"]["use_news_features"] = True
        news_table.to_csv(destination / "news_features.csv", index=False)

    foundation_revision = None

    rows = []
    for cutoff_text in settings["origins"]:
        cutoff = pd.Period(cutoff_text, freq="M")
        print("Отсечка " + cutoff_text, flush=True)
        horizons = backtest.valid_horizons_for_cutoff(cutoff, months, settings["horizons"])
        history = wide.loc[:, :cutoff]
        selected = history.index.get_level_values("territory_id").isin(sample_ids)
        sample = history.loc[selected].dropna()
        if len(sample) == 0:
            raise ValueError("Нет полных историй на отсечке " + cutoff_text)

        for key, values in sample.iterrows():
            history_values = values.to_numpy(dtype=float)
            forecasts = {
                "SeasonalNaive": models_forecast.forecast_seasonal_naive(history_values, horizons),
                "ETS": models_forecast.forecast_ets(history_values, horizons, config["ets_params"]),
                "Prophet_history_aware": models_forecast.forecast_prophet(
                    list(sample.columns), history_values, horizons, config["prophet_params"]
                ),
            }
            # Повторяем старую настройку на той же выборке: это проверяет
            # гипотезу о нестабильной годовой сезонности на короткой истории.
            original_prophet = copy.deepcopy(config["prophet_params"])
            original_prophet["minimum_seasonal_history"] = 0
            forecasts["Prophet_original_setting"] = models_forecast.forecast_prophet(
                list(sample.columns), history_values, horizons, original_prophet
            )
            for model, predictions in forecasts.items():
                vectors = {}
                for horizon, value in predictions.items():
                    vectors[horizon] = np.array([
                        value,
                    ])
                add_forecasts(rows, model, cutoff, horizons, [
                    key,
                ], vectors, wide)

        print("Глобальная модель", flush=True)
        model, columns = backtest.train_one_step_global_model(
            wide, static, calendar, months, cutoff, config
        )
        forecasts = backtest.recursive_global_forecast(
            model, sample, static.reindex(sample.index), calendar,
            cutoff, max(horizons), config
        )
        add_forecasts(rows, "LightGBM_checked", cutoff, horizons, sample.index, forecasts, wide)

        if news_table is not None:
            print("Глобальная модель с сообщениями ЦБ", flush=True)
            news_model, columns = backtest.train_one_step_global_model(
                wide, news_static, calendar, months, cutoff, news_config, news_table
            )
            news_forecasts = backtest.recursive_global_forecast(
                news_model, sample, news_static.reindex(sample.index), calendar,
                cutoff, max(horizons), news_config, news_table
            )
            add_forecasts(rows, "LightGBM_CBR_events", cutoff, horizons, sample.index, news_forecasts, wide)

        if settings["foundation"]["enabled"]:
            saved_sample = sample.copy()
            saved_sample.columns = saved_sample.columns.astype(str)
            saved_sample.to_parquet(destination / ("foundation_input_" + cutoff_text + ".parquet"))
        pd.DataFrame(rows).to_csv(destination / "predictions.csv", index=False)

    results = pd.DataFrame(rows)
    if settings["foundation"]["enabled"]:
        # На macOS PyTorch и LightGBM могут загружать разные OpenMP.
        # Отдельный процесс предотвращает конфликт нативных библиотек.
        command = [
            sys.executable,
            str(ROOT / "scripts" / "foundation_worker.py"),
        ]
        subprocess.run(command, check=True)
        foundation_results = pd.read_csv(destination / "foundation_predictions.csv")
        results = pd.concat([
            results,
            foundation_results,
        ], ignore_index=True)
        with open(destination / "foundation.json", encoding="utf-8") as source:
            foundation_revision = json.load(source)["revision"]
    results.to_csv(destination / "predictions.csv", index=False)
    common = backtest.filter_to_common_prediction_keys(results)
    metrics = backtest.compute_metrics_by_model_and_horizon(common)
    metrics.to_csv(destination / "metrics.csv", index=False)
    parts = []
    for category, category_rows in common.groupby("category"):
        category_metrics = backtest.compute_metrics_by_model_and_horizon(category_rows)
        category_metrics["category"] = category
        parts.append(category_metrics)
    pd.concat(parts, ignore_index=True).to_csv(destination / "metrics_by_category.csv", index=False)
    versions = {}
    for name in [
        "pandas",
        "numpy",
        "prophet",
        "lightgbm",
        "statsmodels",
        "chronos-forecasting",
        "torch",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    summary = {
        "status": "completed",
        "seconds": time.time() - start,
        "territories_requested": settings["sample_size"],
        "versions": versions,
        "foundation_revision": foundation_revision,
        "evaluation": "exploratory; not an untouched final test",
        "availability_assumption": "Расходы месяца доступны на его конец. История фактических публикаций отсутствует.",
        "training_scope": "LightGBM использует все МО с известными к отсечке целями; локальные модели используют свой ряд.",
    }
    with open(destination / "run.json", "w", encoding="utf-8") as target:
        json.dump(summary, target, ensure_ascii=False, indent=2)
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

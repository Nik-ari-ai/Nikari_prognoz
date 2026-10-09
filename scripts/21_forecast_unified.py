"""Единая прогнозная таблица: все модели (включая CatBoost и Chronos) на всех
четырёх горизонтах, с разбросом MAE по МО (медиана и 95%-интервал).

По приоритету обратной связи:
  1) ГЛАВНЫЙ результат — pooled по ВСЕМ категориям (а не одна выбранная
     категория): так честнее и нет post-hoc выбора;
  2) добавлен CatBoost ИМЕННО в прогнозную таблицу (тот же рекурсивный
     глобальный протокол, что и у LightGBM) — через фабрику, не трогая
     аудируемый predictions.csv;
  3) Chronos включён во все четыре горизонта;
  4) добавлен разброс MAE по МО (медиана и 95%-перцентильный интервал).

CatBoost считается на ТОМ ЖЕ составе территорий, что в сохранённых прогнозах
(outputs/checked/evaluated_territories.csv), поэтому ключи совпадают, и сравнение
идёт на общих ключах.
"""

import os
import sys
import json
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import backtest
from src import calendar_features
from src import data_loading
from src import features
from src import models_forecast
from src import utils


def select_strategy_on_early(early_mae_by_model):
    """Выбрать стратегию ТОЛЬКО по ранним фактам (поправка аудита v16).

    Правило выбора зафиксировано явно: из списка кандидатов берётся модель с
    наименьшим MAE на РАННИХ origins (2023-12, 2024-03, 2024-06). Уточнение
    (аудит v17): для реально оцениваемых поздно h=1/h=3 метки ранних origins
    доступны до первого позднего выпуска при допущении о доступности расходов
    на конец месяца; для h=6/h=12 поздней проверки нет вовсе (их последние
    ранние метки лежат в 2024-12, позднее первого общего origin 2024-09),
    поэтому там это «выбор на ранних origins без доступной поздней проверки».
    Функция ЧИСТАЯ: зависит только от ранних MAE, поэтому любое изменение
    ПОЗДНИХ фактов НЕ может изменить выбор. Поздний срез используется отдельно
    только для ИЗМЕРЕНИЯ результата этого выбора.

    early_mae_by_model : {model_name: early_mae}. Возвращает имя выбранной модели
    (или None, если словарь пуст).
    """
    best_model = None
    best_mae = None
    for model_name in sorted(early_mae_by_model.keys()):
        candidate_mae = early_mae_by_model[model_name]
        if candidate_mae is None:
            continue
        if best_mae is None or candidate_mae < best_mae:
            best_mae = candidate_mae
            best_model = model_name
    return best_model


def mae_on_cutoffs(common_frame, model_name, horizon, cutoffs):
    """MAE одной модели на заданном горизонте и наборе отсечек (или None)."""
    import numpy as _np
    subset = common_frame[(common_frame["model"] == model_name)
                          & (common_frame["horizon"] == horizon)
                          & (common_frame["cutoff_month"].isin(cutoffs))]
    if len(subset) == 0:
        return None
    return float(_np.mean(_np.abs(subset["actual"].to_numpy(dtype=float)
                                  - subset["predicted"].to_numpy(dtype=float))))


def make_catboost_forecaster(use_log1p):
    """Фабрика глобального CatBoost-прогнозатора (тот же интерфейс, что LightGBM)."""
    catboost_parameters = {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "loss_function": "RMSE",
        "random_seed": 42,
        "verbose": 0,
        "allow_writing_files": False,
    }
    return models_forecast.GlobalCatBoostForecaster(catboost_parameters, use_log1p)


def add_forecast_rows(collected_rows, model_name, cutoff, horizons, index, forecasts, wide):
    """Записать прогнозы с проверяемым фактом (как в scripts/08)."""
    for position, key in enumerate(index):
        for horizon in horizons:
            target_month = cutoff + horizon
            actual_value = wide.loc[key, target_month]
            prediction_value = float(np.asarray(forecasts[horizon])[position])
            if pd.isna(actual_value):
                continue
            if not np.isfinite(prediction_value):
                raise ValueError("Нечисловой прогноз модели " + model_name)
            collected_rows.append({
                "territory_id": int(key[0]),
                "category": key[1],
                "model": model_name,
                "cutoff_month": str(cutoff),
                "horizon": horizon,
                "actual": float(actual_value),
                "predicted": max(0.0, prediction_value),
            })


def compute_catboost_predictions():
    """Прогнать глобальный CatBoost тем же протоколом и составом, что в scripts/08."""
    settings = utils.load_yaml_config(ROOT / "configs" / "checked.yaml")
    config = utils.load_yaml_config(ROOT / "configs" / "forecast.yaml")
    data_config = utils.load_yaml_config(ROOT / "configs" / "data.yaml")

    consumption = data_loading.load_consumption_dataframe(data_config)
    dictionary = data_loading.load_territory_dictionary(data_config)
    wide = features.prepare_wide_value_matrix(consumption)
    months = list(wide.columns)

    # Те же отключения признаков, что в проверенном прогоне.
    config["features"]["use_market_access"] = False
    config["features"]["use_region"] = False
    config["features"]["use_coordinates"] = False
    config["features"]["use_holiday_features"] = False
    config["features"]["use_news_features"] = False
    config["prophet_params"]["minimum_seasonal_history"] = 24
    config["backtest"]["min_train_months"] = 6

    market = pd.DataFrame(columns=["territory_id", "market_access"])
    static, codes = features.build_static_features(wide, market, dictionary, config)
    calendar = calendar_features.build_calendar_features_table(months)

    # Состав — авторитетные оценочные территории (совпадает с сохранёнными прогнозами).
    evaluated = pd.read_csv(ROOT / "outputs" / "checked" / "evaluated_territories.csv")
    evaluated_column = None
    for column_name in evaluated.columns:
        if "terr" in column_name.lower():
            evaluated_column = column_name
            break
    evaluated_ids = set()
    for one_identifier in evaluated[evaluated_column]:
        evaluated_ids.add(int(one_identifier))

    collected_rows = []
    for cutoff_text in settings["origins"]:
        cutoff = pd.Period(cutoff_text, freq="M")
        print("CatBoost: отсечка " + cutoff_text, flush=True)
        horizons = backtest.valid_horizons_for_cutoff(cutoff, months, settings["horizons"])
        history = wide.loc[:, :cutoff]
        selected = history.index.get_level_values("territory_id").isin(evaluated_ids)
        sample = history.loc[selected].dropna()
        if len(sample) == 0:
            continue

        model, columns = backtest.train_one_step_global_model(
            wide, static, calendar, months, cutoff, config,
            forecaster_factory=make_catboost_forecaster
        )
        if model is None:
            continue
        forecasts = backtest.recursive_global_forecast(
            model, sample, static.reindex(sample.index), calendar,
            cutoff, max(horizons), config
        )
        add_forecast_rows(collected_rows, "CatBoost_global", cutoff, horizons,
                          sample.index, forecasts, wide)

    return pd.DataFrame(collected_rows)


def per_territory_mae_spread(frame):
    """Разброс MAE по МО: медиана и 95%-перцентильный интервал по территориям.

    Для каждой территории считаем её MAE (среднее |факт-прогноз| по её ключам),
    затем возвращаем медиану и перцентили 2.5 / 97.5 по территориям.
    """
    per_territory_errors = {}
    for _, row in frame.iterrows():
        territory_id = int(row["territory_id"])
        absolute_error = abs(float(row["actual"]) - float(row["predicted"]))
        per_territory_errors.setdefault(territory_id, []).append(absolute_error)

    territory_mae_values = []
    for territory_id in per_territory_errors:
        territory_mae_values.append(float(np.mean(per_territory_errors[territory_id])))
    territory_mae_values = np.asarray(territory_mae_values, dtype=float)
    if len(territory_mae_values) == 0:
        return None, None, None
    median_value = float(np.median(territory_mae_values))
    low_value = float(np.quantile(territory_mae_values, 0.025))
    high_value = float(np.quantile(territory_mae_values, 0.975))
    return median_value, low_value, high_value


def main(output_dir=None):
    if output_dir is None:
        destination = ROOT / "outputs" / "headline"
    else:
        destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    print("Читаю 7 сохранённых моделей из predictions.csv (к ним добавим CatBoost)...", flush=True)
    saved = pd.read_csv(ROOT / "outputs" / "checked" / "predictions.csv")

    print("Считаю CatBoost тем же рекурсивным протоколом (8-я модель)...", flush=True)
    catboost_predictions = compute_catboost_predictions()
    catboost_predictions.to_csv(destination / "catboost_predictions.csv", index=False)

    # --- Заранее заданный простой ансамбль CatBoost/LightGBM (без подбора весов)
    # Строки ансамбля считаются для прозрачности и входят в таблицу как ОДИН из
    # девяти кандидатов. Веса фиксированы 0.5/0.5 заранее (не оптимизируются по
    # тесту). Выбор стратегии (включая ансамбль) делается НИЖЕ только по ранним
    # фактам (select_strategy_on_early); поздний срез в выборе не участвует.
    key_columns = ["territory_id", "category", "cutoff_month", "horizon"]
    catboost_for_ensemble = catboost_predictions[key_columns + ["actual", "predicted"]].rename(
        columns={"predicted": "predicted_cb"})
    lightgbm_for_ensemble = saved[saved["model"] == "LightGBM_checked"][key_columns + ["predicted"]].rename(
        columns={"predicted": "predicted_lgb"})
    ensemble_merged = pd.merge(catboost_for_ensemble, lightgbm_for_ensemble, on=key_columns)
    ensemble_rows = []
    for _, row in ensemble_merged.iterrows():
        ensemble_rows.append({
            "territory_id": int(row["territory_id"]),
            "category": str(row["category"]),
            "model": "Ensemble_CatBoost_LightGBM",
            "cutoff_month": str(row["cutoff_month"]),
            "horizon": int(row["horizon"]),
            "actual": float(row["actual"]),
            "predicted": 0.5 * float(row["predicted_cb"]) + 0.5 * float(row["predicted_lgb"]),
        })
    ensemble_predictions = pd.DataFrame(ensemble_rows)

    combined = pd.concat([saved, catboost_predictions, ensemble_predictions], ignore_index=True)

    # Общие ключи среди ВСЕХ моделей (честное сравнение на одинаковых точках).
    common = backtest.filter_to_common_prediction_keys(combined)

    # --- Единая pooled-таблица по ВСЕМ категориям ---------------------------
    unified_rows = []
    spread_rows = []
    for (model_name, horizon), group in common.groupby(["model", "horizon"]):
        actual_values = group["actual"].to_numpy(dtype=float)
        predicted_values = group["predicted"].to_numpy(dtype=float)
        mae_value = float(np.mean(np.abs(actual_values - predicted_values)))
        ss_res = float(np.sum((actual_values - predicted_values) ** 2))
        ss_tot = float(np.sum((actual_values - np.mean(actual_values)) ** 2))
        r2_value = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        median_mae, low_mae, high_mae = per_territory_mae_spread(group)
        unified_rows.append({
            "model": model_name,
            "horizon": int(horizon),
            "MAE": mae_value,
            "R2": r2_value,
            "n_points": int(len(group)),
            "mae_median_per_mo": median_mae,
            "mae_p2_5_per_mo": low_mae,
            "mae_p97_5_per_mo": high_mae,
        })
        spread_rows.append({
            "model": model_name, "horizon": int(horizon),
            "mae_median_per_mo": median_mae,
            "mae_p2_5_per_mo": low_mae, "mae_p97_5_per_mo": high_mae,
        })

    unified_table = pd.DataFrame(unified_rows).sort_values(by=["horizon", "MAE"]).reset_index(drop=True)
    unified_table.to_csv(destination / "forecast_unified_all_categories.csv", index=False)
    pd.DataFrame(spread_rows).to_csv(destination / "forecast_mae_spread.csv", index=False)

    # --- ТРИ РАЗДЕЛЁННЫХ ПОНЯТИЯ (поправка аудита v16) ------------------------
    # Аудит v16 потребовал полностью развести выбор и оценку. Больше нет гейта,
    # который использовал бы поздние факты в решении. Вместо него:
    #   (1) benchmark_best_by_horizon — ОПИСАТЕЛЬНЫЙ лучший результат полной
    #       ретроспективной таблицы (argmin MAE по всем общим ключам). Это НЕ
    #       независимый результат и НЕ выбранная до проверки стратегия, а просто
    #       минимум по уже просмотренным данным.
    #   (2) selected_on_early — СТРАТЕГИЯ, выбранная ТОЛЬКО по ранним origins
    #       (2023-12, 2024-03, 2024-06). Для оцениваемых поздно h=1/h=3 их метки
    #       доступны до первого позднего выпуска; для h=6/h=12 поздней проверки
    #       нет вовсе — это «выбор на ранних origins без поздней проверки». Список
    #       кандидатов (девять моделей, включая ансамбль и запасные одиночные) и
    #       правило (argmin РАННЕГО MAE) зафиксированы заранее. Выбор делает
    #       чистая функция select_strategy_on_early — она зависит только от ранних
    #       MAE, поэтому изменение поздних фактов НЕ может его изменить.
    #   (3) late_evaluation — ИЗМЕРЕНИЕ выбранной стратегии на позднем срезе без
    #       пересмотра выбора. Поздний проигрыш тоже публикуется и НЕ отменяет
    #       раннее решение. Для горизонтов без зрелых поздних меток (h=6, h=12) —
    #       статус "not_evaluated_late": отсутствие проверки не доказывает ни
    #       пользу, ни бесполезность.
    # Ретроспективное разбиение уже просмотренных данных НЕ является нетронутым
    # финальным тестом; для него нужен новый период (см. no_untouched_final_test).
    all_cutoffs = sorted(common["cutoff_month"].unique().tolist())
    if len(all_cutoffs) > 2:
        early_cutoffs = all_cutoffs[:-2]
        late_cutoffs = all_cutoffs[-2:]
    else:
        early_cutoffs = all_cutoffs
        late_cutoffs = []

    candidate_models = sorted(common["model"].unique().tolist())
    horizons_present = []
    for horizon_raw in common["horizon"].unique().tolist():
        horizons_present.append(int(horizon_raw))
    horizons_present = sorted(horizons_present)

    benchmark_best_by_horizon = {}
    selected_on_early = {}
    late_evaluation = {}

    for horizon_value in horizons_present:
        # (1) Описательный победитель полной таблицы (минимум MAE на общих ключах).
        horizon_rows = unified_table[unified_table["horizon"] == horizon_value]
        benchmark_row = horizon_rows.sort_values(by="MAE").iloc[0]
        benchmark_best_by_horizon[horizon_value] = {
            "model": str(benchmark_row["model"]),
            "mae_full_table": round(float(benchmark_row["MAE"]), 1),
        }

        # (2) Выбор стратегии ТОЛЬКО по ранним фактам среди всех девяти кандидатов.
        early_mae_by_model = {}
        for model_name in candidate_models:
            early_mae_by_model[model_name] = mae_on_cutoffs(
                common, model_name, horizon_value, early_cutoffs)
        chosen_model = select_strategy_on_early(early_mae_by_model)
        chosen_early_mae = early_mae_by_model.get(chosen_model)
        selected_on_early[horizon_value] = {
            "model": chosen_model,
            "mae_early": round(chosen_early_mae, 1) if chosen_early_mae is not None else None,
        }

        # (3) Измерение выбранной стратегии на позднем срезе (выбор НЕ меняем).
        chosen_late_mae = mae_on_cutoffs(common, chosen_model, horizon_value, late_cutoffs)
        if chosen_late_mae is None:
            late_evaluation[horizon_value] = {
                "model": chosen_model,
                "status": "not_evaluated_late",
                "mae_late": None,
                "note": ("нет зрелых поздних меток для этого горизонта; выбор не "
                         "проверялся на позднем срезе (отсутствие проверки не доказывает "
                         "ни пользу, ни бесполезность)"),
            }
        else:
            # Справочно: лучший поздний MAE среди всех кандидатов. На ВЫБОР не влияет,
            # служит только для честной констатации, подтвердился выбор или уступил.
            best_late_value = None
            best_late_model = None
            for model_name in candidate_models:
                late_value = mae_on_cutoffs(common, model_name, horizon_value, late_cutoffs)
                if late_value is not None and (best_late_value is None or late_value < best_late_value):
                    best_late_value = late_value
                    best_late_model = model_name
            if best_late_model == chosen_model:
                outcome = "на позднем срезе выбор оказался лучшим"
            else:
                outcome = ("на позднем срезе выбор уступил модели " + str(best_late_model)
                           + " (её поздний MAE " + str(round(best_late_value, 1)) + ")")
            late_evaluation[horizon_value] = {
                "model": chosen_model,
                "status": "evaluated_late",
                "mae_late": round(chosen_late_mae, 1),
                "best_late_reference": {
                    "model": best_late_model,
                    "mae_late": round(best_late_value, 1) if best_late_value is not None else None,
                },
                "outcome": outcome,
                "note": ("поздний срез только измеряет неизменный выбор; результат "
                         "(в т.ч. проигрыш) не пересматривает раннее решение"),
            }

    # Обратная совместимость: winner_by_horizon = ОПИСАТЕЛЬНЫЙ бенчмарк-минимум
    # (не независимый результат; стратегия до проверки — в selected_on_early).
    winner_by_horizon = {}
    for horizon_value in horizons_present:
        winner_by_horizon[horizon_value] = benchmark_best_by_horizon[horizon_value]["model"]

    # Текст-вывод генерируется ИЗ рассчитанных полей (аудит v16: не хардкодить).
    selected_summary_parts = []
    for horizon_value in horizons_present:
        chosen = selected_on_early[horizon_value]["model"]
        early_mae = selected_on_early[horizon_value]["mae_early"]
        late_info = late_evaluation[horizon_value]
        if late_info["status"] == "not_evaluated_late":
            piece = ("h=" + str(horizon_value) + ": выбран " + str(chosen)
                     + " (ранний MAE " + str(early_mae) + "), поздней оценки нет")
        else:
            piece = ("h=" + str(horizon_value) + ": выбран " + str(chosen)
                     + " (ранний MAE " + str(early_mae) + "), поздний MAE "
                     + str(late_info["mae_late"]) + " — " + late_info["outcome"])
        selected_summary_parts.append(piece)
    generated_note = ("Три разведённых понятия (аудит v16). "
                      "benchmark_best_by_horizon — описательный минимум полной "
                      "ретроспективной таблицы (НЕ независимый результат). "
                      "selected_on_early — стратегия, выбранная только по ранним "
                      "фактам из девяти кандидатов правилом argmin раннего MAE. "
                      "late_evaluation — измерение этого неизменного выбора на "
                      "позднем срезе (проигрыш тоже публикуется; h=6/h=12 — "
                      "not_evaluated_late, зрелых поздних меток нет). Поздний срез "
                      "в выборе НЕ участвует. Детали выбора: "
                      + "; ".join(selected_summary_parts) + ".")

    info = {
        "description": "Единая pooled-таблица по всем категориям; общие ключи всех моделей.",
        "models": candidate_models,
        "horizons": horizons_present,
        "selection_candidates": candidate_models,
        "selection_rule": ("argmin MAE на ранних отсечках (early_cutoffs); поздние факты "
                           "в выборе не участвуют — выбор делает чистая функция "
                           "select_strategy_on_early"),
        "early_cutoffs": early_cutoffs,
        "late_cutoffs": late_cutoffs,
        "benchmark_best_by_horizon": benchmark_best_by_horizon,
        "selected_on_early": selected_on_early,
        "late_evaluation": late_evaluation,
        "winner_by_horizon": winner_by_horizon,
        "winner_by_horizon_meaning": ("ОПИСАТЕЛЬНЫЙ победитель полной ретроспективной "
                                      "таблицы (argmin MAE). Это НЕ независимый результат и "
                                      "НЕ стратегия, выбранная до проверки; см. "
                                      "selected_on_early и late_evaluation"),
        "catboost_added": True,
        "ensemble_in_table": True,
        "ensemble_type": "простое заранее заданное среднее 0.5*CatBoost + 0.5*LightGBM (веса не подбирались)",
        "no_untouched_final_test": ("ретроспективное разбиение уже просмотренных данных НЕ "
                                    "является нетронутым финальным тестом; для него нужен "
                                    "новый период. Ранний выбор честен, но свежей проверки нет"),
        "note": generated_note,
    }
    with open(destination / "forecast_unified_summary.json", "w", encoding="utf-8") as target_file:
        json.dump(info, target_file, ensure_ascii=False, indent=2)

    print("\n=== ЕДИНАЯ POOLED-ТАБЛИЦА (все категории, общие ключи) ===", flush=True)
    show = unified_table.copy()
    for column_name in ["MAE", "mae_median_per_mo", "mae_p2_5_per_mo", "mae_p97_5_per_mo"]:
        rounded_column_values = []
        for original_value in show[column_name]:
            if original_value is not None and not pd.isna(original_value):
                rounded_column_values.append(round(original_value, 1))
            else:
                rounded_column_values.append(None)
        show[column_name] = rounded_column_values
    rounded_r2_values = []
    for original_r2_value in show["R2"]:
        rounded_r2_values.append(round(original_r2_value, 4))
    show["R2"] = rounded_r2_values
    print(show.to_string(index=False), flush=True)
    print("\nРазведённые понятия (аудит v16):", flush=True)
    for horizon_value in horizons_present:
        benchmark_model = benchmark_best_by_horizon[horizon_value]["model"]
        chosen_model = selected_on_early[horizon_value]["model"]
        late_info = late_evaluation[horizon_value]
        if late_info["status"] == "not_evaluated_late":
            late_text = "поздняя оценка отсутствует"
        else:
            late_text = "поздний MAE " + str(late_info["mae_late"]) + " (" + late_info["outcome"] + ")"
        print("  h=" + str(horizon_value)
              + ": бенчмарк-минимум=" + benchmark_model
              + " | выбран по ранним=" + str(chosen_model)
              + " | " + late_text, flush=True)


if __name__ == "__main__":
    main()

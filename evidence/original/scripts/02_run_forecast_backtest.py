# ===========================================================================
# Скрипт: сравнение моделей прогнозирования методом скользящего начала.
#
# Что делает:
#   1) загружает данные и строит панель;
#   2) выбирает представительную выборку территорий ("турнир моделей");
#   3) гоняет пообъектные модели (SeasonalNaive, Prophet, ETS) на выборке;
#   4) гоняет глобальную модель LightGBM (обучение на всей панели, оценка на
#      той же выборке);
#   5) считает MAE, R^2, sMAPE по каждой паре (модель, горизонт);
#   6) сохраняет детальные прогнозы и итоговую таблицу метрик.
#
# Запуск из корня репозитория:
#     python scripts/02_run_forecast_backtest.py
# ===========================================================================

import os
import sys
import time

# Добавляем корень репозитория в путь поиска модулей, чтобы работал импорт src.
PATH_TO_THIS_FILE = os.path.abspath(__file__)
PATH_TO_SCRIPTS_DIR = os.path.dirname(PATH_TO_THIS_FILE)
PATH_TO_REPOSITORY_ROOT = os.path.dirname(PATH_TO_SCRIPTS_DIR)
if PATH_TO_REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, PATH_TO_REPOSITORY_ROOT)

import pandas as pd

from src import utils
from src import data_loading
from src import calendar_features
from src import features as features_module
from src import backtest


def main():
    start_time = time.time()

    # ---- Загрузка конфигов -------------------------------------------------
    data_config = utils.load_yaml_config(
        utils.make_absolute_path("configs/data.yaml")
    )
    forecast_config = utils.load_yaml_config(
        utils.make_absolute_path("configs/forecast.yaml")
    )

    # ---- Пути для результатов и лога прогресса -----------------------------
    metrics_directory = utils.make_absolute_path(
        forecast_config["output"]["metrics_dir"]
    )
    forecasts_directory = utils.make_absolute_path(
        forecast_config["output"]["forecasts_dir"]
    )
    utils.ensure_directory_exists(metrics_directory)
    utils.ensure_directory_exists(forecasts_directory)

    progress_log_path = os.path.join(metrics_directory, "backtest_progress.log")
    # Очищаем лог перед запуском.
    with open(progress_log_path, "w", encoding="utf-8") as opened_file:
        opened_file.write("Старт бэктеста прогнозирования\n")

    def log(message_text):
        backtest._append_progress(progress_log_path, message_text)
        print(message_text, flush=True)

    # ---- Данные ------------------------------------------------------------
    log("Загружаю данные...")
    prepared_data = data_loading.load_full_panel(data_config)
    consumption = prepared_data["consumption"]
    market_access = prepared_data["market_access"]
    territory_dictionary = prepared_data["territory_dictionary"]
    all_months = list(prepared_data["all_months"])

    log("Строю широкую матрицу значений по всей панели...")
    full_wide_matrix = features_module.prepare_wide_value_matrix(consumption)

    log("Строю статические признаки и таблицу календаря...")
    static_features, category_to_code = features_module.build_static_features(
        full_wide_matrix, market_access, territory_dictionary, forecast_config
    )
    calendar_table = calendar_features.build_calendar_features_table(all_months)

    # ---- Выборка территорий для турнира ------------------------------------
    number_of_territories = forecast_config["backtest"]["tournament_territories"]
    random_seed = forecast_config["backtest"]["random_seed"]

    if number_of_territories is None:
        selected_territory_ids = sorted(consumption["territory_id"].unique().tolist())
    else:
        selected_territory_ids = backtest.select_stratified_territory_sample(
            consumption, number_of_territories, territory_dictionary, random_seed
        )
    log("Территорий в выборке: " + str(len(selected_territory_ids)))

    # Широкая матрица только для рядов выборки (все 6 категорий этих территорий).
    rows_in_sample_mask = full_wide_matrix.index.get_level_values("territory_id").isin(
        selected_territory_ids
    )
    sample_wide_matrix = full_wide_matrix.loc[rows_in_sample_mask]
    sample_index = sample_wide_matrix.index
    log("Рядов (территория x категория) в выборке: " + str(len(sample_wide_matrix)))

    # ---- Отсечки и горизонты ----------------------------------------------
    list_of_cutoffs = backtest.build_list_of_cutoffs(
        all_months,
        forecast_config["backtest"]["min_train_months"],
        forecast_config["backtest"]["origin_step"],
    )
    list_of_horizons = forecast_config["horizons"]
    log("Отсечки: " + ", ".join(str(month) for month in list_of_cutoffs))
    log("Горизонты: " + ", ".join(str(h) for h in list_of_horizons))

    models_to_run = forecast_config["models"]

    # ---- Пообъектный бэктест -----------------------------------------------
    log("Запускаю пообъектные модели (SeasonalNaive, Prophet, ETS)...")
    per_series_results = backtest.run_per_series_backtest(
        sample_wide_matrix, all_months, list_of_cutoffs, list_of_horizons,
        models_to_run, forecast_config, progress_log_path=progress_log_path,
    )
    per_series_path = os.path.join(forecasts_directory, "backtest_per_series.csv")
    utils.save_dataframe_as_csv(per_series_results, per_series_path)
    log("Пообъектные модели готовы, строк результата: " + str(len(per_series_results)))

    # ---- Глобальная модель LightGBM ----------------------------------------
    all_results_tables = [per_series_results]

    if models_to_run["lightgbm_global"]:
        log("Запускаю глобальную модель LightGBM...")
        global_results = backtest.run_global_model_backtest(
            full_wide_matrix, sample_index, static_features, calendar_table,
            all_months, list_of_cutoffs, list_of_horizons, forecast_config,
            news_features_table=None, progress_log_path=progress_log_path,
        )
        global_path = os.path.join(forecasts_directory, "backtest_global.csv")
        utils.save_dataframe_as_csv(global_results, global_path)
        log("Глобальная модель готова, строк результата: " + str(len(global_results)))
        all_results_tables.append(global_results)

    # ---- Объединение и метрики ---------------------------------------------
    combined_results = pd.concat(all_results_tables, axis=0, ignore_index=True)
    combined_path = os.path.join(forecasts_directory, "backtest_all_predictions.csv")
    utils.save_dataframe_as_csv(combined_results, combined_path)

    # Метрики на всех доступных точках каждой модели.
    metrics_table = backtest.compute_metrics_by_model_and_horizon(combined_results)
    metrics_path = os.path.join(metrics_directory, "forecast_metrics_by_horizon.csv")
    utils.save_dataframe_as_csv(metrics_table, metrics_path)

    # Метрики на ОБЩИХ точках всех моделей (честное сравнение "яблоки к яблокам").
    aligned_results = backtest.filter_to_common_prediction_keys(combined_results)
    aligned_metrics_table = backtest.compute_metrics_by_model_and_horizon(aligned_results)
    aligned_metrics_path = os.path.join(
        metrics_directory, "forecast_metrics_by_horizon_aligned.csv"
    )
    utils.save_dataframe_as_csv(aligned_metrics_table, aligned_metrics_path)

    log("==== МЕТРИКИ НА ВСЕХ ТОЧКАХ (MAE, R2, sMAPE) ====")
    log("\n" + metrics_table.to_string(index=False))
    log("==== МЕТРИКИ НА ОБЩИХ ТОЧКАХ ВСЕХ МОДЕЛЕЙ (честное сравнение) ====")
    log("\n" + aligned_metrics_table.to_string(index=False))

    elapsed_minutes = (time.time() - start_time) / 60.0
    log("Готово. Время выполнения, мин: " + str(round(elapsed_minutes, 1)))


if __name__ == "__main__":
    main()

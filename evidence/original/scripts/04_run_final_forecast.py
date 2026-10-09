# ===========================================================================
# Скрипт: финальный прогноз потребления на 2025 год по всей панели.
#
# Что делает:
#   1) обучает глобальную одношаговую модель LightGBM на ВСЕЙ истории
#      (январь 2023 - декабрь 2024);
#   2) рекурсивно разворачивает прогноз на 12 месяцев вперёд
#      (январь - декабрь 2025) для каждого ряда (территория x категория);
#   3) сохраняет прогнозы и важности признаков для интерпретации.
#
# Запуск:
#     python scripts/04_run_final_forecast.py
# ===========================================================================

import os
import sys

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
    data_config = utils.load_yaml_config(
        utils.make_absolute_path("configs/data.yaml")
    )
    forecast_config = utils.load_yaml_config(
        utils.make_absolute_path("configs/forecast.yaml")
    )

    forecasts_directory = utils.make_absolute_path("outputs/forecasts")
    metrics_directory = utils.make_absolute_path("outputs/metrics")
    utils.ensure_directory_exists(forecasts_directory)
    utils.ensure_directory_exists(metrics_directory)

    print("Загружаю данные...", flush=True)
    prepared_data = data_loading.load_full_panel(data_config)
    consumption = prepared_data["consumption"]
    market_access = prepared_data["market_access"]
    territory_dictionary = prepared_data["territory_dictionary"]
    all_months_observed = list(prepared_data["all_months"])

    print("Строю широкую матрицу и признаки...", flush=True)
    full_wide_matrix = features_module.prepare_wide_value_matrix(consumption)
    static_features, category_to_code = features_module.build_static_features(
        full_wide_matrix, market_access, territory_dictionary, forecast_config
    )

    # Календарь строим с запасом на весь 2025 год (для будущих месяцев прогноза).
    last_observed_month = all_months_observed[-1]
    number_of_future_months = max(forecast_config["horizons"])
    future_months = []
    next_month = last_observed_month
    for step_number in range(number_of_future_months):
        next_month = next_month + 1
        future_months.append(next_month)

    months_for_calendar = all_months_observed + future_months
    calendar_table = calendar_features.build_calendar_features_table(months_for_calendar)

    print("Обучаю глобальную одношаговую модель на всей истории...", flush=True)
    forecaster, feature_column_names = backtest.train_one_step_global_model(
        full_wide_matrix, static_features, calendar_table, all_months_observed,
        last_observed_month, forecast_config, news_features_table=None,
    )

    print("Рекурсивно прогнозирую 12 месяцев вперёд для всей панели...", flush=True)
    predictions_by_horizon = backtest.recursive_global_forecast(
        forecaster, full_wide_matrix, static_features, calendar_table,
        last_observed_month, number_of_future_months, forecast_config,
        news_features_table=None,
    )

    # Собираем длинную таблицу прогнозов.
    collected_rows = []
    for horizon_step, predictions_series in predictions_by_horizon.items():
        forecast_month = last_observed_month + horizon_step
        for series_key in predictions_series.index:
            one_row = {
                "territory_id": series_key[0],
                "category": series_key[1],
                "forecast_month": str(forecast_month),
                "horizon": horizon_step,
                "predicted_value": float(predictions_series.loc[series_key]),
            }
            collected_rows.append(one_row)

    forecast_table = pd.DataFrame(collected_rows)
    forecast_path = os.path.join(forecasts_directory, "future_forecast_2025.csv")
    utils.save_dataframe_as_csv(forecast_table, forecast_path)
    print("Прогноз сохранён: строк " + str(len(forecast_table)), flush=True)

    # Важности признаков одношаговой модели — для интерпретации.
    importances_table = forecaster.get_feature_importances(1)
    importances_path = os.path.join(metrics_directory, "feature_importances.csv")
    utils.save_dataframe_as_csv(importances_table, importances_path)
    print("Важности признаков:", flush=True)
    print(importances_table.to_string(index=False), flush=True)

    print("Готово.", flush=True)


if __name__ == "__main__":
    main()

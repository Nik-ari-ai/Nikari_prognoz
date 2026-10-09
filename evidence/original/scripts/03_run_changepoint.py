# ===========================================================================
# Скрипт: сравнение методов обнаружения точек структурных изменений и
# применение их к реальным агрегированным рядам потребления.
#
# Что делает:
#   1) сравнивает методы (PELT, BinSeg, Window, CUSUM, BOCPD) на синтетических
#      рядах с известными точками разладки -> таблица precision/recall/F1/задержка;
#   2) строит агрегированные реальные ряды (вся страна, по категориям) и ищет
#      в них точки разладки каждым методом;
#   3) запускает онлайн-детектор шоков по остаткам прогноза на национальном ряде;
#   4) сохраняет агрегированные ряды и найденные точки для отчёта и лендинга.
#
# Запуск:
#     python scripts/03_run_changepoint.py
# ===========================================================================

import os
import sys
import copy

PATH_TO_THIS_FILE = os.path.abspath(__file__)
PATH_TO_SCRIPTS_DIR = os.path.dirname(PATH_TO_THIS_FILE)
PATH_TO_REPOSITORY_ROOT = os.path.dirname(PATH_TO_SCRIPTS_DIR)
if PATH_TO_REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, PATH_TO_REPOSITORY_ROOT)

import numpy as np
import pandas as pd

from src import utils
from src import data_loading
from src import changepoint as changepoint_module
from src import models_forecast
from src import events_calendar


def build_aggregated_series(consumption):
    """
    Построить агрегированные по стране ряды потребления:
        - по каждой из 6 категорий (сумма по всем территориям за месяц);
    Возвращает таблицу: month (строка), category, value_total.
    """
    aggregated = (
        consumption.groupby(["category", "month"])["value"]
        .sum()
        .reset_index(name="value_total")
    )
    aggregated["month"] = aggregated["month"].astype(str)
    aggregated = aggregated.sort_values(by=["category", "month"]).reset_index(drop=True)
    return aggregated


def run_synthetic_comparison(changepoint_config):
    """
    Сравнить все методы на синтетике и вернуть таблицу метрик.
    Для синтетики отключаем десезонализацию (у синтетических рядов нет
    сезонности), оставляя только стандартизацию.
    """
    config_for_synthetic = copy.deepcopy(changepoint_config)
    config_for_synthetic["preprocessing"]["deseasonalize"] = False

    synthetic_config = changepoint_config["evaluation"]["synthetic"]
    tolerance = changepoint_config["evaluation"]["tolerance"]

    list_of_methods = []
    methods_flags = changepoint_config["methods"]
    if methods_flags["ruptures_pelt"]:
        list_of_methods.append("ruptures_PELT")
    if methods_flags["ruptures_binseg"]:
        list_of_methods.append("ruptures_BinSeg")
    if methods_flags["ruptures_window"]:
        list_of_methods.append("ruptures_Window")
    if methods_flags["cusum"]:
        list_of_methods.append("CUSUM")
    if methods_flags["bocpd"]:
        list_of_methods.append("BOCPD")

    collected_rows = []
    for method_name in list_of_methods:
        metrics = changepoint_module.evaluate_detector_on_synthetic(
            method_name, config_for_synthetic, synthetic_config, tolerance
        )
        collected_rows.append(metrics)

    metrics_table = pd.DataFrame(collected_rows)
    metrics_table = metrics_table.sort_values(by="f1", ascending=False).reset_index(drop=True)
    return metrics_table


def detect_on_real_aggregated(aggregated_series, changepoint_config):
    """
    Применить все методы обнаружения разладок к каждому агрегированному ряду
    (по категориям). Возвращает таблицу: category, method, detected_months.
    """
    collected_rows = []

    list_of_categories = sorted(aggregated_series["category"].unique().tolist())
    for category_name in list_of_categories:
        category_subset = aggregated_series[
            aggregated_series["category"] == category_name
        ].sort_values(by="month")
        series_values = category_subset["value_total"].values
        list_of_months = category_subset["month"].tolist()

        results_by_method = changepoint_module.detect_changepoints_all_methods(
            series_values, changepoint_config
        )

        for method_name, detected_indices in results_by_method.items():
            detected_months = []
            for index_value in detected_indices:
                if 0 <= index_value < len(list_of_months):
                    detected_months.append(list_of_months[index_value])
            one_row = {
                "category": category_name,
                "method": method_name,
                "detected_months": "; ".join(detected_months),
                "n_detected": len(detected_months),
            }
            collected_rows.append(one_row)

    detections_table = pd.DataFrame(collected_rows)
    return detections_table


def run_online_shock_detector(aggregated_series, changepoint_config):
    """
    Запустить онлайн-детектор шоков по остаткам прогноза на национальном ряде
    категории "Все категории".

    В качестве простого прозрачного прогноза используем сезонный наивный
    прогноз (значение год назад) — он не требует обучения и понятен. Остатки
    (факт минус прогноз) показывают, где потребление отклонилось от ожидаемого.
    """
    online_config = changepoint_config["online_shock_detector"]

    total_series = aggregated_series[
        aggregated_series["category"] == "Все категории"
    ].sort_values(by="month")
    series_values = total_series["value_total"].values
    list_of_months = total_series["month"].tolist()

    # Сезонный наивный прогноз на 1 шаг: ожидаемое значение = значение год назад.
    season_length = 12
    forecast_values = np.full(len(series_values), np.nan, dtype="float64")
    for position in range(len(series_values)):
        if position >= season_length:
            forecast_values[position] = series_values[position - season_length]
        else:
            # Для первого года прошлого нет — берём предыдущее значение.
            if position > 0:
                forecast_values[position] = series_values[position - 1]
            else:
                forecast_values[position] = series_values[position]

    residual_table = changepoint_module.detect_shocks_from_forecast_residuals(
        series_values, forecast_values, online_config
    )
    residual_table["month"] = list_of_months
    residual_table["actual"] = series_values
    residual_table["forecast"] = forecast_values
    return residual_table


def main():
    data_config = utils.load_yaml_config(
        utils.make_absolute_path("configs/data.yaml")
    )
    changepoint_config = utils.load_yaml_config(
        utils.make_absolute_path("configs/changepoint.yaml")
    )

    metrics_directory = utils.make_absolute_path("outputs/metrics")
    forecasts_directory = utils.make_absolute_path("outputs/forecasts")
    external_directory = utils.make_absolute_path("data/external")
    utils.ensure_directory_exists(metrics_directory)
    utils.ensure_directory_exists(forecasts_directory)
    utils.ensure_directory_exists(external_directory)

    print("Загружаю данные...", flush=True)
    prepared_data = data_loading.load_full_panel(data_config)
    consumption = prepared_data["consumption"]

    print("Строю агрегированные ряды...", flush=True)
    aggregated_series = build_aggregated_series(consumption)
    aggregated_path = os.path.join(forecasts_directory, "aggregated_series.csv")
    utils.save_dataframe_as_csv(aggregated_series, aggregated_path)

    print("Сравниваю методы обнаружения разладок на синтетике...", flush=True)
    synthetic_metrics = run_synthetic_comparison(changepoint_config)
    synthetic_path = os.path.join(metrics_directory, "changepoint_synthetic_metrics.csv")
    utils.save_dataframe_as_csv(synthetic_metrics, synthetic_path)
    print(synthetic_metrics.to_string(index=False), flush=True)

    print("Ищу точки разладки в реальных агрегированных рядах...", flush=True)
    real_detections = detect_on_real_aggregated(aggregated_series, changepoint_config)
    real_path = os.path.join(metrics_directory, "changepoint_real_detections.csv")
    utils.save_dataframe_as_csv(real_detections, real_path)
    print(real_detections.to_string(index=False), flush=True)

    print("Запускаю онлайн-детектор шоков по остаткам прогноза...", flush=True)
    shock_table = run_online_shock_detector(aggregated_series, changepoint_config)
    shock_path = os.path.join(forecasts_directory, "shock_alerts_national.csv")
    utils.save_dataframe_as_csv(shock_table, shock_path)

    alert_months = shock_table[shock_table["is_alert"]]["month"].tolist()
    print("Месяцы с ранним сигналом шока (национальный ряд):", alert_months, flush=True)

    # Сохраняем календарь реальных событий рядом (для сверки в отчёте/лендинге).
    events_table = events_calendar.build_events_dataframe()
    events_path = os.path.join(external_directory, "events_calendar.csv")
    events_table.to_csv(events_path, index=False, encoding="utf-8-sig")

    print("Готово. Результаты сохранены в outputs/.", flush=True)


if __name__ == "__main__":
    main()

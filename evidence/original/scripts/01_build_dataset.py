# ===========================================================================
# Скрипт: подготовка данных и первичный анализ (EDA).
#
# Что делает:
#   1) загружает и фильтрует панель расходов;
#   2) сохраняет короткую сводку по данным (JSON);
#   3) считает и сохраняет агрегированные по стране ряды (для графиков);
#   4) сохраняет календарь событий.
#
# Запуск:
#     python scripts/01_build_dataset.py
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
from src import events_calendar


def main():
    data_config = utils.load_yaml_config(
        utils.make_absolute_path("configs/data.yaml")
    )

    processed_directory = utils.make_absolute_path("data/processed")
    external_directory = utils.make_absolute_path("data/external")
    metrics_directory = utils.make_absolute_path("outputs/metrics")
    forecasts_directory = utils.make_absolute_path("outputs/forecasts")
    for one_directory in [processed_directory, external_directory,
                          metrics_directory, forecasts_directory]:
        utils.ensure_directory_exists(one_directory)

    print("Загружаю и готовлю данные...", flush=True)
    prepared_data = data_loading.load_full_panel(data_config)
    consumption = prepared_data["consumption"]
    summary = prepared_data["summary"]

    # Сохраняем сводку по данным.
    summary_path = os.path.join(metrics_directory, "dataset_summary.json")
    utils.save_dictionary_as_json(summary, summary_path)
    print("Сводка по данным:", flush=True)
    for key_name, value in summary.items():
        print("   " + str(key_name) + ": " + str(value), flush=True)

    # Сохраняем обработанную панель в parquet (компактно).
    processed_panel_path = os.path.join(processed_directory, "panel.parquet")
    panel_to_save = consumption.copy()
    panel_to_save["month"] = panel_to_save["month"].astype(str)
    panel_to_save.to_parquet(processed_panel_path, index=False)
    print("Обработанная панель сохранена: " + processed_panel_path, flush=True)

    # Агрегированные по стране ряды по категориям (для графиков лендинга).
    aggregated = (
        consumption.groupby(["category", "month"])["value"]
        .sum()
        .reset_index(name="value_total")
    )
    aggregated["month"] = aggregated["month"].astype(str)
    aggregated_path = os.path.join(forecasts_directory, "aggregated_series.csv")
    utils.save_dataframe_as_csv(aggregated, aggregated_path)
    print("Агрегированные ряды сохранены: " + aggregated_path, flush=True)

    # Календарь событий.
    events_table = events_calendar.build_events_dataframe()
    events_path = os.path.join(external_directory, "events_calendar.csv")
    events_table.to_csv(events_path, index=False, encoding="utf-8-sig")
    print("Календарь событий сохранён: " + events_path, flush=True)

    print("Готово.", flush=True)


if __name__ == "__main__":
    main()

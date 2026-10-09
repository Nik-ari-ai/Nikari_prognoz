# ===========================================================================
# Скрипт: демонстрация согласования новостей/событий с данными СберИндекса.
#
# Что делает:
#   1) берёт календарь реальных датированных событий (ставка ЦБ, курс рубля и
#      т.д.);
#   2) строит месячный новостной сигнал по регионам с корректной временной и
#      пространственной привязкой и защитой от заглядывания в будущее;
#   3) сохраняет новостной признак в файл, готовый для подключения в
#      прогнозную модель (флаг use_news_features в configs/forecast.yaml).
#
# Запуск:
#     python scripts/05_news_demo.py
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
from src import news_alignment


def main():
    data_config = utils.load_yaml_config(
        utils.make_absolute_path("configs/data.yaml")
    )
    news_config = utils.load_yaml_config(
        utils.make_absolute_path("configs/news.yaml")
    )

    external_directory = utils.make_absolute_path("data/external")
    utils.ensure_directory_exists(external_directory)

    print("Загружаю данные (для списка месяцев и регионов)...", flush=True)
    prepared_data = data_loading.load_full_panel(data_config)
    consumption = prepared_data["consumption"]
    territory_dictionary = prepared_data["territory_dictionary"]
    list_of_months = list(prepared_data["all_months"])

    # Список регионов, которые реально встречаются в данных расходов.
    territory_ids_in_data = consumption["territory_id"].unique().tolist()
    dictionary_subset = territory_dictionary[
        territory_dictionary["territory_id"].isin(territory_ids_in_data)
    ]
    list_of_region_codes = sorted(dictionary_subset["region_code"].dropna().unique().tolist())
    list_of_region_codes = [int(code) for code in list_of_region_codes]
    print("Регионов в данных: " + str(len(list_of_region_codes)), flush=True)

    # Сохраняем и загружаем календарь событий.
    events_path = os.path.join(external_directory, "events_calendar.csv")
    events_calendar.save_events_calendar(events_path)
    events_table = news_alignment.load_events_table(events_path)
    print("Событий в календаре: " + str(len(events_table)), flush=True)

    print("Строю месячный новостной сигнал по регионам...", flush=True)
    news_features_table = news_alignment.build_monthly_news_signal(
        events_table, list_of_months, list_of_region_codes, news_config
    )

    news_features_path = os.path.join(external_directory, "news_features.csv")
    news_alignment.save_news_features(news_features_table, news_features_path)
    print("Новостной признак сохранён: " + news_features_path, flush=True)

    # Показываем пример: федеральный сигнал по месяцам (берём любой регион,
    # так как федеральные события одинаковы для всех).
    example_region = list_of_region_codes[0]
    example_signal = news_features_table[
        news_features_table["region_code"] == example_region
    ].sort_values(by="month")
    example_signal_display = example_signal.copy()
    example_signal_display["month"] = example_signal_display["month"].astype(str)
    print("Пример новостного сигнала (федеральные события, один регион):", flush=True)
    print(example_signal_display.to_string(index=False), flush=True)

    # Демонстрация защиты от заглядывания в будущее.
    origin_month_example = list_of_months[17]  # например, июнь 2024
    allowed = news_alignment.verify_no_look_ahead(news_features_table, origin_month_example)
    print(
        "Для прогноза на момент "
        + str(origin_month_example)
        + " доступно строк новостного сигнала: "
        + str(len(allowed)),
        flush=True,
    )

    print("Готово.", flush=True)


if __name__ == "__main__":
    main()

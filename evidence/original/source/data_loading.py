# ===========================================================================
# Загрузка данных СберИндекса и построение панельного датасета.
#
# Что здесь происходит:
#   1) читаем основной файл расходов consumption.parquet;
#   2) читаем вспомогательные файлы (market_access, справочник территорий);
#   3) приводим даты к месячным периодам;
#   4) строим "панель" — для каждой пары (территория, категория) непрерывный
#      ряд по месяцам;
#   5) отбираем ряды достаточной длины.
# ===========================================================================

import numpy as np
import pandas as pd

from src import utils


def load_consumption_dataframe(data_config):
    """
    Прочитать основной датасет расходов и привести колонки к стандартным именам.

    Возвращает таблицу с колонками:
        territory_id : int
        category     : str
        month        : Period[M]  (месячный период)
        value        : float      (оценка расходов жителя МО, руб.)
    """
    path_to_consumption = utils.make_absolute_path(
        data_config["paths"]["consumption_file"]
    )
    raw_consumption = pd.read_parquet(path_to_consumption)

    # Имена колонок берём из конфига, чтобы код не ломался при переименованиях.
    column_date = data_config["columns"]["date"]
    column_territory = data_config["columns"]["territory_id"]
    column_category = data_config["columns"]["category"]
    column_value = data_config["columns"]["value"]

    consumption = pd.DataFrame()
    consumption["territory_id"] = raw_consumption[column_territory].astype("int64")
    consumption["category"] = raw_consumption[column_category].astype("string")
    consumption["value"] = raw_consumption[column_value].astype("float64")

    # Превращаем строку даты "2023-01" в месячный период.
    consumption["month"] = pd.PeriodIndex(
        raw_consumption[column_date].astype("string"), freq="M"
    )

    # Упорядочиваем строки — так удобнее строить лаги дальше.
    consumption = consumption.sort_values(
        by=["territory_id", "category", "month"]
    ).reset_index(drop=True)

    return consumption


def load_market_access_dataframe(data_config):
    """
    Прочитать индекс доступности рынков (статический признак по каждому МО).
    Возвращает таблицу: territory_id, market_access.
    """
    path_to_market_access = utils.make_absolute_path(
        data_config["paths"]["market_access_file"]
    )
    market_access = pd.read_parquet(path_to_market_access)
    market_access["territory_id"] = market_access["territory_id"].astype("int64")
    market_access["market_access"] = market_access["market_access"].astype("float64")
    return market_access


def load_territory_dictionary(data_config):
    """
    Прочитать справочник территорий и оставить одну актуальную версию на
    каждый territory_id (самую свежую по году начала действия записи).

    Возвращает таблицу с колонками:
        territory_id, region_code, region_name,
        center_latitude, center_longitude, municipal_district_type
    """
    path_to_dictionary = utils.make_absolute_path(
        data_config["paths"]["territory_dict_file"]
    )
    raw_dictionary = pd.read_excel(path_to_dictionary)

    # В справочнике несколько версий одного МО (поле year_from). Берём самую
    # свежую версию: сортируем по году начала и оставляем последнюю строку.
    sorted_dictionary = raw_dictionary.sort_values(by=["territory_id", "year_from"])
    latest_version_per_territory = sorted_dictionary.drop_duplicates(
        subset=["territory_id"], keep="last"
    )

    territory_dictionary = pd.DataFrame()
    territory_dictionary["territory_id"] = (
        latest_version_per_territory["territory_id"].astype("int64")
    )
    territory_dictionary["region_code"] = (
        latest_version_per_territory["region_code"].astype("int64")
    )
    territory_dictionary["region_name"] = (
        latest_version_per_territory["region_name"].astype("string")
    )
    territory_dictionary["center_latitude"] = (
        latest_version_per_territory["municipal_district_center_lat"].astype("float64")
    )
    territory_dictionary["center_longitude"] = (
        latest_version_per_territory["municipal_district_center_lon"].astype("float64")
    )
    territory_dictionary["municipal_district_type"] = (
        latest_version_per_territory["municipal_district_type"].astype("string")
    )

    territory_dictionary = territory_dictionary.reset_index(drop=True)
    return territory_dictionary


def build_complete_monthly_index(consumption):
    """
    Построить полный список месяцев от минимального до максимального,
    встречающегося в данных. Нужен, чтобы находить пропуски в рядах.
    """
    earliest_month = consumption["month"].min()
    latest_month = consumption["month"].max()
    complete_month_range = pd.period_range(
        start=earliest_month, end=latest_month, freq="M"
    )
    return complete_month_range


def select_series_with_enough_history(consumption, data_config):
    """
    Оставить только те ряды (пары территория-категория), у которых достаточно
    месяцев с данными. Порог задан в конфиге (min_months_per_series).

    Возвращает отфильтрованную таблицу расходов.
    """
    minimum_required_months = data_config["filtering"]["min_months_per_series"]

    # Считаем число месяцев для каждой пары (территория, категория).
    months_count_per_series = (
        consumption.groupby(["territory_id", "category"])["month"]
        .nunique()
        .reset_index(name="months_available")
    )

    # Отбираем пары, где месяцев достаточно.
    series_passing_filter = months_count_per_series[
        months_count_per_series["months_available"] >= minimum_required_months
    ]

    # Оставляем в основной таблице только прошедшие фильтр ряды.
    key_of_passing_series = series_passing_filter[["territory_id", "category"]]
    filtered_consumption = consumption.merge(
        key_of_passing_series, on=["territory_id", "category"], how="inner"
    )

    return filtered_consumption


def describe_dataset(consumption):
    """
    Собрать короткую сводку по датасету для лога и отчёта.
    Возвращает словарь с ключевыми числами.
    """
    summary = {}
    summary["n_rows"] = int(len(consumption))
    summary["n_territories"] = int(consumption["territory_id"].nunique())
    summary["n_categories"] = int(consumption["category"].nunique())
    summary["n_months"] = int(consumption["month"].nunique())
    summary["month_min"] = str(consumption["month"].min())
    summary["month_max"] = str(consumption["month"].max())
    summary["value_mean"] = float(consumption["value"].mean())
    summary["value_median"] = float(consumption["value"].median())
    summary["categories"] = sorted(
        [str(category) for category in consumption["category"].unique()]
    )
    return summary


def load_full_panel(data_config):
    """
    Главная функция модуля: загрузить и подготовить всё сразу.

    Возвращает словарь:
        "consumption"          — отфильтрованная таблица расходов (панель),
        "market_access"        — статический индекс по МО,
        "territory_dictionary" — справочник территорий,
        "all_months"           — полный список месяцев,
        "summary"              — сводка по данным.
    """
    consumption = load_consumption_dataframe(data_config)
    consumption = select_series_with_enough_history(consumption, data_config)

    market_access = load_market_access_dataframe(data_config)
    territory_dictionary = load_territory_dictionary(data_config)
    all_months = build_complete_monthly_index(consumption)
    summary = describe_dataset(consumption)

    prepared_data = {
        "consumption": consumption,
        "market_access": market_access,
        "territory_dictionary": territory_dictionary,
        "all_months": all_months,
        "summary": summary,
    }
    return prepared_data

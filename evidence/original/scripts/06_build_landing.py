# ===========================================================================
# Скрипт: сборка интерактивного лендинга из реальных результатов.
#
# Читает готовые CSV/JSON из outputs/ и data/external, формирует компактный
# JSON с данными и вставляет его в HTML-шаблон (reports/landing_template.html),
# заменяя маркер __INJECT_DATA__. Итог: outputs/landing/index.html.
#
# Так числа на странице всегда соответствуют реально посчитанным результатам
# (никакого ручного копирования).
#
# Запуск:
#     python scripts/06_build_landing.py
# ===========================================================================

import os
import sys
import json

PATH_TO_THIS_FILE = os.path.abspath(__file__)
PATH_TO_SCRIPTS_DIR = os.path.dirname(PATH_TO_THIS_FILE)
PATH_TO_REPOSITORY_ROOT = os.path.dirname(PATH_TO_SCRIPTS_DIR)
if PATH_TO_REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, PATH_TO_REPOSITORY_ROOT)

import pandas as pd

from src import utils


def read_csv_safely(path_to_csv):
    """Прочитать CSV, если он есть; иначе вернуть пустую таблицу."""
    if os.path.exists(path_to_csv):
        return pd.read_csv(path_to_csv, encoding="utf-8-sig")
    return pd.DataFrame()


def build_data_dictionary():
    """
    Собрать словарь со всеми данными для лендинга из файлов outputs/.
    """
    metrics_dir = utils.make_absolute_path("outputs/metrics")
    forecasts_dir = utils.make_absolute_path("outputs/forecasts")
    external_dir = utils.make_absolute_path("data/external")

    data_for_page = {}

    # --- Сводка по данным ---------------------------------------------------
    summary_path = os.path.join(metrics_dir, "dataset_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as opened_file:
            data_for_page["summary"] = json.load(opened_file)
    else:
        data_for_page["summary"] = {}

    # --- Метрики прогноза (выровненные, честное сравнение) ------------------
    forecast_metrics = read_csv_safely(
        os.path.join(metrics_dir, "forecast_metrics_by_horizon_aligned.csv")
    )
    forecast_rows = []
    for _, row in forecast_metrics.iterrows():
        forecast_rows.append({
            "model": str(row["model"]),
            "horizon": int(row["horizon"]),
            "MAE": float(row["MAE"]),
            "R2": float(row["R2"]),
            "sMAPE": float(row["sMAPE"]),
            "n_points": int(row["n_points"]),
        })
    data_for_page["forecast_metrics"] = forecast_rows

    # --- Метрики CPD на синтетике ------------------------------------------
    cpd_metrics = read_csv_safely(
        os.path.join(metrics_dir, "changepoint_synthetic_metrics.csv")
    )
    cpd_rows = []
    for _, row in cpd_metrics.iterrows():
        cpd_rows.append({
            "method": str(row["method"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "mean_abs_delay": float(row["mean_abs_delay"]),
        })
    data_for_page["cpd_metrics"] = cpd_rows

    # --- Реальные детекции по категориям -----------------------------------
    real_detections = read_csv_safely(
        os.path.join(metrics_dir, "changepoint_real_detections.csv")
    )
    detection_rows = []
    for _, row in real_detections.iterrows():
        detection_rows.append({
            "category": str(row["category"]),
            "method": str(row["method"]),
            "detected_months": str(row["detected_months"]) if pd.notna(row["detected_months"]) else "",
            "n_detected": int(row["n_detected"]),
        })
    data_for_page["real_detections"] = detection_rows

    # --- Агрегированные ряды (для графиков) --------------------------------
    aggregated = read_csv_safely(os.path.join(forecasts_dir, "aggregated_series.csv"))
    series_by_category = {}
    if len(aggregated) > 0:
        for category_name, subset in aggregated.groupby("category"):
            subset = subset.sort_values(by="month")
            series_by_category[str(category_name)] = {
                "months": [str(month) for month in subset["month"].tolist()],
                "values": [float(value) for value in subset["value_total"].tolist()],
            }
    data_for_page["aggregated_series"] = series_by_category

    # --- Онлайн-детектор шоков (национальный ряд) --------------------------
    shock_table = read_csv_safely(os.path.join(forecasts_dir, "shock_alerts_national.csv"))
    shock_rows = []
    for _, row in shock_table.iterrows():
        shock_rows.append({
            "month": str(row["month"]),
            "actual": float(row["actual"]),
            "forecast": float(row["forecast"]),
            "standardized_residual": float(row["standardized_residual"]),
            "is_alert": bool(row["is_alert"]),
        })
    data_for_page["shock_series"] = shock_rows

    # --- Важности признаков ------------------------------------------------
    importances = read_csv_safely(os.path.join(metrics_dir, "feature_importances.csv"))
    importance_rows = []
    for _, row in importances.iterrows():
        importance_rows.append({
            "feature": str(row["feature"]),
            "importance": float(row["importance"]),
        })
    data_for_page["feature_importances"] = importance_rows

    # --- Новостной сигнал (один регион, федеральные события) ---------------
    news_features = read_csv_safely(os.path.join(external_dir, "news_features.csv"))
    news_rows = []
    if len(news_features) > 0:
        first_region = sorted(news_features["region_code"].unique().tolist())[0]
        news_subset = news_features[news_features["region_code"] == first_region].sort_values(by="month")
        for _, row in news_subset.iterrows():
            news_rows.append({
                "month": str(row["month"]),
                "news_signal": float(row["news_signal"]),
            })
    data_for_page["news_signal"] = news_rows

    # --- Календарь событий --------------------------------------------------
    events = read_csv_safely(os.path.join(external_dir, "events_calendar.csv"))
    event_rows = []
    for _, row in events.iterrows():
        event_rows.append({
            "date": str(row["date"])[:10],
            "month": str(row["month"]),
            "title": str(row["title"]),
            "event_type": str(row["event_type"]),
            "direction": int(row["direction"]),
            "intensity": int(row["intensity"]),
        })
    data_for_page["events"] = event_rows

    return data_for_page


def main():
    data_for_page = build_data_dictionary()

    template_path = utils.make_absolute_path("reports/landing_template.html")
    with open(template_path, "r", encoding="utf-8") as opened_file:
        template_html = opened_file.read()

    data_as_json = json.dumps(data_for_page, ensure_ascii=False)
    injection_script = "window.__DATA__ = " + data_as_json + ";"

    output_html = template_html.replace("/*__INJECT_DATA__*/", injection_script)

    landing_dir = utils.make_absolute_path("outputs/landing")
    utils.ensure_directory_exists(landing_dir)
    output_path = os.path.join(landing_dir, "index.html")
    with open(output_path, "w", encoding="utf-8") as opened_file:
        opened_file.write(output_html)

    print("Лендинг собран: " + output_path, flush=True)
    print("Размер, КБ: " + str(round(len(output_html) / 1024, 1)), flush=True)


if __name__ == "__main__":
    main()

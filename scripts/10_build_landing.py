# ===========================================================================
# Скрипт: сборка интерактивного лендинга из ПРОВЕРЕННЫХ (исправленных)
# результатов. Читает outputs/checked, outputs/audit, outputs/detectors,
# data/external и reports, формирует JSON и вставляет его в HTML-шаблон
# reports/landing_template.html (маркер __INJECT_DATA__).
#
# Итог: outputs/landing/index.html. Все числа берутся из посчитанных файлов,
# ничего не вписывается вручную.
#
# Запуск:
#     python scripts/10_build_landing.py
# ===========================================================================

import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd


def read_csv_safely(path_to_csv):
    if os.path.exists(path_to_csv):
        return pd.read_csv(path_to_csv, encoding="utf-8-sig")
    return pd.DataFrame()


def read_json_safely(path_to_json):
    if os.path.exists(path_to_json):
        with open(path_to_json, "r", encoding="utf-8") as source:
            return json.load(source)
    return {}


def metrics_table_to_rows(metrics_table):
    collected_rows = []
    for _, row in metrics_table.iterrows():
        one = {
            "model": str(row["model"]),
            "horizon": int(row["horizon"]),
            "MAE": float(row["MAE"]),
            "R2": float(row["R2"]),
            "n_points": int(row["n_points"]),
        }
        # sMAPE присутствует не во всех таблицах.
        if "sMAPE" in metrics_table.columns:
            one["sMAPE"] = float(row["sMAPE"])
        else:
            one["sMAPE"] = None
        collected_rows.append(one)
    return collected_rows


def build_data_dictionary():
    data_for_page = {}

    # --- Метрики проверенного эксперимента ---------------------------------
    checked_metrics = read_csv_safely(ROOT / "outputs" / "checked" / "metrics.csv")
    data_for_page["checked_metrics"] = metrics_table_to_rows(checked_metrics)

    # --- Метрики при истории >= 12 месяцев (ответ на завышенный выигрыш) ---
    after12 = read_csv_safely(ROOT / "outputs" / "audit" / "metrics_after_12_months.csv")
    data_for_page["after12_metrics"] = metrics_table_to_rows(after12)

    # --- Исходные (старые) метрики для сравнения "до/после" ----------------
    recomputed = read_csv_safely(ROOT / "outputs" / "audit" / "recomputed_metrics.csv")
    data_for_page["original_metrics"] = metrics_table_to_rows(recomputed)

    # --- Метрики последовательных детекторов (синтетика) -------------------
    detectors = read_csv_safely(ROOT / "outputs" / "detectors" / "synthetic_test_metrics.csv")
    detector_rows = []
    for _, row in detectors.iterrows():
        detector_rows.append({
            "method": str(row["method"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "delay": float(row["mean_detection_delay"]),
            "false_per_100": float(row["false_alarms_per_100_points"]),
        })
    data_for_page["detector_metrics"] = detector_rows

    # --- События ЦБ (реальные, с источниками) ------------------------------
    cbr = read_csv_safely(ROOT / "data" / "external" / "cbr_rate_events.csv")
    cbr_rows = []
    for _, row in cbr.iterrows():
        cbr_rows.append({
            "event_date": str(row["event_date"]),
            "rate_change_pp": float(row["rate_change_pp"]),
            "new_rate_pct": float(row["new_rate_pct"]),
            "source_url": str(row["source_url"]),
        })
    data_for_page["cbr_events"] = cbr_rows

    # --- Реальный пример (МО 2007) -----------------------------------------
    example = read_csv_safely(ROOT / "reports" / "real_example.csv")
    example_rows = []
    for _, row in example.iterrows():
        example_rows.append({
            "model": str(row["model"]),
            "horizon": int(row["horizon"]),
            "actual": float(row["actual"]),
            "predicted": float(row["predicted"]),
        })
    data_for_page["example_rows"] = example_rows

    # --- Протокол и воспроизводимость --------------------------------------
    run_info = read_json_safely(ROOT / "outputs" / "checked" / "run.json")
    audit_info = read_json_safely(ROOT / "outputs" / "audit" / "audit.json")
    data_for_page["run_info"] = run_info
    data_for_page["audit_info"] = audit_info

    # --- Независимый финальный тест ----------------------------------------
    data_for_page["final_test"] = read_json_safely(ROOT / "outputs" / "final_test" / "final_test.json")
    final_val = read_csv_safely(ROOT / "outputs" / "final_test" / "validation_metrics.csv")
    final_fin = read_csv_safely(ROOT / "outputs" / "final_test" / "final_test_metrics.csv")
    data_for_page["final_test_validation"] = metrics_table_to_rows(final_val) if len(final_val) else []
    data_for_page["final_test_final"] = metrics_table_to_rows(final_fin) if len(final_fin) else []

    # --- Реальная оценка детекторов (слепые метки) -------------------------
    real_det = read_csv_safely(ROOT / "outputs" / "shocks" / "real_detector_eval.csv")
    real_det_rows = []
    for _, row in real_det.iterrows():
        real_det_rows.append({
            "method": str(row["method"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "delay": float(row["mean_detection_delay"]),
        })
    data_for_page["real_detector_eval"] = real_det_rows
    data_for_page["shocks_summary"] = read_json_safely(ROOT / "outputs" / "shocks" / "shocks_summary.json")

    # --- Модель риска события ----------------------------------------------
    data_for_page["risk_eval"] = read_json_safely(ROOT / "outputs" / "risk" / "risk_eval.json")

    # --- Раннее предупреждение: rolling-origin + базовые правила (scripts/15)
    # Короткий прогрев даёт несколько ВРЕМЕННЫХ обучающих срезов; обученную
    # модель сравниваем с простыми правилами при ОДИНАКОВОМ бюджете тревог.
    risk_tscv_summary = read_json_safely(ROOT / "outputs" / "risk_tscv" / "summary.json")
    data_for_page["risk_tscv_summary"] = risk_tscv_summary

    risk_tscv_methods = read_csv_safely(ROOT / "outputs" / "risk_tscv" / "methods_comparison.csv")
    risk_tscv_rows = []
    for _, row in risk_tscv_methods.iterrows():
        def numeric_or_none(column_name):
            value = row[column_name]
            if pd.isna(value):
                return None
            return float(value)
        risk_tscv_rows.append({
            "method": str(row["method"]),
            "total_alarms": int(row["total_alarms"]),
            "total_hits": int(row["total_hits"]),
            "total_false_alarms": int(row["total_false_alarms"]),
            "total_events": int(row["total_events"]),
            "event_precision": numeric_or_none("event_precision"),
            "event_recall": numeric_or_none("event_recall"),
            "false_per_100": numeric_or_none("false_alarms_per_100_mo_months"),
            "mean_lead": numeric_or_none("mean_lead_time_months"),
        })
    data_for_page["risk_tscv_methods"] = risk_tscv_rows

    # --- Like-for-like (scripts/18): модель против правил на событиях scripts/13
    likeforlike = read_csv_safely(ROOT / "outputs" / "risk_likeforlike" / "methods_comparison.csv")
    likeforlike_rows = []
    for _, row in likeforlike.iterrows():
        def lf_num(column_name):
            value = row[column_name]
            if pd.isna(value):
                return None
            return float(value)
        likeforlike_rows.append({
            "method": str(row["method"]),
            "event_hits": int(row["event_hits"]),
            "onset_events_eligible": int(row["onset_events_eligible"]),
            "event_recall": lf_num("event_recall"),
            "event_precision": lf_num("event_precision"),
        })
    data_for_page["likeforlike_methods"] = likeforlike_rows

    # --- Устойчивый сдвиг уровня (scripts/19): редкость и не-декабрьскость -----
    levelshift_summary = read_json_safely(ROOT / "outputs" / "risk_levelshift" / "summary.json")
    levelshift_methods = read_csv_safely(ROOT / "outputs" / "risk_levelshift" / "methods_comparison.csv")
    best_rule_recall = None
    model_recall = None
    for _, row in levelshift_methods.iterrows():
        recall_value = row["event_recall"]
        if pd.isna(recall_value):
            continue
        if str(row["method"]) == "logistic":
            model_recall = float(recall_value)
        else:
            if best_rule_recall is None or float(recall_value) > best_rule_recall:
                best_rule_recall = float(recall_value)
    data_for_page["levelshift_summary"] = {
        "total": levelshift_summary.get("level_shift_onsets_total"),
        "december": levelshift_summary.get("december_onsets"),
        "distribution": levelshift_summary.get("month_distribution", {}),
        "logistic_trained": levelshift_summary.get("logistic_trained_in_any_mature_fold"),
        "scale_reference_cutoff": levelshift_summary.get("scale_reference_cutoff"),
        "model_recall": model_recall,
        "best_rule_recall": best_rule_recall,
    }

    # --- Единая pooled-таблица прогноза (scripts/21): главный результат ------
    unified = read_csv_safely(ROOT / "outputs" / "headline" / "forecast_unified_all_categories.csv")
    unified_summary = read_json_safely(ROOT / "outputs" / "headline" / "forecast_unified_summary.json")
    unified_rows = []
    for _, row in unified.iterrows():
        def unified_num(column_name):
            value = row[column_name]
            if pd.isna(value):
                return None
            return float(value)
        unified_rows.append({
            "model": str(row["model"]),
            "horizon": int(row["horizon"]),
            "MAE": unified_num("MAE"),
            "R2": unified_num("R2"),
            "mae_median_per_mo": unified_num("mae_median_per_mo"),
            "mae_p2_5_per_mo": unified_num("mae_p2_5_per_mo"),
            "mae_p97_5_per_mo": unified_num("mae_p97_5_per_mo"),
        })
    data_for_page["forecast_unified"] = unified_rows
    # winner_by_horizon = ОПИСАТЕЛЬНЫЙ бенчмарк-минимум (не независимый результат).
    data_for_page["forecast_unified_winners"] = unified_summary.get("winner_by_horizon", {})
    # Аудит v16: выбор и оценка разведены — пробрасываем все три поля на лендинг.
    data_for_page["forecast_unified_benchmark_best"] = unified_summary.get("benchmark_best_by_horizon", {})
    data_for_page["forecast_unified_selected_on_early"] = unified_summary.get("selected_on_early", {})
    data_for_page["forecast_unified_late_evaluation"] = unified_summary.get("late_evaluation", {})

    # --- Прикладной шок: robust rolling backtest (scripts/20) ---------------
    seasonal_summary = read_json_safely(ROOT / "outputs" / "risk_seasonal" / "summary.json")
    seasonal_methods = read_csv_safely(ROOT / "outputs" / "risk_seasonal" / "methods_comparison.csv")
    seasonal_rows = []
    for _, row in seasonal_methods.iterrows():
        def seasonal_num(column_name):
            value = row[column_name]
            if pd.isna(value):
                return None
            return float(value)
        seasonal_rows.append({
            "method": str(row["method"]),
            "event_recall": seasonal_num("event_recall"),
            "event_precision": seasonal_num("event_precision"),
            "pr_auc": seasonal_num("pr_auc"),
            "pr_auc_ci_low": seasonal_num("pr_auc_ci_low"),
            "pr_auc_ci_high": seasonal_num("pr_auc_ci_high"),
        })
    data_for_page["seasonal_methods"] = seasonal_rows
    data_for_page["seasonal_summary"] = {
        "onsets_total": seasonal_summary.get("onsets_total"),
        "december_share": seasonal_summary.get("december_share"),
        "comparable_folds": seasonal_summary.get("comparable_folds", []),
        "pooled_positive_rows": seasonal_summary.get("pooled_positive_rows"),
        "holdout_origin": seasonal_summary.get("holdout_origin"),
    }

    # Детализация по фолдам: обучающие срезы ТОЛЬКО зрелых фолдов (незрелый
    # ноябрь показываем отдельно как pending, его точность не оценивается).
    risk_tscv_folds = read_csv_safely(ROOT / "outputs" / "risk_tscv" / "folds_detail.csv")
    mature_train_months_by_origin = {}
    for _, row in risk_tscv_folds.iterrows():
        is_mature_fold = bool(row["mature"]) if "mature" in risk_tscv_folds.columns else True
        if is_mature_fold:
            mature_train_months_by_origin[str(row["test_origin"])] = int(row["train_months"])
    data_for_page["risk_tscv_train_slices"] = mature_train_months_by_origin

    # Зрелые и незрелые (pending) фолды, доля декабрьских onset — из summary.
    data_for_page["risk_tscv_mature_folds"] = risk_tscv_summary.get("mature_folds_evaluated", [])
    data_for_page["risk_tscv_pending_folds"] = risk_tscv_summary.get("pending_folds_not_evaluated", [])
    december_by_fold = risk_tscv_summary.get("december_concentration_by_fold", {})
    december_shares = []
    for origin_key in december_by_fold:
        one_fold = december_by_fold[origin_key]
        share_value = one_fold.get("december_share")
        if share_value is not None:
            december_shares.append(float(share_value))
    if len(december_shares) > 0:
        data_for_page["risk_tscv_december_min"] = min(december_shares)
        data_for_page["risk_tscv_december_max"] = max(december_shares)
    else:
        data_for_page["risk_tscv_december_min"] = None
        data_for_page["risk_tscv_december_max"] = None

    # --- Главный результат по цели «Все категории» -------------------------
    headline = read_csv_safely(ROOT / "outputs" / "headline" / "headline_lgbm_vs_prophet_total.csv")
    headline_rows = []
    for _, row in headline.iterrows():
        headline_rows.append({
            "horizon": int(row["horizon"]),
            "mae_prophet": float(row["MAE_a"]),
            "mae_lgbm": float(row["MAE_b"]),
            "reduction_pct": float(row["MAE_reduction_pct"]),
            "r2_prophet": float(row["R2_a"]),
            "r2_lgbm": float(row["R2_b"]),
            "n_points": int(row["n_points"]),
        })
    data_for_page["headline_total"] = headline_rows

    # Объединение всех категорий (где Prophet может выигрывать на h=12).
    pooled = read_csv_safely(ROOT / "outputs" / "headline" / "forecast_pooled_all_categories.csv")
    data_for_page["pooled_all"] = metrics_table_to_rows(pooled) if len(pooled) else []

    # Чистая абляция ЦБ/ИПЦ.
    ablation = read_csv_safely(ROOT / "outputs" / "headline" / "cpi_ablation.csv")
    ablation_rows = []
    for _, row in ablation.iterrows():
        entry = {"horizon": int(row["horizon"])}
        for col in ablation.columns:
            if col == "horizon":
                continue
            value = row[col]
            entry[col] = (float(value) if pd.notna(value) else None)
        ablation_rows.append(entry)
    data_for_page["cpi_ablation"] = ablation_rows

    # Фактическое покрытие по горизонтам (из checked_metrics n_points).
    coverage = {}
    for row in data_for_page["checked_metrics"]:
        coverage[str(row["horizon"])] = row["n_points"]
    data_for_page["coverage"] = coverage

    return data_for_page


def main():
    data_for_page = build_data_dictionary()

    template_path = ROOT / "reports" / "landing_template.html"
    with open(template_path, "r", encoding="utf-8") as source:
        template_html = source.read()

    data_as_json = json.dumps(data_for_page, ensure_ascii=False)
    injection = "window.__DATA__ = " + data_as_json + ";"
    output_html = template_html.replace("/*__INJECT_DATA__*/", injection)

    landing_dir = ROOT / "outputs" / "landing"
    landing_dir.mkdir(parents=True, exist_ok=True)
    output_path = landing_dir / "index.html"
    with open(output_path, "w", encoding="utf-8") as target:
        target.write(output_html)

    print("Лендинг собран: " + str(output_path))
    print("Размер, КБ: " + str(round(len(output_html) / 1024, 1)))


if __name__ == "__main__":
    main()

"""Раннее предупреждение сезонно-нормализованного шока: множество rolling-фолдов,
отдельный holdout, сравнение многих методов при равном бюджете, PR-AUC и
бутстрэп-доверительные интервалы.

Отвечает на приоритет аудита: прежние 25 событий на двух фолдах — мало для
сильного заявления. Здесь прикладное целевое событие (src/seasonal_shock.py),
каузальная разметка, непересекающиеся rolling-фолды (горизонт 1) + финальный
holdout, и сравнение методов:
  seasonal_naive, last_change, volatility, random top-K,
  logistic, LightGBM, CatBoost — при ОДИНАКОВОМ бюджете тревог.

Метрики: событийные recall/precision/ложные-на-100 МО-месяцев/опережение, а также
строчный PR-AUC (average precision) с бутстрэп 95% ДИ.

Честная рамка: один 2024 год, нет независимого будущего года; результат
исследовательский. Декабрьские календарные пики исключены нормализацией по
общему движению.
"""

import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import utils
from src import data_loading
from src import macro
from src import news
from src import seasonal_shock
from src import risk_eval


def build_feature_table(signal_table, common_by_month, onset_any,
                        cbr_month_change, cpi_signal_by_month, horizon):
    """Собрать каузальные признаки на месяц t и метку onset в t+1..t+horizon."""
    table = signal_table.sort_values(["territory_id", "month"]).reset_index(drop=True)
    onset_lookup = set(onset_any)

    collected_rows = []
    for territory_id, group in table.groupby("territory_id"):
        group = group.sort_values("month").reset_index(drop=True)
        months_here = list(group["month"])
        z_series = group["z"].to_numpy(dtype=float)
        r_series = group["r"].to_numpy(dtype=float)

        for position in range(len(group)):
            current_month = months_here[position]

            # Trailing волатильность z по [t-3, t-1] (строго прошлое).
            window_start = position - 3
            if window_start < 0:
                window_start = 0
            if position - 1 >= window_start and position >= 1:
                trailing_values = z_series[window_start:position]
                if len(trailing_values) >= 2:
                    trailing_z_vol = float(np.std(trailing_values))
                else:
                    trailing_z_vol = 0.0
            else:
                trailing_z_vol = 0.0

            onset_label = 0
            for step in range(1, horizon + 1):
                future_month = current_month + step
                if (int(territory_id), str(future_month)) in onset_lookup:
                    onset_label = 1
                    break

            collected_rows.append({
                "territory_id": int(territory_id),
                "month": str(current_month),
                "z": float(z_series[position]),
                "r": float(r_series[position]),
                "trailing_z_vol": trailing_z_vol,
                "common_movement": float(common_by_month.get(str(current_month), 0.0)),
                "rate_change": float(cbr_month_change.get(str(current_month), 0.0)),
                "cpi_signal": float(cpi_signal_by_month.get(current_month, 0.0)),
                "onset_label": int(onset_label),
            })
    result = pd.DataFrame(collected_rows)
    result["month_period"] = pd.PeriodIndex(result["month"].astype(str), freq="M")
    return result


def seasonal_naive_scores(test_frame, onset_any, horizon):
    """Сезонный наивный скор: был ли шок в тот же месяц ГОД назад.

    Для выпуска t (горизонт 1) целевой месяц — t+1; смотрим (МО, t+1-12).
    Скор = 1.0, если год назад в этом месяце был onset, иначе 0.0.
    """
    onset_lookup = set(onset_any)
    scores = []
    for _, row in test_frame.iterrows():
        release_month = pd.Period(str(row["month"]), freq="M")
        target_month = release_month + horizon
        last_year_month = target_month - 12
        if (int(row["territory_id"]), str(last_year_month)) in onset_lookup:
            scores.append(1.0)
        else:
            scores.append(0.0)
    return np.asarray(scores, dtype="float64")


def train_and_score_models(train, test, feature_columns):
    """Обучить logistic/LightGBM/CatBoost и вернуть словарь тестовых скоров.

    Если обучение невозможно (один класс / мало примеров), метод пропускается.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    train_positive = int(train["onset_label"].sum())
    both_classes = len(np.unique(train["onset_label"])) == 2
    if train_positive < 5 or not both_classes:
        return {}

    train_x = train[feature_columns].to_numpy(dtype=float)
    test_x = test[feature_columns].to_numpy(dtype=float)
    train_y = train["onset_label"].to_numpy(dtype=int)

    model_scores = {}

    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)
    logistic = LogisticRegression(class_weight="balanced", max_iter=1000)
    logistic.fit(train_x_scaled, train_y)
    model_scores["logistic"] = logistic.predict_proba(test_x_scaled)[:, 1]

    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=31, learning_rate=0.05, n_jobs=2,
        random_state=42, deterministic=True, force_col_wise=True,
        class_weight="balanced", verbose=-1
    )
    lgb_model.fit(train_x, train_y)
    model_scores["lightgbm"] = lgb_model.predict_proba(test_x)[:, 1]

    cat_model = CatBoostClassifier(
        iterations=200, depth=6, learning_rate=0.05, random_seed=42,
        verbose=0, allow_writing_files=False, auto_class_weights="Balanced"
    )
    cat_model.fit(train_x, train_y)
    model_scores["catboost"] = cat_model.predict_proba(test_x)[:, 1]

    return model_scores


def evaluate_one_fold(test, scores, territory_ids, budget_rate, horizon, onset_any):
    """Равный бюджет top-K + событийная оценка. Возвращает (метрики, маска, alarms)."""
    budget_k = max(int(round(len(test) * budget_rate)), 1)
    alarm_mask = risk_eval.top_k_alarm_mask(np.asarray(scores, dtype="float64"), budget_k, territory_ids)
    metrics = risk_eval.event_level_evaluation(
        test[["territory_id", "month"]], np.asarray(scores, dtype="float64"),
        None, horizon, onset_any, alarm_mask=alarm_mask
    )
    return metrics, alarm_mask, int(np.sum(alarm_mask))


def bootstrap_pr_auc(labels_by_method_rows, scores_by_method, territory_ids_rows,
                     n_samples, seed, confidence):
    """КЛАСТЕРНЫЙ бутстрэп-ДИ для PR-AUC (average precision).

    ПОПРАВКА ОБРАТНОЙ СВЯЗИ: наблюдения одного МО зависимы (панельные данные),
    поэтому ресэмпл отдельных СТРОК даёт слишком оптимистичные интервалы. Здесь
    ресэмплируются КЛАСТЕРЫ — territory_id: на каждой бутстрэп-итерации с
    возвращением выбираются МО, и берутся ВСЕ строки выбранных МО. Это учитывает
    внутрикластерную зависимость и честнее отражает неопределённость.

    Один и тот же ресэмпл кластеров применяется ко всем методам (парное сравнение).
    Возвращает словарь method -> (point, low, high).
    """
    from sklearn.metrics import average_precision_score
    labels = np.asarray(labels_by_method_rows, dtype=int)
    territories = np.asarray(territory_ids_rows)
    random_generator = np.random.default_rng(seed)

    # Индексы строк, сгруппированные по МО (кластеру).
    rows_by_territory = {}
    for row_index in range(len(territories)):
        territory_value = int(territories[row_index])
        rows_by_territory.setdefault(territory_value, []).append(row_index)
    unique_territories = sorted(rows_by_territory.keys())
    territory_count = len(unique_territories)

    point_estimates = {}
    for method_name in scores_by_method:
        method_scores = np.asarray(scores_by_method[method_name], dtype="float64")
        if labels.sum() > 0:
            point_estimates[method_name] = float(average_precision_score(labels, method_scores))
        else:
            point_estimates[method_name] = None

    bootstrap_values = {}
    for method_name in scores_by_method:
        bootstrap_values[method_name] = []

    for _ in range(n_samples):
        # Выбираем МО (кластеры) с возвращением и собираем все их строки.
        sampled_territory_positions = random_generator.integers(0, territory_count, size=territory_count)
        sampled_indices = []
        for one_position in sampled_territory_positions:
            territory_value = unique_territories[int(one_position)]
            sampled_indices.extend(rows_by_territory[territory_value])
        sampled_indices = np.asarray(sampled_indices, dtype=int)
        sampled_labels = labels[sampled_indices]
        if sampled_labels.sum() == 0:
            continue
        for method_name in scores_by_method:
            sampled_scores = np.asarray(scores_by_method[method_name], dtype="float64")[sampled_indices]
            bootstrap_values[method_name].append(
                float(average_precision_score(sampled_labels, sampled_scores))
            )

    lower_quantile = (1.0 - confidence) / 2.0
    upper_quantile = 1.0 - lower_quantile
    result = {}
    for method_name in scores_by_method:
        values = np.asarray(bootstrap_values[method_name], dtype="float64")
        if len(values) > 0:
            low = float(np.quantile(values, lower_quantile))
            high = float(np.quantile(values, upper_quantile))
        else:
            low = None
            high = None
        result[method_name] = (point_estimates[method_name], low, high)
    return result


def main(output_dir=None):
    data_config = utils.load_yaml_config(ROOT / "configs" / "data.yaml")
    risk_config = utils.load_yaml_config(ROOT / "configs" / "risk_seasonal.yaml")

    target = risk_config["target_definition"]
    reference_cutoff = target["scale_reference_cutoff"]
    persistence = int(target["persistence"])
    persist_quantile = float(target["persist_quantile"])
    single_quantile = float(target["single_quantile"])
    horizon = int(target["horizon"])
    test_origins = list(risk_config["protocol"]["test_origins"])
    holdout_origin_text = risk_config["protocol"]["holdout_origin"]
    budget_rate = float(risk_config["alarm_budget"]["target_alarm_rate"])
    bootstrap_samples = int(risk_config["bootstrap"]["samples"])
    bootstrap_seed = int(risk_config["bootstrap"]["seed"])
    bootstrap_confidence = float(risk_config["bootstrap"]["confidence"])
    feature_columns = list(risk_config["feature_columns"])

    if output_dir is None:
        destination = ROOT / "outputs" / "risk_seasonal"
    else:
        destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    print("Загружаю данные и строю сезонно-нормализованный сигнал...", flush=True)
    consumption = data_loading.load_consumption_dataframe(data_config)
    total = consumption[consumption["category"] == "Все категории"]
    wide_total = total.pivot_table(index="territory_id", columns="month", values="value")
    wide_total = wide_total.sort_index(axis=1).dropna(axis=0, how="any")
    last_observed_month = max(wide_total.columns)

    mom_table = seasonal_shock.build_month_over_month_table(wide_total)
    common_by_month = seasonal_shock.compute_common_movement(mom_table)
    signal_table, pooled_floor = seasonal_shock.build_seasonal_signal(
        mom_table, common_by_month, reference_cutoff
    )
    q_persist, q_single = seasonal_shock.estimate_quantile_thresholds(
        signal_table, reference_cutoff, persist_quantile, single_quantile
    )
    onset_positive, onset_negative, onset_any = seasonal_shock.label_seasonal_shock_onsets(
        signal_table, q_persist, q_single, persistence
    )

    december_onsets = 0
    month_distribution = {}
    for _, month_str in onset_any:
        if str(month_str).endswith("-12"):
            december_onsets += 1
        month_distribution[str(month_str)] = month_distribution.get(str(month_str), 0) + 1
    print("Сезонных шоков (onset): " + str(len(onset_any))
          + " (+" + str(len(onset_positive)) + " / -" + str(len(onset_negative))
          + "); декабрьских: " + str(december_onsets)
          + " (" + ("%.1f%%" % (100.0 * december_onsets / max(len(onset_any), 1))) + ")", flush=True)

    months = list(wide_total.columns)
    events = pd.read_csv(ROOT / "data" / "external" / "cbr_rate_events.csv")
    news_table = news.monthly_rate_changes(events, months, [0])
    cbr_month_change = {}
    for _, row in news_table.iterrows():
        cbr_month_change[str(row["month"])] = float(row["news_signal"])
    cpi_table = macro.load_cpi_table(ROOT / "data" / "external" / "rosstat_cpi.csv")
    cpi_signal_by_month = macro.build_cpi_signal_by_month(cpi_table, months)

    dataset = build_feature_table(
        signal_table, common_by_month, onset_any,
        cbr_month_change, cpi_signal_by_month, horizon
    )

    # ---- Набор методов -----------------------------------------------------
    ml_methods = ["logistic", "lightgbm", "catboost"]
    rule_methods = ["seasonal_naive", "last_change", "volatility", "random"]
    all_methods = ml_methods + rule_methods

    random_generator_for_scores = np.random.default_rng(777)

    accumulators = {}
    for method_name in all_methods:
        accumulators[method_name] = {"hits": 0, "false": 0, "events": 0,
                                     "alarms": 0, "test_rows": 0, "leads": []}
    fold_rows = []
    comparable_origins = []
    pooled_rows_labels = []
    pooled_rows_territories = []
    pooled_rows_scores = {}
    for method_name in all_methods:
        pooled_rows_scores[method_name] = []

    def score_rules(test_frame, territory_ids_local):
        rule_scores = {}
        rule_scores["last_change"] = np.abs(test_frame["r"].to_numpy(dtype=float))
        rule_scores["volatility"] = test_frame["trailing_z_vol"].to_numpy(dtype=float)
        rule_scores["seasonal_naive"] = seasonal_naive_scores(test_frame, onset_any, horizon)
        rule_scores["random"] = random_generator_for_scores.random(len(test_frame))
        return rule_scores

    def run_origin(origin_text):
        test_origin = pd.Period(origin_text, freq="M")
        # Созревание: onset в t+1..t+h подтверждается к t+h+(persistence-1).
        label_fully_observed_at = test_origin + horizon + (persistence - 1)
        is_mature = label_fully_observed_at <= pd.Period(str(last_observed_month), freq="M")
        last_train_month = test_origin - (horizon + persistence)
        train = dataset[dataset["month_period"] <= last_train_month].copy()
        test = dataset[dataset["month_period"] == test_origin].copy()
        return test_origin, is_mature, train, test

    # ---- Rolling-фолды ------------------------------------------------------
    for origin_text in test_origins:
        test_origin, is_mature, train, test = run_origin(origin_text)
        if len(train) == 0 or len(test) == 0 or not is_mature:
            continue
        territory_ids = test["territory_id"].to_numpy()
        model_scores = train_and_score_models(train, test, feature_columns)
        is_comparable = len(model_scores) == len(ml_methods)
        if is_comparable:
            comparable_origins.append(origin_text)
        rule_scores = score_rules(test, territory_ids)

        all_scores = {}
        for method_name in rule_methods:
            all_scores[method_name] = rule_scores[method_name]
        for method_name in model_scores:
            all_scores[method_name] = model_scores[method_name]

        if is_comparable:
            pooled_rows_labels.extend(test["onset_label"].tolist())
            pooled_rows_territories.extend(test["territory_id"].tolist())
            for method_name in all_methods:
                pooled_rows_scores[method_name].extend(list(np.asarray(all_scores[method_name], dtype="float64")))

        for method_name in all_methods:
            if method_name not in all_scores:
                continue
            metrics, alarm_mask, n_alarms = evaluate_one_fold(
                test, all_scores[method_name], territory_ids, budget_rate, horizon, onset_any
            )
            fold_rows.append({
                "test_origin": origin_text, "fold_type": "rolling",
                "mature": True, "comparable": bool(is_comparable),
                "method": method_name, "alarms": n_alarms,
                "event_hits": metrics["event_hits"],
                "event_false_alarms": metrics["event_false_alarms"],
                "onset_events_eligible": metrics["onset_events_eligible"],
                "event_recall": metrics["event_recall"],
                "event_precision": metrics["event_precision"],
                "mean_lead_time_months": metrics["mean_lead_time_months"],
            })
            if is_comparable:
                accumulators[method_name]["hits"] += metrics["event_hits"]
                accumulators[method_name]["false"] += metrics["event_false_alarms"]
                accumulators[method_name]["events"] += metrics["onset_events_eligible"]
                accumulators[method_name]["alarms"] += n_alarms
                accumulators[method_name]["test_rows"] += len(test)
                if metrics["mean_lead_time_months"] is not None:
                    for _ in range(metrics["event_hits"]):
                        accumulators[method_name]["leads"].append(metrics["mean_lead_time_months"])

    # ---- Бутстрэп PR-AUC по объединённым сопоставимым строкам ---------------
    pr_auc_ci = bootstrap_pr_auc(
        pooled_rows_labels, pooled_rows_scores, pooled_rows_territories,
        bootstrap_samples, bootstrap_seed, bootstrap_confidence
    )

    # ---- Сводка по методам (сопоставимые rolling-фолды) ---------------------
    summary_rows = []
    for method_name in all_methods:
        acc = accumulators[method_name]
        detections = acc["hits"] + acc["false"]
        precision = acc["hits"] / detections if detections > 0 else None
        recall = acc["hits"] / acc["events"] if acc["events"] > 0 else None
        false_per_100 = 100.0 * acc["false"] / acc["test_rows"] if acc["test_rows"] > 0 else None
        mean_lead = float(np.mean(acc["leads"])) if len(acc["leads"]) > 0 else None
        point, low, high = pr_auc_ci.get(method_name, (None, None, None))
        summary_rows.append({
            "method": method_name,
            "total_alarms": acc["alarms"],
            "total_hits": acc["hits"],
            "total_events": acc["events"],
            "event_recall": recall,
            "event_precision": precision,
            "false_alarms_per_100_mo_months": false_per_100,
            "mean_lead_time_months": mean_lead,
            "pr_auc": point,
            "pr_auc_ci_low": low,
            "pr_auc_ci_high": high,
        })
    summary_table = pd.DataFrame(summary_rows).sort_values(
        by="pr_auc", ascending=False, na_position="last"
    )
    summary_table.to_csv(destination / "methods_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(destination / "folds_detail.csv", index=False)

    # ---- Финальный holdout (отдельно) ---------------------------------------
    holdout_rows = []
    holdout_origin, holdout_mature, holdout_train, holdout_test = run_origin(holdout_origin_text)
    if len(holdout_train) > 0 and len(holdout_test) > 0 and holdout_mature:
        holdout_territory_ids = holdout_test["territory_id"].to_numpy()
        holdout_model_scores = train_and_score_models(holdout_train, holdout_test, feature_columns)
        holdout_rule_scores = score_rules(holdout_test, holdout_territory_ids)
        holdout_all = {}
        for method_name in rule_methods:
            holdout_all[method_name] = holdout_rule_scores[method_name]
        for method_name in holdout_model_scores:
            holdout_all[method_name] = holdout_model_scores[method_name]
        for method_name in all_methods:
            if method_name not in holdout_all:
                continue
            metrics, alarm_mask, n_alarms = evaluate_one_fold(
                holdout_test, holdout_all[method_name], holdout_territory_ids,
                budget_rate, horizon, onset_any
            )
            holdout_rows.append({
                "method": method_name, "alarms": n_alarms,
                "event_hits": metrics["event_hits"],
                "onset_events_eligible": metrics["onset_events_eligible"],
                "event_recall": metrics["event_recall"],
                "event_precision": metrics["event_precision"],
                "mean_lead_time_months": metrics["mean_lead_time_months"],
            })
    pd.DataFrame(holdout_rows).to_csv(destination / "holdout.csv", index=False)

    info = {
        "target": "сезонно-нормализованный шок (отклонение МО от общего движения и своего уровня)",
        "formulation_id": risk_config.get("formulation_id"),
        "config_version": risk_config.get("version"),
        "definition": {"scale_reference_cutoff": reference_cutoff,
                       "persistence": persistence, "persist_quantile": persist_quantile,
                       "single_quantile": single_quantile, "horizon": horizon},
        "q_persist": q_persist, "q_single": q_single, "pooled_floor": pooled_floor,
        "onsets_total": int(len(onset_any)),
        "onsets_positive": int(len(onset_positive)),
        "onsets_negative": int(len(onset_negative)),
        "december_onsets": int(december_onsets),
        "december_share": float(december_onsets) / float(max(len(onset_any), 1)),
        "month_distribution": dict(sorted(month_distribution.items())),
        "rolling_folds": test_origins,
        "comparable_folds": comparable_origins,
        "holdout_origin": holdout_origin_text,
        "pooled_comparable_rows": int(len(pooled_rows_labels)),
        "pooled_positive_rows": int(np.sum(pooled_rows_labels)) if len(pooled_rows_labels) else 0,
        "bootstrap_samples": bootstrap_samples,
        "metrics_note": ("Событийные recall/precision/ложные-на-100/опережение при равном "
                         "бюджете top-K; PR-AUC (average precision) с КЛАСТЕРНЫМ (по territory_id) бутстрэп "
                         + str(int(100 * bootstrap_confidence)) + "% ДИ. Агрегат — по сопоставимым "
                         "фолдам (где обучились все ML-модели). Декабрь исключён нормализацией "
                         "по общему движению."),
        "honest_caveat": ("Один 2024 год, нет независимого будущего года. Результат "
                          "исследовательский: смотрите пересечение доверительных интервалов "
                          "PR-AUC, прежде чем заявлять превосходство какого-либо метода."),
    }
    with open(destination / "summary.json", "w", encoding="utf-8") as target_file:
        json.dump(info, target_file, ensure_ascii=False, indent=2)

    print("\n=== СЕЗОННЫЙ ШОК: сравнение методов (сопоставимые фолды, равный бюджет) ===", flush=True)
    show = summary_table.copy()
    for column_name in ["event_recall", "event_precision", "false_alarms_per_100_mo_months",
                        "mean_lead_time_months", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high"]:
        rounded_column_values = []
        for original_value in show[column_name]:
            if original_value is not None and not pd.isna(original_value):
                rounded_column_values.append(round(original_value, 4))
            else:
                rounded_column_values.append(None)
        show[column_name] = rounded_column_values
    print(show.to_string(index=False), flush=True)
    print("\nСопоставимые rolling-фолды: " + str(comparable_origins), flush=True)
    print("Пулинг строк (сопоставимые): " + str(len(pooled_rows_labels))
          + ", из них положительных: " + str(int(np.sum(pooled_rows_labels)) if len(pooled_rows_labels) else 0), flush=True)
    if len(holdout_rows) > 0:
        print("\n=== ФИНАЛЬНЫЙ HOLDOUT (" + holdout_origin_text + ", отдельно) ===", flush=True)
        holdout_show = pd.DataFrame(holdout_rows).sort_values(by="event_recall", ascending=False, na_position="last")
        print(holdout_show.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

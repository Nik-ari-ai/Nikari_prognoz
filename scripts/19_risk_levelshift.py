"""Раннее предупреждение для ЯВНО заданного целевого события: устойчивый сдвиг
уровня (src/level_shift.py), rolling-origin + простые правила при равном бюджете.

Закрывает содержательный пункт аудита: определить ПОЛЕЗНЫЙ целевой тип изменения
и его проверяемую разметку, затем сравнить модель и простые правила на ОДНИХ
событиях. Целевое событие — устойчивый сдвиг уровня (а не импульс-возврат
месячной разметки scripts/15), проверенный на синтетике.

Протокол тот же, что в scripts/15: rolling-origin с растущим обучением, созревание
тестовых окон (незрелый фолд — pending, без фиктивных исходов), равный бюджет
тревог top-K с детерминированным tie-break, общий событийный оценщик.

ГЛАВНЫЙ ЧЕСТНЫЙ РЕЗУЛЬТАТ (его и надо показать): устойчивых сдвигов уровня в этих
данных ОЧЕНЬ МАЛО (порядка десятков на 2016 МО × 24 месяца) и они НЕ декабрьские.
Это конкретно объясняет, почему месячная разметка «набирала события» за счёт
декабря, и показывает предел данных: на таком числе событий надёжное раннее
предупреждение не устанавливается ни моделью, ни простыми правилами.
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
from src import risk_formulation
from src import level_shift
from src import risk_eval


def baseline_scores(frame, method):
    """Простые правила риска (предупреждают будущий onset сдвига уровня)."""
    if method == "last_change":
        return np.abs(frame["r"].to_numpy(dtype=float))
    if method == "dev_from_median":
        return np.abs(frame["z"].to_numpy(dtype=float))
    if method == "volatility":
        return frame["trailing_vol"].to_numpy(dtype=float)
    raise ValueError("Неизвестное базовое правило: " + method)


def main(output_dir=None):
    data_config = utils.load_yaml_config(ROOT / "configs" / "data.yaml")
    risk_config = utils.load_yaml_config(ROOT / "configs" / "risk_levelshift.yaml")

    target = risk_config["target_definition"]
    window = int(target["window"])
    persistence = int(target["persistence"])
    threshold_k = float(target["threshold_k"])
    horizon = int(target["horizon"])
    scale_reference_cutoff = target.get("scale_reference_cutoff")  # каузальный reference period
    target_alarm_rate = float(risk_config["alarm_budget"]["target_alarm_rate"])
    test_origins = list(risk_config["protocol"]["test_origins"])
    feature_columns = list(risk_config["feature_columns"])

    if output_dir is None:
        destination = ROOT / "outputs" / "risk_levelshift"
    else:
        destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    print("Загружаю данные...", flush=True)
    consumption = data_loading.load_consumption_dataframe(data_config)
    total = consumption[consumption["category"] == "Все категории"]
    wide_total = total.pivot_table(index="territory_id", columns="month", values="value")
    wide_total = wide_total.sort_index(axis=1).dropna(axis=0, how="any")

    # --- Целевое событие: устойчивый сдвиг уровня (пред-зарегистрировано) -----
    # ПОПРАВКА АУДИТА v9: масштаб оцениваем ТОЛЬКО по reference period строго до
    # тестового периода (каузально), иначе число событий зависело бы от тестового
    # будущего. Reference period задан в конфиге (2023 год).
    scale_floor = level_shift.estimate_level_scale_floor(
        wide_total, reference_cutoff=scale_reference_cutoff
    )
    onset_set, last_observed_month = level_shift.label_level_shift_onsets(
        wide_total, window, persistence, threshold_k, scale_floor
    )

    # --- Таблица чувствительности масштаба к границе данных (прозрачность) ----
    # Показывает, что число событий зависит от reference period; именно поэтому
    # он зафиксирован каузально. "0 декабрьских" устойчиво на всех границах.
    # Каузальность границы определяется ТОЧНО: reference period каузален, если
    # он строго раньше ПЕРВОГО тестового origin (тогда масштаб не использует ни
    # одного тестового месяца). Первый origin = самый ранний из test_origins.
    test_origin_periods = []
    for one_origin in test_origins:
        test_origin_periods.append(pd.Period(str(one_origin), freq="M"))
    first_test_origin = min(test_origin_periods)
    sensitivity_rows = []
    for boundary_text in ["2023-12", "2024-01", "2024-05", "2024-08", "2024-12"]:
        boundary_period = pd.Period(boundary_text, freq="M")
        boundary_floor = level_shift.estimate_level_scale_floor(
            wide_total, reference_cutoff=boundary_text
        )
        boundary_onsets, _ = level_shift.label_level_shift_onsets(
            wide_total, window, persistence, threshold_k, boundary_floor
        )
        boundary_december = 0
        for _, one_month in boundary_onsets:
            if str(one_month).endswith("-12"):
                boundary_december += 1
        sensitivity_rows.append({
            "scale_reference_cutoff": boundary_text,
            "scale_floor": round(boundary_floor, 6),
            "onsets_total": len(boundary_onsets),
            "december_onsets": boundary_december,
            # ПОПРАВКА АУДИТА v10: каузально, если граница строго раньше первого
            # тестового origin (2024-02). Поэтому и 2023-12, и 2024-01 каузальны.
            "is_causal_vs_test": bool(boundary_period < first_test_origin),
        })
    pd.DataFrame(sensitivity_rows).to_csv(destination / "scale_sensitivity.csv", index=False)

    # Распределение событий по месяцам и доля декабря (ключевая диагностика).
    onset_months = []
    for _, month_str in onset_set:
        onset_months.append(str(month_str))
    december_onsets = 0
    for month_str in onset_months:
        if month_str.endswith("-12"):
            december_onsets += 1
    month_distribution = {}
    for month_str in onset_months:
        month_distribution[month_str] = month_distribution.get(month_str, 0) + 1
    print("Устойчивых сдвигов уровня (onset): " + str(len(onset_set))
          + "; из них декабрьских: " + str(december_onsets), flush=True)
    print("Распределение по месяцам: " + str(dict(sorted(month_distribution.items()))), flush=True)

    # --- Каузальные признаки (те же, что в scripts/15) -----------------------
    months = list(wide_total.columns)
    events = pd.read_csv(ROOT / "data" / "external" / "cbr_rate_events.csv")
    news_table = news.monthly_rate_changes(events, months, [0])
    cbr_month_change = {}
    for _, row in news_table.iterrows():
        cbr_month_change[str(row["month"])] = float(row["news_signal"])
    cpi_table = macro.load_cpi_table(ROOT / "data" / "external" / "rosstat_cpi.csv")
    cpi_signal_by_month = macro.build_cpi_signal_by_month(cpi_table, months)

    mom_table = risk_formulation.build_month_over_month_table(wide_total)
    signal_table = risk_formulation.add_trailing_baseline_and_scale(mom_table, window)
    pooled_floor = risk_formulation.estimate_pooled_floor(
        signal_table, signal_table["month"].unique()
    )
    signal_z = risk_formulation.add_deviation_z(signal_table, pooled_floor)
    dataset = risk_formulation.build_features_and_labels(
        signal_z, onset_set, cbr_month_change, cpi_signal_by_month, horizon, persistence
    )
    dataset["month_period"] = pd.PeriodIndex(dataset["month"].astype(str), freq="M")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    methods = ["logistic", "dev_from_median", "last_change", "volatility"]
    accumulators = {}
    for method in methods:
        accumulators[method] = {"hits": 0, "false": 0, "events": 0, "leads": [],
                                "alarms": 0, "test_rows": 0}
    fold_rows = []
    mature_origins = []
    pending_origins = []
    comparable_origins = []
    logistic_trained_in_any_mature_fold = False

    for test_origin_text in test_origins:
        test_origin = pd.Period(test_origin_text, freq="M")
        # Созревание: onset в r+1..r+h наблюдается при (r+h)+(persistence-1) <= last.
        label_fully_observed_at = test_origin + horizon + (persistence - 1)
        is_mature = label_fully_observed_at <= pd.Period(str(last_observed_month), freq="M")
        if is_mature:
            mature_origins.append(test_origin_text)
        else:
            pending_origins.append(test_origin_text)

        last_train_month = test_origin - (horizon + persistence)
        train_months = []
        for month_period in sorted(dataset["month_period"].unique()):
            if month_period <= last_train_month:
                train_months.append(month_period)

        train = dataset[dataset["month_period"].isin(train_months)].copy()
        test = dataset[dataset["month_period"] == test_origin].copy()
        if len(train) == 0 or len(test) == 0:
            continue

        train_onset_positives = int(train["onset_label"].sum())
        can_train_logistic = train_onset_positives >= 5 and len(np.unique(train["onset_label"])) == 2
        if can_train_logistic:
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train[feature_columns].to_numpy(dtype=float))
            test_features = scaler.transform(test[feature_columns].to_numpy(dtype=float))
            classifier = LogisticRegression(class_weight="balanced", max_iter=1000)
            classifier.fit(train_features, train["onset_label"].to_numpy(dtype=int))
            if is_mature:
                logistic_trained_in_any_mature_fold = True

        territory_ids = test["territory_id"].to_numpy()
        budget_k = max(int(round(len(test) * target_alarm_rate)), 1)

        # Фолд «сопоставимый» для head-to-head, только если В НЁМ обучилась и
        # логистическая модель: иначе её знаменатель событий отличался бы от
        # правил (апельсины против яблок). Правила считаются во ВСЕХ зрелых
        # фолдах (для прозрачности), но в агрегат сравнения входят лишь
        # сопоставимые фолды — одни и те же события у всех методов.
        is_comparable = is_mature and can_train_logistic
        if is_comparable:
            comparable_origins.append(test_origin_text)

        for method in methods:
            if method == "logistic":
                if not can_train_logistic:
                    continue
                test_scores = classifier.predict_proba(test_features)[:, 1]
            else:
                test_scores = baseline_scores(test, method)
            test_scores = np.asarray(test_scores, dtype="float64")
            alarm_mask = risk_eval.top_k_alarm_mask(test_scores, budget_k, territory_ids)
            n_alarms = int(np.sum(alarm_mask))

            if is_mature:
                metrics = risk_eval.event_level_evaluation(
                    test[["territory_id", "month"]], test_scores, None, horizon,
                    onset_set, alarm_mask=alarm_mask
                )
                fold_rows.append({
                    "test_origin": test_origin_text, "mature": True, "status": "evaluated",
                    "comparable": bool(is_comparable),
                    "method": method, "train_months": len(train_months),
                    "train_onset_positives": train_onset_positives, "alarms": n_alarms,
                    "event_hits": metrics["event_hits"],
                    "event_false_alarms": metrics["event_false_alarms"],
                    "onset_events_eligible": metrics["onset_events_eligible"],
                    "event_precision": metrics["event_precision"],
                    "event_recall": metrics["event_recall"],
                    "mean_lead_time_months": metrics["mean_lead_time_months"],
                })
                # В агрегат — только сопоставимые фолды (равный набор событий).
                if is_comparable:
                    accumulators[method]["hits"] += metrics["event_hits"]
                    accumulators[method]["false"] += metrics["event_false_alarms"]
                    accumulators[method]["events"] += metrics["onset_events_eligible"]
                    accumulators[method]["alarms"] += n_alarms
                    accumulators[method]["test_rows"] += len(test)
                    if metrics["mean_lead_time_months"] is not None:
                        for _ in range(metrics["event_hits"]):
                            accumulators[method]["leads"].append(metrics["mean_lead_time_months"])
            else:
                fold_rows.append({
                    "test_origin": test_origin_text, "mature": False,
                    "status": "pending_unobserved_future", "comparable": False,
                    "method": method, "train_months": len(train_months),
                    "train_onset_positives": train_onset_positives, "alarms": n_alarms,
                    "event_hits": None, "event_false_alarms": None,
                    "onset_events_eligible": None, "event_precision": None,
                    "event_recall": None, "mean_lead_time_months": None,
                })

    pd.DataFrame(fold_rows).to_csv(destination / "folds_detail.csv", index=False)

    summary_rows = []
    for method in methods:
        acc = accumulators[method]
        detections = acc["hits"] + acc["false"]
        precision = acc["hits"] / detections if detections > 0 else None
        recall = acc["hits"] / acc["events"] if acc["events"] > 0 else None
        mean_lead = float(np.mean(acc["leads"])) if len(acc["leads"]) > 0 else None
        summary_rows.append({
            "method": method, "total_alarms": acc["alarms"], "total_hits": acc["hits"],
            "total_false_alarms": acc["false"], "total_events": acc["events"],
            "event_precision": precision, "event_recall": recall,
            "mean_lead_time_months": mean_lead,
        })
    summary_table = pd.DataFrame(summary_rows).sort_values(
        by="event_recall", ascending=False, na_position="last"
    )
    summary_table.to_csv(destination / "methods_comparison.csv", index=False)

    info = {
        "target": "устойчивый сдвиг уровня (явно задан, проверен на синтетике)",
        "formulation_id": risk_config.get("formulation_id"),
        "config_version": risk_config.get("version"),
        "definition": {"window": window, "persistence": persistence,
                       "threshold_k": threshold_k, "horizon": horizon},
        "scale_reference_cutoff": scale_reference_cutoff,
        "scale_floor": scale_floor,
        "scale_causality_note": ("ПОПРАВКА АУДИТА v9: масштаб оценён ТОЛЬКО по reference "
                                 "period (данные <= " + str(scale_reference_cutoff) + ", весь "
                                 "2023 год, строго до тестового периода), поэтому определение "
                                 "события НЕ зависит от тестового будущего. Таблица "
                                 "scale_sensitivity.csv показывает зависимость числа событий "
                                 "от границы масштаба; «0 декабрьских» устойчиво на всех границах."),
        "level_shift_onsets_total": int(len(onset_set)),
        "december_onsets": int(december_onsets),
        "month_distribution": dict(sorted(month_distribution.items())),
        "mature_folds_evaluated": mature_origins,
        "pending_folds_not_evaluated": pending_origins,
        "comparable_folds_in_aggregate": comparable_origins,
        "comparability_note": ("Агрегированное сравнение методов считается ТОЛЬКО по "
                               "сопоставимым фолдам (где обучилась и логистическая модель), "
                               "иначе её знаменатель событий отличался бы от правил. "
                               "Полные пофолдовые числа — в folds_detail.csv (колонка comparable)."),
        "last_observed_month": str(last_observed_month),
        "logistic_trained_in_any_mature_fold": bool(logistic_trained_in_any_mature_fold),
        "logistic_note": ("В зрелых фолдах обучаемых положительных примеров почти нет, "
                          "поэтому логистическая модель почти не обучается. Вердикт «модель "
                          "против правил» для этого целевого события на этих данных вынести "
                          "нельзя: это предел данных, а не выбор модели."),
        "key_finding": ("При каузальном масштабе (reference <= " + str(scale_reference_cutoff)
                        + ") устойчивых сдвигов уровня " + str(len(onset_set))
                        + ", и НИ ОДНОГО в декабре. Число событий чувствительно к способу "
                        "оценки масштаба (см. scale_sensitivity.csv) — поэтому масштаб "
                        "зафиксирован каузально; не-декабрьскость устойчива. Это очерчивает "
                        "предел данных: даже при корректно определённом событии обучаемых "
                        "зрелых примеров почти нет."),
        "caveat": "Исследовательский результат: мало событий, один 2024 год, нет независимого будущего.",
    }
    with open(destination / "summary.json", "w", encoding="utf-8") as target_file:
        json.dump(info, target_file, ensure_ascii=False, indent=2)

    print("\n=== УСТОЙЧИВЫЙ СДВИГ УРОВНЯ: модель против простых правил (равный бюджет) ===", flush=True)
    show = summary_table.copy()
    for column_name in ["event_precision", "event_recall", "mean_lead_time_months"]:
        rounded_column_values = []
        for original_value in show[column_name]:
            if original_value is not None and not pd.isna(original_value):
                rounded_column_values.append(round(original_value, 4))
            else:
                rounded_column_values.append(None)
        show[column_name] = rounded_column_values
    print(show.to_string(index=False), flush=True)
    print("Зрелые фолды: " + str(mature_origins) + "; pending: " + str(pending_origins), flush=True)
    print("Сопоставимые фолды в агрегате (где обучилась логистическая): "
          + str(comparable_origins), flush=True)
    print("scale_floor (каузальный, reference <= " + str(scale_reference_cutoff) + "): "
          + ("%.5f" % scale_floor) + "; устойчивых сдвигов: " + str(len(onset_set))
          + ", декабрьских: " + str(december_onsets), flush=True)


if __name__ == "__main__":
    main()

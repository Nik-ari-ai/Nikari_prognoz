"""Эксперимент раннего предупреждения: rolling-origin + базовые правила.

Отвечает на три пункта плана, но ЧЕСТНО, с поправками независимого аудита v5:

  1. Постановка с МЕНЬШИМ прогревом (src/risk_formulation.py): месячная разность
     и короткое скользящее окно. Сигнал существует с августа 2023, обучающих
     ВРЕМЕННЫХ срезов становится несколько.
  2. Простые конкуренты сравниваются с обученной моделью при ОДИНАКОВОМ бюджете
     тревог (top-K на каждом выпуске, детерминированный отбор ровно K).
  3. Оценка — последовательная (train на прошлом -> test на следующем периоде),
     rolling-origin с РАСТУЩИМ обучением.

ПОПРАВКА АУДИТА v5 (критичная): фолд оценивается ТОЛЬКО если его метка полностью
наблюдается. История заканчивается декабрём 2024, поэтому выпуск ноября 2024 с
горизонтом 3 (окно onset до февраля 2025, подтверждение до марта 2025) НЕ созрел
и исключается из всех метрик и знаменателей. Он показывается как pending —
реальные прогнозы возможны, но их точность на этих данных не проверяема.

ПОПРАВКА АУДИТА v5 (методическая, в summary и отчёте): короткая месячная разметка
ловит в основном ДЕКАБРЬСКИЙ календарный всплеск-возврат (две противоположные
большие месячные разности), а не несколько независимых типов структурных шоков.
Поэтому «больше обучающих срезов» НЕ означает «больше независимых событий».
Доля декабрьских onset по фолдам считается и выводится явно.

Определение шока, протокол и бюджет берутся из configs/risk_tscv.yaml
(пред-регистрация) и НЕ подбираются под результат. Это исследовательский
эксперимент: один 2024 год и короткая история ограничивают выводы.
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
from src import risk_eval


def baseline_scores(frame, method):
    """Оценки простых правил риска (предупреждают будущий onset)."""
    if method == "last_change":
        return np.abs(frame["r"].to_numpy(dtype=float))
    if method == "dev_from_median":
        return np.abs(frame["z"].to_numpy(dtype=float))
    if method == "volatility":
        return frame["trailing_vol"].to_numpy(dtype=float)
    raise ValueError("Неизвестное базовое правило: " + method)


def december_share_of_onsets(onset_set):
    """Доля декабрьских onset среди всех начал эпизодов (диагностика разметки).

    Короткая месячная разметка подозрительно концентрируется на декабре
    (праздничный всплеск расходов с возвратом). Эта доля делает концентрацию
    явной в отчёте, не пряча её за ростом числа обучающих срезов.
    """
    onset_months = []
    for _, month_str in onset_set:
        onset_months.append(str(month_str))
    total = len(onset_months)
    if total == 0:
        return None, 0, 0
    december_count = 0
    for month_str in onset_months:
        if month_str.endswith("-12"):
            december_count += 1
    return december_count / total, december_count, total


def main(output_dir=None):
    data_config = utils.load_yaml_config(ROOT / "configs" / "data.yaml")
    risk_config = utils.load_yaml_config(ROOT / "configs" / "risk_tscv.yaml")

    # Пред-зарегистрированные параметры (из версионируемого конфига).
    shock = risk_config["shock_definition"]
    window = int(shock["window"])
    horizon = int(shock["horizon"])
    min_duration = int(shock["min_duration"])
    threshold_k = float(shock["threshold_k"])
    target_alarm_rate = float(risk_config["alarm_budget"]["target_alarm_rate"])
    test_origins = list(risk_config["protocol"]["test_origins"])
    feature_columns = list(risk_config["feature_columns"])

    if output_dir is None:
        destination = ROOT / "outputs" / "risk_tscv"
    else:
        destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    print("Загружаю данные...", flush=True)
    consumption = data_loading.load_consumption_dataframe(data_config)
    total = consumption[consumption["category"] == "Все категории"]
    wide_total = total.pivot_table(index="territory_id", columns="month", values="value")
    wide_total = wide_total.sort_index(axis=1).dropna(axis=0, how="any")

    # Внешние признаки.
    months = list(wide_total.columns)
    events = pd.read_csv(ROOT / "data" / "external" / "cbr_rate_events.csv")
    news_table = news.monthly_rate_changes(events, months, [0])
    cbr_month_change = {}
    for _, row in news_table.iterrows():
        cbr_month_change[str(row["month"])] = float(row["news_signal"])
    cpi_table = macro.load_cpi_table(ROOT / "data" / "external" / "rosstat_cpi.csv")
    cpi_signal_by_month = macro.build_cpi_signal_by_month(cpi_table, months)

    print("Строю сигнал с коротким прогревом (окно " + str(window) + ")...", flush=True)
    mom_table = risk_formulation.build_month_over_month_table(wide_total)
    signal_table = risk_formulation.add_trailing_baseline_and_scale(mom_table, window)
    signal_month_min = str(signal_table["month"].min())
    signal_month_max = str(signal_table["month"].max())
    # Последний НАБЛЮДАЕМЫЙ месяц сигнала — граница созревания меток.
    last_observed_month = max(signal_table["month"].unique())
    print("Сигнал доступен с " + signal_month_min + " по " + signal_month_max, flush=True)
    print("Последний наблюдаемый месяц: " + str(last_observed_month), flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    methods = ["logistic", "dev_from_median", "last_change", "volatility"]
    # Накопители событийных исходов по методам (суммируем ТОЛЬКО по зрелым фолдам).
    accumulators = {}
    for method in methods:
        accumulators[method] = {"hits": 0, "false": 0, "events": 0, "leads": [],
                                "alarms": 0, "test_rows": 0}

    fold_rows = []
    pending_rows = []
    december_by_fold = {}

    for test_origin_text in test_origins:
        test_origin = pd.Period(test_origin_text, freq="M")

        # --- ПОПРАВКА v5: созревание метки -----------------------------------
        # Метка выпуска t известна полностью только когда наблюдается месяц
        # подтверждения последнего возможного onset: t + horizon + (min_duration-1).
        label_fully_observed_at = test_origin + horizon + (min_duration - 1)
        is_mature = label_fully_observed_at <= last_observed_month

        # Обучающие месяцы: метка созрела строго до теста.
        last_train_month = test_origin - (horizon + min_duration)
        train_months = []
        for month_period in sorted(signal_table["month"].unique()):
            if month_period <= last_train_month:
                train_months.append(month_period)
        if len(train_months) == 0:
            continue

        # Порог масштаба — по обучающим месяцам (каузально).
        pooled_floor = risk_formulation.estimate_pooled_floor(signal_table, train_months)
        signal_z = risk_formulation.add_deviation_z(signal_table, pooled_floor)
        labeled, onset_set = risk_formulation.label_onsets(signal_z, threshold_k, min_duration)

        # Диагностика: доля декабрьских onset в разметке этого фолда.
        december_share, december_count, onset_total = december_share_of_onsets(onset_set)
        december_by_fold[test_origin_text] = {
            "december_share": december_share,
            "december_onsets": december_count,
            "onset_total": onset_total,
        }

        dataset = risk_formulation.build_features_and_labels(
            labeled, onset_set, cbr_month_change, cpi_signal_by_month, horizon, min_duration
        )
        dataset["month_period"] = pd.PeriodIndex(dataset["month"].astype(str), freq="M")

        train = dataset[dataset["month_period"].isin(train_months)].copy()
        test = dataset[dataset["month_period"] == test_origin].copy()
        if len(train) == 0 or len(test) == 0:
            continue

        train_onset_positives = int(train["onset_label"].sum())

        # Обучаем логистическую модель (если есть оба класса и достаточно примеров).
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train[feature_columns].to_numpy(dtype=float))
        test_features = scaler.transform(test[feature_columns].to_numpy(dtype=float))
        can_train_logistic = train_onset_positives >= 5 and len(np.unique(train["onset_label"])) == 2
        if can_train_logistic:
            classifier = LogisticRegression(class_weight="balanced", max_iter=1000)
            classifier.fit(train_features, train["onset_label"].to_numpy(dtype=int))

        # РАВНЫЙ бюджет тревог: ровно K тревог у каждого метода (top-K по оценке
        # ТЕКУЩЕГО выпуска, детерминированный tie-break по territory_id).
        budget_k = int(round(len(test) * target_alarm_rate))
        budget_k = max(budget_k, 1)
        territory_ids = test["territory_id"].to_numpy()

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

            # ПОПРАВКА аудита v6: событийный оценщик вызывается ТОЛЬКО для зрелого
            # фолда. Для pending исходы НЕизвестны (часть будущего отсутствует),
            # поэтому hits/false/events/precision/recall/lead остаются пустыми
            # (None), а не записываются как «0 попаданий / N ложных».
            if is_mature:
                metrics = risk_eval.event_level_evaluation(
                    test[["territory_id", "month"]], test_scores, None, horizon,
                    onset_set, alarm_mask=alarm_mask
                )
                fold_rows.append({
                    "test_origin": test_origin_text,
                    "mature": True,
                    "status": "evaluated",
                    "method": method,
                    "train_months": len(train_months),
                    "train_onset_positives": train_onset_positives,
                    "december_share_in_labeling": december_share,
                    "alarms": n_alarms,
                    "event_hits": metrics["event_hits"],
                    "event_false_alarms": metrics["event_false_alarms"],
                    "onset_events_eligible": metrics["onset_events_eligible"],
                    "event_precision": metrics["event_precision"],
                    "event_recall": metrics["event_recall"],
                    "mean_lead_time_months": metrics["mean_lead_time_months"],
                    "label_fully_observed_at": str(label_fully_observed_at),
                })
                # Только зрелые фолды входят в агрегированные метрики.
                accumulators[method]["hits"] += metrics["event_hits"]
                accumulators[method]["false"] += metrics["event_false_alarms"]
                accumulators[method]["events"] += metrics["onset_events_eligible"]
                accumulators[method]["alarms"] += n_alarms
                accumulators[method]["test_rows"] += len(test)
                if metrics["mean_lead_time_months"] is not None:
                    for _ in range(metrics["event_hits"]):
                        accumulators[method]["leads"].append(metrics["mean_lead_time_months"])
            else:
                # Незрелый фолд: фиксируем только число тревог; исходы — пустые.
                fold_rows.append({
                    "test_origin": test_origin_text,
                    "mature": False,
                    "status": "pending_unobserved_future",
                    "method": method,
                    "train_months": len(train_months),
                    "train_onset_positives": train_onset_positives,
                    "december_share_in_labeling": december_share,
                    "alarms": n_alarms,
                    "event_hits": None,
                    "event_false_alarms": None,
                    "onset_events_eligible": None,
                    "event_precision": None,
                    "event_recall": None,
                    "mean_lead_time_months": None,
                    "label_fully_observed_at": str(label_fully_observed_at),
                })
                pending_rows.append({
                    "test_origin": test_origin_text,
                    "method": method,
                    "alarms": n_alarms,
                    "label_fully_observed_at": str(label_fully_observed_at),
                    "status": "pending_unobserved_future",
                })

    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(destination / "folds_detail.csv", index=False)
    pending_table = pd.DataFrame(pending_rows)
    pending_table.to_csv(destination / "pending_folds.csv", index=False)

    # Сводка по методам (агрегируем исходы ТОЛЬКО по зрелым фолдам).
    summary_rows = []
    for method in methods:
        acc = accumulators[method]
        detections = acc["hits"] + acc["false"]
        precision = acc["hits"] / detections if detections > 0 else None
        recall = acc["hits"] / acc["events"] if acc["events"] > 0 else None
        false_per_100 = 100.0 * acc["false"] / acc["test_rows"] if acc["test_rows"] > 0 else None
        mean_lead = float(np.mean(acc["leads"])) if len(acc["leads"]) > 0 else None
        summary_rows.append({
            "method": method,
            "total_alarms": acc["alarms"],
            "total_hits": acc["hits"],
            "total_false_alarms": acc["false"],
            "total_events": acc["events"],
            "event_precision": precision,
            "event_recall": recall,
            "false_alarms_per_100_mo_months": false_per_100,
            "mean_lead_time_months": mean_lead,
        })
    summary_table = pd.DataFrame(summary_rows).sort_values(
        by="event_recall", ascending=False, na_position="last"
    )
    summary_table.to_csv(destination / "methods_comparison.csv", index=False)

    mature_origins = []
    pending_origins = []
    for test_origin_text in test_origins:
        test_origin = pd.Period(test_origin_text, freq="M")
        label_fully_observed_at = test_origin + horizon + (min_duration - 1)
        if label_fully_observed_at <= last_observed_month:
            mature_origins.append(test_origin_text)
        else:
            pending_origins.append(test_origin_text)

    info = {
        "formulation": "месячная разность + скользящее окно " + str(window) + " (короткий прогрев)",
        "formulation_id": risk_config.get("formulation_id"),
        "config_version": risk_config.get("version"),
        "shock_definition": {"window": window, "threshold_k": threshold_k,
                             "min_duration": min_duration, "horizon": horizon},
        "protocol": "rolling-origin с растущим обучением; равный бюджет тревог top-K на каждом выпуске",
        "alarm_policy": ("top-K на ТЕКУЩЕМ выпуске (сравнение ранжирования при равной "
                         "нагрузке); это НЕ замороженный абсолютный порог с ранней "
                         "валидации и НЕ утечка будущих меток (все оценки — одного "
                         "уже наблюдаемого месяца); совпадения разрешаются по territory_id"),
        "target_alarm_rate": target_alarm_rate,
        "test_origins_pre_registered": test_origins,
        "last_observed_month": str(last_observed_month),
        "mature_folds_evaluated": mature_origins,
        "pending_folds_not_evaluated": pending_origins,
        "maturation_rule": "фолд оценивается, только если origin + horizon + (min_duration-1) <= последний наблюдаемый месяц",
        "signal_available_from": signal_month_min,
        "december_concentration_by_fold": december_by_fold,
        "label_semantics_caveat": (
            "Короткая месячная разметка ловит преимущественно декабрьский "
            "всплеск-возврат (две противоположные большие месячные разности), а не "
            "несколько независимых типов структурных шоков. Поэтому рост числа "
            "обучающих срезов НЕ означает рост числа независимых событий."),
        "comparability_caveat": (
            "Метрики этой постановки НЕсопоставимы напрямую с onset-моделью "
            "scripts/13 (разные целевые события, сплиты и политика тревог)."),
        "pre_registration_note": (
            "pre_registered=true и конфиг делают постановку явной и версионируемой, "
            "но сами по себе НЕ являются доказательством независимой "
            "предварительной регистрации."),
        "pre_registered": True,
        "caveat": ("Исследовательский эксперимент: один 2024 год, короткая история, "
                   "мало событий; устойчивых выводов об обобщении нет. Разница 4 против "
                   "2 попаданий — это два события, а не надёжно установленное превосходство."),
    }
    with open(destination / "summary.json", "w", encoding="utf-8") as target:
        json.dump(info, target, ensure_ascii=False, indent=2)

    print("\n=== СРАВНЕНИЕ МЕТОДОВ (rolling-origin, равный бюджет, ТОЛЬКО зрелые фолды) ===", flush=True)
    show = summary_table.copy()
    for column_name in ["event_precision", "event_recall", "false_alarms_per_100_mo_months", "mean_lead_time_months"]:
        rounded_column_values = []
        for original_value in show[column_name]:
            if original_value is not None and not pd.isna(original_value):
                rounded_column_values.append(round(original_value, 4))
            else:
                rounded_column_values.append(None)
        show[column_name] = rounded_column_values
    print(show.to_string(index=False), flush=True)
    print("\nЗрелые фолды (оценены): " + str(mature_origins), flush=True)
    print("Незрелые фолды (pending, НЕ оценены): " + str(pending_origins), flush=True)
    mature_fold_table = fold_table[fold_table["mature"]] if len(fold_table) else fold_table
    if len(mature_fold_table):
        print("Обучающих срезов по зрелым фолдам (мес): "
              + str(sorted(mature_fold_table["train_months"].unique().tolist())), flush=True)
    print("Доля декабрьских onset по фолдам:", flush=True)
    for origin_text in test_origins:
        if origin_text in december_by_fold and december_by_fold[origin_text]["december_share"] is not None:
            share = december_by_fold[origin_text]["december_share"]
            print("  " + origin_text + ": " + ("%.2f%%" % (100.0 * share)), flush=True)


if __name__ == "__main__":
    main()

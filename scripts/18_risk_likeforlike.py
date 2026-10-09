"""Like-for-like: обученная onset-модель против простых правил на ЕЁ СОБСТВЕННЫХ
событиях (разметка scripts/13: годовая лог-разность), один протокол и бюджет.

Закрывает прямое замечание аудита: сравнение простых правил в scripts/15 сделано
на ДРУГОЙ (месячной) разметке и НЕ отвечает, превосходит ли обученная модель
простые правила на СОБЫТИЯХ И СПЛИТАХ scripts/13. Здесь всё берётся ровно как в
scripts/13 (та же калибровка, разметка, обучающий/тестовый сплит, признаки и
обученная модель), но ДОБАВЛЕНО: три простых правила риска и честное сравнение
всех методов при ОДИНАКОВОМ бюджете тревог (детерминированный top-K) общим
событийным оценщиком.

Честная рамка: это одно тестовое окно (июнь–август 2024) — следствие годового
прогрева разметки, из-за которого созревает единственный обучающий срез. Выводы
исследовательские: мало событий, один переход года.
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
from src import shock_labeling
from src import macro
from src import news
from src import risk_model
from src import risk_eval


# Те же константы, что в scripts/13 (чтобы события и сплиты совпадали).
TEST_START = pd.Period("2024-06", freq="M")
HORIZON = 3
MIN_DURATION = 2
THRESHOLD_SIGMA = 3.0
TARGET_ALARM_RATE = 0.10
FEATURE_COLUMNS = ["yoy_t", "yoy_change", "mom_t", "mom_prev",
                   "recent_volatility", "rate_change", "cpi_signal"]


def baseline_scores(frame, method):
    """Простые правила риска на признаках scripts/13 (предупреждают будущий onset).

    last_change    — величина последнего МЕСЯЧНОГО изменения |mom_t|;
    dev_from_median— величина годового отклонения |yoy_t| (сигнал этой разметки);
    volatility     — недавняя волатильность месячных изменений.
    """
    if method == "last_change":
        return np.abs(frame["mom_t"].to_numpy(dtype=float))
    if method == "dev_from_median":
        return np.abs(frame["yoy_t"].to_numpy(dtype=float))
    if method == "volatility":
        return frame["recent_volatility"].to_numpy(dtype=float)
    raise ValueError("Неизвестное базовое правило: " + method)


def main(output_dir=None):
    data_config = utils.load_yaml_config(ROOT / "configs" / "data.yaml")
    if output_dir is None:
        destination = ROOT / "outputs" / "risk_likeforlike"
    else:
        destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    print("Загружаю данные и воспроизвожу разметку scripts/13...", flush=True)
    consumption = data_loading.load_consumption_dataframe(data_config)
    total = consumption[consumption["category"] == "Все категории"]
    wide_total = total.pivot_table(index="territory_id", columns="month", values="value")
    wide_total = wide_total.sort_index(axis=1).dropna(axis=0, how="any")

    yoy_table = shock_labeling.compute_year_over_year_log_change(wide_total)
    yoy_months = pd.PeriodIndex(yoy_table["month"].astype(str), freq="M")

    # Калибровка разметки — пред-тестовое окно (как в scripts/13).
    calibration_yoy = yoy_table[yoy_months < TEST_START]
    median_by_territory_map = {}
    for territory_id, group in calibration_yoy.groupby("territory_id"):
        median_by_territory_map[int(territory_id)] = float(group["yoy_log_change"].median())

    centered = []
    for _, row in calibration_yoy.iterrows():
        centered.append(row["yoy_log_change"] - median_by_territory_map.get(int(row["territory_id"]), 0.0))
    centered = np.asarray(centered, dtype=float)
    mad = float(np.median(np.abs(centered - np.median(centered))))
    robust_sigma = 1.4826 * mad

    labels = shock_labeling.label_shocks_with_fixed_stats(
        yoy_table, median_by_territory_map, robust_sigma, THRESHOLD_SIGMA, MIN_DURATION
    )
    onset_set = shock_labeling.extract_episode_onsets(labels)

    months = list(wide_total.columns)
    events = pd.read_csv(ROOT / "data" / "external" / "cbr_rate_events.csv")
    news_table = news.monthly_rate_changes(events, months, [0])
    cbr_month_change = {}
    for _, row in news_table.iterrows():
        cbr_month_change[str(row["month"])] = float(row["news_signal"])
    cpi_table = macro.load_cpi_table(ROOT / "data" / "external" / "rosstat_cpi.csv")
    cpi_signal_by_month = macro.build_cpi_signal_by_month(cpi_table, months)

    dataset = risk_model.build_risk_dataset(
        wide_total, labels, cbr_month_change, cpi_signal_by_month,
        horizon_months=HORIZON, onset_set=onset_set, min_duration=MIN_DURATION
    )
    dataset["month_period"] = pd.PeriodIndex(dataset["month"].astype(str), freq="M")

    last_mature_train_month = TEST_START - (HORIZON + MIN_DURATION)
    matured = dataset[dataset["label_matured"]].copy()
    train = matured[matured["month_period"] <= last_mature_train_month].copy()
    test = matured[matured["month_period"] >= TEST_START].copy()

    test_months = sorted(test["month"].unique().tolist())
    print("Обучающий срез: <= " + str(last_mature_train_month)
          + "; тестовое окно: " + str(test_months), flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train[FEATURE_COLUMNS].to_numpy(dtype=float))
    test_features = scaler.transform(test[FEATURE_COLUMNS].to_numpy(dtype=float))
    classifier = LogisticRegression(class_weight="balanced", max_iter=1000)
    classifier.fit(train_features, train["onset_label"].to_numpy(dtype=int))

    territory_ids = test["territory_id"].to_numpy()
    budget_k = int(round(len(test) * TARGET_ALARM_RATE))
    budget_k = max(budget_k, 1)

    methods = ["logistic", "last_change", "dev_from_median", "volatility"]
    summary_rows = []
    for method in methods:
        if method == "logistic":
            test_scores = classifier.predict_proba(test_features)[:, 1]
        else:
            test_scores = baseline_scores(test, method)
        test_scores = np.asarray(test_scores, dtype="float64")

        # Равный бюджет тревог: ровно K тревог у каждого метода (детерминированно).
        alarm_mask = risk_eval.top_k_alarm_mask(test_scores, budget_k, territory_ids)
        n_alarms = int(np.sum(alarm_mask))
        metrics = risk_eval.event_level_evaluation(
            test[["territory_id", "month"]], test_scores, None, HORIZON,
            onset_set, alarm_mask=alarm_mask
        )
        summary_rows.append({
            "method": method,
            "alarms": n_alarms,
            "event_hits": metrics["event_hits"],
            "event_false_alarms": metrics["event_false_alarms"],
            "onset_events_eligible": metrics["onset_events_eligible"],
            "event_precision": metrics["event_precision"],
            "event_recall": metrics["event_recall"],
            "false_alarms_per_100_mo_months": metrics["false_alarms_per_100_mo_months"],
            "mean_lead_time_months": metrics["mean_lead_time_months"],
        })

    summary_table = pd.DataFrame(summary_rows).sort_values(
        by="event_recall", ascending=False, na_position="last"
    )
    summary_table.to_csv(destination / "methods_comparison.csv", index=False)

    info = {
        "labeling": "scripts/13 (годовая лог-разность, порог 3σ, длительность 2)",
        "purpose": "like-for-like: обученная модель против простых правил на ЕЁ событиях и сплитах",
        "test_window": test_months,
        "train_up_to": str(last_mature_train_month),
        "alarm_policy": "равный бюджет top-K на тестовом окне (детерминированный tie-break)",
        "target_alarm_rate": TARGET_ALARM_RATE,
        "onset_events_total": int(len(onset_set)),
        "note": ("Одно тестовое окно июнь–август 2024 — следствие годового прогрева "
                 "разметки (созревает один обучающий срез). Исследовательский результат."),
    }
    with open(destination / "summary.json", "w", encoding="utf-8") as target:
        json.dump(info, target, ensure_ascii=False, indent=2)

    print("\n=== LIKE-FOR-LIKE на событиях scripts/13 (равный бюджет) ===", flush=True)
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


if __name__ == "__main__":
    main()

"""Модель риска НАЧАЛА шока (onset): корректный протокол и честный статус.

Третий и четвёртый раунды аудита выявили:
  1. Скрипт падал (100 * None) до записи результата, а запуск поверх старых
     файлов оставлял прежний JSON — создавалась иллюзия успеха.
  2. Ноль шоков был следствием ВЫРОЖДЕННОЙ калибровки: масштаб разметки
     оценивался по одному месяцу (январь 2024), где центрированные значения
     строго нули, поэтому масштаб подставлялся аварийной константой 1.0.
  3. Событийная оценка была заявлена, но не реализована.

Исправления:
  - период КАЛИБРОВКИ разметки отделён от допустимых ОБУЧАЮЩИХ строк: масштаб и
    медианы оцениваются по всем доступным ДО теста месяцам (пред-тестовое окно),
    что не использует тестовые данные и не вырождается;
  - вырожденность калибровки детектируется явно (масштаб из данных или статус
    calibration_degenerate);
  - любое форматирование None защищено; статусы различимы; старый результат
    удаляется в начале; JSON пишется строго (allow_nan=False);
  - реализована СОБЫТИЙНАЯ оценка: порог берётся на обучении, предупреждение
    засчитывается только строго ДО начала эпизода, сопоставление один-к-одному,
    считаются event precision/recall, ложные тревоги на 100 МО-месяцев и
    распределение опережения (lead time).

Честная формулировка результата на этих данных:
  калибровка теперь корректна, но созревших обучающих срезов слишком мало —
  эффективность раннего предупреждения НЕ оценена. Это не доказательство
  невозможности задачи на 24 месяцах, а ограничение данной постановки и объёма.
"""

import os
import sys
import json
import math
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
from src.risk_eval import event_level_evaluation


TEST_START = pd.Period("2024-06", freq="M")
HORIZON = 3
MIN_DURATION = 2
THRESHOLD_SIGMA = 3.0
MIN_TRAIN_ONSETS = 5     # минимум положительных примеров начала шока для обучения
MIN_TRAIN_ROWS = 50


def write_json_strict(obj, path):
    """Записать JSON строго (без NaN/Inf). Если что-то не сериализуется —
    явно упасть, а не записать повреждённый файл."""
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    with open(path, "w", encoding="utf-8") as target:
        target.write(text)


def main(output_dir=None):
    """Запуск модели риска. output_dir позволяет писать результаты во ВРЕМЕННЫЙ
    каталог (для изоляции тестов), не затрагивая рабочие outputs/risk и MANIFEST."""
    data_config = utils.load_yaml_config(ROOT / "configs" / "data.yaml")
    if output_dir is None:
        destination = ROOT / "outputs" / "risk"
    else:
        destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    # Чистый старт: убираем прежний результат, чтобы неудача не маскировалась.
    result_path = destination / "risk_eval.json"
    if result_path.exists():
        os.remove(result_path)

    print("Загружаю данные и метки...", flush=True)
    consumption = data_loading.load_consumption_dataframe(data_config)
    total = consumption[consumption["category"] == "Все категории"]
    wide_total = total.pivot_table(index="territory_id", columns="month", values="value")
    wide_total = wide_total.sort_index(axis=1).dropna(axis=0, how="any")
    months = list(wide_total.columns)

    yoy_table = shock_labeling.compute_year_over_year_log_change(wide_total)
    yoy_months = pd.PeriodIndex(yoy_table["month"].astype(str), freq="M")

    # --- КАЛИБРОВКА разметки: пред-тестовое окно (без тестовых данных) -------
    # Отдельно от допустимых обучающих строк: масштаб оцениваем по ВСЕМ
    # доступным месяцам строго до теста, чтобы он идентифицировался по данным.
    calibration_mask = yoy_months < TEST_START
    calibration_yoy = yoy_table[calibration_mask]
    calibration_months = sorted(calibration_yoy["month"].unique().tolist())

    median_by_territory_map = {}
    for territory_id, group in calibration_yoy.groupby("territory_id"):
        median_by_territory_map[int(territory_id)] = float(group["yoy_log_change"].median())

    centered = []
    for _, row in calibration_yoy.iterrows():
        centered.append(row["yoy_log_change"] - median_by_territory_map.get(int(row["territory_id"]), 0.0))
    centered = np.asarray(centered, dtype=float)
    mad = float(np.median(np.abs(centered - np.median(centered)))) if len(centered) > 0 else 0.0
    robust_sigma_from_data = 1.4826 * mad
    calibration_degenerate = not (robust_sigma_from_data > 1e-9)
    robust_sigma = robust_sigma_from_data if not calibration_degenerate else float(np.std(centered)) if len(centered) > 0 else 0.0

    print("Калибровочные месяцы: " + str(calibration_months), flush=True)
    print("Масштаб разметки (robust_sigma) из данных: "
          + ("%.5f" % robust_sigma if robust_sigma > 0 else "0 (вырожден)"), flush=True)

    base_result = {
        "target": "onset (начало нового эпизода шока)",
        "test_start": str(TEST_START),
        "horizon_months": HORIZON,
        "min_duration": MIN_DURATION,
        "threshold_sigma": THRESHOLD_SIGMA,
        "calibration_months": calibration_months,
        "calibration_observations": int(len(centered)),
        "robust_sigma_from_data": robust_sigma_from_data,
        "calibration_degenerate": bool(calibration_degenerate),
        "max_abs_centered": float(np.max(np.abs(centered))) if len(centered) > 0 else 0.0,
        "label_maturation": "метка окна t+1..t+h подтверждается в t+h+(min_duration-1)",
        "metric_note": "average_precision_score — это average precision, а не трапецеидальная площадь под PR-кривой.",
    }

    if calibration_degenerate or robust_sigma <= 0:
        base_result["status"] = "calibration_degenerate"
        base_result["honest_conclusion"] = (
            "Масштаб разметки не идентифицируется по доступной калибровке — "
            "раннее предупреждение в этой постановке НЕ оценено. Это ограничение "
            "конфигурации и объёма данных, а не доказательство невозможности задачи."
        )
        write_json_strict(base_result, result_path)
        print("СТАТУС: calibration_degenerate — результат сохранён.", flush=True)
        return

    # --- Разметка по зафиксированным на калибровке статистикам --------------
    labels = shock_labeling.label_shocks_with_fixed_stats(
        yoy_table, median_by_territory_map, robust_sigma, THRESHOLD_SIGMA, MIN_DURATION
    )
    onset_set = shock_labeling.extract_episode_onsets(labels)

    # Внешние признаки.
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
    # Пометим сам месяц как onset (для событийной оценки).
    # Явный цикл по строкам вместо apply(lambda ...): для каждой строки берём
    # пару (территория, месяц) и проверяем, входит ли она в множество onset'ов.
    is_onset_month_flags = []
    for _, dataset_row in dataset.iterrows():
        territory_identifier = int(dataset_row["territory_id"])
        month_label = str(dataset_row["month"])
        is_onset_month_flags.append((territory_identifier, month_label) in onset_set)
    dataset["is_onset_month"] = is_onset_month_flags
    dataset.to_csv(destination / "risk_dataset.csv", index=False)

    # Допустимые обучающие строки: метка созрела строго до теста.
    last_mature_train_month = TEST_START - (HORIZON + MIN_DURATION)
    dataset["month_period"] = pd.PeriodIndex(dataset["month"].astype(str), freq="M")
    matured = dataset[dataset["label_matured"]].copy()
    train = matured[matured["month_period"] <= last_mature_train_month].copy()
    test = matured[matured["month_period"] >= TEST_START].copy()

    # Распределения по сплитам (для прозрачности, как просил аудит).
    base_result.update({
        "last_mature_train_month": str(last_mature_train_month),
        "train_months": sorted(train["month"].unique().tolist()),
        "test_months": sorted(test["month"].unique().tolist()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "onset_positives_train": int(train["onset_label"].sum()),
        "onset_positives_test": int(test["onset_label"].sum()),
        "shock_months_total": int(labels["is_shock"].sum()),
        "onset_events_total": int(len(onset_set)),
    })

    print("Обучающие месяцы (созревшие метки): " + str(base_result["train_months"]), flush=True)
    print("onset-положительных train/test: "
          + str(base_result["onset_positives_train"]) + " / "
          + str(base_result["onset_positives_test"]), flush=True)

    # --- Достаточно ли данных для честного обучения? -----------------------
    if base_result["onset_positives_train"] < MIN_TRAIN_ONSETS or base_result["train_rows"] < MIN_TRAIN_ROWS:
        base_result["status"] = "insufficient_training_data"
        base_result["event_evaluation"] = "not_reached"
        base_result["honest_conclusion"] = (
            "Калибровка разметки теперь корректна (масштаб оценён по "
            + str(base_result["calibration_observations"]) + " наблюдениям "
            "пред-тестового окна). Но созревших обучающих срезов слишком мало "
            "(" + str(len(base_result["train_months"])) + " мес, "
            + str(base_result["onset_positives_train"]) + " примеров начала шока), "
            "поэтому onset-модель не обучается и её событийная оценка не "
            "выполняется. Эффективность раннего предупреждения в этой постановке "
            "НЕ оценена — это ограничение объёма данных и выбранной конфигурации "
            "(годовая разность, горизонт 3, подтверждение 2), а НЕ доказательство "
            "невозможности задачи на 24 месяцах."
        )
        write_json_strict(base_result, result_path)
        print("СТАТУС: insufficient_training_data — результат сохранён.", flush=True)
        print(base_result["honest_conclusion"], flush=True)
        return

    # --- Обучение onset-модели и СОБЫТИЙНАЯ оценка --------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score

    feature_columns = ["yoy_t", "yoy_change", "mom_t", "mom_prev",
                       "recent_volatility", "rate_change", "cpi_signal"]
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train[feature_columns].to_numpy(dtype=float))
    test_features = scaler.transform(test[feature_columns].to_numpy(dtype=float))
    classifier = LogisticRegression(class_weight="balanced", max_iter=1000)
    classifier.fit(train_features, train["onset_label"].to_numpy(dtype=int))

    train_scores = classifier.predict_proba(train_features)[:, 1]
    test_scores = classifier.predict_proba(test_features)[:, 1]

    # Порог выбираем ТОЛЬКО на обучении.
    chosen_threshold = float(np.quantile(train_scores, 0.9))

    test_onset = test["onset_label"].to_numpy(dtype=int)
    average_precision = float(average_precision_score(test_onset, test_scores)) if test_onset.sum() > 0 else None

    event_metrics = event_level_evaluation(test, test_scores, chosen_threshold, HORIZON, onset_set)

    base_result["status"] = "trained"
    base_result["average_precision_onset_test"] = average_precision
    base_result["alarm_threshold_from_train_q90"] = chosen_threshold
    base_result["event_evaluation"] = event_metrics
    base_result["honest_conclusion"] = (
        "Onset-модель обучена; приведена СОБЫТИЙНАЯ оценка. Результат следует "
        "считать разведочным: один годовой переход и мало событий не позволяют "
        "заявлять надёжное раннее предупреждение."
    )
    write_json_strict(base_result, result_path)
    print("СТАТУС: trained. Событийная оценка:", flush=True)
    print(json.dumps(event_metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

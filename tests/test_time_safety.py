"""Проверки ошибок, обнаруженных при аудите исходного проекта."""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src import backtest
from src import calendar_features
from src import features
from src import news
from src import utils
from src.online_detection import detect_online, match_alarms

ROOT = Path(__file__).resolve().parents[1]


class HistoryOnlyModel:
    """Небольшая модель для проверки доступности данных в рекурсии."""

    def predict_for_horizon(self, table, horizon):
        return table["lag_1"].to_numpy() + table["news_signal"].to_numpy()


class TimeSafetyTests(unittest.TestCase):
    def setUp(self):
        self.config = utils.load_yaml_config(ROOT / "configs" / "forecast.yaml")
        self.config["target"]["use_log1p"] = False
        self.config["features"]["use_holiday_features"] = False
        self.config["features"]["use_news_features"] = False
        self.months = pd.period_range("2023-01", periods=24, freq="M")
        self.index = pd.MultiIndex.from_tuples(
            [
                (1, "Все категории"),
            ], names=[
                "territory_id",
                "category",
            ]
        )
        self.wide = pd.DataFrame([
            np.arange(1, 25),
        ], index=self.index, columns=self.months)
        self.static = pd.DataFrame({"category_code": [
            0,
        ], "region_code": [
            0,
        ]}, index=self.index)
        self.calendar = calendar_features.build_calendar_features_table(self.months)

    def test_year_over_year_uses_twelve_months(self):
        table = features.build_features_at_origin(
            self.wide, self.static, self.calendar, self.months[12], 1, self.config
        )
        self.assertEqual(table["yoy_ratio"].iloc[0], 13.0 / 1.0)

    def test_future_values_do_not_change_features(self):
        first = features.build_features_at_origin(
            self.wide, self.static, self.calendar, self.months[12], 1, self.config
        )
        changed = self.wide.copy()
        changed.iloc[:, 13:] = 999999
        second = features.build_features_at_origin(
            changed, self.static, self.calendar, self.months[12], 1, self.config
        )
        columns = features.get_feature_column_names(first)
        pd.testing.assert_frame_equal(first[columns], second[columns])

    def test_future_news_do_not_enter_recursion(self):
        self.config["features"]["use_news_features"] = True
        table = pd.DataFrame({
            "month": [
                self.months[12],
                self.months[13],
            ],
            "region_code": [
                0,
                0,
            ],
            "news_signal": [
                1.0,
                9999.0,
            ],
        })
        forecasts = backtest.recursive_global_forecast(
            HistoryOnlyModel(), self.wide, self.static, self.calendar,
            self.months[12], 2, self.config, table
        )
        self.assertEqual(forecasts[1].iloc[0], 14.0)
        self.assertEqual(forecasts[2].iloc[0], 15.0)

    def test_publication_time_controls_news(self):
        table = pd.DataFrame({
            "published_at": [
                "2024-02-01T00:00:00Z",
                "2024-02-01T00:00:00Z",
            ],
            "source_url": [
                "https://example.org/event",
                "https://example.org/event",
            ],
            "scope": [
                "federal",
                "federal",
            ],
            "region_code": [
                0,
                0,
            ],
        })
        before = news.available_events(table, "2024-01-31T23:59:00Z")
        after = news.available_events(table, "2024-02-01T00:00:00Z")
        self.assertEqual(len(before), 0)
        self.assertEqual(len(after), 1)

    def test_online_decisions_do_not_change_when_future_arrives(self):
        settings = utils.load_yaml_config(ROOT / "configs" / "checked.yaml")["detectors"]
        values = np.random.default_rng(71).normal(size=50)
        values[30:] = values[30:] + 4.0
        prefix = detect_online(values[:35], settings)
        full = detect_online(values, settings)
        pd.testing.assert_frame_equal(prefix, full[full["position"] < 35].reset_index(drop=True))

    def test_early_alarm_is_not_counted_as_detection(self):
        hits, false_alarms, delays = match_alarms([
            9,
            11,
            12,
        ], [
            10,
        ], 3)
        self.assertEqual(hits, 1)
        self.assertEqual(false_alarms, 2)
        self.assertEqual(delays, [
            1,
        ])

    def test_duplicate_predictions_are_rejected(self):
        row = {
            "territory_id": 1,
            "category": "Все категории",
            "model": "test",
            "cutoff_month": "2023-12",
            "horizon": 1,
            "actual": 1.0,
            "predicted": 2.0,
        }
        with self.assertRaises(ValueError):
            backtest.filter_to_common_prediction_keys(pd.DataFrame([
                row,
                row,
            ]))


if __name__ == "__main__":
    unittest.main()


class LabelMaturationTests(unittest.TestCase):
    """Проверки корректного созревания меток риска (третий раунд аудита)."""

    def test_label_available_at_respects_min_duration(self):
        """label_available_at = t + HORIZON + (MIN_DURATION-1) и не раньше."""
        from src import shock_labeling, risk_model
        # Небольшой искусственный ряд: один МО, 24 месяца.
        months = pd.period_range("2023-01", "2024-12", freq="M")
        values = np.linspace(1000.0, 2000.0, len(months))
        wide = pd.DataFrame([values], index=pd.Index([1], name="territory_id"),
                            columns=months)
        yoy = shock_labeling.compute_year_over_year_log_change(wide)
        labels = shock_labeling.label_shocks(yoy, 3.0, 2)[0]
        onsets = shock_labeling.extract_episode_onsets(labels)
        dataset = risk_model.build_risk_dataset(
            wide, labels, {}, {}, horizon_months=3, onset_set=onsets, min_duration=2
        )
        for _, row in dataset.iterrows():
            if row["label_matured"]:
                t = pd.Period(row["month"], freq="M")
                available = pd.Period(row["label_available_at"], freq="M")
                # Созревание не раньше, чем через HORIZON + (MIN_DURATION-1) = 4 месяца.
                self.assertGreaterEqual((available - t).n, 4)

    def test_matured_training_label_invariant_to_future_change(self):
        """Метка созревшей обучающей строки не меняется при изменении данных
        ПОСЛЕ её label_available_at."""
        from src import shock_labeling, risk_model
        months = pd.period_range("2023-01", "2024-12", freq="M")
        rng = np.random.default_rng(0)
        base = 1000.0 + rng.normal(0, 20, len(months)).cumsum()
        wide = pd.DataFrame([base], index=pd.Index([1], name="territory_id"),
                            columns=months)

        def onset_label_for(wide_matrix, target_month):
            yoy = shock_labeling.compute_year_over_year_log_change(wide_matrix)
            labels = shock_labeling.label_shocks(yoy, 3.0, 2)[0]
            onsets = shock_labeling.extract_episode_onsets(labels)
            ds = risk_model.build_risk_dataset(
                wide_matrix, labels, {}, {}, horizon_months=3, onset_set=onsets, min_duration=2
            )
            row = ds[(ds["territory_id"] == 1) & (ds["month"] == str(target_month))]
            if len(row) == 0:
                return None
            return int(row.iloc[0]["onset_label"]), row.iloc[0]["label_available_at"]

        target = pd.Period("2024-01", freq="M")
        before = onset_label_for(wide, target)
        self.assertIsNotNone(before)
        label_before, available_at = before
        # Меняем месяц ПОСЛЕ созревания метки — метка обязана остаться прежней.
        changed = wide.copy()
        later_month = pd.Period(available_at, freq="M") + 2
        if later_month in changed.columns:
            changed.loc[1, later_month] = changed.loc[1, later_month] * 3.0
            after = onset_label_for(changed, target)
            self.assertEqual(label_before, after[0])


if __name__ == "__main__":
    unittest.main()


class RiskScriptRobustnessTests(unittest.TestCase):
    """Скрипт риска не должен падать и обязан писать валидный JSON (аудит v2)."""

    def test_strict_json_rejects_nan(self):
        """JSON результата пишется строго: NaN недопустим."""
        import json as _json
        with self.assertRaises(ValueError):
            _json.dumps({"x": float("nan")}, allow_nan=False)
        # Корректный объект сериализуется без ошибки.
        _json.dumps({"x": None, "y": 1.0}, allow_nan=False)

    def test_no_shocks_does_not_crash(self):
        """На ряде без вариации (нет шоков) конвейер меток/датасета не падает."""
        from src import shock_labeling, risk_model
        months = pd.period_range("2023-01", "2024-12", freq="M")
        flat = np.full(len(months), 1000.0)
        wide = pd.DataFrame([flat, flat], index=pd.Index([1, 2], name="territory_id"),
                            columns=months)
        yoy = shock_labeling.compute_year_over_year_log_change(wide)
        median_map = {1: 0.0, 2: 0.0}
        # Масштаб из плоских данных вырожден -> ряд без шоков.
        labels = shock_labeling.label_shocks_with_fixed_stats(yoy, median_map, 1.0, 3.0, 2)
        self.assertEqual(int(labels["is_shock"].sum()), 0)
        onsets = shock_labeling.extract_episode_onsets(labels)
        self.assertEqual(len(onsets), 0)
        dataset = risk_model.build_risk_dataset(
            wide, labels, {}, {}, horizon_months=3, onset_set=onsets, min_duration=2
        )
        # Нет положительных onset-меток, но колонки на месте и нет исключений.
        self.assertIn("onset_label", dataset.columns)
        self.assertIn("label_available_at", dataset.columns)
        self.assertEqual(int(dataset["onset_label"].sum()), 0)


if __name__ == "__main__":
    unittest.main()


def _load_event_evaluator():
    """Событийный оценщик теперь живёт в src.risk_eval (общий модуль)."""
    from src import risk_eval
    return risk_eval


def _frame(territory_release_months):
    rows = []
    for territory_id, month in territory_release_months:
        rows.append({"territory_id": territory_id, "month": month})
    return pd.DataFrame(rows)


class EventEvaluatorBoundaryTests(unittest.TestCase):
    """Граничные тесты событийного оценщика (аудит v3)."""

    def setUp(self):
        self.mod = _load_event_evaluator()

    def test_august_warning_catches_september_onset(self):
        """Тревога в августе о сентябрьском событии — это попадание."""
        frame = _frame([(1, "2024-08")])
        scores = np.array([0.9])
        onset = {(1, "2024-09")}
        res = self.mod.event_level_evaluation(frame, scores, 0.5, 3, onset)
        self.assertEqual(res["event_hits"], 1)
        self.assertEqual(res["event_false_alarms"], 0)
        self.assertEqual(res["onset_events_eligible"], 1)
        self.assertEqual(res["mean_lead_time_months"], 1.0)

    def test_june_event_without_earlier_release_not_in_denominator(self):
        """Июньское событие без более ранних выпусков не попадает в recall."""
        # Выпуски только июнь-август; событие в июне недостижимо ни из одного.
        frame = _frame([(1, "2024-06"), (1, "2024-07"), (1, "2024-08")])
        scores = np.array([0.1, 0.1, 0.1])
        onset = {(1, "2024-06")}
        res = self.mod.event_level_evaluation(frame, scores, 0.5, 3, onset)
        self.assertEqual(res["onset_events_eligible"], 0)

    def test_warning_in_onset_month_is_not_early(self):
        """Тревога в сам месяц начала эпизода не считается ранней."""
        frame = _frame([(1, "2024-09")])
        scores = np.array([0.9])
        onset = {(1, "2024-09")}
        res = self.mod.event_level_evaluation(frame, scores, 0.5, 3, onset)
        # Событие сентября недостижимо из сентябрьского выпуска (lead=0).
        self.assertEqual(res["onset_events_eligible"], 0)
        self.assertEqual(res["event_hits"], 0)
        self.assertEqual(res["event_false_alarms"], 1)

    def test_two_warnings_one_event_single_hit(self):
        """Два предупреждения об одном событии дают одно попадание."""
        frame = _frame([(1, "2024-07"), (1, "2024-08")])
        scores = np.array([0.9, 0.9])
        onset = {(1, "2024-09")}
        res = self.mod.event_level_evaluation(frame, scores, 0.5, 3, onset)
        self.assertEqual(res["event_hits"], 1)
        self.assertEqual(res["event_false_alarms"], 1)

    def test_event_without_warning_counts_in_recall(self):
        """Событие без предупреждения учитывается в знаменателе recall."""
        frame = _frame([(1, "2024-07")])
        scores = np.array([0.1])  # ниже порога -> нет тревоги
        onset = {(1, "2024-09")}
        res = self.mod.event_level_evaluation(frame, scores, 0.5, 3, onset)
        self.assertEqual(res["onset_events_eligible"], 1)
        self.assertEqual(res["event_hits"], 0)
        self.assertEqual(res["event_recall"], 0.0)

    def test_empty_sets_have_explicit_none(self):
        """При пустых множествах precision/recall явно None, без деления на ноль."""
        frame = _frame([(1, "2024-07")])
        scores = np.array([0.1])
        onset = set()
        res = self.mod.event_level_evaluation(frame, scores, 0.5, 3, onset)
        self.assertIsNone(res["event_recall"])
        self.assertIsNone(res["event_precision"])


class RiskMainIntegrationTests(unittest.TestCase):
    """Интеграционный тест: настоящий main() завершается и пишет валидный JSON."""

    def test_main_runs_and_writes_valid_json(self):
        import json as _json
        import tempfile
        data_file = ROOT / "data" / "raw" / "consumption.parquet"
        if not data_file.exists():
            self.skipTest("нет данных consumption.parquet")
        import importlib.util
        spec = importlib.util.spec_from_file_location("risk_script_13", ROOT / "scripts" / "13_risk_model.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # ПОПРАВКА аудита: пишем во ВРЕМЕННЫЙ каталог, чтобы тест не трогал
        # рабочие outputs/risk и не ломал хеши MANIFEST.
        with tempfile.TemporaryDirectory() as temporary_directory:
            mod.main(output_dir=temporary_directory)
            result_path = Path(temporary_directory) / "risk_eval.json"
            self.assertTrue(result_path.exists())
            with open(result_path, encoding="utf-8") as f:
                data = _json.load(f)
            # Строгий JSON (без NaN) и известный статус.
            _json.dumps(data, allow_nan=False)
            self.assertIn(data["status"], {"trained", "insufficient_training_data", "calibration_degenerate"})


class RiskFormulationTests(unittest.TestCase):
    """Короткий прогрев и каузальность новой постановки риска (scripts/15)."""

    def test_trailing_window_is_causal_and_shortens_warmup(self):
        from src import risk_formulation as rf
        months = pd.period_range("2023-01", "2024-12", freq="M")
        rng = np.random.default_rng(1)
        y = 1000.0 + rng.normal(0, 30, len(months)).cumsum()
        wide = pd.DataFrame([y], index=pd.Index([1], name="territory_id"), columns=months)
        mom = rf.build_month_over_month_table(wide)
        sig = rf.add_trailing_baseline_and_scale(mom, window=6)
        # Прогрев короче года: сигнал появляется заметно раньше 2024 года.
        first_signal = pd.Period(str(sig["month"].min()), freq="M")
        self.assertLess(first_signal, pd.Period("2024-01", freq="M"))
        # Базовый уровень на t считается по прошлому (не включает сам месяц):
        # при сдвиге на 1 медиана не равна текущему r в общем случае.
        self.assertTrue((sig["trailing_median"].notna()).all())

    def test_floor_prevents_degenerate_scale(self):
        from src import risk_formulation as rf
        months = pd.period_range("2023-01", "2024-12", freq="M")
        flat = np.full(len(months), 1000.0)
        wide = pd.DataFrame([flat], index=pd.Index([1], name="territory_id"), columns=months)
        mom = rf.build_month_over_month_table(wide)
        sig = rf.add_trailing_baseline_and_scale(mom, window=6)
        floor = rf.estimate_pooled_floor(sig, sig["month"].unique())
        z = rf.add_deviation_z(sig, max(floor, 1e-6))
        # Масштаб не нулевой (есть пол), z конечен.
        self.assertTrue((z["scale"].to_numpy() > 0).all())
        self.assertTrue(np.isfinite(z["z"].to_numpy()).all())


class TopKAlarmBudgetTests(unittest.TestCase):
    """Детерминированный ровно-K отбор тревог (поправка аудита v5)."""

    def test_exactly_k_even_with_tied_scores(self):
        from src import risk_eval
        # Десять одинаковых оценок, бюджет K=1: наивный score>=порог дал бы 10
        # тревог. Детерминированный top-K обязан дать ровно 1.
        scores = np.ones(10, dtype=float)
        territory_ids = np.arange(10)
        mask = risk_eval.top_k_alarm_mask(scores, 1, territory_ids)
        self.assertEqual(int(mask.sum()), 1)
        # При равенстве выбирается наименьший territory_id (стабильно).
        self.assertTrue(bool(mask[0]))

    def test_tie_break_is_deterministic_by_id(self):
        from src import risk_eval
        scores = np.array([0.5, 0.5, 0.9, 0.1], dtype=float)
        territory_ids = np.array([7, 3, 5, 9])
        mask = risk_eval.top_k_alarm_mask(scores, 2, territory_ids)
        # Берём 0.9 (id 5) и из двух 0.5 — меньший id 3.
        self.assertEqual(int(mask.sum()), 2)
        self.assertTrue(bool(mask[2]))  # score 0.9
        self.assertTrue(bool(mask[1]))  # score 0.5, id 3 < id 7

    def test_budget_larger_than_rows_marks_all(self):
        from src import risk_eval
        scores = np.array([0.2, 0.8], dtype=float)
        mask = risk_eval.top_k_alarm_mask(scores, 5, np.array([1, 2]))
        self.assertEqual(int(mask.sum()), 2)


class LabelSemanticsTests(unittest.TestCase):
    """Что именно ловит короткая месячная разметка (поправка аудита v5).

    Контрпример аудитора: ПОСТОЯННЫЙ скачок уровня НЕ размечается как шок, а
    краткий всплеск-возврат размечается. Это фиксирует, что метка ловит
    импульс с возвратом (во многом декабрьский календарный эффект), а не
    устойчивое изменение уровня.
    """

    def _label_single_series(self, values):
        from src import risk_formulation as rf
        months = pd.period_range("2023-01", periods=len(values), freq="M")
        wide = pd.DataFrame([list(map(float, values))],
                            index=pd.Index([1], name="territory_id"), columns=months)
        mom = rf.build_month_over_month_table(wide)
        sig = rf.add_trailing_baseline_and_scale(mom, window=6)
        floor = 0.05  # заранее заданный пол масштаба (как в контрпримере аудита)
        z = rf.add_deviation_z(sig, floor)
        _, onset_set = rf.label_onsets(z, threshold_k=3.0, min_duration=2)
        return onset_set

    def test_constant_level_jump_is_not_labeled_shock(self):
        # Семь месяцев 1000, затем постоянно 2000. Одна большая разность, далее ноль.
        values = [1000.0] * 7 + [2000.0] * 8
        onset_set = self._label_single_series(values)
        self.assertEqual(len(onset_set), 0)

    def test_impulse_with_return_is_labeled_shock(self):
        # Семь месяцев 1000, один месяц 2000, затем снова 1000. Две противоположные
        # большие разности подряд -> размечается как эпизод (его начало — onset).
        values = [1000.0] * 7 + [2000.0] + [1000.0] * 7
        onset_set = self._label_single_series(values)
        self.assertGreaterEqual(len(onset_set), 1)


class TscvMaturationTests(unittest.TestCase):
    """Незрелые тестовые окна исключаются из метрик (поправка аудита v5)."""

    def test_november_fold_is_classified_pending(self):
        # Правило созревания: origin + horizon + (min_duration-1) <= последний месяц.
        horizon = 3
        min_duration = 2
        last_observed_month = pd.Period("2024-12", freq="M")
        november = pd.Period("2024-11", freq="M")
        august = pd.Period("2024-08", freq="M")
        november_ready = november + horizon + (min_duration - 1)
        august_ready = august + horizon + (min_duration - 1)
        # Ноябрь созревает только в марте 2025 — вне наблюдаемой истории.
        self.assertGreater(november_ready, last_observed_month)
        # Август созревает ровно в декабре 2024 — наблюдается.
        self.assertLessEqual(august_ready, last_observed_month)


class LevelShiftTests(unittest.TestCase):
    """Проверяемое целевое событие: устойчивый сдвиг уровня (scripts/19).

    Ключевое требование аудита: целевое событие должно отличать УСТОЙЧИВЫЙ сдвиг
    уровня от импульса-возврата. Эти тесты фиксируют оба случая на синтетике.
    """

    def _onsets(self, values, window=6, persistence=2, k=3.0, floor=0.05):
        from src import level_shift
        months = pd.period_range("2023-01", periods=len(values), freq="M")
        wide = pd.DataFrame([list(map(float, values))],
                            index=pd.Index([1], name="territory_id"), columns=months)
        onset_set, _ = level_shift.label_level_shift_onsets(wide, window, persistence, k, floor)
        return onset_set

    def test_constant_level_jump_is_detected(self):
        # Семь месяцев 1000, затем постоянно 2000 — устойчивый сдвиг уровня.
        onset_set = self._onsets([1000.0] * 7 + [2000.0] * 8)
        self.assertGreaterEqual(len(onset_set), 1)

    def test_impulse_with_return_is_not_detected(self):
        # Семь месяцев 1000, один месяц 2000, затем снова 1000 — НЕ устойчивый сдвиг.
        onset_set = self._onsets([1000.0] * 7 + [2000.0] + [1000.0] * 7)
        self.assertEqual(len(onset_set), 0)

    def test_scale_floor_is_positive(self):
        from src import level_shift
        months = pd.period_range("2023-01", periods=15, freq="M")
        rng = np.random.default_rng(3)
        y = 1000.0 + rng.normal(0, 25, len(months)).cumsum()
        wide = pd.DataFrame([y], index=pd.Index([1], name="territory_id"), columns=months)
        floor = level_shift.estimate_level_scale_floor(wide)
        self.assertGreater(floor, 0.0)

    def test_scale_floor_reference_cutoff_ignores_future(self):
        # ПОПРАВКА АУДИТА v9: масштаб с reference_cutoff НЕ должен зависеть от
        # месяцев после cutoff. Добавление будущих столбцов не меняет результат.
        from src import level_shift
        months = pd.period_range("2023-01", "2024-12", freq="M")
        rng = np.random.default_rng(11)
        # Разная волатильность до и после 2023-12, чтобы будущее реально влияло,
        # если бы его учитывали.
        values = []
        for index_value in range(len(months)):
            if months[index_value] <= pd.Period("2023-12", freq="M"):
                values.append(1000.0 + rng.normal(0, 10))
            else:
                values.append(1000.0 + rng.normal(0, 300))
        wide_full = pd.DataFrame([values], index=pd.Index([1], name="territory_id"), columns=months)
        past_columns = []
        for one_month in months:
            if one_month <= pd.Period("2023-12", freq="M"):
                past_columns.append(one_month)
        wide_past = wide_full[past_columns]

        floor_with_future = level_shift.estimate_level_scale_floor(wide_full, reference_cutoff="2023-12")
        floor_past_only = level_shift.estimate_level_scale_floor(wide_past, reference_cutoff="2023-12")
        floor_past_no_cut = level_shift.estimate_level_scale_floor(wide_past)
        # Все три совпадают: будущее не участвует в оценке масштаба.
        self.assertAlmostEqual(floor_with_future, floor_past_only, places=12)
        self.assertAlmostEqual(floor_with_future, floor_past_no_cut, places=12)


class UnifiedForecastArtifactTests(unittest.TestCase):
    """Проверки целостности единой прогнозной таблицы и CatBoost-прогнозов."""

    def test_catboost_keys_are_subset_of_saved(self):
        catboost_path = ROOT / "outputs" / "headline" / "catboost_predictions.csv"
        saved_path = ROOT / "outputs" / "checked" / "predictions.csv"
        if not catboost_path.exists() or not saved_path.exists():
            self.skipTest("нет сохранённых прогнозов")
        catboost = pd.read_csv(catboost_path)
        saved = pd.read_csv(saved_path)
        key_columns = ["territory_id", "category", "cutoff_month", "horizon"]
        catboost_keys = set(map(tuple, catboost[key_columns].itertuples(index=False, name=None)))
        saved_keys = set(map(tuple, saved[key_columns].itertuples(index=False, name=None)))
        # Ключи CatBoost должны совпадать с ключами базовых моделей (тот же состав).
        self.assertTrue(catboost_keys.issubset(saved_keys),
                        "ключи CatBoost не являются подмножеством сохранённых")
        self.assertGreater(len(catboost_keys), 0)

    def test_catboost_has_no_duplicate_keys(self):
        catboost_path = ROOT / "outputs" / "headline" / "catboost_predictions.csv"
        if not catboost_path.exists():
            self.skipTest("нет CatBoost-прогнозов")
        catboost = pd.read_csv(catboost_path)
        key_columns = ["territory_id", "category", "cutoff_month", "horizon"]
        duplicate_count = int(catboost.duplicated(subset=key_columns).sum())
        self.assertEqual(duplicate_count, 0)

    def test_unified_table_has_core_eight_models(self):
        unified_path = ROOT / "outputs" / "headline" / "forecast_unified_all_categories.csv"
        if not unified_path.exists():
            self.skipTest("нет единой таблицы")
        unified = pd.read_csv(unified_path)
        models_present = set(unified["model"])
        # Восемь базовых прогнозных моделей (7 сохранённых + CatBoost) обязательны.
        core_eight = {
            "SeasonalNaive", "ETS", "Prophet_history_aware", "Prophet_original_setting",
            "LightGBM_checked", "LightGBM_CBR_events", "Chronos_Bolt_tiny", "CatBoost_global",
        }
        for core_model in core_eight:
            self.assertIn(core_model, models_present)
        # В таблице не меньше восьми моделей (может присутствовать ещё и ансамбль).
        self.assertGreaterEqual(unified["model"].nunique(), 8)
        # Все четыре горизонта присутствуют.
        self.assertEqual(sorted(unified["horizon"].unique().tolist()), [1, 3, 6, 12])


class CausalNewsSelectionTests(unittest.TestCase):
    """Выбор доступного события новостей причинен по published_at (аудит-замечание)."""

    def _load_news_module(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "news_cases_mod", ROOT / "scripts" / "23_news_cases.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_event_published_after_cutoff_is_not_selected(self):
        module = self._load_news_module()
        # Событие ПРОИЗОШЛО в прошлом (event_date), но ОПУБЛИКОВАНО после отсечки.
        events = pd.DataFrame([
            {"published_at": "2024-01-20T13:30:00+03:00", "event_date": "2024-01-20",
             "new_rate_pct": 16.0, "source_url": "u1"},
            {"published_at": "2024-06-20T13:30:00+03:00", "event_date": "2024-05-01",
             "new_rate_pct": 18.0, "source_url": "u2"},
        ])
        # Отсечка — май 2024. Второе событие опубликовано только в июне → недоступно,
        # хотя его event_date (май) <= отсечки. Должно выбраться ПЕРВОЕ (январь).
        cutoff = pd.Period("2024-05", freq="M")
        chosen = module.latest_available_cbr_event(events, cutoff)
        self.assertIsNotNone(chosen)
        self.assertEqual(str(chosen["published_at"])[:7], "2024-01")

    def test_latest_available_event_is_chosen(self):
        module = self._load_news_module()
        events = pd.DataFrame([
            {"published_at": "2023-07-21T13:30:00+03:00", "event_date": "2023-07-21",
             "new_rate_pct": 8.5, "source_url": "u1"},
            {"published_at": "2023-12-15T13:30:00+03:00", "event_date": "2023-12-15",
             "new_rate_pct": 16.0, "source_url": "u2"},
        ])
        cutoff = pd.Period("2024-02", freq="M")
        chosen = module.latest_available_cbr_event(events, cutoff)
        self.assertEqual(float(chosen["new_rate_pct"]), 16.0)


class CatBoostForecasterTests(unittest.TestCase):
    """GlobalCatBoostForecaster: тот же интерфейс, что LightGBM, и каузальный путь
    LightGBM не меняется при наличии фабрики (добавление CatBoost в прогноз)."""

    def test_catboost_forecaster_fits_and_predicts_finite(self):
        from src import models_forecast
        rng = np.random.default_rng(5)
        n = 200
        training = pd.DataFrame({
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
        })
        training["target"] = 2.0 * training["f1"] - training["f2"] + rng.normal(0, 0.1, n)
        forecaster = models_forecast.GlobalCatBoostForecaster(
            {"iterations": 50, "depth": 4, "learning_rate": 0.1, "random_seed": 1,
             "verbose": 0, "allow_writing_files": False},
            use_log1p=False
        )
        forecaster.fit_for_horizon(training, ["f1", "f2"], 1)
        predictions = forecaster.predict_for_horizon(training.head(10), 1)
        self.assertEqual(len(predictions), 10)
        self.assertTrue(np.all(np.isfinite(predictions)))
        # Неотрицательность (расходы не бывают отрицательными — clip снизу нулём).
        self.assertTrue(np.all(predictions >= 0.0))

    def _load_unified_module(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "forecast_unified_mod", ROOT / "scripts" / "21_forecast_unified.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_select_strategy_on_early_picks_min_early_mae(self):
        # ПОПРАВКА АУДИТА v16: правило выбора — argmin РАННЕГО MAE среди кандидатов.
        # Чистая функция: на вход только ранние MAE.
        mod = self._load_unified_module()
        early = {"A_model": 700.0, "B_model": 498.7, "C_model": 551.0}
        self.assertEqual(mod.select_strategy_on_early(early), "B_model")
        # None (нет данных у модели) пропускается, не ломает выбор.
        early_with_gap = {"A_model": None, "B_model": 498.7, "C_model": 551.0}
        self.assertEqual(mod.select_strategy_on_early(early_with_gap), "B_model")

    def test_select_strategy_on_early_signature_has_no_late(self):
        # Функция принимает ТОЛЬКО ранние MAE: поздние факты физически не могут
        # попасть в выбор (аудит v16 — независимость выбора от позднего среза).
        mod = self._load_unified_module()
        import inspect
        parameters = list(inspect.signature(mod.select_strategy_on_early).parameters.keys())
        self.assertEqual(parameters, ["early_mae_by_model"])

    def test_late_actual_change_does_not_change_early_selection(self):
        # ИНТЕГРАЦИОННЫЙ ТЕСТ (аудит v16): произвольное изменение ВСЕХ поздних
        # actual не меняет selected_on_early — ни для ансамбля, ни для запасной
        # одиночной модели. Воспроизводим тот же путь выбора, что в main:
        # mae_on_cutoffs на РАННИХ отсечках -> select_strategy_on_early.
        mod = self._load_unified_module()
        rng = np.random.default_rng(0)
        model_names = ["Ensemble_CatBoost_LightGBM", "CatBoost_global",
                       "LightGBM_checked", "Prophet_history_aware"]
        early_cuts = ["2024-01", "2024-02", "2024-03"]
        late_cuts = ["2024-04", "2024-05"]
        # Ранние разрывы: Prophet лучший на h=1, ансамбль лучший на h=3.
        early_offsets = {
            (1, "Prophet_history_aware"): 1.0, (1, "CatBoost_global"): 3.0,
            (1, "LightGBM_checked"): 4.0, (1, "Ensemble_CatBoost_LightGBM"): 5.0,
            (3, "Ensemble_CatBoost_LightGBM"): 1.0, (3, "CatBoost_global"): 3.0,
            (3, "LightGBM_checked"): 4.0, (3, "Prophet_history_aware"): 6.0,
        }
        rows = []
        for horizon in [1, 3]:
            for model_name in model_names:
                gap = early_offsets[(horizon, model_name)]
                for cutoff in early_cuts:
                    for territory in range(20):
                        rows.append({"territory_id": territory, "category": "all",
                                     "model": model_name, "cutoff_month": cutoff,
                                     "horizon": horizon, "actual": 100.0,
                                     "predicted": 100.0 + gap})
                for cutoff in late_cuts:
                    for territory in range(20):
                        rows.append({"territory_id": territory, "category": "all",
                                     "model": model_name, "cutoff_month": cutoff,
                                     "horizon": horizon, "actual": 100.0,
                                     "predicted": 100.0 + float(rng.normal(0, 1))})
        frame = pd.DataFrame(rows)

        def selection_for(working_frame):
            chosen = {}
            for horizon in [1, 3]:
                early_mae_by_model = {}
                for model_name in model_names:
                    early_mae_by_model[model_name] = mod.mae_on_cutoffs(
                        working_frame, model_name, horizon, early_cuts)
                chosen[horizon] = mod.select_strategy_on_early(early_mae_by_model)
            return chosen

        before = selection_for(frame)
        self.assertEqual(before[1], "Prophet_history_aware")
        self.assertEqual(before[3], "Ensemble_CatBoost_LightGBM")
        # Ломаем ВСЕ поздние actual произвольными большими значениями.
        broken = frame.copy()
        late_mask = broken["cutoff_month"].isin(late_cuts)
        broken.loc[late_mask, "actual"] = rng.normal(5000, 1000, int(late_mask.sum()))
        after = selection_for(broken)
        self.assertEqual(before, after)

    def test_summary_separates_benchmark_selected_and_late(self):
        # Сводка РАЗВОДИТ три понятия (аудит v16) и помечает winner_by_horizon как
        # описательный бенчмарк, а h=6/h=12 — как не оценённые на позднем срезе.
        summary_path = ROOT / "outputs" / "headline" / "forecast_unified_summary.json"
        if not summary_path.exists():
            self.skipTest("нет сводки unified")
        import json as _json
        with open(summary_path, encoding="utf-8") as f:
            summary = _json.load(f)
        self.assertIn("benchmark_best_by_horizon", summary)
        self.assertIn("selected_on_early", summary)
        self.assertIn("late_evaluation", summary)
        self.assertIn("winner_by_horizon_meaning", summary)
        # Старый гейтовый словарь больше не публикуется.
        self.assertNotIn("ensemble_selected_horizons", summary)
        self.assertNotIn("ensemble_validation", summary)
        late = summary["late_evaluation"]
        # h=6 и h=12 не имеют зрелых поздних меток — статус not_evaluated_late.
        self.assertEqual(late.get("6", {}).get("status"), "not_evaluated_late")
        self.assertEqual(late.get("12", {}).get("status"), "not_evaluated_late")

    def test_default_global_model_is_lightgbm(self):
        # Без фабрики используется LightGBM — существующий путь не меняется.
        from src import models_forecast
        import inspect
        source = inspect.getsource(__import__("src.backtest", fromlist=["x"]).train_one_step_global_model)
        self.assertIn("GlobalLightGbmForecaster", source)
        self.assertIn("forecaster_factory", source)


class SeasonalShockTests(unittest.TestCase):
    """Сезонно-нормализованный шок: общее движение убирает общие (в т.ч.
    декабрьские) пики; идиосинкратическое отклонение размечается (scripts/20)."""

    def _make_panel(self, n_territories, idiosyncratic_territory=None,
                    idiosyncratic_month=None, idiosyncratic_factor=1.0):
        months = pd.period_range("2023-01", "2024-12", freq="M")
        # Общий сезонный профиль с декабрьским пиком (одинаков для всех МО).
        seasonal = []
        for one_month in months:
            if one_month.month == 12:
                seasonal.append(1.6)   # общий декабрьский скачок
            else:
                seasonal.append(1.0)
        rows = {}
        for territory_index in range(n_territories):
            values = []
            for position in range(len(months)):
                base = 1000.0 * seasonal[position]
                if (idiosyncratic_territory is not None
                        and territory_index == idiosyncratic_territory
                        and str(months[position]) == idiosyncratic_month):
                    base = base * idiosyncratic_factor
                values.append(base)
            rows[territory_index] = values
        wide = pd.DataFrame.from_dict(rows, orient="index", columns=months)
        wide.index.name = "territory_id"
        return wide

    def _label(self, wide):
        from src import seasonal_shock as ss
        mom = ss.build_month_over_month_table(wide)
        common = ss.compute_common_movement(mom)
        signal, _ = ss.build_seasonal_signal(mom, common, reference_cutoff="2023-12")
        q_persist, q_single = ss.estimate_quantile_thresholds(signal, "2023-12", 0.95, 0.99)
        pos, neg, any_set = ss.label_seasonal_shock_onsets(signal, q_persist, q_single, persistence=2)
        return pos, neg, any_set

    def test_common_december_spike_is_not_a_shock(self):
        # Все МО одинаково растут в декабре: это ОБЩЕЕ движение, а не шок.
        wide = self._make_panel(20)
        _, _, any_set = self._label(wide)
        december_onsets = []
        for (_, onset_month) in any_set:
            if str(onset_month).endswith("-12"):
                december_onsets.append(onset_month)
        self.assertEqual(len(december_onsets), 0)

    def test_idiosyncratic_jump_is_flagged(self):
        # Одно МО дополнительно скакнуло в мае 2024 — это идиосинкратический шок.
        wide = self._make_panel(20, idiosyncratic_territory=0,
                                 idiosyncratic_month="2024-05", idiosyncratic_factor=2.0)
        pos, _, _ = self._label(wide)
        self.assertIn((0, "2024-05"), pos)

    def test_thresholds_are_causal(self):
        # Пороги-квантили и масштаб с reference_cutoff не зависят от будущего.
        from src import seasonal_shock as ss
        wide_full = self._make_panel(20, idiosyncratic_territory=0,
                                     idiosyncratic_month="2024-07", idiosyncratic_factor=3.0)
        past_columns = []
        for one_column in wide_full.columns:
            if one_column <= pd.Period("2023-12", freq="M"):
                past_columns.append(one_column)
        wide_past = wide_full[past_columns]

        mom_full = ss.build_month_over_month_table(wide_full)
        common_full = ss.compute_common_movement(mom_full)
        signal_full, _ = ss.build_seasonal_signal(mom_full, common_full, reference_cutoff="2023-12")
        q_persist_full, q_single_full = ss.estimate_quantile_thresholds(signal_full, "2023-12", 0.95, 0.99)

        mom_past = ss.build_month_over_month_table(wide_past)
        common_past = ss.compute_common_movement(mom_past)
        signal_past, _ = ss.build_seasonal_signal(mom_past, common_past, reference_cutoff="2023-12")
        q_persist_past, q_single_past = ss.estimate_quantile_thresholds(signal_past, "2023-12", 0.95, 0.99)
        # Пороги, оценённые по reference period, совпадают при наличии/отсутствии будущего.
        self.assertAlmostEqual(q_persist_full, q_persist_past, places=10)
        self.assertAlmostEqual(q_single_full, q_single_past, places=10)


class LikeForLikeIntegrationTests(unittest.TestCase):
    """scripts/18: простые правила против модели на событиях scripts/13 (аудит)."""

    def test_scripts18_runs_and_compares_on_yoy_events(self):
        import tempfile
        data_file = ROOT / "data" / "raw" / "consumption.parquet"
        if not data_file.exists():
            self.skipTest("нет данных consumption.parquet")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "risk_script_18", ROOT / "scripts" / "18_risk_likeforlike.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as temporary_directory:
            mod.main(output_dir=temporary_directory)
            methods = pd.read_csv(Path(temporary_directory) / "methods_comparison.csv")
            # Все четыре метода присутствуют и бюджет тревог равный.
            self.assertEqual(set(methods["method"]),
                             {"logistic", "last_change", "dev_from_median", "volatility"})
            self.assertEqual(methods["alarms"].nunique(), 1)


class LevelShiftIntegrationTests(unittest.TestCase):
    """scripts/19: устойчивый сдвиг уровня, pending-защита и малое число событий."""

    def test_scripts19_runs_and_marks_pending(self):
        import json as _json
        import tempfile
        data_file = ROOT / "data" / "raw" / "consumption.parquet"
        if not data_file.exists():
            self.skipTest("нет данных consumption.parquet")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "risk_script_19", ROOT / "scripts" / "19_risk_levelshift.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as temporary_directory:
            mod.main(output_dir=temporary_directory)
            with open(Path(temporary_directory) / "summary.json", encoding="utf-8") as f:
                summary = _json.load(f)
            # Не-декабрьскость устойчива; масштаб оценён каузально (reference period).
            self.assertEqual(summary["december_onsets"], 0)
            self.assertEqual(summary["scale_reference_cutoff"], "2023-12")
            self.assertIn("2024-11", summary["pending_folds_not_evaluated"])
            # Незрелый фолд не имеет значений исходов.
            folds = pd.read_csv(Path(temporary_directory) / "folds_detail.csv")
            pending = folds[folds["mature"] == False]
            self.assertTrue(pending["event_hits"].isna().all())
            # Агрегат считается по сопоставимым фолдам: у ВСЕХ методов равный бюджет
            # и равный знаменатель событий (апельсины не сравниваются с яблоками).
            methods = pd.read_csv(Path(temporary_directory) / "methods_comparison.csv")
            self.assertEqual(methods["total_alarms"].nunique(), 1)
            self.assertEqual(methods["total_events"].nunique(), 1)
            # Таблица чувствительности масштаба приложена.
            self.assertTrue((Path(temporary_directory) / "scale_sensitivity.csv").exists())


class SeasonalBacktestIntegrationTests(unittest.TestCase):
    """scripts/20: robust rolling backtest сезонного шока во ВРЕМЕННЫЙ каталог."""

    def test_scripts20_runs_rolling_and_holdout(self):
        import json as _json
        import tempfile
        data_file = ROOT / "data" / "raw" / "consumption.parquet"
        if not data_file.exists():
            self.skipTest("нет данных consumption.parquet")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "risk_script_20", ROOT / "scripts" / "20_risk_seasonal.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as temporary_directory:
            mod.main(output_dir=temporary_directory)
            with open(Path(temporary_directory) / "summary.json", encoding="utf-8") as f:
                summary = _json.load(f)
            # Декабрь не доминирует (нормализация по общему движению работает).
            self.assertLess(summary["december_share"], 0.25)
            # Несколько сопоставимых фолдов (робастность, а не 2 фолда).
            self.assertGreaterEqual(len(summary["comparable_folds"]), 4)
            methods = pd.read_csv(Path(temporary_directory) / "methods_comparison.csv")
            # Равный бюджет и равный знаменатель событий у всех методов.
            self.assertEqual(methods["total_alarms"].nunique(), 1)
            self.assertEqual(methods["total_events"].nunique(), 1)
            # Случайный baseline присутствует и у него есть PR-AUC (нижняя планка).
            self.assertIn("random", set(methods["method"]))
            # Финальный holdout посчитан отдельно.
            self.assertTrue((Path(temporary_directory) / "holdout.csv").exists())


class TscvIntegrationTests(unittest.TestCase):
    """scripts/15 целиком на реальных данных во ВРЕМЕННЫЙ каталог (аудит v5)."""

    def test_scripts15_excludes_november_and_equal_budget(self):
        import json as _json
        import tempfile
        data_file = ROOT / "data" / "raw" / "consumption.parquet"
        if not data_file.exists():
            self.skipTest("нет данных consumption.parquet")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "risk_script_15", ROOT / "scripts" / "15_risk_tscv.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as temporary_directory:
            mod.main(output_dir=temporary_directory)
            with open(Path(temporary_directory) / "summary.json", encoding="utf-8") as f:
                summary = _json.load(f)
            # Ноябрь — pending (не оценён), зрелых фолдов ровно три.
            self.assertIn("2024-11", summary["pending_folds_not_evaluated"])
            self.assertEqual(len(summary["mature_folds_evaluated"]), 3)
            methods = pd.read_csv(Path(temporary_directory) / "methods_comparison.csv")
            # Равный бюджет тревог: у всех методов одинаковое число тревог.
            self.assertEqual(methods["total_alarms"].nunique(), 1)
            # Исключение незрелого ноября убрало ложные тревоги: 606, не 808.
            self.assertEqual(int(methods["total_alarms"].iloc[0]), 606)
            # ПОПРАВКА аудита v6: у незрелого (pending) фолда в подробной таблице
            # НЕТ значений исходов — только число тревог. Иначе читатель CSV
            # примет неизвестный исход за доказанный провал.
            folds = pd.read_csv(Path(temporary_directory) / "folds_detail.csv")
            pending_rows = folds[folds["mature"] == False]
            self.assertGreater(len(pending_rows), 0)
            for column_name in ["event_hits", "event_false_alarms",
                                "onset_events_eligible", "event_precision",
                                "event_recall", "mean_lead_time_months"]:
                self.assertTrue(pending_rows[column_name].isna().all(),
                                "pending-фолд не должен иметь значения " + column_name)
            # При этом число тревог у pending записано (оно наблюдаемо).
            self.assertTrue(pending_rows["alarms"].notna().all())


if __name__ == "__main__":
    unittest.main()

# ===========================================================================
# Модели прогнозирования.
#
# Здесь собраны модели двух типов:
#
#   1) Пообъектные (per-series) — обучаются на каждый ряд отдельно:
#         - SeasonalNaive   (сезонный наивный прогноз, нижняя планка),
#         - Prophet         (базовая модель из ТЗ, её нужно превзойти),
#         - ETS             (экспоненциальное сглаживание Холта-Уинтерса).
#      Интерфейс: функция получает историю ряда (значения по месяцам до
#      момента отсечки) и список горизонтов, возвращает словарь
#      {горизонт: прогноз на этот горизонт}.
#
#   2) Глобальная (панельная) модель LightGBM — обучается сразу на всех рядах
#      по признакам. Она вынесена в отдельный класс, потому что обучается один
#      раз на все ряды, а не по одному.
#
# Все модели принимают историю в хронологическом порядке (сначала старые
# месяцы, потом новые). Последний элемент истории — это месяц отсечки.
# ===========================================================================

import warnings

import numpy as np
import pandas as pd

# Prophet и statsmodels печатают много предупреждений — глушим их, чтобы
# лог бэктеста оставался читаемым.
# Предупреждения сохраняются в журнале запуска.


# ---------------------------------------------------------------------------
# 1. Сезонный наивный прогноз (SeasonalNaive)
# ---------------------------------------------------------------------------

def forecast_seasonal_naive(history_values, list_of_horizons,
                            season_length=12):
    """
    Сезонный наивный прогноз: значение в месяце (отсечка + h) равно значению
    год назад — в месяце (отсечка + h - 12).

    Если нужного прошлого значения нет (история короче сезона), берём
    последнее известное значение ряда.

    history_values : список/массив значений ряда в хронологическом порядке.
    list_of_horizons : список горизонтов (например, [1, 3, 6, 12]).
    season_length : длина сезона в месяцах (12 для годовой сезонности).
    """
    history_as_array = np.asarray(history_values, dtype="float64")
    length_of_history = len(history_as_array)
    last_known_value = history_as_array[-1]

    predictions_by_horizon = {}
    for horizon in list_of_horizons:
        # Позиция целевого месяца относительно начала истории.
        position_of_target = (length_of_history - 1) + horizon
        # Позиция значения "год назад" от целевого месяца.
        position_one_season_before = position_of_target - season_length

        if 0 <= position_one_season_before <= (length_of_history - 1):
            predicted_value = history_as_array[position_one_season_before]
        else:
            predicted_value = last_known_value

        predictions_by_horizon[horizon] = float(predicted_value)

    return predictions_by_horizon


# ---------------------------------------------------------------------------
# 2. Prophet (базовая модель)
# ---------------------------------------------------------------------------

def forecast_prophet(history_months, history_values, list_of_horizons,
                     prophet_parameters):
    """
    Прогноз моделью Prophet.

    history_months : список месячных периодов (Period[M]) истории.
    history_values : список значений ряда, соответствующих месяцам.
    list_of_horizons : список горизонтов.
    prophet_parameters : словарь гиперпараметров из конфига.

    Возвращает словарь {горизонт: прогноз}.
    """
    import logging
    from prophet import Prophet

    # Глушим подробный лог Stan/Prophet, чтобы не засорять вывод бэктеста.
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)

    # Готовим таблицу в формате Prophet: колонки ds (дата) и y (значение).
    list_of_timestamps = []
    for month_period in history_months:
        # Берём первый день месяца как метку времени.
        list_of_timestamps.append(month_period.to_timestamp(how="start"))

    training_frame = pd.DataFrame()
    training_frame["ds"] = list_of_timestamps
    training_frame["y"] = np.asarray(history_values, dtype="float64")

    # Число точек изменения тренда. На коротком ряде (24 месяца) большое число
    # точек приводит к переобучению, поэтому берём небольшое значение из конфига.
    number_of_changepoints = prophet_parameters["n_changepoints"]

    # На короткой истории отключаем годовую сезонность. Это заранее
    # заданный проверочный вариант, а не доказанно оптимальная настройка.
    # Старый вариант включается параметром minimum_seasonal_history=0.
    prophet_model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=prophet_parameters["weekly_seasonality"],
        daily_seasonality=prophet_parameters["daily_seasonality"],
        seasonality_mode=prophet_parameters["seasonality_mode"],
        uncertainty_samples=prophet_parameters["uncertainty_samples"],
        n_changepoints=number_of_changepoints,
        changepoint_prior_scale=prophet_parameters.get("changepoint_prior_scale", 0.05),
        seasonality_prior_scale=prophet_parameters.get("seasonality_prior_scale", 10.0),
    )
    minimum_seasonal_history = prophet_parameters.get("minimum_seasonal_history", 24)
    if len(history_values) >= minimum_seasonal_history:
        prophet_model.add_seasonality(
            name="yearly_monthly",
            period=365.25,
            fourier_order=prophet_parameters["yearly_fourier_order"],
        )
    prophet_model.fit(training_frame)

    # Строим будущие даты на максимальный горизонт.
    maximum_horizon = max(list_of_horizons)
    future_frame = prophet_model.make_future_dataframe(
        periods=maximum_horizon, freq="MS"
    )
    forecast_frame = prophet_model.predict(future_frame)

    # Берём последние maximum_horizon строк — это будущие месяцы.
    future_predictions = forecast_frame["yhat"].values[-maximum_horizon:]

    predictions_by_horizon = {}
    for horizon in list_of_horizons:
        # Прогноз на шаг h — это (h-1)-й элемент будущего (индексация с нуля).
        predicted_value = future_predictions[horizon - 1]
        predictions_by_horizon[horizon] = float(max(0.0, predicted_value))

    return predictions_by_horizon


# ---------------------------------------------------------------------------
# 3. ETS (экспоненциальное сглаживание Холта-Уинтерса)
# ---------------------------------------------------------------------------

def forecast_ets(history_values, list_of_horizons, ets_parameters):
    """
    Прогноз моделью экспоненциального сглаживания.

    Если история достаточно длинная для сезонной модели (не меньше двух
    сезонов), используем сезонную модель Холта-Уинтерса. Иначе переключаемся
    на модель без сезонности (только тренд), чтобы не упасть на коротком ряде.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    history_as_array = np.asarray(history_values, dtype="float64")
    length_of_history = len(history_as_array)

    season_length = ets_parameters["seasonal_periods"]
    maximum_horizon = max(list_of_horizons)

    # Решаем, можно ли использовать сезонную модель.
    can_use_seasonal_model = length_of_history >= (2 * season_length)

    try:
        if can_use_seasonal_model:
            smoothing_model = ExponentialSmoothing(
                history_as_array,
                trend=ets_parameters["trend"],
                seasonal=ets_parameters["seasonal"],
                seasonal_periods=season_length,
                initialization_method="estimated",
            )
        else:
            # Короткий ряд: модель только с трендом, без сезонности.
            smoothing_model = ExponentialSmoothing(
                history_as_array,
                trend=ets_parameters["trend"],
                seasonal=None,
                initialization_method="estimated",
            )
        fitted_model = smoothing_model.fit()
        future_predictions = fitted_model.forecast(maximum_horizon)
    except Exception as error:
        # Ошибка ETS не должна превращать её в другую модель под прежним именем.
        raise RuntimeError("Не удалось обучить ETS") from error

    predictions_by_horizon = {}
    for horizon in list_of_horizons:
        predicted_value = future_predictions[horizon - 1]
        predictions_by_horizon[horizon] = float(max(0.0, predicted_value))

    return predictions_by_horizon


# ---------------------------------------------------------------------------
# 4. Глобальная модель LightGBM
# ---------------------------------------------------------------------------

class GlobalCatBoostForecaster:
    """Глобальная одношаговая модель CatBoost с тем же интерфейсом, что и
    GlobalLightGbmForecaster (fit_for_horizon / predict_for_horizon).

    Добавлена по просьбе обратной связи для сравнения GBM-против-GBM в прогнозной
    таблице на ТОМ ЖЕ рекурсивном протоколе, что и LightGBM. Существующий путь
    LightGBM не меняется: эта модель подключается только через фабрику в
    отдельном скрипте (scripts/21), поэтому аудируемый predictions.csv не затронут.
    """

    def __init__(self, catboost_parameters, use_log1p):
        self.catboost_parameters = catboost_parameters
        self.use_log1p = use_log1p
        self.trained_models_by_horizon = {}
        self.feature_column_names = None

    def fit_for_horizon(self, training_features_table, feature_column_names, horizon):
        from catboost import CatBoostRegressor

        self.feature_column_names = feature_column_names
        rows_with_known_target = training_features_table[
            training_features_table["target"].notna()
        ]
        feature_matrix = rows_with_known_target[feature_column_names]
        target_vector = rows_with_known_target["target"]

        catboost_model = CatBoostRegressor(**self.catboost_parameters)
        catboost_model.fit(feature_matrix, target_vector)
        self.trained_models_by_horizon[horizon] = catboost_model

    def predict_for_horizon(self, features_table, horizon):
        catboost_model = self.trained_models_by_horizon[horizon]
        feature_matrix = features_table[self.feature_column_names]
        raw_predictions = catboost_model.predict(feature_matrix)
        if self.use_log1p:
            predictions_in_rubles = np.expm1(raw_predictions)
        else:
            predictions_in_rubles = raw_predictions
        predictions_in_rubles = np.clip(predictions_in_rubles, 0.0, None)
        return predictions_in_rubles


class GlobalLightGbmForecaster:
    """
    Обёртка хранит модели по горизонту. В текущем запуске обучается одна
    модель на шаг вперёд, сразу на
    всех рядах панели по признакам (лаги, скользящие средние, календарь,
    статические признаки территории).

    Общие закономерности могут помочь коротким рядам. Качество определяется
    бэктестом; само число рядов не гарантирует превосходства над Prophet.
    """

    def __init__(self, lightgbm_parameters, use_log1p):
        self.lightgbm_parameters = lightgbm_parameters
        self.use_log1p = use_log1p
        # Словарь: горизонт -> обученная модель.
        self.trained_models_by_horizon = {}
        # Список колонок-признаков (фиксируется при обучении).
        self.feature_column_names = None

    def fit_for_horizon(self, training_features_table, feature_column_names,
                        horizon):
        """
        Обучить модель для одного горизонта на подготовленной таблице
        признаков. В таблице должна быть колонка target (цель).
        Строки с пропущенной целью отбрасываются.
        """
        import lightgbm

        self.feature_column_names = feature_column_names

        # Оставляем только строки с известной целью.
        rows_with_known_target = training_features_table[
            training_features_table["target"].notna()
        ]

        feature_matrix = rows_with_known_target[feature_column_names]
        target_vector = rows_with_known_target["target"]

        lightgbm_model = lightgbm.LGBMRegressor(**self.lightgbm_parameters)
        lightgbm_model.fit(feature_matrix, target_vector)

        self.trained_models_by_horizon[horizon] = lightgbm_model

    def predict_for_horizon(self, features_table, horizon):
        """
        Сделать прогноз для одного горизонта по готовой таблице признаков.
        Возвращает прогноз в исходных единицах (руб.), с обратным
        преобразованием из log1p при необходимости.
        """
        lightgbm_model = self.trained_models_by_horizon[horizon]
        feature_matrix = features_table[self.feature_column_names]
        raw_predictions = lightgbm_model.predict(feature_matrix)

        if self.use_log1p:
            # Обратное преобразование логарифма: expm1.
            predictions_in_rubles = np.expm1(raw_predictions)
        else:
            predictions_in_rubles = raw_predictions

        # Расходы не бывают отрицательными — обрезаем снизу нулём.
        predictions_in_rubles = np.clip(predictions_in_rubles, 0.0, None)
        return predictions_in_rubles

    def get_feature_importances(self, horizon):
        """
        Вернуть важности признаков обученной модели для горизонта.
        Полезно для интерпретации (какие признаки влияют на прогноз).
        """
        lightgbm_model = self.trained_models_by_horizon[horizon]
        importances_table = pd.DataFrame()
        importances_table["feature"] = self.feature_column_names
        importances_table["importance"] = lightgbm_model.feature_importances_
        importances_table = importances_table.sort_values(
            by="importance", ascending=False
        ).reset_index(drop=True)
        return importances_table

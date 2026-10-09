# ===========================================================================
# Обнаружение точек структурных изменений (changepoint detection, CPD).
#
# Здесь собраны и сравниваются несколько методов:
#   1) ruptures PELT    — точный поиск разладок за линейное время;
#   2) ruptures BinSeg  — бинарная сегментация (жадный поиск);
#   3) ruptures Window  — оконный детектор;
#   4) CUSUM            — накопленные суммы отклонений (классика, онлайн);
#   5) BOCPD            — байесовское онлайн-обнаружение (Adams & MacKay, 2007).
#
# Дополнительно реализован онлайн-детектор шоков по остаткам прогноза —
# он связывает прогнозирование и обнаружение разладок: если факт сильно и
# устойчиво отклоняется от прогноза, это ранний сигнал структурного сдвига.
#
# Все детекторы возвращают список индексов точек изменения внутри ряда
# (индекс — позиция месяца в ряде, считая с нуля).
# ===========================================================================

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Предобработка ряда перед поиском разладок
# ---------------------------------------------------------------------------

def deseasonalize_monthly_series(series_values, season_length=12):
    """
    Убрать годовую сезонность сезонной разностью: из каждого значения вычитаем
    значение за тот же месяц год назад.

    Для первых season_length точек прошлого года нет — там оставляем приращение
    относительно среднего (чтобы не терять начало ряда).

    Возвращает массив той же длины, что и исходный ряд.
    """
    values_array = np.asarray(series_values, dtype="float64")
    length_of_series = len(values_array)
    deseasonalized = np.zeros(length_of_series, dtype="float64")

    overall_mean = np.mean(values_array)

    for position in range(length_of_series):
        if position >= season_length:
            deseasonalized[position] = (
                values_array[position] - values_array[position - season_length]
            )
        else:
            deseasonalized[position] = values_array[position] - overall_mean

    return deseasonalized


def standardize_series(series_values):
    """
    Привести ряд к нулевому среднему и единичному стандартному отклонению.
    Если разброс нулевой, возвращаем ряд из нулей.
    """
    values_array = np.asarray(series_values, dtype="float64")
    mean_value = np.mean(values_array)
    standard_deviation = np.std(values_array)

    if standard_deviation == 0.0:
        return np.zeros_like(values_array)

    standardized = (values_array - mean_value) / standard_deviation
    return standardized


def preprocess_series(series_values, preprocessing_config):
    """
    Применить к ряду предобработку согласно конфигу: сначала (при необходимости)
    убрать сезонность, затем (при необходимости) стандартизировать.
    """
    processed_values = np.asarray(series_values, dtype="float64")

    if preprocessing_config["deseasonalize"]:
        processed_values = deseasonalize_monthly_series(processed_values)

    if preprocessing_config["standardize"]:
        processed_values = standardize_series(processed_values)

    return processed_values


# ---------------------------------------------------------------------------
# 1-3. Методы из библиотеки ruptures
# ---------------------------------------------------------------------------

def detect_with_ruptures_pelt(series_values, ruptures_config):
    """
    Поиск точек разладки алгоритмом PELT из библиотеки ruptures.
    Возвращает список индексов точек изменения (без последней точки-конца ряда).
    """
    import ruptures

    signal_as_column = np.asarray(series_values, dtype="float64").reshape(-1, 1)

    pelt_detector = ruptures.Pelt(
        model=ruptures_config["model_cost"],
        min_size=ruptures_config["min_size"],
    )
    pelt_detector.fit(signal_as_column)
    breakpoints = pelt_detector.predict(pen=ruptures_config["penalty"])

    # ruptures возвращает последнюю точку = длина ряда; её убираем.
    changepoints = []
    for breakpoint_index in breakpoints:
        if breakpoint_index < len(series_values):
            changepoints.append(int(breakpoint_index))
    return changepoints


def detect_with_ruptures_binseg(series_values, ruptures_config):
    """
    Поиск точек разладки методом бинарной сегментации (Binary Segmentation).
    """
    import ruptures

    signal_as_column = np.asarray(series_values, dtype="float64").reshape(-1, 1)

    binseg_detector = ruptures.Binseg(
        model=ruptures_config["model_cost"],
        min_size=ruptures_config["min_size"],
    )
    binseg_detector.fit(signal_as_column)
    breakpoints = binseg_detector.predict(pen=ruptures_config["penalty"])

    changepoints = []
    for breakpoint_index in breakpoints:
        if breakpoint_index < len(series_values):
            changepoints.append(int(breakpoint_index))
    return changepoints


def detect_with_ruptures_window(series_values, ruptures_config):
    """
    Поиск точек разладки оконным методом (Window-based).
    """
    import ruptures

    signal_as_column = np.asarray(series_values, dtype="float64").reshape(-1, 1)

    window_detector = ruptures.Window(
        width=ruptures_config["window_width"],
        model=ruptures_config["model_cost"],
        min_size=ruptures_config["min_size"],
    )
    window_detector.fit(signal_as_column)
    breakpoints = window_detector.predict(pen=ruptures_config["penalty"])

    changepoints = []
    for breakpoint_index in breakpoints:
        if breakpoint_index < len(series_values):
            changepoints.append(int(breakpoint_index))
    return changepoints


# ---------------------------------------------------------------------------
# 4. CUSUM (накопленные суммы отклонений)
# ---------------------------------------------------------------------------

def detect_with_cusum(series_values, cusum_config):
    """
    Двусторонний CUSUM-детектор сдвига среднего.

    Ряд предполагается уже стандартизованным (среднее около 0, СКО около 1).
    Накапливаем положительные и отрицательные отклонения от нуля за вычетом
    "мёртвой зоны" (drift). Когда накопленная сумма превышает порог, объявляем
    точку изменения и сбрасываем накопители.

        S_high[t] = max(0, S_high[t-1] + x[t] - drift)
        S_low[t]  = min(0, S_low[t-1]  + x[t] + drift)
        тревога, если S_high > threshold или S_low < -threshold

    threshold и drift заданы в единицах стандартного отклонения.
    """
    values_array = np.asarray(series_values, dtype="float64")
    length_of_series = len(values_array)

    threshold = cusum_config["threshold_sigma"]
    drift = cusum_config["drift_sigma"]

    # Работаем с первыми разностями ряда. Сдвиг УРОВНЯ ряда превращается в
    # одиночный всплеск разности, поэтому CUSUM по разностям быстро ловит
    # момент сдвига и не накапливает ложные тревоги внутри стабильного отрезка
    # (где разности колеблются около нуля).
    first_differences = np.diff(values_array)

    # Масштабируем разности по их стандартному отклонению, чтобы пороги в
    # сигмах имели одинаковый смысл для разных рядов.
    difference_std = np.std(first_differences)
    if difference_std == 0.0:
        return []
    scaled_differences = first_differences / difference_std

    cumulative_high = 0.0
    cumulative_low = 0.0
    detected_changepoints = []

    for difference_position in range(len(scaled_differences)):
        current_value = scaled_differences[difference_position]

        cumulative_high = max(0.0, cumulative_high + current_value - drift)
        cumulative_low = min(0.0, cumulative_low + current_value + drift)

        alarm_triggered = False
        if cumulative_high > threshold:
            alarm_triggered = True
        if cumulative_low < -threshold:
            alarm_triggered = True

        if alarm_triggered:
            # Разность с индексом i соответствует изменению между точками i и
            # i+1 исходного ряда, поэтому точка разладки — это position + 1.
            changepoint_position = difference_position + 1
            detected_changepoints.append(int(changepoint_position))
            # Сбрасываем накопители после срабатывания.
            cumulative_high = 0.0
            cumulative_low = 0.0

    return detected_changepoints


# ---------------------------------------------------------------------------
# 5. BOCPD (Bayesian Online Changepoint Detection)
# ---------------------------------------------------------------------------

def detect_with_bocpd(series_values, bocpd_config):
    """
    Байесовское онлайн-обнаружение точек разладки (Adams & MacKay, 2007)
    с нормально-гамма сопряжённым априори (неизвестные среднее и дисперсия,
    предиктивное распределение — распределение Стьюдента).

    На каждом шаге поддерживается распределение по "длине текущего отрезка
    стабильности" (run length). Если наиболее вероятная длина отрезка резко
    падает к нулю, значит произошла разладка.

    Возвращает список индексов точек изменения.
    """
    from scipy import stats

    values_array = np.asarray(series_values, dtype="float64")
    length_of_series = len(values_array)

    # Постоянная hazard-функция: вероятность разладки на каждом шаге.
    expected_run_length = bocpd_config["expected_run_length"]
    hazard_probability = 1.0 / float(expected_run_length)

    # Параметры априорного нормально-гамма распределения.
    mu0 = bocpd_config["mu0"]
    kappa0 = bocpd_config["kappa0"]
    alpha0 = bocpd_config["alpha0"]
    beta0 = bocpd_config["beta0"]

    probability_threshold = bocpd_config["probability_threshold"]

    # Массивы параметров для каждой возможной длины отрезка.
    # Начинаем с одного состояния (длина отрезка = 0).
    mu_parameters = np.array([mu0], dtype="float64")
    kappa_parameters = np.array([kappa0], dtype="float64")
    alpha_parameters = np.array([alpha0], dtype="float64")
    beta_parameters = np.array([beta0], dtype="float64")

    # Распределение по длине отрезка. Сейчас вся масса в длине 0.
    run_length_distribution = np.array([1.0], dtype="float64")

    detected_changepoints = []
    previous_most_probable_run_length = 0

    for position in range(length_of_series):
        current_value = values_array[position]

        # Предиктивная вероятность текущего наблюдения для каждой длины отрезка
        # (распределение Стьюдента с параметрами из нормально-гамма).
        degrees_of_freedom = 2.0 * alpha_parameters
        predictive_scale = np.sqrt(
            beta_parameters * (kappa_parameters + 1.0)
            / (alpha_parameters * kappa_parameters)
        )
        predictive_probability = stats.t.pdf(
            current_value,
            df=degrees_of_freedom,
            loc=mu_parameters,
            scale=predictive_scale,
        )

        # Вероятность роста отрезка (разладки не было).
        growth_probabilities = (
            run_length_distribution * predictive_probability * (1.0 - hazard_probability)
        )
        # Вероятность разладки (длина отрезка сбрасывается в 0).
        changepoint_probability = np.sum(
            run_length_distribution * predictive_probability * hazard_probability
        )

        # Новое распределение по длине отрезка.
        new_run_length_distribution = np.empty(len(growth_probabilities) + 1)
        new_run_length_distribution[0] = changepoint_probability
        new_run_length_distribution[1:] = growth_probabilities

        # Нормируем, чтобы сумма вероятностей была равна 1.
        total_mass = np.sum(new_run_length_distribution)
        if total_mass > 0.0:
            new_run_length_distribution = new_run_length_distribution / total_mass
        run_length_distribution = new_run_length_distribution

        # Обновляем параметры нормально-гамма для каждой длины отрезка.
        new_mu = np.empty(len(mu_parameters) + 1)
        new_kappa = np.empty(len(kappa_parameters) + 1)
        new_alpha = np.empty(len(alpha_parameters) + 1)
        new_beta = np.empty(len(beta_parameters) + 1)

        # Состояние "длина 0" — это априорные параметры.
        new_mu[0] = mu0
        new_kappa[0] = kappa0
        new_alpha[0] = alpha0
        new_beta[0] = beta0

        # Остальные состояния — обновление по текущему наблюдению.
        new_kappa[1:] = kappa_parameters + 1.0
        new_mu[1:] = (kappa_parameters * mu_parameters + current_value) / (kappa_parameters + 1.0)
        new_alpha[1:] = alpha_parameters + 0.5
        new_beta[1:] = beta_parameters + (
            kappa_parameters * (current_value - mu_parameters) ** 2
        ) / (2.0 * (kappa_parameters + 1.0))

        mu_parameters = new_mu
        kappa_parameters = new_kappa
        alpha_parameters = new_alpha
        beta_parameters = new_beta

        # Наиболее вероятная длина отрезка на текущем шаге.
        most_probable_run_length = int(np.argmax(run_length_distribution))

        # Признак разладки: наиболее вероятная длина отрезка РЕЗКО
        # сбрасывается к малому значению после того, как отрезок уже заметно
        # вырос. Это означает, что модель решила: начался новый режим.
        #   - до сброса отрезок должен быть не короче minimum_segment_before
        #     (чтобы не реагировать на мелкий шум в самом начале отрезка);
        #   - после сброса длина отрезка становится маленькой (<= 1).
        minimum_segment_before = 2
        if position > 0:
            run_length_was_long_enough = (
                previous_most_probable_run_length >= minimum_segment_before
            )
            run_length_reset_to_small = most_probable_run_length <= 1
            if run_length_was_long_enough and run_length_reset_to_small:
                detected_changepoints.append(int(position))

        previous_most_probable_run_length = most_probable_run_length

    return detected_changepoints


# ---------------------------------------------------------------------------
# Единый интерфейс запуска всех методов
# ---------------------------------------------------------------------------

def detect_changepoints_all_methods(series_values, changepoint_config):
    """
    Запустить все включённые в конфиге методы на одном ряде (после общей
    предобработки) и вернуть словарь: имя метода -> список индексов разладок.
    """
    methods_flags = changepoint_config["methods"]
    ruptures_config = changepoint_config["ruptures_params"]
    cusum_config = changepoint_config["cusum_params"]
    bocpd_config = changepoint_config["bocpd_params"]
    preprocessing_config = changepoint_config["preprocessing"]

    processed_series = preprocess_series(series_values, preprocessing_config)

    results_by_method = {}

    if methods_flags["ruptures_pelt"]:
        results_by_method["ruptures_PELT"] = detect_with_ruptures_pelt(
            processed_series, ruptures_config
        )
    if methods_flags["ruptures_binseg"]:
        results_by_method["ruptures_BinSeg"] = detect_with_ruptures_binseg(
            processed_series, ruptures_config
        )
    if methods_flags["ruptures_window"]:
        results_by_method["ruptures_Window"] = detect_with_ruptures_window(
            processed_series, ruptures_config
        )
    if methods_flags["cusum"]:
        results_by_method["CUSUM"] = detect_with_cusum(
            processed_series, cusum_config
        )
    if methods_flags["bocpd"]:
        results_by_method["BOCPD"] = detect_with_bocpd(
            processed_series, bocpd_config
        )

    return results_by_method


# ---------------------------------------------------------------------------
# Генерация синтетических рядов с известными точками разладки
# ---------------------------------------------------------------------------

def generate_synthetic_series_with_changepoints(series_length, number_of_changepoints,
                                                noise_sigma, random_generator):
    """
    Сгенерировать один синтетический ряд с кусочно-постоянным средним и
    известными точками разладки.

    На каждом отрезке среднее постоянно, между отрезками оно скачком меняется.
    Поверх добавляется гауссов шум. Такой ряд имитирует структурные сдвиги
    уровня потребления.

    Возвращает кортеж (значения_ряда, список_истинных_точек_разладки).
    """
    # Выбираем позиции точек разладки так, чтобы они не были слишком близко
    # к краям и друг к другу.
    minimum_gap = max(4, series_length // (number_of_changepoints + 2))
    true_changepoints = []
    candidate_position = minimum_gap
    for changepoint_number in range(number_of_changepoints):
        # Немного случайно сдвигаем позицию, сохраняя минимальный зазор.
        jitter = int(random_generator.integers(-2, 3))
        position = candidate_position + jitter
        if position < 2:
            position = 2
        if position > series_length - 2:
            position = series_length - 2
        true_changepoints.append(position)
        candidate_position = candidate_position + minimum_gap

    true_changepoints = sorted(set(true_changepoints))

    # Задаём уровни на отрезках. Каждый следующий уровень отличается от
    # предыдущего на заметную величину (несколько сигм шума).
    segment_levels = []
    current_level = float(random_generator.normal(0.0, 1.0))
    segment_levels.append(current_level)
    for changepoint_number in range(len(true_changepoints)):
        level_jump = float(random_generator.normal(0.0, 1.0))
        # Усиливаем скачок, чтобы он был различим на фоне шума.
        if level_jump >= 0:
            level_jump = level_jump + 3.0 * noise_sigma
        else:
            level_jump = level_jump - 3.0 * noise_sigma
        current_level = current_level + level_jump
        segment_levels.append(current_level)

    # Собираем ряд: для каждой точки берём уровень её отрезка плюс шум.
    series_values = np.zeros(series_length, dtype="float64")
    current_segment_index = 0
    for position in range(series_length):
        # Если дошли до очередной точки разладки, переходим на следующий отрезок.
        if (current_segment_index < len(true_changepoints)
                and position >= true_changepoints[current_segment_index]):
            current_segment_index = current_segment_index + 1
        level_here = segment_levels[current_segment_index]
        noise_here = float(random_generator.normal(0.0, noise_sigma))
        series_values[position] = level_here + noise_here

    return series_values, true_changepoints


def match_detected_to_true(detected_changepoints, true_changepoints, tolerance):
    """
    Сопоставить найденные точки разладки с истинными с допуском tolerance.

    Истинная точка считается "пойманной", если рядом с ней (в пределах
    tolerance) есть найденная точка. Каждую найденную точку используем не
    более одного раза.

    Возвращает кортеж (число_попаданий, число_ложных_срабатываний,
    список_задержек_по_пойманным_точкам).
    """
    remaining_detected = list(detected_changepoints)
    number_of_hits = 0
    detection_delays = []

    for true_point in true_changepoints:
        best_match_index = None
        best_distance = None
        # Ищем ближайшую найденную точку в пределах допуска.
        for detected_index in range(len(remaining_detected)):
            detected_point = remaining_detected[detected_index]
            distance = abs(detected_point - true_point)
            if distance <= tolerance:
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_match_index = detected_index

        if best_match_index is not None:
            number_of_hits = number_of_hits + 1
            matched_point = remaining_detected[best_match_index]
            # Задержка: насколько позже истинной точки сработал детектор
            # (отрицательная задержка = сработал раньше, тоже допустимо).
            detection_delays.append(matched_point - true_point)
            # Убираем использованную найденную точку.
            remaining_detected.pop(best_match_index)

    number_of_false_alarms = len(remaining_detected)
    return number_of_hits, number_of_false_alarms, detection_delays


def evaluate_detector_on_synthetic(method_name, changepoint_config,
                                   synthetic_config, tolerance):
    """
    Оценить один метод обнаружения разладок на наборе синтетических рядов.

    Для каждого ряда генерируем известные точки разладки, запускаем детектор и
    считаем попадания/пропуски/ложные срабатывания. По всем рядам агрегируем в
    precision, recall, F1 и среднюю задержку детекции.

    Возвращает словарь с метриками качества детектора.
    """
    random_generator = np.random.default_rng(synthetic_config["random_seed"])

    number_of_series = synthetic_config["n_series"]
    series_length = synthetic_config["series_length"]
    number_of_changepoints = synthetic_config["n_changepoints"]
    noise_sigma = synthetic_config["noise_sigma"]

    total_hits = 0
    total_true_points = 0
    total_false_alarms = 0
    all_detection_delays = []

    for series_number in range(number_of_series):
        series_values, true_changepoints = generate_synthetic_series_with_changepoints(
            series_length, number_of_changepoints, noise_sigma, random_generator
        )

        # Запускаем конкретный метод (через общий интерфейс, но берём один метод).
        results_by_method = run_single_method(
            method_name, series_values, changepoint_config
        )
        detected_changepoints = results_by_method

        hits, false_alarms, delays = match_detected_to_true(
            detected_changepoints, true_changepoints, tolerance
        )

        total_hits = total_hits + hits
        total_true_points = total_true_points + len(true_changepoints)
        total_false_alarms = total_false_alarms + false_alarms
        for delay_value in delays:
            all_detection_delays.append(delay_value)

    # Precision = доля верных среди всех сработавших.
    number_of_detections = total_hits + total_false_alarms
    if number_of_detections > 0:
        precision = total_hits / number_of_detections
    else:
        precision = 0.0

    # Recall = доля пойманных среди всех истинных точек.
    if total_true_points > 0:
        recall = total_hits / total_true_points
    else:
        recall = 0.0

    # F1 — гармоническое среднее precision и recall.
    if (precision + recall) > 0:
        f1_score = 2.0 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0

    if len(all_detection_delays) > 0:
        mean_absolute_delay = float(np.mean(np.abs(all_detection_delays)))
    else:
        mean_absolute_delay = float("nan")

    metrics = {
        "method": method_name,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "mean_abs_delay": mean_absolute_delay,
        "total_hits": total_hits,
        "total_true": total_true_points,
        "total_false_alarms": total_false_alarms,
    }
    return metrics


def run_single_method(method_name, series_values, changepoint_config):
    """
    Запустить ОДИН метод обнаружения разладок по его имени.
    Используется в оценке на синтетике, где важно гонять методы по отдельности.
    """
    ruptures_config = changepoint_config["ruptures_params"]
    cusum_config = changepoint_config["cusum_params"]
    bocpd_config = changepoint_config["bocpd_params"]
    preprocessing_config = changepoint_config["preprocessing"]

    processed_series = preprocess_series(series_values, preprocessing_config)

    if method_name == "ruptures_PELT":
        return detect_with_ruptures_pelt(processed_series, ruptures_config)
    if method_name == "ruptures_BinSeg":
        return detect_with_ruptures_binseg(processed_series, ruptures_config)
    if method_name == "ruptures_Window":
        return detect_with_ruptures_window(processed_series, ruptures_config)
    if method_name == "CUSUM":
        return detect_with_cusum(processed_series, cusum_config)
    if method_name == "BOCPD":
        return detect_with_bocpd(processed_series, bocpd_config)

    raise ValueError("Неизвестный метод обнаружения разладок: " + str(method_name))


# ---------------------------------------------------------------------------
# Онлайн-детектор шоков по остаткам прогноза
# ---------------------------------------------------------------------------

def detect_shocks_from_forecast_residuals(actual_values, forecast_values,
                                          online_config):
    """
    Ранний детектор шоков на основе ошибок прогноза.

    Идея интеграции прогноза и обнаружения разладок:
        Пока ряд ведёт себя как обычно, прогноз хорошо предсказывает факт,
        и остатки (факт минус прогноз) малы. Когда начинается структурный
        сдвиг или шок, факт резко отклоняется от прогноза — остаток выходит
        за пределы нормального разброса. Это и есть ранний сигнал.

    На каждом шаге стандартизируем остаток по разбросу остатков в скользящем
    калибровочном окне и поднимаем флаг, если он превышает порог alert_sigma.

    Возвращает таблицу с колонками:
        position, residual, standardized_residual, is_alert.
    """
    actual_array = np.asarray(actual_values, dtype="float64")
    forecast_array = np.asarray(forecast_values, dtype="float64")
    residuals = actual_array - forecast_array

    calibration_window = online_config["calibration_window"]
    alert_threshold = online_config["alert_sigma"]

    length_of_series = len(residuals)
    standardized_residuals = np.zeros(length_of_series, dtype="float64")
    alert_flags = np.zeros(length_of_series, dtype="bool")

    for position in range(length_of_series):
        # Берём окно предыдущих остатков для калибровки разброса.
        window_start = position - calibration_window
        if window_start < 0:
            window_start = 0
        window_residuals = residuals[window_start: position]

        if len(window_residuals) >= 3:
            window_mean = np.mean(window_residuals)
            window_std = np.std(window_residuals)
        else:
            window_mean = 0.0
            window_std = 0.0

        if window_std > 0.0:
            standardized_value = (residuals[position] - window_mean) / window_std
        else:
            standardized_value = 0.0

        standardized_residuals[position] = standardized_value
        if abs(standardized_value) > alert_threshold:
            alert_flags[position] = True

    result_table = pd.DataFrame()
    result_table["position"] = np.arange(length_of_series)
    result_table["residual"] = residuals
    result_table["standardized_residual"] = standardized_residuals
    result_table["is_alert"] = alert_flags
    return result_table

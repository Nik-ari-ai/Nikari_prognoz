# ===========================================================================
# Вспомогательные функции: загрузка конфигов, метрики качества, работа с датами.
#
# Стиль кода намеренно подробный: каждая функция делает одну понятную вещь,
# переменные названы словами, циклы расписаны построчно. Это сделано для
# читаемости и проверяемости, а не для компактности.
# ===========================================================================

import os
import json

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Работа с конфигурационными файлами
# ---------------------------------------------------------------------------

def load_yaml_config(path_to_config_file):
    """
    Прочитать YAML-файл конфигурации и вернуть его как обычный словарь Python.

    path_to_config_file : str
        Путь к файлу .yaml.
    """
    with open(path_to_config_file, "r", encoding="utf-8") as opened_file:
        config_as_dictionary = yaml.safe_load(opened_file)
    return config_as_dictionary


def get_repository_root():
    """
    Вернуть абсолютный путь к корню репозитория.

    Файл utils.py лежит в папке src/, поэтому корень — это родительская папка
    относительно папки src/.
    """
    path_to_this_file = os.path.abspath(__file__)
    path_to_src_folder = os.path.dirname(path_to_this_file)
    path_to_repository_root = os.path.dirname(path_to_src_folder)
    return path_to_repository_root


def make_absolute_path(relative_path):
    """
    Превратить путь, записанный относительно корня репозитория, в абсолютный.
    """
    repository_root = get_repository_root()
    absolute_path = os.path.join(repository_root, relative_path)
    return absolute_path


def ensure_directory_exists(path_to_directory):
    """
    Создать папку, если её ещё нет (включая промежуточные папки).
    """
    if not os.path.exists(path_to_directory):
        os.makedirs(path_to_directory)


# ---------------------------------------------------------------------------
# Метрики качества прогноза
#
# MAE — обязательная метрика по ТЗ.
# R^2 — опциональная метрика по ТЗ.
# Дополнительно считаем sMAPE как устойчивую относительную метрику.
# ---------------------------------------------------------------------------

def mean_absolute_error(actual_values, predicted_values):
    """
    Средняя абсолютная ошибка (Mean Absolute Error, MAE).

        MAE = среднее( |факт - прогноз| )

    Чем меньше — тем лучше. Измеряется в тех же единицах, что и ряд (рубли).

    actual_values, predicted_values : одномерные массивы одинаковой длины.
    """
    actual_array = np.asarray(actual_values, dtype="float64")
    predicted_array = np.asarray(predicted_values, dtype="float64")

    absolute_errors = np.abs(actual_array - predicted_array)
    mae_value = np.mean(absolute_errors)
    return float(mae_value)


def r_squared(actual_values, predicted_values):
    """
    Коэффициент детерминации R^2.

        R^2 = 1 - SS_res / SS_tot
        SS_res = сумма( (факт - прогноз)^2 )
        SS_tot = сумма( (факт - среднее_факта)^2 )

    R^2 = 1   — идеальный прогноз.
    R^2 = 0   — прогноз не лучше простого среднего.
    R^2 < 0   — прогноз хуже простого среднего.

    Если все факты одинаковы (SS_tot = 0), R^2 не определён, возвращаем NaN.
    """
    actual_array = np.asarray(actual_values, dtype="float64")
    predicted_array = np.asarray(predicted_values, dtype="float64")

    mean_of_actual = np.mean(actual_array)

    sum_of_squared_residuals = np.sum((actual_array - predicted_array) ** 2)
    sum_of_squared_total = np.sum((actual_array - mean_of_actual) ** 2)

    if sum_of_squared_total == 0.0:
        return float("nan")

    r2_value = 1.0 - (sum_of_squared_residuals / sum_of_squared_total)
    return float(r2_value)


def symmetric_mean_absolute_percentage_error(actual_values, predicted_values):
    """
    Симметричная средняя абсолютная процентная ошибка (sMAPE), в процентах.

        sMAPE = среднее( |факт - прогноз| / ((|факт| + |прогноз|) / 2) ) * 100

    Устойчива к нулевым значениям лучше, чем обычный MAPE. Пары, где и факт,
    и прогноз равны нулю, пропускаются (там ошибка не определена).
    """
    actual_array = np.asarray(actual_values, dtype="float64")
    predicted_array = np.asarray(predicted_values, dtype="float64")

    numerator = np.abs(actual_array - predicted_array)
    denominator = (np.abs(actual_array) + np.abs(predicted_array)) / 2.0

    # Маска пар, где знаменатель не равен нулю.
    valid_mask = denominator > 0.0
    if np.sum(valid_mask) == 0:
        return float("nan")

    percentage_errors = numerator[valid_mask] / denominator[valid_mask]
    smape_value = np.mean(percentage_errors) * 100.0
    return float(smape_value)


def compute_all_metrics(actual_values, predicted_values):
    """
    Посчитать сразу все метрики и вернуть их словарём.
    Удобно для таблиц сравнения моделей.
    """
    metrics_dictionary = {}
    metrics_dictionary["MAE"] = mean_absolute_error(actual_values, predicted_values)
    metrics_dictionary["R2"] = r_squared(actual_values, predicted_values)
    metrics_dictionary["sMAPE"] = symmetric_mean_absolute_percentage_error(
        actual_values, predicted_values
    )
    metrics_dictionary["n_points"] = int(len(actual_values))
    return metrics_dictionary


# ---------------------------------------------------------------------------
# Работа с месячными датами
# ---------------------------------------------------------------------------

def parse_month_string_to_period(month_string):
    """
    Превратить строку вида "2023-01" в месячный период pandas (Period[M]).
    С периодами удобно арифметически сдвигаться на N месяцев.
    """
    period_value = pd.Period(month_string, freq="M")
    return period_value


def shift_month_period(month_period, number_of_months):
    """
    Сдвинуть месячный период на заданное число месяцев вперёд (или назад,
    если число отрицательное).
    """
    shifted_period = month_period + number_of_months
    return shifted_period


def period_to_month_string(month_period):
    """
    Превратить месячный период обратно в строку "YYYY-MM".
    """
    month_string = str(month_period)
    return month_string


# ---------------------------------------------------------------------------
# Сохранение результатов
# ---------------------------------------------------------------------------

def save_dataframe_as_csv(dataframe, path_to_output_file):
    """
    Сохранить таблицу в CSV в кодировке UTF-8 с BOM, чтобы Excel на русском
    корректно открывал кириллицу.
    """
    directory_of_output = os.path.dirname(path_to_output_file)
    if directory_of_output != "":
        ensure_directory_exists(directory_of_output)
    dataframe.to_csv(path_to_output_file, index=False, encoding="utf-8-sig")


def save_dictionary_as_json(dictionary_to_save, path_to_output_file):
    """
    Сохранить словарь в JSON-файл (с отступами и без экранирования кириллицы).
    """
    directory_of_output = os.path.dirname(path_to_output_file)
    if directory_of_output != "":
        ensure_directory_exists(directory_of_output)
    with open(path_to_output_file, "w", encoding="utf-8") as opened_file:
        json.dump(dictionary_to_save, opened_file, ensure_ascii=False, indent=2)

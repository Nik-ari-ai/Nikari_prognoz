# ===========================================================================
# Календарные признаки на основе официального производственного календаря РФ.
#
# Зачем это нужно:
#   Потребление сильно зависит от числа выходных и праздничных дней в месяце
#   (декабрь и январь — характерные примеры). Эти признаки детерминированы и
#   известны заранее на любой будущий месяц, поэтому их можно использовать в
#   прогнозе без заглядывания в будущее.
#
# Источник чисел: официальный производственный календарь РФ (количество
#   рабочих дней при пятидневной рабочей неделе). Числа фиксированы и
#   публикуются государством, поэтому заданы здесь константами.
# ===========================================================================

import calendar

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Число рабочих дней в каждом месяце по производственному календарю РФ.
# Ключ — (год, номер месяца), значение — число рабочих дней.
# Данные приведены для 2023, 2024 и 2025 годов (2025 нужен для прогноза).
# ---------------------------------------------------------------------------
WORKING_DAYS_BY_YEAR_AND_MONTH = {
    # 2023 год: всего 247 рабочих дней.
    (2023, 1): 17, (2023, 2): 18, (2023, 3): 22, (2023, 4): 20,
    (2023, 5): 20, (2023, 6): 21, (2023, 7): 21, (2023, 8): 23,
    (2023, 9): 21, (2023, 10): 22, (2023, 11): 21, (2023, 12): 21,

    # 2024 год (високосный): всего 248 рабочих дней.
    (2024, 1): 17, (2024, 2): 20, (2024, 3): 20, (2024, 4): 21,
    (2024, 5): 20, (2024, 6): 19, (2024, 7): 23, (2024, 8): 22,
    (2024, 9): 21, (2024, 10): 23, (2024, 11): 21, (2024, 12): 21,

    # 2025 год: всего 247 рабочих дней.
    (2025, 1): 17, (2025, 2): 20, (2025, 3): 21, (2025, 4): 22,
    (2025, 5): 18, (2025, 6): 19, (2025, 7): 23, (2025, 8): 21,
    (2025, 9): 22, (2025, 10): 23, (2025, 11): 19, (2025, 12): 22,
}


def count_days_in_month(year_number, month_number):
    """
    Вернуть число календарных дней в заданном месяце.
    """
    first_weekday, number_of_days = calendar.monthrange(year_number, month_number)
    return number_of_days


def get_working_days(year_number, month_number):
    """
    Вернуть число рабочих дней в месяце по производственному календарю.

    Если для года данных нет (например, запросили год вне таблицы), возвращаем
    приблизительную оценку: число будних дней (понедельник-пятница) в месяце.
    Это разумный запасной вариант, который не ломает расчёт.
    """
    calendar_key = (year_number, month_number)
    if calendar_key in WORKING_DAYS_BY_YEAR_AND_MONTH:
        return WORKING_DAYS_BY_YEAR_AND_MONTH[calendar_key]

    # Запасной вариант: честно считаем будние дни месяца.
    number_of_days = count_days_in_month(year_number, month_number)
    weekday_count = 0
    for day_number in range(1, number_of_days + 1):
        weekday_index = calendar.weekday(year_number, month_number, day_number)
        # weekday_index: 0=понедельник ... 6=воскресенье.
        if weekday_index <= 4:
            weekday_count = weekday_count + 1
    return weekday_count


def build_calendar_features_table(list_of_month_periods):
    """
    Построить таблицу календарных признаков для заданного списка месяцев.

    list_of_month_periods : перечень месячных периодов pandas (Period[M]).

    Возвращает таблицу с колонками:
        month               — месячный период,
        month_number        — номер месяца (1..12), для сезонности,
        days_in_month       — число календарных дней,
        working_days        — число рабочих дней,
        non_working_days    — число нерабочих дней (выходные + праздники),
        share_working_days  — доля рабочих дней в месяце.
    """
    collected_rows = []

    for month_period in list_of_month_periods:
        year_number = month_period.year
        month_number = month_period.month

        number_of_days = count_days_in_month(year_number, month_number)
        number_of_working_days = get_working_days(year_number, month_number)
        number_of_non_working_days = number_of_days - number_of_working_days
        share_of_working_days = number_of_working_days / number_of_days

        one_row = {
            "month": month_period,
            "month_number": month_number,
            "days_in_month": number_of_days,
            "working_days": number_of_working_days,
            "non_working_days": number_of_non_working_days,
            "share_working_days": share_of_working_days,
        }
        collected_rows.append(one_row)

    calendar_features_table = pd.DataFrame(collected_rows)
    return calendar_features_table

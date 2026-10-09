"""Слепая (правило-ориентированная) разметка реальных шоков потребления.

Зачем:
    Чтобы честно оценивать детекторы и обучать модель риска, нужны метки
    шоков, полученные НЕ из самих детекторов. Иначе оценка была бы подгонкой
    под собственный ответ.

Принцип разметки (зафиксирован заранее, не смотрит на детекторы):
    1. Для ряда МО по категории «Все категории» берём годовую лог-разность
       d_t = log(y_t) - log(y_{t-12}). Она убирает годовую сезонность и
       показывает темп роста к тому же месяцу прошлого года.
    2. У каждого МО вычитаем его медиану d (убираем постоянное различие
       уровней роста между МО): e_t = d_t - median_MO(d).
    3. Устойчивый масштаб шума оцениваем по всей панели робастно:
       sigma = 1.4826 * MAD(e) по всем (МО, месяц).
    4. Месяц t объявляется шоком, если |e_t| > THRESH * sigma И то же условие
       выполняется в соседнем месяце (длительность >= MIN_DURATION). Это
       отсекает одиночные выбросы и ловит именно устойчивые сдвиги.

Это ПРОКСИ-определение шока, а не абсолютная истина. Оно полностью
детерминировано и не зависит от детекторов, поэтому пригодно как эталон.
"""

import numpy as np
import pandas as pd


def compute_year_over_year_log_change(wide_matrix_total):
    """Годовая лог-разность для каждого МО и месяца.

    wide_matrix_total: широкая матрица значений «Все категории»
        (строки — territory_id, столбцы — месяцы Period[M]).
    Возвращает таблицу (territory_id, month, yoy_log_change) только там, где
    есть значение 12 месяцев назад.
    """
    month_columns = list(wide_matrix_total.columns)
    collected_rows = []
    for position in range(len(month_columns)):
        if position < 12:
            continue
        current_month = month_columns[position]
        year_ago_month = month_columns[position - 12]
        current_values = wide_matrix_total[current_month]
        year_ago_values = wide_matrix_total[year_ago_month]
        for territory_id in wide_matrix_total.index:
            current_value = current_values.loc[territory_id]
            year_ago_value = year_ago_values.loc[territory_id]
            if pd.isna(current_value) or pd.isna(year_ago_value):
                continue
            if current_value <= 0 or year_ago_value <= 0:
                continue
            yoy_log_change = float(np.log(current_value) - np.log(year_ago_value))
            collected_rows.append({
                "territory_id": int(territory_id),
                "month": str(current_month),
                "yoy_log_change": yoy_log_change,
            })
    return pd.DataFrame(collected_rows)


def label_shocks(yoy_table, threshold_sigma=3.0, min_duration=2):
    """Разметить шоки по заранее заданному правилу.

    yoy_table: таблица из compute_year_over_year_log_change.
    threshold_sigma: во сколько робастных сигм должно быть отклонение.
    min_duration: минимальная длительность превышения порога (в месяцах).

    Возвращает ту же таблицу с добавленными колонками:
        centered (e_t), is_extreme (превышение порога в этот месяц),
        is_shock (устойчивое превышение нужной длительности).
    """
    table = yoy_table.copy()

    # (2) Центрируем по медиане каждого МО.
    median_by_territory = table.groupby("territory_id")["yoy_log_change"].transform("median")
    table["centered"] = table["yoy_log_change"] - median_by_territory

    # (3) Робастный масштаб по всей панели (MAD -> сигма).
    centered_values = table["centered"].to_numpy(dtype=float)
    median_abs_deviation = float(np.median(np.abs(centered_values - np.median(centered_values))))
    robust_sigma = 1.4826 * median_abs_deviation
    if robust_sigma <= 0:
        robust_sigma = float(np.std(centered_values)) or 1.0
    table["robust_sigma"] = robust_sigma

    # Превышение порога в данный месяц.
    table["is_extreme"] = np.abs(table["centered"]) > (threshold_sigma * robust_sigma)

    # (4) Устойчивость: превышение держится не меньше min_duration месяцев
    #     подряд (у того же МО). Отмечаем шоком каждый месяц такой серии.
    table = table.sort_values(["territory_id", "month"]).reset_index(drop=True)
    table["is_shock"] = False

    for territory_id, group in table.groupby("territory_id"):
        extreme_flags = group["is_extreme"].tolist()
        row_indices = group.index.tolist()
        run_length = 0
        run_start_position = 0
        for position in range(len(extreme_flags)):
            if extreme_flags[position]:
                if run_length == 0:
                    run_start_position = position
                run_length = run_length + 1
            else:
                if run_length >= min_duration:
                    for mark_position in range(run_start_position, position):
                        table.loc[row_indices[mark_position], "is_shock"] = True
                run_length = 0
        # Закрываем серию в конце ряда.
        if run_length >= min_duration:
            for mark_position in range(run_start_position, len(extreme_flags)):
                table.loc[row_indices[mark_position], "is_shock"] = True

    return table, robust_sigma


def extract_episode_onsets(labeled_table):
    """Найти НАЧАЛА эпизодов шока (первый месяц непрерывной серии is_shock).

    Возвращает множество пар (territory_id, month_str) — это onset-события.
    Нужно, чтобы отличать НАЧАЛО нового шока от продолжения уже идущего.
    """
    onsets = set()
    sorted_table = labeled_table.sort_values(["territory_id", "month"]).reset_index(drop=True)
    for territory_id, group in sorted_table.groupby("territory_id"):
        previous_was_shock = False
        for _, row in group.iterrows():
            current_is_shock = bool(row["is_shock"])
            if current_is_shock and not previous_was_shock:
                onsets.add((int(territory_id), str(row["month"])))
            previous_was_shock = current_is_shock
    return onsets


def label_shocks_with_fixed_stats(yoy_table, median_by_territory_map,
                                  robust_sigma, threshold_sigma=3.0, min_duration=2):
    """Разметить шоки с ЗАРАНЕЕ ЗАДАННЫМИ статистиками (без заглядывания).

    В отличие от label_shocks, медиана каждого МО и робастная сигма здесь не
    оцениваются по всем данным (включая тест), а передаются снаружи — например,
    посчитанные только по обучающему периоду. Это убирает утечку из теста в
    определение меток.

    median_by_territory_map: dict territory_id -> медиана yoy по обучению.
    robust_sigma: робастная сигма, посчитанная по обучению.
    """
    table = yoy_table.copy()

    centered_values = []
    for _, row in table.iterrows():
        territory_median = median_by_territory_map.get(int(row["territory_id"]), 0.0)
        centered_values.append(row["yoy_log_change"] - territory_median)
    table["centered"] = centered_values

    if robust_sigma <= 0:
        robust_sigma = 1.0
    table["is_extreme"] = np.abs(table["centered"]) > (threshold_sigma * robust_sigma)

    table = table.sort_values(["territory_id", "month"]).reset_index(drop=True)
    table["is_shock"] = False
    for territory_id, group in table.groupby("territory_id"):
        extreme_flags = group["is_extreme"].tolist()
        row_indices = group.index.tolist()
        run_length = 0
        run_start_position = 0
        for position in range(len(extreme_flags)):
            if extreme_flags[position]:
                if run_length == 0:
                    run_start_position = position
                run_length = run_length + 1
            else:
                if run_length >= min_duration:
                    for mark_position in range(run_start_position, position):
                        table.loc[row_indices[mark_position], "is_shock"] = True
                run_length = 0
        if run_length >= min_duration:
            for mark_position in range(run_start_position, len(extreme_flags)):
                table.loc[row_indices[mark_position], "is_shock"] = True

    return table

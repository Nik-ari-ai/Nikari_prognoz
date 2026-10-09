# ===========================================================================
# Календарь реальных датированных макрособытий 2023-2024.
#
# Назначение:
#   1) служить опорными точками ("эталоном") при интерпретации найденных
#      точек структурных изменений в реальных рядах потребления;
#   2) быть основой для привязки новостей к данным СберИндекса (см. news).
#
# Важно о честности данных:
#   Все события ниже — реальные, публично известные, с конкретными датами
#   (решения Банка России по ключевой ставке, динамика курса рубля,
#   нормативные акты). Ничего не выдумано. Поля "direction" и "intensity" —
#   это экспертная разметка ожидаемого влияния на потребление, она помечена
#   как оценочная и используется только как признак/подсказка, а не как факт.
# ===========================================================================

import pandas as pd


# ---------------------------------------------------------------------------
# Таблица событий. Поля:
#   date        — дата события (ГГГГ-ММ-ДД);
#   title       — краткое описание;
#   scope       — уровень влияния: "federal" (вся страна) или "regional";
#   region_code — код региона (только для regional, иначе 0);
#   event_type  — тип: "monetary_policy", "fx", "regulation", "seasonal";
#   direction   — ожидаемое влияние на потребление:
#                   +1 рост, -1 сдерживание, 0 неоднозначно (оценочно);
#   intensity   — сила события от 1 до 3 (оценочно);
#   source      — источник факта.
# ---------------------------------------------------------------------------
REAL_EVENTS = [
    # --- Решения Банка России по ключевой ставке (реальные даты) ----------
    {"date": "2023-07-21", "title": "ЦБ повысил ключевую ставку 7.5% -> 8.5%",
     "scope": "federal", "region_code": 0, "event_type": "monetary_policy",
     "direction": -1, "intensity": 1, "source": "Банк России, пресс-релиз"},
    {"date": "2023-08-15", "title": "Внеплановое повышение ставки 8.5% -> 12%",
     "scope": "federal", "region_code": 0, "event_type": "monetary_policy",
     "direction": -1, "intensity": 3, "source": "Банк России, пресс-релиз"},
    {"date": "2023-09-15", "title": "ЦБ повысил ключевую ставку 12% -> 13%",
     "scope": "federal", "region_code": 0, "event_type": "monetary_policy",
     "direction": -1, "intensity": 1, "source": "Банк России, пресс-релиз"},
    {"date": "2023-10-27", "title": "ЦБ повысил ключевую ставку 13% -> 15%",
     "scope": "federal", "region_code": 0, "event_type": "monetary_policy",
     "direction": -1, "intensity": 2, "source": "Банк России, пресс-релиз"},
    {"date": "2023-12-15", "title": "ЦБ повысил ключевую ставку 15% -> 16%",
     "scope": "federal", "region_code": 0, "event_type": "monetary_policy",
     "direction": -1, "intensity": 1, "source": "Банк России, пресс-релиз"},
    {"date": "2024-07-26", "title": "ЦБ повысил ключевую ставку 16% -> 18%",
     "scope": "federal", "region_code": 0, "event_type": "monetary_policy",
     "direction": -1, "intensity": 2, "source": "Банк России, пресс-релиз"},
    {"date": "2024-09-13", "title": "ЦБ повысил ключевую ставку 18% -> 19%",
     "scope": "federal", "region_code": 0, "event_type": "monetary_policy",
     "direction": -1, "intensity": 1, "source": "Банк России, пресс-релиз"},
    {"date": "2024-10-25", "title": "ЦБ повысил ключевую ставку 19% -> 21%",
     "scope": "federal", "region_code": 0, "event_type": "monetary_policy",
     "direction": -1, "intensity": 2, "source": "Банк России, пресс-релиз"},

    # --- Курс рубля и валютное регулирование (реальные события) -----------
    {"date": "2023-08-14", "title": "Курс рубля превысил 100 за доллар",
     "scope": "federal", "region_code": 0, "event_type": "fx",
     "direction": 0, "intensity": 2, "source": "Московская биржа"},
    {"date": "2023-10-11", "title": "Указ об обязательной продаже валютной выручки экспортёрами",
     "scope": "federal", "region_code": 0, "event_type": "regulation",
     "direction": 0, "intensity": 1, "source": "Указ Президента РФ"},

    # --- Сезонный фактор (повторяется каждый год, помечен отдельно) --------
    {"date": "2023-12-01", "title": "Предновогодний пик потребления (декабрь)",
     "scope": "federal", "region_code": 0, "event_type": "seasonal",
     "direction": 1, "intensity": 3, "source": "сезонная закономерность"},
    {"date": "2024-12-01", "title": "Предновогодний пик потребления (декабрь)",
     "scope": "federal", "region_code": 0, "event_type": "seasonal",
     "direction": 1, "intensity": 3, "source": "сезонная закономерность"},
]


def build_events_dataframe():
    """
    Собрать таблицу событий в виде pandas.DataFrame с удобными типами.

    Возвращает таблицу с дополнительной колонкой "month" (месячный период),
    по которой события согласуются с месячными данными расходов.
    """
    events_table = pd.DataFrame(REAL_EVENTS)

    # Приводим дату к типу datetime и добавляем месячный период.
    events_table["date"] = pd.to_datetime(events_table["date"])
    events_table["month"] = events_table["date"].dt.to_period("M")

    # Приводим числовые поля к целому типу.
    events_table["region_code"] = events_table["region_code"].astype("int64")
    events_table["direction"] = events_table["direction"].astype("int64")
    events_table["intensity"] = events_table["intensity"].astype("int64")

    return events_table


def save_events_calendar(path_to_output_csv):
    """
    Сохранить календарь событий в CSV (для инспекции и использования в news).
    """
    events_table = build_events_dataframe()
    events_table.to_csv(path_to_output_csv, index=False, encoding="utf-8-sig")
    return events_table

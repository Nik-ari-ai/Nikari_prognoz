"""Присоединение опубликованных сообщений к моменту прогноза.

Дата события и дата его публикации могут различаться. Фильтруем именно
published_at. Фактический момент получения сообщения хранится отдельно
в retrieved_at и нужен для аудита сегодняшней выгрузки.
"""

import pandas as pd


def available_events(events, cutoff, region_code=None):
    """Выбрать известные к cutoff сообщения, убрать повторы по URL."""
    table = events.copy()
    table["published_at"] = pd.to_datetime(table["published_at"], utc=True)
    cutoff_time = pd.Timestamp(cutoff)
    if cutoff_time.tzinfo is None:
        raise ValueError("У момента прогноза должен быть указан часовой пояс.")
    table = table[table["published_at"] <= cutoff_time]
    if region_code is None:
        table = table[table["scope"] == "federal"]
    else:
        relevant = table["scope"].eq("federal")
        relevant = relevant | table["region_code"].eq(region_code)
        table = table[relevant]
    table = table.sort_values("published_at")
    return table.drop_duplicates("source_url", keep="first")


def monthly_rate_changes(events, months, regions):
    """Получить сумму опубликованных изменений ставки за месяц по регионам.

Сигнал измеряется в процентных пунктах. Это наблюдавшееся изменение ставки,
а не выдуманная сила влияния новости на расходы. Будущие решения неизвестны.
"""
    rows = []
    for month in months:
        start = month.to_timestamp(how="start").tz_localize("Europe/Moscow")
        end = month.to_timestamp(how="end").tz_localize("Europe/Moscow")
        for region in regions:
            known = available_events(events, end, region)
            known = known[known["published_at"] >= start]
            rows.append({
                "month": month,
                "region_code": int(region),
                "news_signal": float(known["rate_change_pp"].sum()),
                "event_count": int(len(known)),
            })
    return pd.DataFrame(rows)

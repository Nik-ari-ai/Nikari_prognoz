"""Новости как эксперимент: два разобранных кейса МО, где сигнал ЦБ изменил
прогноз, плюс сводка абляции без-новостей / +ЦБ / +ЦБ+ИПЦ.

По приоритету обратной связи. Таблица событий ЦБ уже в полном виде
(`data/external/cbr_rate_events.csv`: published_at, event_date, scope,
region_code, source_url, ...). Абляция уже посчитана
(`outputs/headline/cpi_ablation.csv`). Здесь добавляются конкретные примеры:
два МО, где добавление сигнала ЦБ заметно сдвинуло прогноз к факту (или от него).

Запуск: python scripts/23_news_cases.py  # -> outputs/news/
"""

import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def latest_available_cbr_event(cbr_events, cutoff_period):
    """Последнее событие ЦБ, ДОСТУПНОЕ на момент прогноза (каузально).

    Доступность определяется по ТОЧНОМУ ВРЕМЕНИ ПУБЛИКАЦИИ (published_at): прогноз
    на отсечке видит только то, что опубликовано не позже конца месяца отсечки.
    Среди доступных берётся публикация с САМЫМ ПОЗДНИМ временем published_at
    (поправка аудита v15: при двух публикациях одного месяца выбиралась первая, а
    не последняя — теперь сравнение по полному таймстемпу, а не по месяцу).
    Возвращает строку события или None.
    """
    cutoff_month_end = (cutoff_period + 1).to_timestamp() - pd.Timedelta(seconds=1)
    cutoff_month_end = cutoff_month_end.tz_localize("Europe/Moscow")

    best_event = None
    best_publication_time = None
    for _, event in cbr_events.iterrows():
        publication_time = pd.Timestamp(str(event["published_at"]))
        if publication_time.tzinfo is None:
            publication_time = publication_time.tz_localize("Europe/Moscow")
        if publication_time <= cutoff_month_end:
            if best_publication_time is None or publication_time > best_publication_time:
                best_event = event
                best_publication_time = publication_time
    return best_event


def main(output_dir=None):
    if output_dir is None:
        destination = ROOT / "outputs" / "news"
    else:
        destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(ROOT / "outputs" / "checked" / "predictions.csv")

    without_news = predictions[predictions["model"] == "LightGBM_checked"].copy()
    with_news = predictions[predictions["model"] == "LightGBM_CBR_events"].copy()

    key_columns = ["territory_id", "category", "cutoff_month", "horizon"]
    merged = pd.merge(
        without_news[key_columns + ["actual", "predicted"]],
        with_news[key_columns + ["predicted"]],
        on=key_columns, suffixes=("_no_news", "_with_news")
    )

    merged["error_no_news"] = np.abs(merged["actual"] - merged["predicted_no_news"])
    merged["error_with_news"] = np.abs(merged["actual"] - merged["predicted_with_news"])
    # Насколько добавление новостей ЦБ приблизило прогноз к факту (положительно — помогло).
    merged["improvement"] = merged["error_no_news"] - merged["error_with_news"]
    # Насколько новости вообще сдвинули прогноз.
    merged["shift"] = np.abs(merged["predicted_with_news"] - merged["predicted_no_news"])

    # Берём два кейса: наибольшее улучшение среди заметно сдвинутых прогнозов.
    candidates = merged[merged["shift"] > merged["shift"].median()].copy()
    candidates = candidates.sort_values(by="improvement", ascending=False)

    cbr_events = pd.read_csv(ROOT / "data" / "external" / "cbr_rate_events.csv")

    case_rows = []
    chosen = candidates.head(2)
    for _, row in chosen.iterrows():
        cutoff_period = pd.Period(str(row["cutoff_month"]), freq="M")
        # Последнее событие ЦБ, ДОСТУПНОЕ по дате публикации (каузально).
        active_event = latest_available_cbr_event(cbr_events, cutoff_period)
        case_rows.append({
            "territory_id": int(row["territory_id"]),
            "category": str(row["category"]),
            "cutoff_month": str(row["cutoff_month"]),
            "horizon": int(row["horizon"]),
            "actual": round(float(row["actual"]), 1),
            "predicted_no_news": round(float(row["predicted_no_news"]), 1),
            "predicted_with_cbr": round(float(row["predicted_with_news"]), 1),
            "error_no_news": round(float(row["error_no_news"]), 1),
            "error_with_cbr": round(float(row["error_with_news"]), 1),
            "improvement_rub": round(float(row["improvement"]), 1),
            "latest_cbr_event_date": (str(active_event["event_date"]) if active_event is not None else None),
            "latest_cbr_new_rate_pct": (float(active_event["new_rate_pct"]) if active_event is not None else None),
            "latest_cbr_source_url": (str(active_event["source_url"]) if active_event is not None else None),
        })
    cases_table = pd.DataFrame(case_rows)
    cases_table.to_csv(destination / "news_cases.csv", index=False)

    # Сводка абляции (копия для удобства + эффект ЦБ).
    ablation = pd.read_csv(ROOT / "outputs" / "headline" / "cpi_ablation.csv")
    ablation.to_csv(destination / "news_ablation.csv", index=False)

    info = {
        "events_table": "data/external/cbr_rate_events.csv (схема: published_at, event_date, scope, region_code, source_url, ...)",
        "events_count": int(len(cbr_events)),
        "ablation": "без-новостей / +ЦБ / +ЦБ+ИПЦ (outputs/headline/cpi_ablation.csv)",
        "ablation_note": ("Чистый эффект ИПЦ неустойчив (+4.2% на h=1, -16.3% на h=3); "
                          "сигнал ЦБ помогает скромно и неравномерно по горизонтам."),
        "cases_note": ("Два МО, где добавление сигнала ЦБ сильнее всего приблизило прогноз "
                       "к факту (news_cases.csv). Сигнал ЦБ доступен только на момент прогноза "
                       "(published_at), будущих сообщений модель не видит."),
        "regional_news_status": ("Полный региональный новостной архив с историей дат публикации "
                                 "офлайн недоступен (сеть окружения ограничена аллоулистом); "
                                 "основной поток — официальные события ЦБ + ИПЦ Росстата."),
    }
    with open(destination / "news_summary.json", "w", encoding="utf-8") as target_file:
        json.dump(info, target_file, ensure_ascii=False, indent=2)

    print("=== НОВОСТИ: два кейса МО (сигнал ЦБ сдвинул прогноз к факту) ===", flush=True)
    print(cases_table.to_string(index=False), flush=True)
    print("\nАбляция (без-новостей / +ЦБ / +ЦБ+ИПЦ):", flush=True)
    print(ablation.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

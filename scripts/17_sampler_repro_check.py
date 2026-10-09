# ===========================================================================
# Скрипт: проверка воспроизводимости состава выборки сэмплером.
#
# Повторяет ТОЧНЫЙ вызов стратифицированного сэмплера из scripts/08 (данные до
# первой отсечки, seed и sample_size из configs/checked.yaml) и сверяет 120
# кандидатов с авторитетными 111 оценочными МО (evaluated_territories.csv).
#
# Зачем: членство в наборе кандидатов ЧУВСТВИТЕЛЬНО к версиям numpy/pandas
# (разный порядок/поток ГСЧ). Под закреплёнными версиями (numpy 2.4.1,
# pandas 2.3.3) воспроизводятся все 111; под более новыми один МО смещается.
# Этот скрипт печатает версии окружения и фактический результат, поэтому
# оговорку о version-sensitivity можно предъявить логом, а не на словах.
#
# Запуск из корня проекта:
#     python scripts/17_sampler_repro_check.py
# ===========================================================================

import sys
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy
import pandas as pd

from src import utils
from src import data_loading
from src import backtest


def main(output_path=None):
    data_config = utils.load_yaml_config(ROOT / "configs" / "data.yaml")
    checked_config = utils.load_yaml_config(ROOT / "configs" / "checked.yaml")

    consumption = data_loading.load_consumption_dataframe(data_config)
    territory_dictionary = data_loading.load_territory_dictionary(data_config)
    first_origin = pd.Period(checked_config["origins"][0], freq="M")
    known = consumption[consumption["month"] <= first_origin]

    sample_ids = backtest.select_stratified_territory_sample(
        known, checked_config["sample_size"], territory_dictionary,
        checked_config["random_seed"]
    )
    candidate_ids = []
    for one_identifier in sample_ids:
        candidate_ids.append(int(one_identifier))
    candidate_ids = sorted(candidate_ids)

    evaluated = pd.read_csv(ROOT / "outputs" / "checked" / "evaluated_territories.csv")
    evaluated_column = None
    for column_name in evaluated.columns:
        if "terr" in column_name.lower():
            evaluated_column = column_name
            break
    evaluated_ids = []
    for one_identifier in evaluated[evaluated_column]:
        evaluated_ids.append(int(one_identifier))
    evaluated_ids = sorted(evaluated_ids)

    candidate_set = set(candidate_ids)
    evaluated_set = set(evaluated_ids)
    missing_from_candidates = sorted(evaluated_set - candidate_set)
    extra_candidates = sorted(candidate_set - evaluated_set)

    report = {
        "environment": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pd.__version__,
        },
        "seed": checked_config["random_seed"],
        "sample_size": checked_config["sample_size"],
        "first_origin": str(first_origin),
        "candidate_count": len(candidate_ids),
        "evaluated_count": len(evaluated_ids),
        "all_evaluated_present": len(missing_from_candidates) == 0,
        "evaluated_missing_from_candidates": missing_from_candidates,
        "extra_candidates": extra_candidates,
        "candidate_ids": candidate_ids,
    }

    if output_path is None:
        output_path = ROOT / "outputs" / "checked" / "sampler_repro_check.json"
    with open(output_path, "w", encoding="utf-8") as target:
        json.dump(report, target, ensure_ascii=False, indent=2)

    print("Окружение: python " + report["environment"]["python"]
          + ", numpy " + report["environment"]["numpy"]
          + ", pandas " + report["environment"]["pandas"], flush=True)
    print("Кандидатов: " + str(report["candidate_count"])
          + "; все 111 оценочных присутствуют: "
          + str(report["all_evaluated_present"]), flush=True)
    if len(missing_from_candidates) > 0:
        print("Отсутствуют среди кандидатов: " + str(missing_from_candidates), flush=True)
    print("Дополнительные кандидаты: " + str(extra_candidates), flush=True)
    print("Отчёт сохранён: " + str(output_path), flush=True)


if __name__ == "__main__":
    main()

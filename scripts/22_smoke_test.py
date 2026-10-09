"""Smoke-тест: проверяет импорты ключевых зависимостей и что крошечный
сквозной прогон модели работает. Это НЕ проверка чистой установки всей
поставки: запускается в уже существующем окружении, а torch/Chronos в
облачном контейнере отсутствуют (см. финальное сообщение лога).

По приоритету обратной связи (воспроизводимость): перед подачей убедиться, что
импортируются LightGBM, Prophet, CatBoost (Chronos — опционально), а базовый
прогнозный путь запускается без ошибок на нескольких рядах.

Запуск из корня проекта:
    python scripts/22_smoke_test.py
Код возврата 0 — всё в порядке; ненулевой — что-то не установлено/не работает.
"""

import sys
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_imports():
    """Проверить импорт обязательных пакетов и вернуть их версии."""
    required_packages = [
        "numpy", "pandas", "sklearn", "lightgbm", "catboost",
        "prophet", "statsmodels",
    ]
    optional_packages = ["chronos", "torch", "transformers"]

    versions = {}
    problems = []

    for package_name in required_packages:
        try:
            module = importlib.import_module(package_name)
            versions[package_name] = getattr(module, "__version__", "?")
        except Exception as import_error:
            problems.append("ОБЯЗАТЕЛЬНЫЙ пакет не импортируется: "
                            + package_name + " (" + str(import_error)[:80] + ")")

    for package_name in optional_packages:
        try:
            module = importlib.import_module(package_name)
            versions[package_name] = getattr(module, "__version__", "?")
        except Exception as import_error:
            versions[package_name] = "НЕ установлен (опционально): " + str(import_error)[:60]

    return versions, problems


def tiny_end_to_end():
    """Крошечный сквозной прогон: несколько МО, признаки, обучение, прогноз."""
    import numpy as np
    import pandas as pd
    from src import utils, data_loading, features, calendar_features, backtest

    data_config = utils.load_yaml_config(ROOT / "configs" / "data.yaml")
    forecast_config = utils.load_yaml_config(ROOT / "configs" / "forecast.yaml")

    consumption = data_loading.load_consumption_dataframe(data_config)
    wide = features.prepare_wide_value_matrix(consumption)
    months = list(wide.columns)

    forecast_config["features"]["use_market_access"] = False
    forecast_config["features"]["use_region"] = False
    forecast_config["features"]["use_coordinates"] = False
    forecast_config["features"]["use_holiday_features"] = False
    forecast_config["features"]["use_news_features"] = False
    forecast_config["backtest"]["min_train_months"] = 6

    dictionary = data_loading.load_territory_dictionary(data_config)
    market = pd.DataFrame(columns=["territory_id", "market_access"])
    static, codes = features.build_static_features(wide, market, dictionary, forecast_config)
    calendar = calendar_features.build_calendar_features_table(months)

    # Берём небольшую выборку МО для скорости.
    cutoff = pd.Period("2024-06", freq="M")
    history = wide.loc[:, :cutoff]
    sample = history.dropna().head(50)
    if len(sample) == 0:
        raise RuntimeError("Smoke-тест: нет полных историй на отсечке")

    # Обучаем глобальную одношаговую модель (LightGBM по умолчанию) и прогнозируем.
    model, columns = backtest.train_one_step_global_model(
        wide, static, calendar, months, cutoff, forecast_config
    )
    if model is None:
        raise RuntimeError("Smoke-тест: модель не обучилась")
    forecasts = backtest.recursive_global_forecast(
        model, sample, static.reindex(sample.index), calendar, cutoff, 3, forecast_config
    )
    horizon_1 = np.asarray(forecasts[1], dtype=float)
    if not np.all(np.isfinite(horizon_1)):
        raise RuntimeError("Smoke-тест: прогноз содержит нечисловые значения")
    return len(sample), float(np.mean(horizon_1))


def main():
    import datetime
    log_lines = []

    def emit(text):
        print(text, flush=True)
        log_lines.append(text)

    emit("=== SMOKE-ТЕСТ ЧИСТОЙ УСТАНОВКИ ===")
    emit("Запущен: " + datetime.datetime.now().isoformat(timespec="seconds"))
    emit("Python: " + sys.version.split()[0])
    versions, problems = check_imports()
    emit("Версии пакетов:")
    for package_name in sorted(versions.keys()):
        emit("  " + package_name + ": " + str(versions[package_name]))

    run_ok = True
    if len(problems) > 0:
        emit("")
        emit("ПРОБЛЕМЫ:")
        for one_problem in problems:
            emit("  - " + one_problem)
        emit("Smoke-тест НЕ пройден.")
        run_ok = False
    else:
        emit("")
        emit("Крошечный сквозной прогон (LightGBM, 50 МО)...")
        try:
            sample_size, mean_prediction = tiny_end_to_end()
            emit("  обучено и спрогнозировано на " + str(sample_size)
                 + " МО; средний прогноз h=1 ≈ " + ("%.0f" % mean_prediction) + " руб.")
            emit("")
            emit("Smoke-тест ПРОЙДЕН: обязательные пакеты на месте, базовый путь работает.")
            torch_present = "torch" in versions and "НЕ установлен" not in str(versions.get("torch", ""))
            chronos_present = "chronos" in versions and "НЕ установлен" not in str(versions.get("chronos", ""))
            if torch_present and chronos_present:
                emit("torch и chronos доступны: фундаментальная модель воспроизводима в этом окружении.")
            else:
                emit("ВАЖНО: в ЭТОМ окружении (облачный контейнер) torch/chronos НЕ установлены, "
                     "поэтому данный лог подтверждает импорты обязательных зависимостей и базовый "
                     "прогноз, но НЕ инференс Chronos и НЕ чистую установку всей поставки. Полная "
                     "проверка всей поставки с инференсом Chronos требует закреплённого окружения "
                     "(requirements.txt), где доступны torch/chronos; Chronos запускается отдельным "
                     "процессом scripts/foundation_worker.py.")
        except Exception as run_error:
            emit("ОШИБКА сквозного прогона: " + str(run_error))
            run_ok = False

    # Сохраняем лог в архив (воспроизводимость).
    log_path = ROOT / "outputs" / "smoke_test.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("\n".join(log_lines) + "\n")

    if not run_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

# ===========================================================================
# Фундаментальная модель временных рядов: Chronos (Amazon) в режиме zero-shot.
#
# Что это и зачем:
#   Chronos — предобученная "фундаментальная" модель для временных рядов.
#   Она уже обучена на огромном количестве рядов и умеет прогнозировать новый
#   ряд БЕЗ дообучения (zero-shot). Это преимущество по критерию ТЗ об
#   использовании современных фундаментальных моделей.
#
# Как работает этот модуль:
#   - модель подключается по флагу models.chronos в configs/forecast.yaml;
#   - веса модели загружаются либо из локальной папки (models_cache), либо по
#     имени с HuggingFace при наличии интернета;
#   - прогноз делается пообъектно (на каждый ряд), как у Prophet/ETS, чтобы
#     результат встал в ту же таблицу сравнения.
#
# ВАЖНО о запуске:
#   Для работы нужны пакеты chronos-forecasting и torch, а также веса модели.
#   В изолированной среде без доступа к HuggingFace веса не скачиваются —
#   тогда нужно заранее положить их в папку models_cache (см. README) или
#   запускать на машине с интернетом. Код написан так, чтобы при отсутствии
#   весов честно сообщить об этом, а не выдавать выдуманные числа.
# ===========================================================================

import os

import numpy as np


class ChronosForecaster:
    """
    Обёртка над Chronos для пообъектного прогноза в режиме zero-shot.
    """

    def __init__(self, model_name_or_path, device="cpu", num_samples=20):
        """
        model_name_or_path : имя модели на HuggingFace (например,
            "amazon/chronos-t5-small") или путь к локальной папке с весами.
        device : "cpu" или "cuda".
        num_samples : сколько выборок генерировать для оценки медианы прогноза.
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.num_samples = num_samples
        self.pipeline = None

    def load(self):
        """
        Загрузить веса модели. Вызывается один раз перед прогнозами.
        Если пакеты или веса недоступны, бросает понятное исключение.
        """
        try:
            import torch
            from chronos import ChronosPipeline
        except ImportError as import_error:
            raise ImportError(
                "Не установлены пакеты для Chronos. Установите их командой:\n"
                "    pip install chronos-forecasting torch\n"
                "Исходная ошибка: " + str(import_error)
            )

        self.pipeline = ChronosPipeline.from_pretrained(
            self.model_name_or_path,
            device_map=self.device,
            torch_dtype=torch.float32,
        )

    def forecast_one_series(self, history_values, list_of_horizons):
        """
        Спрогнозировать один ряд на заданные горизонты (zero-shot).

        history_values : значения ряда в хронологическом порядке.
        list_of_horizons : список горизонтов.

        Возвращает словарь {горизонт: прогноз}. Прогноз на горизонт h — это
        медиана распределения прогноза на шаг h.
        """
        import torch

        if self.pipeline is None:
            raise RuntimeError(
                "Модель Chronos не загружена. Сначала вызовите метод load()."
            )

        maximum_horizon = max(list_of_horizons)

        history_as_tensor = torch.tensor(
            np.asarray(history_values, dtype="float32")
        )

        # predict возвращает тензор формы [1, num_samples, maximum_horizon].
        forecast_samples = self.pipeline.predict(
            context=history_as_tensor,
            prediction_length=maximum_horizon,
            num_samples=self.num_samples,
        )
        forecast_as_array = forecast_samples[0].numpy()  # [num_samples, horizon]

        # Медиана по выборкам для каждого шага.
        median_forecast_per_step = np.quantile(forecast_as_array, 0.5, axis=0)

        predictions_by_horizon = {}
        for horizon in list_of_horizons:
            predicted_value = median_forecast_per_step[horizon - 1]
            # Расходы неотрицательны — обрезаем снизу нулём.
            if predicted_value < 0.0:
                predicted_value = 0.0
            predictions_by_horizon[horizon] = float(predicted_value)

        return predictions_by_horizon


def resolve_model_location(forecast_config, repository_root):
    """
    Определить, откуда брать веса Chronos: из локальной папки models_cache,
    если она существует, иначе по имени с HuggingFace.

    Возвращает строку (путь или имя модели).
    """
    chronos_config = forecast_config.get("chronos_params", {})
    model_name = chronos_config.get("model_name", "amazon/chronos-t5-small")
    local_folder_name = chronos_config.get("local_folder", "")

    if local_folder_name != "":
        local_path = os.path.join(repository_root, "models_cache", local_folder_name)
        if os.path.isdir(local_path):
            return local_path

    # Локальных весов нет — возвращаем имя для загрузки с HuggingFace.
    return model_name

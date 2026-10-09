"""Отдельный процесс Chronos-Bolt: не импортирует LightGBM."""

from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "models_cache"))

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline
from huggingface_hub import snapshot_download
from src import utils


def main():
    settings = utils.load_yaml_config(ROOT / "configs" / "checked.yaml")
    destination = ROOT / "outputs" / "checked"
    torch.set_num_threads(4)
    torch.manual_seed(settings["random_seed"])
    location = snapshot_download(
        repo_id=settings["foundation"]["name"],
        revision=settings["foundation"]["revision"],
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
        ],
    )
    pipeline = BaseChronosPipeline.from_pretrained(
        location, device_map="cpu", torch_dtype=torch.float32
    )
    raw = pd.read_parquet(ROOT / "data" / "raw" / "consumption.parquet")
    wide = raw.pivot(index=[
        "territory_id",
        "category",
    ], columns="date", values="value")
    rows = []
    for cutoff_text in settings["origins"]:
        print("Chronos-Bolt " + cutoff_text, flush=True)
        cutoff = pd.Period(cutoff_text, freq="M")
        horizons = []
        for horizon in settings["horizons"]:
            if str(cutoff + horizon) in wide.columns:
                horizons.append(horizon)
        sample = pd.read_parquet(destination / ("foundation_input_" + cutoff_text + ".parquet"))
        batch_size = settings["foundation"]["batch_size"]
        for begin in range(0, len(sample), batch_size):
            batch = sample.iloc[begin:begin + batch_size]
            context = torch.tensor(batch.to_numpy(dtype=np.float32))
            with torch.inference_mode():
                quantiles, means = pipeline.predict_quantiles(
                    context, prediction_length=max(horizons), quantile_levels=[
                        0.5,
                    ]
                )
            median = quantiles[:, :, 0].detach().cpu().numpy()
            for position, key in enumerate(batch.index):
                for horizon in horizons:
                    actual = wide.loc[key, str(cutoff + horizon)]
                    if pd.isna(actual):
                        continue
                    prediction = float(median[position, horizon - 1])
                    if not np.isfinite(prediction):
                        raise ValueError("Chronos вернул нечисловой прогноз.")
                    rows.append({
                        "territory_id": int(key[0]),
                        "category": key[1],
                        "model": "Chronos_Bolt_tiny",
                        "cutoff_month": cutoff_text,
                        "horizon": horizon,
                        "actual": float(actual),
                        "predicted": max(0.0, prediction),
                    })
    pd.DataFrame(rows).to_csv(destination / "foundation_predictions.csv", index=False)
    metadata = {
        "model": settings["foundation"]["name"],
        "revision": Path(location).name,
        "device": "cpu",
        "quantile": 0.5,
        "training_data_contamination": "Absence of SberIndex in pretraining is not independently verified.",
    }
    with open(destination / "foundation.json", "w", encoding="utf-8") as target:
        json.dump(metadata, target, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

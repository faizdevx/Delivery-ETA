"""Load the saved model bundle and make predictions from raw trip inputs."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from . import config


class InputError(ValueError):
    """Raised when a prediction request contains values the model cannot score."""


@lru_cache(maxsize=2)
def load_bundle(path: str | None = None) -> dict:
    p = Path(path) if path else config.MODEL_PATH
    if not p.exists():
        raise FileNotFoundError(f"model not found at {p}; run `python scripts/train_model.py`")
    return joblib.load(p)


def validate_record(record: dict, bundle: dict) -> None:
    """Reject inputs outside what the model was trained on (unknown categories)."""
    if record["pickup_zone"] not in bundle["pickup_zones"]:
        raise InputError(f"unknown pickup_zone: {record['pickup_zone']!r}")
    if record["dropoff_zone"] not in bundle["dropoff_zones"]:
        raise InputError(f"unknown dropoff_zone: {record['dropoff_zone']!r}")
    if record["color"] not in bundle["colors"]:
        raise InputError(f"unknown color: {record['color']!r}")


def predict_records(records: list[dict], bundle: dict | None = None) -> list[float]:
    """records: dicts with keys pickup, pickup_zone, dropoff_zone, passengers, color."""
    bundle = bundle or load_bundle()
    for r in records:
        validate_record(r, bundle)
    frame = pd.DataFrame(records)[bundle["input_columns"]]
    frame["pickup"] = pd.to_datetime(frame["pickup"])
    return [float(v) for v in bundle["pipeline"].predict(frame)]


def predict_one(pickup, pickup_zone: str, dropoff_zone: str, passengers: int,
                color: str, bundle: dict | None = None) -> float:
    record = {"pickup": pickup, "pickup_zone": pickup_zone, "dropoff_zone": dropoff_zone,
              "passengers": passengers, "color": color}
    return predict_records([record], bundle)[0]

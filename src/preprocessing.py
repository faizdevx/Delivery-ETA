"""Cleaning, target construction and chronological splitting."""
from __future__ import annotations

import pandas as pd

from . import config


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """target = (dropoff - pickup) in minutes. Timestamps are naive local times."""
    out = df.copy()
    out["pickup"] = pd.to_datetime(out["pickup"])
    out["dropoff"] = pd.to_datetime(out["dropoff"])
    out[config.TARGET] = (out["dropoff"] - out["pickup"]).dt.total_seconds() / 60.0
    return out


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop invalid rows, sort chronologically, keep a `row_id` for traceability.

    Exclusions (documented in DATASET.md):
      * duration <= 0 minutes
      * duration > 180 minutes
      * passengers outside 0..6
      * exact duplicate rows (none in the pinned file)
    Missing zones are NOT dropped; they stay missing (handled natively by the model).
    """
    df = add_target(df)
    df["row_id"] = range(len(df))
    n0 = len(df)
    report: dict = {"n_input": n0}

    dup = df.drop(columns="row_id").duplicated()
    report["excluded_duplicates"] = int(dup.sum())
    df = df[~dup]

    bad_dur = (df[config.TARGET] <= config.MIN_DURATION_EXCLUSIVE) | (
        df[config.TARGET] > config.MAX_DURATION_INCLUSIVE
    )
    report["excluded_invalid_duration"] = int(bad_dur.sum())
    df = df[~bad_dur]

    lo, hi = config.PASSENGERS_RANGE
    bad_pax = (df["passengers"] < lo) | (df["passengers"] > hi)
    report["excluded_passengers_out_of_range"] = int(bad_pax.sum())
    df = df[~bad_pax]

    df = df.sort_values(["pickup", "row_id"], kind="mergesort").reset_index(drop=True)
    report["n_clean"] = int(len(df))
    return df, report


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-sorted frame: oldest -> train, next -> validation, newest -> test."""
    if not df["pickup"].is_monotonic_increasing:
        raise ValueError("frame must be sorted by pickup time")
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return (
        df.iloc[:n_train].reset_index(drop=True),
        df.iloc[n_train : n_train + n_val].reset_index(drop=True),
        df.iloc[n_train + n_val :].reset_index(drop=True),
    )


def build_zone_borough_map(df: pd.DataFrame) -> dict[str, str]:
    """Zone -> borough lookup learned from the data (pickup and dropoff columns).

    Raises if a zone maps to more than one borough.
    """
    pairs = pd.concat(
        [
            df[["pickup_zone", "pickup_borough"]].set_axis(["zone", "borough"], axis=1),
            df[["dropoff_zone", "dropoff_borough"]].set_axis(["zone", "borough"], axis=1),
        ]
    ).dropna()
    nunique = pairs.groupby("zone")["borough"].nunique()
    if (nunique > 1).any():
        raise ValueError(f"zones mapped to several boroughs: {list(nunique[nunique > 1].index)}")
    return {str(k): str(v) for k, v in
            pairs.drop_duplicates().set_index("zone")["borough"].items()}

"""Loading and validating the raw dataset. Nothing here modifies or invents data."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from . import config


class DataValidationError(ValueError):
    """Raised when the raw data does not match the expected schema."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path | None = None, verify_checksum: bool = True) -> pd.DataFrame:
    """Read the raw CSV. Fails loudly if it is missing or differs from the pinned file."""
    path = Path(path) if path is not None else config.RAW_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` or place the "
            "file there manually (see DATASET.md)."
        )
    if verify_checksum:
        digest = file_sha256(path)
        if digest != config.DATASET_SHA256:
            raise DataValidationError(
                f"{path} has SHA-256 {digest}, expected {config.DATASET_SHA256}."
            )
    return pd.read_csv(path)


def validate_raw(df: pd.DataFrame) -> dict:
    """Check schema and basic integrity. Raises on fatal problems, returns findings.

    Fatal: missing columns, empty frame, unparseable timestamps, null timestamps,
    non-numeric numeric columns. Non-fatal findings (e.g. dropoff <= pickup) are
    returned in the report and handled by cleaning.
    """
    problems: list[str] = []
    missing = [c for c in config.RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")
    if len(df) == 0:
        raise DataValidationError("dataset is empty")

    for col in ("passengers", "distance", "fare", "tip", "tolls", "total"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"column '{col}' is not numeric")
    pickup = pd.to_datetime(df["pickup"], errors="coerce")
    dropoff = pd.to_datetime(df["dropoff"], errors="coerce")
    if pickup.isna().any() or dropoff.isna().any():
        problems.append(
            f"unparseable/null timestamps: pickup={int(pickup.isna().sum())}, "
            f"dropoff={int(dropoff.isna().sum())}"
        )
    if problems:
        raise DataValidationError("; ".join(problems))

    duration = (dropoff - pickup).dt.total_seconds() / 60.0
    lo, hi = config.PASSENGERS_RANGE
    return {
        "n_rows": int(len(df)),
        "n_exact_duplicate_rows": int(df.duplicated().sum()),
        "n_dropoff_not_after_pickup": int((duration <= 0).sum()),
        "n_duration_over_max": int((duration > config.MAX_DURATION_INCLUSIVE).sum()),
        "n_passengers_out_of_range": int(
            ((df["passengers"] < lo) | (df["passengers"] > hi)).sum()
        ),
        "missing_values": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
        "pickup_min": str(pickup.min()),
        "pickup_max": str(pickup.max()),
    }

"""Central configuration: paths, dataset pin, split rules, seeds."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# --- Dataset pin (see DATASET.md for provenance and its limits) -------------
RAW_FILENAME = "taxis.csv"
RAW_PATH = DATA_RAW / RAW_FILENAME
SOURCE_META_PATH = DATA_RAW / "SOURCE.json"
DATASET_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv"
# SHA-256 of the file as retrieved on 2026-10-07. A mismatch means the upstream
# file changed (or the local file was edited) and everything downstream must be
# re-verified before use.
DATASET_SHA256 = "08d6d71784dbaa2651fee37fc03389754194c05d72d2d19cbc2c799dea6ac09d"

# --- Schema -----------------------------------------------------------------
RAW_COLUMNS = [
    "pickup", "dropoff", "passengers", "distance", "fare", "tip", "tolls",
    "total", "color", "payment", "pickup_zone", "dropoff_zone",
    "pickup_borough", "dropoff_borough",
]
TARGET = "duration_min"

# Columns the model is allowed to see: all are known when the trip starts.
INPUT_COLUMNS = ["pickup", "pickup_zone", "dropoff_zone", "passengers", "color"]

# Columns rejected as inputs because they are only known after the trip ends.
LEAKY_COLUMNS = {
    "dropoff": "defines the target (target = dropoff - pickup)",
    "distance": "metered distance actually driven; only known after the trip",
    "fare": "metered fare, computed from distance and time travelled",
    "tip": "paid at the end of the trip",
    "tolls": "incurred on the route actually taken",
    "total": "sum of fare, tip, tolls (post-trip)",
    "payment": "payment type is recorded at trip end",
}

# Raw columns not taken from the file as inputs, but re-derived inside the model
# from the (allowed) zone columns; not leaky because zones are known at pickup.
DERIVED_NOT_LEAKY = {
    "pickup_borough": "derived from pickup_zone via a zone->borough lookup",
    "dropoff_borough": "derived from dropoff_zone via a zone->borough lookup",
}

# --- Cleaning / splitting ---------------------------------------------------
MIN_DURATION_EXCLUSIVE = 0.0   # minutes; durations <= 0 are not physical trips
MAX_DURATION_INCLUSIVE = 180.0  # minutes; none exceed this in the pinned file
PASSENGERS_RANGE = (0, 6)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # test gets the remaining chronologically-last 15%

SEED = 42
MODEL_VERSION = "gbt-v1"
MODEL_PATH = MODELS_DIR / "model.joblib"

# Trip Duration Predictor

> **Leakage-aware machine learning pipeline for predicting trip duration before a trip starts.**

Gradient-boosted-tree regression of **trip duration in minutes**, with chronological evaluation, baselines, hyperparameter tuning, error analysis, a FastAPI service, and a small web UI.

<p align-"center">

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-Validation-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-Testing-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-Linting-D7FF64?style=for-the-badge&logo=ruff&logoColor=black)
![MyPy](https://img.shields.io/badge/MyPy-Type_Checking-2A6DB2?style=for-the-badge&logo=mypy&logoColor=white)
![Make](https://img.shields.io/badge/Make-Automation-427819?style=for-the-badge&logo=gnu&logoColor=white)
</p>

<p align="center">
  <strong>4.80 min MAE</strong> ·
  <strong>7.13 min RMSE</strong> ·
  <strong>0.587 R²</strong>
</p>

---

## Project Snapshot

| Area | Details |
|---|---|
| **Task** | Trip-duration regression |
| **Model** | `HistGradientBoostingRegressor` |
| **Primary metric** | MAE |
| **Test MAE** | **4.80 min** |
| **Test RMSE** | **7.13 min** |
| **Test R²** | **0.587** |
| **Test set** | 965 trips |
| **Data** | 6,433-row NYC taxi sample |
| **Evaluation** | Chronological 70/15/15 split |
| **Serving** | FastAPI + web UI |
| **Python** | Tested on 3.13.16 |

---

## Honest Scope

> **Important:** This project is intentionally named **Trip Duration Predictor** rather than a delivery-specific product name.
>
> The underlying data is **NYC taxi trips from March 2019**, not Indian or food-delivery data. I could not find, verify and download a real Indian or food-delivery dataset: the one commonly used for this idea is described by Kaggle as simulated, so it was rejected, and the official NYC sources were unreachable from my build environment.
>
> The project therefore demonstrates **ETA-style regression on a real taxi sample as a stand-in**. Nothing here says anything about Indian delivery times.

---

## Why This Project Exists

The core question is simple:

> **How long will a trip take before it starts, using only information that is available at request time?**

That constraint matters because some of the strongest fields in the raw data are only known after the trip has begun or ended.

The project is therefore designed around **leakage-aware feature selection and future-facing evaluation**, rather than optimizing a random train/test split.

---

## Dataset

`taxis.csv` comes from the `seaborn-data` repository and contains a **6,433-row sample** attributed by its host to the NYC Taxi & Limousine Commission's TLC Trip Record Data.

Full provenance, license status, and verification limits are documented in [`DATASET.md`](DATASET.md).

### Verification limits

The following could not be verified:

- the sampling method
- any modifications
- the license
- agreement with the original TLC files

The original host was unreachable from the build environment.

The raw file is **not committed**. [`scripts/download_data.py`](scripts/download_data.py) downloads it and checks a pinned SHA-256.

### Why this dataset?

It was the only candidate that could be both downloaded and examined from the build environment and that provided:

- recorded start and end timestamps
- zone identifiers
- passenger count

Its weaknesses are material:

- small dataset
- secondary source
- undocumented sampling step
- one city
- one month

Treat the reported results as a **demonstration of method on limited data**, not as evidence about real delivery ETA.

---

## Target

The target is:

```text
duration_min = (dropoff − pickup) in minutes
```

Rows with duration `<= 0` are excluded (**6 rows**). See [`DATASET.md`](DATASET.md).

---

## Features

### Inputs available at pickup

| Feature | Role |
|---|---|
| `pickup` | Pickup timestamp |
| `pickup_zone` | Pickup zone |
| `dropoff_zone` | Destination zone |
| `passengers` | Passenger count |
| `color` | Taxi color |

### Engineered features

The model derives:

- hour
- hour represented with sin/cos
- day of week
- day of week represented with sin/cos
- weekend flag
- boroughs from a zone lookup
- optional smoothed out-of-fold route-duration statistics

Per-feature formula, rationale, and leakage risk are documented in:

- `src/features.py` (`FEATURE_DOCS`)
- `reports/metrics/model_metrics.json`

### Rejected as leaky

The following are deliberately excluded from the product model:

`dropoff`, `distance`, `fare`, `tip`, `tolls`, `total`, `payment`

The reasons are documented in `src/config.py` and the notebook's leakage table.

---

## Methodology

```text
Raw CSV
   │
   ├── checksum + schema validation
   │
   ├── cleaning
   │
   ├── chronological 70 / 15 / 15 split
   │
   ├── feature transformer
   │      └── fitted on training data only
   │
   ├── baseline models
   │
   ├── gradient-boosted regression
   │
   ├── time-series cross-validation
   │
   └── untouched final test evaluation
```

### Baselines

Two baselines are included:

1. Median of training durations
2. Ridge regression on time, borough, and taxi color

### Final model

`sklearn.ensemble.HistGradientBoostingRegressor` with native categorical zones.

No XGBoost or LightGBM dependency was added because nothing in this dataset justified the extra dependency.

### Hyperparameter tuning

Random search over **30 configurations** covering:

- learning rate
- number of iterations
- maximum leaves
- minimum samples per leaf
- L2 regularization
- loss
- whether to use route statistics

Selection metric: **MAE**

Cross-validation: **forward-chaining `TimeSeriesSplit`**, using the training split only.

The configuration with the lowest mean CV MAE was selected without manual overriding. If route statistics did not win, that result is retained rather than hidden.

### Evaluation protocol

- Validation scores come from a train-only fit.
- The final model is refit on train + validation.
- The untouched final 15% is scored once as the test set.
- The split is chronological because the model is intended for future trips.

---

## Results

### Held-out test set

**n = 965**, chronologically last 15% of cleaned data.

MAE and RMSE are in minutes. The MAE interval is a **95% percentile bootstrap over 1,000 resamples** of the test trips.

| Model | MAE | MAE 95% CI | RMSE | R² | Median AE |
|---|---:|---:|---:|---:|---:|
| Baseline: median of training durations | 7.55 | 7.01 – 8.13 | 11.52 | -0.077 | 5.31 |
| Baseline: ridge regression (time, borough, color) | 7.51 | 7.07 – 7.98 | 10.24 | 0.149 | 5.99 |
| **Gradient Boosted Trees (`HistGradientBoostingRegressor`)** | **4.80** | **4.44 – 5.13** | **7.13** | **0.587** | **3.27** |

**MAE reduction vs. median baseline:** 2.76 min  
**Paired bootstrap 95% CI:** 2.35 – 3.20 min

### Validation split

| Model | Val MAE | Val RMSE | Val R² |
|---|---:|---:|---:|
| Baseline: median of training durations | 7.43 | 11.08 | -0.059 |
| Baseline: ridge regression (time, borough, color) | 7.32 | 9.97 | 0.142 |
| **Gradient Boosted Trees (`HistGradientBoostingRegressor`)** | **4.78** | **7.13** | **0.562** |

### Reference experiments

These experiments use the same evaluation protocol and are **not** the product configuration.

| Experiment | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| GBT route stats on ablation | 4.84 | 7.09 | 0.591 |
| GBT + metered `distance` **(LEAKY)** | 2.90 | 4.37 | 0.845 |

The leaky result is included to show how much performance the unavailable-at-request-time `distance` field would buy. It is **not** used by the product model.

### Selected configuration

```json
{
  "learning_rate": 0.1,
  "max_iter": 200,
  "max_leaf_nodes": 16,
  "min_samples_leaf": 20,
  "l2_regularization": 10.0,
  "loss": "squared_error",
  "use_route_stats": false
}
```

Tuning details:

- random search: 30 configs
- `TimeSeriesSplit(4)`
- scoring: MAE
- best CV MAE: **6.14 ± 0.74**
- seed: `42`
- trained: `2026-10-07T04:02:31+00:00` (UTC)
- full log: `reports/metrics/tuning_results.csv`

---

## Feature Importance

Permutation importance is measured on the held-out test split as the increase in MAE after shuffling each input column. Values are mean ± standard deviation over **20 repeats**.

| Input column | MAE increase |
|---|---:|
| `dropoff_zone` | **5.659 ± 0.130** |
| `pickup_zone` | **5.502 ± 0.165** |
| `pickup` | 0.740 ± 0.071 |
| `color` | 0.037 ± 0.011 |
| `passengers` | 0.004 ± 0.004 |

Permutation importance does **not** show that a feature causes longer trips. Correlated inputs can split or hide importance.

The strongest predictive signals are the zones, which is expected because distance is excluded and zones are the main spatial signal available to the model.

---

## Error Analysis

All error-analysis results below come from the **965-trip test split**. See `reports/metrics/evaluation_metrics.json`.

### Absolute-error distribution

| Percentile | Absolute error |
|---|---:|
| p50 | 3.27 min |
| p75 | 5.90 min |
| p90 | 10.65 min |
| p95 | 15.11 min |
| p99 | 28.58 min |

Additional test-set behavior:

- Over-predicted: **541 trips**
- Under-predicted: **424 trips**
- Mean signed error (`pred − actual`): **0.18 min**
- Worst 5% threshold: **15.11 min**
- Worst 5% count: **49 trips**
- Mean actual duration in worst 5%: **32.94 min**
- Mean actual duration overall: **13.99 min**
- 59% of worst 5% are under-predictions
- Missing-zone rate: **6%** in worst 5% vs **1%** overall

For the 20 worst records, metered-distance / duration speed has:

- median: **13.3 mph**
- minimum: **0.0 mph**
- maximum: **31.0 mph**

Overall median speed is **10.0 mph**.

`distance` is used only for diagnosis here, not as a model input.

### Bias by actual duration

Positive signed error means over-prediction.

| Actual duration | n | MAE | Mean (`pred − actual`) |
|---|---:|---:|---:|
| `<=5` min | 145 | 4.78 | 4.64 |
| `5-10` min | 292 | 3.29 | 2.55 |
| `10-20` min | 332 | 3.57 | 0.41 |
| `20-40` min | 162 | 7.24 | -4.90 |
| `>40` min | 34 | 18.07 | -17.17 |

### MAE by pickup hour

| Pickup hour | n | MAE | Mean (`pred − actual`) |
|---|---:|---:|---:|
| `00-05` | 101 | 3.69 | 1.31 |
| `06-09` | 106 | 4.45 | 1.15 |
| `10-15` | 303 | 5.15 | 0.27 |
| `16-19` | 224 | 5.35 | 0.37 |
| `20-23` | 231 | 4.44 | -1.05 |

### MAE by taxi color

| Taxi color | n | MAE | Mean (`pred − actual`) |
|---|---:|---:|---:|
| green | 151 | 6.44 | 1.26 |
| yellow | 814 | 4.49 | -0.02 |

### MAE by day type

| Day type | n | MAE | Mean (`pred − actual`) |
|---|---:|---:|---:|
| weekday | 563 | 5.09 | 0.34 |
| weekend | 402 | 4.39 | -0.04 |

### Worst 8 test predictions

`distance` is shown for diagnosis only and is **not** a model input.

| Pickup | Pickup zone | Dropoff zone | Distance (mi) | Actual | Predicted |
|---|---|---|---:|---:|---:|
| 2019-03-27 14:46:32 | Westchester Village/Unionport | Riverdale/North Riverdale/Fieldston | 9.00 | 56.2 | 17.1 |
| 2019-03-30 20:14:44 | JFK Airport | JFK Airport | 18.91 | 46.7 | 11.7 |
| 2019-03-29 17:32:20 | JFK Airport | Cobble Hill | 26.92 | 81.5 | 47.8 |
| 2019-03-28 15:55:12 | Fort Greene | Brownsville | 4.88 | 51.8 | 18.2 |
| 2019-03-28 14:34:59 | Flatbush/Ditmas Park | Flatbush/Ditmas Park | 0.00 | 3.6 | 35.6 |
| 2019-03-31 21:03:02 | JFK Airport | Mount Hope | 23.30 | 62.0 | 31.6 |
| 2019-03-30 14:14:00 | Erasmus | Upper West Side North | 13.47 | 56.3 | 26.4 |
| 2019-03-30 22:02:43 | Greenwich Village South | Upper West Side South | 6.79 | 47.1 | 17.7 |

Generated figures are stored in:

```text
reports/figures/
├── actual_vs_predicted
├── residual_distribution
├── mae_by_group
├── permutation_importance
└── bias_by_duration
```

### Interpretation

The model **over-predicts short trips and under-predicts long ones**, which is the usual shrinkage behavior of a regressor with limited signal.

Long errors involve airport and cross-borough trips, where the same pair of zones can take very different amounts of time. The model has no traffic, weather, or route information.

Some of the worst records look like data artefacts, such as a zero-distance trip with a multi-minute duration, but their cause has not been verified.

---

## Running Locally

Tested on **Python 3.13.16**.

`pyproject.toml` allows Python `>=3.11`, but other Python versions have not been tested.

### 1. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For notebook and development tools:

```bash
pip install -r requirements-dev.txt
```

### 3. Download and verify the data

```bash
python scripts/download_data.py
```

### 4. Train

```bash
python scripts/train_model.py
```

This writes:

```text
models/model.joblib
reports/metrics/*
```

### 5. Evaluate

```bash
python scripts/evaluate_model.py
```

### 6. Run tests

```bash
python -m pytest -q
```

### 7. Start the API

```bash
uvicorn app.main:app --reload
```

Then open:

```text
http://localhost:8000/
```

API documentation:

```text
http://localhost:8000/docs
```

### Restricted-network fallback

If the download is blocked, place the file manually at:

```text
data/raw/taxis.csv
```

Then run:

```bash
python scripts/download_data.py --verify-only
```

---

## Useful Commands

Run a prediction from the CLI:

```bash
python -m src.predict --pickup "2019-03-15 18:30" --pickup-zone "Midtown Center" \
    --dropoff-zone "JFK Airport" --passengers 1 --color yellow
```

List available zones:

```bash
python -m src.predict --list-zones
```

Rebuild and execute the EDA notebook:

```bash
python scripts/build_eda_notebook.py
```

Refresh generated README blocks:

```bash
python scripts/update_readme.py
```

Lint and type-check:

```bash
ruff check . && mypy
```

Run the complete pipeline:

```bash
make all
```

This runs:

```text
data → train → evaluate → notebook → readme → test
```

---

## API

### Example request

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"pickup_datetime": "2019-03-15T18:30:00", "pickup_zone": "Midtown Center", "dropoff_zone": "JFK Airport", "passengers": 1, "taxi_color": "yellow"}'
```

### Example response

Captured from the repository's model:

```json
{
  "predicted_trip_duration_minutes": 52.39,
  "units": "minutes",
  "model_version": "gbt-v1"
}
```

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Web UI / JSON |
| `GET` | `/health` | Health check |
| `GET` | `/options` | Available input options |
| `POST` | `/predict` | Trip-duration prediction |
| `GET` | `/docs` | API documentation |

The API returns a **point estimate only**. No prediction interval is returned because none has been validated.

The API returns HTTP `422` for:

- unknown zones
- out-of-range passengers
- timezone-aware datetimes
- extra fields

---

## Docker

```bash
docker compose up --build
```

Serves on:

```text
http://localhost:8000
```

Or:

```bash
docker build -t trip-duration-predictor .
docker run -p 8000:8000 trip-duration-predictor
```

### Verification note

Docker was **not verified** because there was no Docker daemon in the build environment, so the image was not built.

A clean virtualenv with only `requirements.txt` plus `src/`, `app/`, and `models/` was verified to serve `/health` and `/predict`, matching what the Dockerfile copies.

The image expects:

```text
models/model.joblib
```

to exist. It is committed.

---

## Project Structure

```text
trip-duration-predictor/
│
├── app/
│   ├── FastAPI application
│   ├── Pydantic schemas
│   └── static web UI
│
├── data/
│   ├── raw/          downloaded data
│   └── processed/    derived data
│
├── models/
│   └── model.joblib
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── reports/
│   ├── metrics/
│   └── figures/
│
├── scripts/
│   ├── download_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── build_eda_notebook.py
│   └── update_readme.py
│
├── src/
│   ├── config
│   ├── data
│   ├── preprocessing
│   ├── features
│   ├── train
│   ├── evaluate
│   ├── inference
│   └── predict
│
├── tests/
│   ├── data
│   ├── features
│   ├── model
│   └── API
│
├── DATASET.md
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

Downloaded and derived data are git-ignored.

---

## Limitations

This repository is deliberately explicit about what the model **does not** establish.

- Taxi trips, one city, one month, a **6,433-row secondary sample**.
- **Not delivery data and not Indian data.**
- Spatial information is zone names only.
- No coordinates, route, traffic, or weather.
- Rare zone pairs dominate.
- Unseen zones are rejected by the API.
- Test set contains **965 trips**.
- Short trips (`<1 min`) and implausible-speed records are retained, as documented in `DATASET.md`.
- `dropoff_zone` must be known up front. This fits an ETA-to-a-given-destination use case but not open-ended dispatch.
- The model version tag `gbt-v1` is a label, not a validation claim.

---

## Data License

**Not verified.**

See [`DATASET.md`](DATASET.md).

The MIT license in [`LICENSE`](LICENSE) covers the repository's **code only**.

---

## Model Card & Responsible Use

### Intended use

Learning and portfolio demonstration of a **leakage-aware ETA regression workflow**.

### Not intended for

- production routing or dispatch
- pricing
- driver evaluation
- safety-relevant decisions
- predictions outside NYC taxi trips

### Geographic limitation

New York City zones only.

### Distribution shift

The model was trained on trips from:

```text
2019-02-28 → 2019-03-31
```

Other seasons, years, events, road changes, and policies are not represented, and nothing was tested on them.

### Model limitations

The model produces point estimates with errors of several minutes, is biased toward the mean on very short and very long trips, and has **no uncertainty estimate**.

It has **not** been validated for production.

---

## Reproducibility

- Seed is fixed at `42` (`src/config.py`, `SEED = 42`).
- Retraining the same data with the pinned library versions reproduced identical metrics in the recorded runs.
- Dependencies are pinned to the versions tested for this project.

---

## Repository Status

This project is a **portfolio-scale demonstration of an end-to-end ML workflow**:

```text
data validation
      ↓
data cleaning
      ↓
leakage-aware features
      ↓
chronological evaluation
      ↓
baseline comparison
      ↓
model tuning
      ↓
error analysis
      ↓
trained artifact
      ↓
FastAPI service
      ↓
web UI
```

The emphasis is on **reproducible methodology and honest evaluation**, not on claiming production-grade ETA accuracy from a limited dataset.

# Bharat Delivery ETA

Gradient-boosted-tree regression of **trip duration in minutes**, with a leakage-aware pipeline, baselines,
tuning, error analysis, a FastAPI service and a small web UI.

> **Read this first (honest scope).** The name says "delivery" and "Bharat", but the data is **NYC taxi trips**
> (March 2019). I could not find, verify and download a real Indian or food-delivery dataset: the one commonly
> used for this idea is described by Kaggle as simulated, so it was rejected, and the official NYC sources were
> unreachable from my build environment. The project therefore demonstrates ETA regression on a real taxi sample as a
> stand-in. Nothing here says anything about Indian delivery times.

## Problem

Predict how long a trip will take **before it starts**, given only information available at request time.
That "before it starts" constraint is the hard part: the strongest predictor in the file (metered distance)
is only known after the trip, so it is excluded (see the leaky reference experiment in Results).

## Dataset

`taxis.csv` from the `seaborn-data` repository, a 6,433-row sample attributed by its host to the NYC Taxi &
Limousine Commission's TLC Trip Record Data. Full provenance, license status and verification limits are in
[`DATASET.md`](DATASET.md). Summary of what is **not** verified: the sampling method, any modifications, the
license, and agreement with the original TLC files (the original host was unreachable from my environment).
The raw file is not committed; `scripts/download_data.py` downloads it and checks a pinned SHA-256.

## Why this dataset

It is the only candidate I could both download and examine from my environment that has recorded start and end
timestamps (so the target is derived, not invented), zone identifiers and a passenger count. Its weaknesses are
material: it is small, comes from a secondary source with an undocumented sampling step, and covers one city and one
month. Treat the results below as a demonstration of method on limited data, not as evidence about real delivery ETA.

## Features

Model inputs (all known at pickup): `pickup` time, `pickup_zone`, `dropoff_zone`, `passengers`, taxi `color`.
Engineered inside the model: hour (plain and sin/cos), day of week (plain and sin/cos), weekend flag, boroughs
from a zone lookup, and optional smoothed out-of-fold route-duration statistics. Per-feature formula, reason
and leakage risk: `src/features.py` (`FEATURE_DOCS`) and `reports/metrics/model_metrics.json`.

Rejected as leaky: `dropoff` (defines the target), `distance`, `fare`, `tip`, `tolls`, `total`, `payment`.
Reasons are in `src/config.py` and the notebook's leakage table.

## Target

`duration_min = (dropoff − pickup)` in minutes. Rows with duration ≤ 0 are excluded (6 rows). See `DATASET.md`.

## Approach

- **Pipeline:** raw CSV → checksum + schema validation → cleaning → chronological 70/15/15 split → feature
  transformer (fitted on training data only) → model → evaluation. A chronological split is used because the
  model would be applied to future trips.
- **Baselines:** median of training durations; ridge regression on time, borough and taxi color.
- **Gradient boosting:** `sklearn.ensemble.HistGradientBoostingRegressor` with native categorical zones.
  No XGBoost/LightGBM: nothing here justified the extra dependency.
- **Tuning:** random search (30 configs) over learning rate, iterations, leaves, min leaf size, L2, loss and
  whether to use route statistics, scored by MAE with forward-chaining `TimeSeriesSplit` on the training split
  only. The configuration with the lowest mean CV MAE was selected; no manual overriding. If route statistics
  did not win, that is reported rather than hidden (see the ablation row).
- **Evaluation protocol:** validation scores come from a train-only fit. Test scores come from a refit on
  train+validation with the chosen parameters, scored once on the untouched final 15% of trips.

## Results

<!-- RESULTS:START -->
Test split (n=965, chronologically last 15% of cleaned data). MAE and RMSE in minutes; MAE interval is a 95% percentile bootstrap (1000 resamples) over test trips.

| Model | MAE | MAE 95% CI | RMSE | R² | Median AE |
|---|---:|---:|---:|---:|---:|
| Baseline: median of training durations | 7.55 | 7.01 – 8.13 | 11.52 | -0.077 | 5.31 |
| Baseline: ridge regression (time, borough, color) | 7.51 | 7.07 – 7.98 | 10.24 | 0.149 | 5.99 |
| **Gradient Boosted Trees (HistGradientBoostingRegressor)** | 4.80 | 4.44 – 5.13 | 7.13 | 0.587 | 3.27 |

MAE reduction of the GBT vs. the median baseline: 2.76 min (paired bootstrap 95% CI 2.35 – 3.20).

Validation-split scores (model fit on train only; used for selection checks):

| Model | Val MAE | Val RMSE | Val R² |
|---|---:|---:|---:|
| Baseline: median of training durations | 7.43 | 11.08 | -0.059 |
| Baseline: ridge regression (time, borough, color) | 7.32 | 9.97 | 0.142 |
| **Gradient Boosted Trees (HistGradientBoostingRegressor)** | 4.78 | 7.13 | 0.562 |

Reference experiments (same protocol; **not** the product):

| Experiment | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| gbt route stats on ablation | 4.84 | 7.09 | 0.591 |
| GBT + metered `distance` (LEAKY: unknown before the trip; shows what leakage would buy) | 2.90 | 4.37 | 0.845 |

Chosen parameters: `{"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 16, "min_samples_leaf": 20, "l2_regularization": 10.0, "loss": "squared_error", "use_route_stats": false}`. Search: random search, 30 configs, TimeSeriesSplit(4) on the training split, scored by MAE; best CV MAE 6.14 ± 0.74 (fold std). Full log: `reports/metrics/tuning_results.csv`. Trained 2026-10-07T04:02:31+00:00 (UTC), seed 42.
<!-- RESULTS:END -->

## Feature importance

<!-- IMPORTANCE:START -->
Permutation importance on the test split (increase in MAE, minutes, when one input column is shuffled; mean ± std over 20 repeats):

| Input column | MAE increase |
|---|---:|
| `dropoff_zone` | 5.659 ± 0.130 |
| `pickup_zone` | 5.502 ± 0.165 |
| `pickup` | 0.740 ± 0.071 |
| `color` | 0.037 ± 0.011 |
| `passengers` | 0.004 ± 0.004 |
<!-- IMPORTANCE:END -->

Permutation importance measures how much held-out error grows when a column is shuffled in this model. It
does **not** show that a column causes longer trips, and importance of correlated inputs can be split or
hidden. Zones were the strongest predictive inputs here, which is expected given that distance is excluded
and zones are the only spatial signal.

## Error analysis

<!-- ERRORS:START -->
All numbers from the test split (n=965); see `reports/metrics/evaluation_metrics.json`.

- Absolute-error percentiles (min): p50=3.27, p75=5.90, p90=10.65, p95=15.11, p99=28.58.
- Over-predicted 541 trips, under-predicted 424; mean signed error (pred − actual) = 0.18 min.
- The worst 5% of errors (|error| ≥ 15.11 min, n=49) have mean actual duration 32.94 min vs 13.99 overall, and 59% of them are under-predictions. 6% have a missing zone vs 1% overall.
- For the 20 worst records, the metered-distance / duration speed has median 13.3 mph (min 0.0, max 31.0); overall median 10.0 mph. (`distance` is used only for this diagnosis.)

**Bias by actual duration** (positive = over-prediction):

| Actual duration (min) | n | MAE | Mean (pred − actual) |
|---|---:|---:|---:|
| <=5 | 145 | 4.78 | 4.64 |
| 5-10 | 292 | 3.29 | 2.55 |
| 10-20 | 332 | 3.57 | 0.41 |
| 20-40 | 162 | 7.24 | -4.90 |
| >40 | 34 | 18.07 | -17.17 |

**MAE by pickup hour:**

| Pickup hour | n | MAE | Mean (pred − actual) |
|---|---:|---:|---:|
| 00-05 | 101 | 3.69 | 1.31 |
| 06-09 | 106 | 4.45 | 1.15 |
| 10-15 | 303 | 5.15 | 0.27 |
| 16-19 | 224 | 5.35 | 0.37 |
| 20-23 | 231 | 4.44 | -1.05 |

**MAE by taxi color:**

| Taxi color | n | MAE | Mean (pred − actual) |
|---|---:|---:|---:|
| green | 151 | 6.44 | 1.26 |
| yellow | 814 | 4.49 | -0.02 |

**MAE by day type:**

| Day type | n | MAE | Mean (pred − actual) |
|---|---:|---:|---:|
| weekday | 563 | 5.09 | 0.34 |
| weekend | 402 | 4.39 | -0.04 |

**Worst 8 test predictions** (`distance` shown for diagnosis only, not a model input):

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

Figures: `reports/figures/` (actual_vs_predicted, residual_distribution, mae_by_group, permutation_importance, bias_by_duration).
<!-- ERRORS:END -->

Reading these numbers (correlation, not causation): the model over-predicts short trips and under-predicts
long ones, the usual shrinkage of a regressor with limited signal. Long errors involve airport and cross-borough
trips, where the same pair of zones can take very different times, and the model has no traffic, weather or
route information. Some of the worst records look like data artefacts (for example a zero-distance trip with a
multi-minute duration), but I have not verified their cause.

## Running locally

Tested on Python 3.13.16 (`pyproject.toml` allows ≥3.11, which I have not tested).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # add requirements-dev.txt for notebook/lint tools

python scripts/download_data.py      # downloads taxis.csv, verifies SHA-256
python scripts/train_model.py        # ~1 min; writes models/model.joblib, reports/metrics/*
python scripts/evaluate_model.py     # test metrics, error analysis, figures
python -m pytest -q                  # needs the data and trained model above
uvicorn app.main:app --reload        # UI at http://localhost:8000/ , docs at /docs
```

If the download is blocked (as it can be in restricted networks), place the file manually at `data/raw/taxis.csv`
and run `python scripts/download_data.py --verify-only`.

Other commands:

```bash
python -m src.predict --pickup "2019-03-15 18:30" --pickup-zone "Midtown Center" \
    --dropoff-zone "JFK Airport" --passengers 1 --color yellow
python -m src.predict --list-zones
python scripts/build_eda_notebook.py   # rebuilds and executes notebooks/01_eda.ipynb
python scripts/update_readme.py        # refreshes the generated README blocks from reports/
ruff check . && mypy                   # lint and type checks (requirements-dev.txt)
```

`make all` runs data → train → evaluate → notebook → readme → test (requires `make`).

## API example

<!-- API_EXAMPLE:START -->
```bash
curl -s -X POST http://localhost:8000/predict -H 'content-type: application/json' \
  -d '{"pickup_datetime": "2019-03-15T18:30:00", "pickup_zone": "Midtown Center", "dropoff_zone": "JFK Airport", "passengers": 1, "taxi_color": "yellow"}'
```

Response (captured from this repository's model):

```json
{
  "predicted_trip_duration_minutes": 52.39,
  "units": "minutes",
  "model_version": "gbt-v1"
}
```
<!-- API_EXAMPLE:END -->

Endpoints: `GET /` (UI for browsers, JSON for API clients), `GET /health`, `GET /options`, `POST /predict`,
`GET /docs`. The response is a point estimate only. No prediction interval is returned because none has been
validated. Unknown zones, out-of-range passengers, timezone-aware datetimes and extra fields return HTTP 422.

## Docker

```bash
docker compose up --build            # serves on http://localhost:8000
# or: docker build -t bharat-delivery-eta . && docker run -p 8000:8000 bharat-delivery-eta
```

**Not verified:** there was no Docker daemon in my build environment, so the image has not been built. I did verify
that a clean virtualenv with only `requirements.txt` plus `src/`, `app/` and `models/` serves `/health` and
`/predict`, which is what the Dockerfile copies. The image expects `models/model.joblib` to exist (it is committed).

## Project structure

```
data/raw, data/processed   downloaded / derived data (git-ignored)
notebooks/01_eda.ipynb     executed EDA; findings are printed by code
src/                       config, data, preprocessing, features, train, evaluate, inference, predict
app/                       FastAPI app, Pydantic schemas, static web UI
scripts/                   download_data, train_model, evaluate_model, build_eda_notebook, update_readme
models/model.joblib        trained pipeline + metadata bundle
reports/metrics, figures   generated metrics (JSON/CSV) and plots
tests/                     pytest suite (data, features, model, API)
```

## Limitations

- Taxi trips, one city, one month, a 6,433-row secondary sample. Not delivery data, not Indian data.
- Spatial information is zone names only; no coordinates, route, traffic or weather.
- Rare zone pairs dominate; unseen zones are rejected by the API.
- Test set is 965 trips; see the bootstrap intervals above for how uncertain the metrics are.
- Short trips (<1 min) and implausible-speed records are retained (documented in `DATASET.md`).
- `dropoff_zone` must be known up front, which fits an "ETA to a given destination" use but not open-ended dispatch.
- The model version tag `gbt-v1` is a label, not a validation claim.

## Data license

**Not verified.** See `DATASET.md`. The MIT license in `LICENSE` covers this repository's code only.

## Model card / responsible use

- **Intended use:** learning, portfolio demonstration of a leakage-aware ETA regression workflow.
- **Not intended for:** production routing or dispatch, pricing, driver evaluation, safety-relevant decisions, or any
  prediction outside New York City taxi trips.
- **Data limitations:** see above and `DATASET.md` (unverified sampling and license).
- **Geographic limitation:** New York City zones only.
- **Distribution shift:** trained on trips from 2019-02-28 to 2019-03-31 only. Other seasons, years, events,
  road changes or policies are not represented, and nothing was tested on them.
- **Model limitations:** point estimates with errors of several minutes (see the table above), biased toward the
  mean on very short and very long trips, no uncertainty estimate. It has not been validated for production.

## Reproducibility

Seeds are fixed (`src/config.py`, `SEED = 42`); retraining the same data with the pinned library versions
reproduced identical metrics in my runs. Dependencies are pinned to the versions I tested.

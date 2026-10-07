"""Reproducible training: baselines, tuning, final model, artifacts, metrics."""
from __future__ import annotations

import json
import platform
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .data import load_raw, validate_raw
from .features import FEATURE_DOCS, TripFeatures
from .preprocessing import build_zone_borough_map, chronological_split, clean

N_TUNING_CONFIGS = 30
CV_SPLITS = 4

SEARCH_SPACE = {
    "learning_rate": [0.02, 0.04, 0.07, 0.1, 0.15],
    "max_iter": [100, 200, 400],
    "max_leaf_nodes": [4, 8, 16, 31],
    "min_samples_leaf": [10, 20, 40, 80],
    "l2_regularization": [0.0, 0.1, 1.0, 10.0],
    "loss": ["squared_error", "absolute_error"],
    "use_route_stats": [True, False],
}


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "median_ae": float(np.median(np.abs(y_true - y_pred))),
        "n": int(len(y_true)),
    }


def make_gbt_pipeline(params: dict, zone_borough: dict, extra_numeric: tuple = ()) -> Pipeline:
    p = dict(params)
    use_route_stats = p.pop("use_route_stats", True)
    return Pipeline([
        ("features", TripFeatures(zone_borough=zone_borough, use_route_stats=use_route_stats,
                                  extra_numeric=extra_numeric)),
        ("model", HistGradientBoostingRegressor(
            categorical_features="from_dtype", early_stopping=False,
            random_state=config.SEED, **p)),
    ])


def make_median_baseline() -> Pipeline:
    return Pipeline([("model", DummyRegressor(strategy="median"))])


def make_ridge_baseline(zone_borough: dict) -> Pipeline:
    numeric = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "passengers"]
    categorical = ["pickup_borough", "dropoff_borough", "color"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([
        ("features", TripFeatures(zone_borough=zone_borough, use_route_stats=False)),
        ("pre", pre),
        ("model", Ridge(alpha=1.0)),
    ])


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    raw = load_raw()
    validation = validate_raw(raw)
    cleaned, clean_report = clean(raw)
    train, val, test = chronological_split(cleaned)
    return train, val, test, validation, clean_report


def _xy(df: pd.DataFrame, extra: tuple = ()):
    return df[config.INPUT_COLUMNS + list(extra)], df[config.TARGET].to_numpy()


def tune(train: pd.DataFrame, zone_borough: dict) -> pd.DataFrame:
    """Random search scored by MAE with forward-chaining CV on the training split."""
    X, y = _xy(train)
    cv = TimeSeriesSplit(n_splits=CV_SPLITS)
    rows = []
    sampler = ParameterSampler(SEARCH_SPACE, n_iter=N_TUNING_CONFIGS, random_state=config.SEED)
    for i, params in enumerate(sampler):
        fold_mae = []
        for tr, te in cv.split(X):
            pipe = make_gbt_pipeline(params, zone_borough)
            pipe.fit(X.iloc[tr], y[tr])
            fold_mae.append(mean_absolute_error(y[te], pipe.predict(X.iloc[te])))
        rows.append({"config": i, **params, "cv_mae_mean": float(np.mean(fold_mae)),
                     "cv_mae_std": float(np.std(fold_mae))})
        print(f"[tune {i + 1}/{N_TUNING_CONFIGS}] cv_mae={np.mean(fold_mae):.3f} {params}")
    return pd.DataFrame(rows).sort_values("cv_mae_mean").reset_index(drop=True)


def main() -> dict:
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    train, val, test, validation, clean_report = prepare_data()
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(config.DATA_PROCESSED / f"{name}.csv", index=False)
    trainval = pd.concat([train, val], ignore_index=True)
    zb_train = build_zone_borough_map(train)
    zb_final = build_zone_borough_map(trainval)

    # ---- tuning (train split only) ---------------------------------------
    tuning = tune(train, zb_train)
    tuning.to_csv(config.METRICS_DIR / "tuning_results.csv", index=False)
    best_row = tuning.iloc[0]
    param_names = list(SEARCH_SPACE)
    best_params = {k: best_row[k].item() if hasattr(best_row[k], "item") else best_row[k]
                   for k in param_names}
    print("best params:", best_params)

    Xtr, ytr = _xy(train)
    Xva, yva = _xy(val)
    Xtv, ytv = _xy(trainval)
    Xte, yte = _xy(test)

    results: dict = {}

    def run(name, builder, extra=()):
        """Validation score: fit on train only. Test score: fit on train+val."""
        cols = list(config.INPUT_COLUMNS) + list(extra)
        m_val = builder(zb_train).fit(train[cols], ytr)
        m_test = builder(zb_final).fit(trainval[cols], ytv)
        results[name] = {
            "validation_fit_on_train": regression_metrics(yva, m_val.predict(val[cols])),
            "test_fit_on_train_plus_val": regression_metrics(yte, m_test.predict(test[cols])),
        }
        print(name, json.dumps(results[name]))
        return m_test

    run("baseline_median", lambda zb: make_median_baseline())
    run("baseline_ridge_time_borough_color", make_ridge_baseline)
    final_model = run("gradient_boosted_trees", lambda zb: make_gbt_pipeline(best_params, zb))

    # Reference experiments (kept in the report even though they are not the product).
    # Ablation: same params with the route-statistics feature switched the other way.
    flipped = {**best_params, "use_route_stats": not best_params["use_route_stats"]}
    flip_name = "on" if flipped["use_route_stats"] else "off"
    run(f"ref_gbt_route_stats_{flip_name}_ablation", lambda zb: make_gbt_pipeline(flipped, zb))
    run("ref_gbt_WITH_metered_distance_LEAKY_not_available_at_prediction_time",
        lambda zb: make_gbt_pipeline(best_params, zb, extra_numeric=("distance",)),
        extra=("distance",))

    # ---- bundle ------------------------------------------------------------
    trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    feats = final_model.named_steps["features"]
    bundle = {
        "pipeline": final_model,
        "model_version": config.MODEL_VERSION,
        "input_columns": config.INPUT_COLUMNS,
        "zone_borough": zb_final,
        "pickup_zones": feats.categories_["pickup_zone"],
        "dropoff_zones": feats.categories_["dropoff_zone"],
        "colors": feats.categories_["color"],
        "trained_at_utc": trained_at,
        "train_pickup_min": str(trainval["pickup"].min()),
        "train_pickup_max": str(trainval["pickup"].max()),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, config.MODEL_PATH)

    gbt_params = final_model.named_steps["model"].get_params()
    metrics = {
        "model_version": config.MODEL_VERSION,
        "training_date_utc": trained_at,
        "target": f"{config.TARGET} = (dropoff - pickup) in minutes",
        "random_seed": config.SEED,
        "sample_counts": {
            "raw_rows": validation["n_rows"],
            "after_cleaning": clean_report["n_clean"],
            "train": len(train), "validation": len(val), "test": len(test),
            "final_fit_train_plus_validation": len(trainval),
        },
        "split": {
            "method": "chronological by pickup time (70/15/15)",
            "train_pickup_range": [str(train["pickup"].min()), str(train["pickup"].max())],
            "validation_pickup_range": [str(val["pickup"].min()), str(val["pickup"].max())],
            "test_pickup_range": [str(test["pickup"].min()), str(test["pickup"].max())],
        },
        "cleaning": clean_report,
        "model_params": {"chosen_by_search": best_params,
                         "full_estimator_params": {k: v for k, v in gbt_params.items()
                                                   if not callable(v)}},
        "tuning": {
            "method": f"random search, {N_TUNING_CONFIGS} configs, "
                      f"TimeSeriesSplit({CV_SPLITS}) on the training split, scored by MAE",
            "search_space": SEARCH_SPACE,
            "best_cv_mae_mean": float(best_row["cv_mae_mean"]),
            "best_cv_mae_std": float(best_row["cv_mae_std"]),
            "full_log": "reports/metrics/tuning_results.csv",
        },
        "metric_protocol": ("validation = fit on train, score on validation; "
                            "test = refit on train+validation with the chosen params, "
                            "score on the untouched chronologically-last test split"),
        "results": results,
        "feature_docs": FEATURE_DOCS,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
    }
    (config.METRICS_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"saved {config.MODEL_PATH} and reports/metrics/model_metrics.json")
    return metrics


if __name__ == "__main__":
    main()

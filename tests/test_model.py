import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.inference import InputError, predict_one, predict_records
from src.preprocessing import build_zone_borough_map
from src.train import make_gbt_pipeline, regression_metrics


def test_bundle_loads_with_expected_contents(bundle):
    for key in ("pipeline", "model_version", "pickup_zones", "dropoff_zones", "colors",
                "zone_borough", "trained_at_utc"):
        assert key in bundle
    assert bundle["model_version"] == config.MODEL_VERSION
    assert bundle["input_columns"] == config.INPUT_COLUMNS


def test_prediction_output_is_finite_floats(bundle, splits):
    records = splits[2].head(25)[config.INPUT_COLUMNS].to_dict("records")
    preds = predict_records(records, bundle)
    assert len(preds) == 25
    assert all(isinstance(p, float) and np.isfinite(p) for p in preds)
    assert all(-10 < p < 200 for p in preds)


def test_single_prediction_and_unknown_zone(bundle):
    z = bundle["pickup_zones"][0]
    p = predict_one("2019-03-15 18:30", z, bundle["dropoff_zones"][0], 1, "yellow", bundle)
    assert np.isfinite(p)
    with pytest.raises(InputError):
        predict_one("2019-03-15 18:30", "Atlantis", z, 1, "yellow", bundle)


def test_model_beats_median_baseline_on_test(bundle, splits):
    test = splits[2]
    trainval = pd.concat(splits[:2], ignore_index=True)
    pred = bundle["pipeline"].predict(test[config.INPUT_COLUMNS])
    gbt = regression_metrics(test[config.TARGET], pred)
    median_mae = float(np.mean(np.abs(test[config.TARGET] - trainval[config.TARGET].median())))
    assert gbt["mae"] < median_mae


def test_saved_metrics_match_reevaluation(bundle, splits):
    saved = json.loads((config.METRICS_DIR / "model_metrics.json").read_text())
    pred = bundle["pipeline"].predict(splits[2][config.INPUT_COLUMNS])
    mae = regression_metrics(splits[2][config.TARGET], pred)["mae"]
    got = saved["results"]["gradient_boosted_trees"]["test_fit_on_train_plus_val"]["mae"]
    assert mae == pytest.approx(got, rel=1e-9)
    assert {"training_date_utc", "model_params", "sample_counts"} <= set(saved)


def test_training_is_deterministic(splits):
    train = splits[0].head(1500)
    zb = build_zone_borough_map(train)
    params = {"learning_rate": 0.1, "max_iter": 30, "max_leaf_nodes": 8, "min_samples_leaf": 20,
              "l2_regularization": 1.0, "loss": "squared_error", "use_route_stats": True}
    X, y = train[config.INPUT_COLUMNS], train[config.TARGET]
    a = make_gbt_pipeline(params, zb).fit(X, y).predict(X)
    b = make_gbt_pipeline(params, zb).fit(X, y).predict(X)
    np.testing.assert_allclose(a, b)

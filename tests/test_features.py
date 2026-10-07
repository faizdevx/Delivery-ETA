import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import TripFeatures
from src.preprocessing import build_zone_borough_map


@pytest.fixture(scope="module")
def fitted(splits):
    train = splits[0]
    zb = build_zone_borough_map(train)
    tf = TripFeatures(zone_borough=zb, use_route_stats=True)
    tf.fit(train[config.INPUT_COLUMNS], train[config.TARGET])
    return tf, train, zb


def test_time_features_known_timestamp(splits):
    train = splits[0]
    zb = build_zone_borough_map(train)
    X = train[config.INPUT_COLUMNS].head(1).copy()
    X["pickup"] = pd.Timestamp("2019-03-16 18:30:00")  # a Saturday
    out = TripFeatures(zone_borough=zb, use_route_stats=False).fit(X).transform(X)
    assert out["hour"].iloc[0] == pytest.approx(18.5)
    assert out["dayofweek"].iloc[0] == 5
    assert out["is_weekend"].iloc[0] == 1.0
    assert out["hour_sin"].iloc[0] == pytest.approx(np.sin(2 * np.pi * 18.5 / 24))


def test_cyclical_encoding_wraps_midnight(splits):
    train = splits[0]
    zb = build_zone_borough_map(train)
    X = train[config.INPUT_COLUMNS].head(2).copy()
    X["pickup"] = [pd.Timestamp("2019-03-12 23:59:00"), pd.Timestamp("2019-03-13 00:01:00")]
    out = TripFeatures(zone_borough=zb, use_route_stats=False).fit(X).transform(X)
    d = np.hypot(out.hour_sin[0] - out.hour_sin[1], out.hour_cos[0] - out.hour_cos[1])
    assert d < 0.02


def test_output_contains_no_leaky_columns(fitted):
    tf, train, _ = fitted
    out = tf.transform(train[config.INPUT_COLUMNS])
    for leaky in config.LEAKY_COLUMNS:
        assert leaky not in out.columns
    assert {"pu_te", "do_te", "route_te"} <= set(out.columns)


def test_unseen_zone_maps_to_missing(fitted):
    tf, train, _ = fitted
    X = train[config.INPUT_COLUMNS].head(3).copy()
    X["pickup_zone"] = "Atlantis"
    out = tf.transform(X)
    assert out["pickup_zone"].isna().all()
    assert out["pickup_borough"].isna().all()
    assert np.isfinite(out["route_te"]).all()


def test_oof_target_encoding_does_not_use_own_target(fitted):
    """Changing one training row's target must not change that row's own OOF encoding."""
    _, train, zb = fitted
    X = train[config.INPUT_COLUMNS].head(600)
    y = train[config.TARGET].head(600).to_numpy().copy()
    base = TripFeatures(zone_borough=zb).fit_transform(X, y)["route_te"].to_numpy()
    y2 = y.copy()
    y2[7] += 500.0
    changed = TripFeatures(zone_borough=zb).fit_transform(X, y2)["route_te"].to_numpy()
    assert changed[7] == pytest.approx(base[7])


def test_feature_building_is_deterministic(fitted):
    _, train, zb = fitted
    X, y = train[config.INPUT_COLUMNS], train[config.TARGET]
    a = TripFeatures(zone_borough=zb).fit_transform(X, y)
    b = TripFeatures(zone_borough=zb).fit_transform(X, y)
    pd.testing.assert_frame_equal(a, b)

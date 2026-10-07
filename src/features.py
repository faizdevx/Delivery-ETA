"""Feature engineering as a leakage-safe scikit-learn transformer.

Input columns (all known at pickup): pickup, pickup_zone, dropoff_zone,
passengers, color. See FEATURE_DOCS for source, formula, reason, leakage risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

from . import config

MISSING_KEY = "__missing__"

FEATURE_DOCS = [
    {"feature": "hour, hour_sin, hour_cos", "source": "pickup",
     "formula": "hour = h + min/60; sin/cos(2*pi*hour/24)",
     "reason": "traffic varies by time of day; cyclical encoding makes 23:59 close to 00:00",
     "leakage_risk": "none: pickup time is known at request time"},
    {"feature": "dayofweek, dow_sin, dow_cos, is_weekend", "source": "pickup",
     "formula": "dayofweek in 0..6; sin/cos(2*pi*d/7); is_weekend = dayofweek >= 5",
     "reason": "weekday/weekend traffic patterns", "leakage_risk": "none"},
    {"feature": "passengers", "source": "passengers",
     "formula": "as reported (driver-entered count)",
     "reason": "available when the trip starts; weak expected signal",
     "leakage_risk": "low: believed to be entered by the driver at pickup (Not verified here)"},
    {"feature": "pickup_zone, dropoff_zone", "source": "pickup_zone, dropoff_zone",
     "formula": "native categorical (missing stays missing)",
     "reason": "the only spatial information allowed (distance is leaky)",
     "leakage_risk": "dropoff zone must be known up front (a destination is required for an ETA)"},
    {"feature": "pickup_borough, dropoff_borough", "source": "pickup_zone, dropoff_zone",
     "formula": "zone->borough lookup learned from the training data",
     "reason": "coarser spatial grouping with more samples per level", "leakage_risk": "none"},
    {"feature": "color", "source": "color", "formula": "yellow | green categorical",
     "reason": "taxi type serves different areas", "leakage_risk": "none"},
    {"feature": "pu_te, do_te, route_te", "source": "pickup_zone, dropoff_zone + training target",
     "formula": "smoothed mean duration per pickup zone / dropoff zone / zone pair: "
                "(sum_y + m*prior)/(n + m), m=smoothing, prior = global mean (zone) or "
                "mean(pu_te, do_te) (route)",
     "reason": "historical route statistics, robust to sparse zone pairs via shrinkage",
     "leakage_risk": "target encoding leaks if a row sees its own target: training rows use "
                     "out-of-fold statistics; stats are fitted on training data only"},
]


class TripFeatures(BaseEstimator, TransformerMixin):
    """Builds model features from the raw input columns."""

    def __init__(
        self,
        zone_borough: dict | None = None,
        use_route_stats: bool = True,
        smoothing: float = 10.0,
        n_splits: int = 5,
        extra_numeric: tuple = (),
        random_state: int = config.SEED,
    ):
        self.zone_borough = zone_borough
        self.use_route_stats = use_route_stats
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.extra_numeric = extra_numeric
        self.random_state = random_state

    # ---- fitting -----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        zb = self.zone_borough or {}
        self.zone_borough_ = dict(zb)
        self.categories_ = {
            "pickup_zone": sorted(X["pickup_zone"].dropna().unique()),
            "dropoff_zone": sorted(X["dropoff_zone"].dropna().unique()),
            "pickup_borough": sorted(set(zb.values())),
            "dropoff_borough": sorted(set(zb.values())),
            "color": sorted(X["color"].dropna().unique()),
        }
        if self.use_route_stats:
            if y is None:
                raise ValueError("y is required when use_route_stats=True")
            self.stats_ = self._compute_stats(X, np.asarray(y, dtype=float))
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        X = pd.DataFrame(X).reset_index(drop=True)
        out = self._base(X)
        if self.use_route_stats:
            # Out-of-fold encoding so a training row never sees its own target.
            y_arr = np.asarray(y, dtype=float)
            te = pd.DataFrame(index=X.index, columns=["pu_te", "do_te", "route_te"], dtype=float)
            kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
            for tr_idx, ho_idx in kf.split(X):
                stats = self._compute_stats(X.iloc[tr_idx], y_arr[tr_idx])
                te.iloc[ho_idx] = self._apply_stats(X.iloc[ho_idx], stats).to_numpy()
            out[["pu_te", "do_te", "route_te"]] = te.to_numpy()
        return out

    def transform(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        out = self._base(X)
        if self.use_route_stats:
            out[["pu_te", "do_te", "route_te"]] = self._apply_stats(X, self.stats_).to_numpy()
        return out

    # ---- internals ---------------------------------------------------------
    def _base(self, X: pd.DataFrame) -> pd.DataFrame:
        pickup = pd.to_datetime(X["pickup"])
        out = pd.DataFrame(index=X.index)
        hour = pickup.dt.hour + pickup.dt.minute / 60.0
        dow = pickup.dt.dayofweek
        out["hour"] = hour.astype(float)
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        out["dayofweek"] = dow.astype(float)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        out["is_weekend"] = (dow >= 5).astype(float)
        out["passengers"] = X["passengers"].astype(float)
        for col in self.extra_numeric:
            out[col] = X[col].astype(float)
        raw = {
            "pickup_zone": X["pickup_zone"],
            "dropoff_zone": X["dropoff_zone"],
            "pickup_borough": X["pickup_zone"].map(self.zone_borough_),
            "dropoff_borough": X["dropoff_zone"].map(self.zone_borough_),
            "color": X["color"],
        }
        for col, values in raw.items():
            cats = self.categories_[col]
            # Values unseen at fit time become NaN (treated as missing by the model).
            out[col] = pd.Categorical(values.where(values.isin(cats)), categories=cats)
        return out

    @staticmethod
    def _keys(X: pd.DataFrame):
        pu = X["pickup_zone"].astype(object).where(X["pickup_zone"].notna(), MISSING_KEY)
        do = X["dropoff_zone"].astype(object).where(X["dropoff_zone"].notna(), MISSING_KEY)
        return pu.astype(str), do.astype(str), (pu.astype(str) + "->" + do.astype(str))

    def _compute_stats(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        pu, do, route = self._keys(X.reset_index(drop=True))
        ys = pd.Series(y)
        return {
            "global_mean": float(ys.mean()),
            "pu": ys.groupby(pu).agg(["sum", "count"]),
            "do": ys.groupby(do).agg(["sum", "count"]),
            "route": ys.groupby(route).agg(["sum", "count"]),
        }

    def _apply_stats(self, X: pd.DataFrame, stats: dict) -> pd.DataFrame:
        pu, do, route = self._keys(X.reset_index(drop=True))
        m, g = self.smoothing, stats["global_mean"]

        def enc(keys, table, prior):
            s = keys.map(table["sum"]).fillna(0.0)
            n = keys.map(table["count"]).fillna(0.0)
            return (s + m * prior) / (n + m)

        pu_te = enc(pu, stats["pu"], g)
        do_te = enc(do, stats["do"], g)
        route_te = enc(route, stats["route"], (pu_te + do_te) / 2.0)
        return pd.DataFrame({"pu_te": pu_te, "do_te": do_te, "route_te": route_te})

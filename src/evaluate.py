"""Evaluation, error analysis, permutation importance and figures."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from . import config  # noqa: E402
from .data import load_raw  # noqa: E402
from .inference import load_bundle  # noqa: E402
from .preprocessing import chronological_split, clean  # noqa: E402
from .train import make_median_baseline, make_ridge_baseline, regression_metrics  # noqa: E402

N_BOOT = 1000
COLOR = "#2a6f97"


def _save(fig, name: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return f"reports/figures/{name}"


def _group_mae(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    out = g.agg(n=("abs_error", "size"), mae=("abs_error", "mean"),
                mean_signed_error=("error", "mean"), mean_actual=("actual", "mean"))
    return out.reset_index()


def bootstrap_ci(y, pred, fn, n_boot=N_BOOT, seed=config.SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = [fn(y[s], pred[s]) for s in (rng.choice(idx, len(idx)) for _ in range(n_boot))]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main() -> dict:
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    model = bundle["pipeline"]
    cleaned, _ = clean(load_raw())
    train, val, test = chronological_split(cleaned)
    trainval = pd.concat([train, val], ignore_index=True)
    X, y = test[config.INPUT_COLUMNS], test[config.TARGET].to_numpy()
    pred = model.predict(X)

    # Baselines refit here with the same protocol as training (fit on train+val).
    zb = bundle["zone_borough"]
    base_median = make_median_baseline().fit(trainval[config.INPUT_COLUMNS],
                                             trainval[config.TARGET])
    base_ridge = make_ridge_baseline(zb).fit(trainval[config.INPUT_COLUMNS],
                                             trainval[config.TARGET])
    preds = {"baseline_median": base_median.predict(X),
             "baseline_ridge_time_borough_color": base_ridge.predict(X),
             "gradient_boosted_trees": pred}
    mae = lambda a, b: float(np.mean(np.abs(a - b)))  # noqa: E731
    metrics = {}
    for name, p in preds.items():
        metrics[name] = {**regression_metrics(y, p), "mae_95ci_bootstrap": bootstrap_ci(y, p, mae)}
    # Paired bootstrap of MAE improvement vs the median baseline.
    rng = np.random.default_rng(config.SEED)
    diffs = []
    for _ in range(N_BOOT):
        s = rng.choice(len(y), len(y))
        diffs.append(mae(y[s], preds["baseline_median"][s]) - mae(y[s], pred[s]))
    improvement = {"mae_reduction_vs_median_baseline": float(np.mean(
        np.abs(y - preds["baseline_median"])) - np.mean(np.abs(y - pred))),
        "paired_bootstrap_95ci": [float(np.percentile(diffs, 2.5)),
                                  float(np.percentile(diffs, 97.5))]}

    # ---- reproducibility check against training-time metrics ---------------
    saved = json.loads((config.METRICS_DIR / "model_metrics.json").read_text())
    saved_mae = saved["results"]["gradient_boosted_trees"]["test_fit_on_train_plus_val"]["mae"]
    repro = {"train_time_test_mae": saved_mae, "reevaluated_test_mae": metrics[
        "gradient_boosted_trees"]["mae"],
        "match": bool(abs(saved_mae - metrics["gradient_boosted_trees"]["mae"]) < 1e-9)}

    # ---- error analysis ----------------------------------------------------
    df = test.copy()
    df["actual"] = y
    df["predicted"] = pred
    df["error"] = df["predicted"] - df["actual"]  # >0 = over-prediction
    df["abs_error"] = df["error"].abs()
    df["hour_bucket"] = pd.cut(df["pickup"].dt.hour, [-1, 5, 9, 15, 19, 23],
                               labels=["00-05", "06-09", "10-15", "16-19", "20-23"])
    df["weekend"] = np.where(df["pickup"].dt.dayofweek >= 5, "weekend", "weekday")
    df["borough_pair"] = (df["pickup_borough"].fillna("?") + " -> "
                          + df["dropoff_borough"].fillna("?"))
    df["actual_bin"] = pd.cut(df["actual"], [0, 5, 10, 20, 40, 180],
                              labels=["<=5", "5-10", "10-20", "20-40", ">40"])
    df["dist_bin"] = pd.cut(df["distance"], [-0.01, 1, 2, 4, 8, 100],
                            labels=["<=1mi", "1-2", "2-4", "4-8", ">8"])
    cols = ["row_id", "pickup", "pickup_zone", "dropoff_zone", "color", "passengers",
            "distance", "actual", "predicted", "error"]
    df.sort_values("abs_error").head(10)[cols].to_csv(
        config.METRICS_DIR / "best_predictions.csv", index=False)
    df.sort_values("abs_error", ascending=False).head(20)[cols].to_csv(
        config.METRICS_DIR / "worst_predictions.csv", index=False)

    thresh = float(np.percentile(df["abs_error"], 95))
    hi = df[df["abs_error"] >= thresh]
    analysis: dict = {
        "n_test": int(len(df)),
        "mean_signed_error": float(df["error"].mean()),
        "n_overpredicted": int((df["error"] > 0).sum()),
        "n_underpredicted": int((df["error"] < 0).sum()),
        "abs_error_percentiles": {str(q): float(np.percentile(df["abs_error"], q))
                                  for q in (50, 75, 90, 95, 99)},
        "high_error_threshold_p95_minutes": thresh,
        "high_error_records": {
            "n": int(len(hi)),
            "share_underpredicted": float((hi["error"] < 0).mean()),
            "mean_actual_duration": float(hi["actual"].mean()),
            "overall_mean_actual_duration": float(df["actual"].mean()),
            "median_distance_miles": float(hi["distance"].median()),
            "overall_median_distance_miles": float(df["distance"].median()),
            "n_missing_zone": int(hi[["pickup_zone", "dropoff_zone"]].isna().any(axis=1).sum()),
            "share_missing_zone": float(
                hi[["pickup_zone", "dropoff_zone"]].isna().any(axis=1).mean()),
            "overall_share_missing_zone": float(
                df[["pickup_zone", "dropoff_zone"]].isna().any(axis=1).mean()),
        },
        "worst_20_note": "see reports/metrics/worst_predictions.csv; 'distance' is shown for "
                         "diagnosis only and is NOT a model input",
        "by_group": {},
    }
    # Diagnostic: implied speed (metered distance / duration) for the worst records.
    worst = df.sort_values("abs_error", ascending=False).head(20)
    analysis["worst_20_implied_speed_mph"] = {
        "median": float((worst["distance"] / (worst["actual"] / 60)).median()),
        "min": float((worst["distance"] / (worst["actual"] / 60)).min()),
        "max": float((worst["distance"] / (worst["actual"] / 60)).max()),
        "overall_median": float((df["distance"] / (df["actual"] / 60)).median()),
    }
    for by in ("hour_bucket", "weekend", "color", "borough_pair", "passengers",
               "actual_bin", "dist_bin"):
        analysis["by_group"][by] = json.loads(_group_mae(df, by).to_json(orient="records"))

    # ---- permutation importance on raw model inputs ------------------------
    pi = permutation_importance(model, X, y, scoring="neg_mean_absolute_error",
                                n_repeats=20, random_state=config.SEED)
    imp = pd.DataFrame({"feature": config.INPUT_COLUMNS,
                        "mae_increase": pi.importances_mean,
                        "std": pi.importances_std}).sort_values("mae_increase", ascending=False)
    imp.to_csv(config.METRICS_DIR / "permutation_importance.csv", index=False)

    # ---- figures -----------------------------------------------------------
    figs = []
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y, pred, s=8, alpha=0.4, color=COLOR)
    lim = max(y.max(), pred.max()) * 1.02
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect prediction")
    ax.set(xlabel="Actual duration (min)", ylabel="Predicted duration (min)",
           title="Actual vs predicted (test split)", xlim=(0, lim), ylim=(0, lim))
    ax.legend()
    figs.append(_save(fig, "actual_vs_predicted.png"))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(df["error"], bins=60, color=COLOR)
    ax.axvline(0, color="k", lw=1)
    ax.set(xlabel="Predicted - actual (min); >0 = over-prediction", ylabel="Trips",
           title="Residual distribution (test split)")
    figs.append(_save(fig, "residual_distribution.png"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    h = _group_mae(df, "hour_bucket")
    axes[0].bar(h["hour_bucket"].astype(str), h["mae"], color=COLOR)
    axes[0].set(xlabel="Pickup hour", ylabel="MAE (min)", title="MAE by pickup hour bucket")
    for i, (m, n) in enumerate(zip(h["mae"], h["n"], strict=True)):
        axes[0].text(i, m, f"n={n}", ha="center", va="bottom", fontsize=8)
    b = _group_mae(df, "borough_pair").query("n >= 20").sort_values("mae")
    axes[1].barh(b["borough_pair"], b["mae"], color=COLOR)
    axes[1].set(xlabel="MAE (min)", title="MAE by borough pair (n >= 20)")
    figs.append(_save(fig, "mae_by_group.png"))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    imp_plot = imp.iloc[::-1]
    ax.barh(imp_plot["feature"], imp_plot["mae_increase"], xerr=imp_plot["std"], color=COLOR)
    ax.set(xlabel="Increase in test MAE when column is shuffled (min)",
           title="Permutation importance (raw inputs)")
    figs.append(_save(fig, "permutation_importance.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    a = _group_mae(df, "actual_bin")
    ax.bar(a["actual_bin"].astype(str), a["mean_signed_error"], color=COLOR)
    ax.axhline(0, color="k", lw=1)
    ax.set(xlabel="Actual duration bin (min)", ylabel="Mean (predicted - actual)",
           title="Bias by actual duration")
    figs.append(_save(fig, "bias_by_duration.png"))

    out = {"split": "test (chronologically last 15%)", "metrics": metrics,
           "improvement_vs_median_baseline": improvement, "reproducibility_check": repro,
           "permutation_importance": json.loads(imp.to_json(orient="records")),
           "error_analysis": analysis, "figures": figs}
    (config.METRICS_DIR / "evaluation_metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"metrics": metrics, "improvement": improvement, "repro": repro}, indent=1))
    return out


if __name__ == "__main__":
    main()

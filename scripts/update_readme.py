"""Fill the generated blocks of README.md from reports/metrics/*.json and a live API call."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from src import config  # noqa: E402

README = config.ROOT / "README.md"
M = json.loads((config.METRICS_DIR / "model_metrics.json").read_text())
E = json.loads((config.METRICS_DIR / "evaluation_metrics.json").read_text())


def f(x, nd=2):
    return f"{x:.{nd}f}"


def results_block() -> str:
    names = {"baseline_median": "Baseline: median of training durations",
             "baseline_ridge_time_borough_color": "Baseline: ridge regression (time, borough, color)",
             "gradient_boosted_trees": "**Gradient Boosted Trees (HistGradientBoostingRegressor)**"}
    n_test = E["metrics"]["gradient_boosted_trees"]["n"]
    lines = [f"Test split (n={n_test}, chronologically last 15% of cleaned data). MAE and RMSE in minutes; "
             "MAE interval is a 95% percentile bootstrap (1000 resamples) over test trips.\n",
             "| Model | MAE | MAE 95% CI | RMSE | R² | Median AE |", "|---|---:|---:|---:|---:|---:|"]
    for k, label in names.items():
        m = E["metrics"][k]
        lo, hi = m["mae_95ci_bootstrap"]
        lines.append(f"| {label} | {f(m['mae'])} | {f(lo)} – {f(hi)} | {f(m['rmse'])} | "
                     f"{f(m['r2'], 3)} | {f(m['median_ae'])} |")
    imp = E["improvement_vs_median_baseline"]
    lo, hi = imp["paired_bootstrap_95ci"]
    lines.append(f"\nMAE reduction of the GBT vs. the median baseline: {f(imp['mae_reduction_vs_median_baseline'])}"
                 f" min (paired bootstrap 95% CI {f(lo)} – {f(hi)}).\n")
    r = M["results"]
    lines.append("Validation-split scores (model fit on train only; used for selection checks):\n")
    lines += ["| Model | Val MAE | Val RMSE | Val R² |", "|---|---:|---:|---:|"]
    for k, label in names.items():
        v = r[k]["validation_fit_on_train"]
        lines.append(f"| {label} | {f(v['mae'])} | {f(v['rmse'])} | {f(v['r2'], 3)} |")
    lines.append("\nReference experiments (same protocol; **not** the product):\n")
    lines += ["| Experiment | Test MAE | Test RMSE | Test R² |", "|---|---:|---:|---:|"]
    for k, v in r.items():
        if k.startswith("ref_"):
            t = v["test_fit_on_train_plus_val"]
            label = ("GBT + metered `distance` (LEAKY: unknown before the trip; shows what leakage would buy)"
                     if "LEAKY" in k else k.replace("ref_", "").replace("_", " "))
            lines.append(f"| {label} | {f(t['mae'])} | {f(t['rmse'])} | {f(t['r2'], 3)} |")
    p = M["model_params"]["chosen_by_search"]
    lines.append(f"\nChosen parameters: `{json.dumps(p)}`. Search: {M['tuning']['method']}; best CV MAE "
                 f"{f(M['tuning']['best_cv_mae_mean'])} ± {f(M['tuning']['best_cv_mae_std'])} "
                 f"(fold std). Full log: `reports/metrics/tuning_results.csv`. "
                 f"Trained {M['training_date_utc']} (UTC), seed {M['random_seed']}.")
    return "\n".join(lines)


def error_block() -> str:
    a = E["error_analysis"]
    g = a["by_group"]
    hi = a["high_error_records"]
    L = [f"All numbers from the test split (n={a['n_test']}); see `reports/metrics/evaluation_metrics.json`.\n",
         "- Absolute-error percentiles (min): " + ", ".join(
             f"p{q}={f(v)}" for q, v in a["abs_error_percentiles"].items()) + ".",
         f"- Over-predicted {a['n_overpredicted']} trips, under-predicted {a['n_underpredicted']}; "
         f"mean signed error (pred − actual) = {f(a['mean_signed_error'])} min.",
         f"- The worst 5% of errors (|error| ≥ {f(a['high_error_threshold_p95_minutes'])} min, n={hi['n']}) "
         f"have mean actual duration {f(hi['mean_actual_duration'])} min vs {f(hi['overall_mean_actual_duration'])} "
         f"overall, and {f(hi['share_underpredicted'] * 100, 0)}% of them are under-predictions. "
         f"{f(hi['share_missing_zone'] * 100, 0)}% have a missing zone vs "
         f"{f(hi['overall_share_missing_zone'] * 100, 0)}% overall.",
         f"- For the 20 worst records, the metered-distance / duration speed has median "
         f"{f(a['worst_20_implied_speed_mph']['median'], 1)} mph (min {f(a['worst_20_implied_speed_mph']['min'], 1)}, "
         f"max {f(a['worst_20_implied_speed_mph']['max'], 1)}); overall median "
         f"{f(a['worst_20_implied_speed_mph']['overall_median'], 1)} mph. (`distance` is used only for this diagnosis.)\n",
         "**Bias by actual duration** (positive = over-prediction):\n",
         "| Actual duration (min) | n | MAE | Mean (pred − actual) |", "|---|---:|---:|---:|"]
    for r in g["actual_bin"]:
        L.append(f"| {r['actual_bin']} | {r['n']} | {f(r['mae'])} | {f(r['mean_signed_error'])} |")
    for key, title in (("hour_bucket", "Pickup hour"), ("color", "Taxi color"), ("weekend", "Day type")):
        L += [f"\n**MAE by {title.lower()}:**\n", f"| {title} | n | MAE | Mean (pred − actual) |", "|---|---:|---:|---:|"]
        for r in g[key]:
            L.append(f"| {r[key]} | {r['n']} | {f(r['mae'])} | {f(r['mean_signed_error'])} |")
    L += ["\n**Worst 8 test predictions** (`distance` shown for diagnosis only, not a model input):\n",
          "| Pickup | Pickup zone | Dropoff zone | Distance (mi) | Actual | Predicted |", "|---|---|---|---:|---:|---:|"]
    w = pd.read_csv(config.METRICS_DIR / "worst_predictions.csv").head(8)
    for _, r in w.iterrows():
        L.append(f"| {r['pickup']} | {r['pickup_zone']} | {r['dropoff_zone']} | {f(r['distance'])} | "
                 f"{f(r['actual'], 1)} | {f(r['predicted'], 1)} |")
    L.append("\nFigures: `reports/figures/` (actual_vs_predicted, residual_distribution, mae_by_group, "
             "permutation_importance, bias_by_duration).")
    return "\n".join(L)


def importance_block() -> str:
    L = ["Permutation importance on the test split (increase in MAE, minutes, when one input column is "
         "shuffled; mean ± std over 20 repeats):\n", "| Input column | MAE increase |", "|---|---:|"]
    for r in E["permutation_importance"]:
        L.append(f"| `{r['feature']}` | {f(r['mae_increase'], 3)} ± {f(r['std'], 3)} |")
    return "\n".join(L)


def api_block() -> str:
    body = {"pickup_datetime": "2019-03-15T18:30:00", "pickup_zone": "Midtown Center",
            "dropoff_zone": "JFK Airport", "passengers": 1, "taxi_color": "yellow"}
    with TestClient(app) as c:
        resp = c.post("/predict", json=body)
    assert resp.status_code == 200, resp.text
    curl = ("curl -s -X POST http://localhost:8000/predict -H 'content-type: application/json' \\\n"
            f"  -d '{json.dumps(body)}'")
    return f"```bash\n{curl}\n```\n\nResponse (captured from this repository's model):\n\n" \
           f"```json\n{json.dumps(resp.json(), indent=2)}\n```"


def replace(text: str, tag: str, content: str) -> str:
    pat = re.compile(rf"(<!-- {tag}:START -->).*?(<!-- {tag}:END -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"marker {tag} missing in README.md")
    return pat.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(2)}", text)


def main() -> None:
    text = README.read_text()
    for tag, fn in (("RESULTS", results_block), ("ERRORS", error_block),
                    ("IMPORTANCE", importance_block), ("API_EXAMPLE", api_block)):
        text = replace(text, tag, fn())
    README.write_text(text)
    print("README.md updated")


if __name__ == "__main__":
    main()

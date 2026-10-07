"""Build and execute notebooks/01_eda.ipynb. All findings are printed by code from the data."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]

cells = []
md = lambda t: cells.append(new_markdown_cell(t))  # noqa: E731
code = lambda t: cells.append(new_code_cell(t))  # noqa: E731

md("""# Exploratory data analysis: Bharat Delivery ETA

Dataset: `taxis.csv` from `mwaskom/seaborn-data`, a sample of NYC TLC trip records (see `DATASET.md`
for provenance and what is *not verified*). Every number below is computed by the cell that prints it.
Observations are printed by code, not typed by hand.""")
code("""import sys, warnings
sys.path.insert(0, '..')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
%matplotlib inline
from src import config
from src.data import load_raw, validate_raw
from src.preprocessing import add_target, clean
plt.rcParams.update({'figure.figsize': (7, 4), 'axes.grid': True, 'grid.alpha': .3})
raw = load_raw()           # verifies the pinned SHA-256
df = add_target(raw)
T = config.TARGET""")
md("## 1. Shape, types, validation")
code("""print('shape:', raw.shape)
print(raw.dtypes.to_string())
import json; print(json.dumps(validate_raw(raw), indent=2))""")
md("## 2. Missing values and duplicates")
code("""miss = raw.isna().sum(); miss = miss[miss > 0]
print((pd.DataFrame({'n_missing': miss, 'pct': (miss / len(raw) * 100).round(2)})).to_string())
print('exact duplicate rows:', raw.duplicated().sum())
print('rows with missing pickup_zone but present borough:', (raw.pickup_zone.isna() & raw.pickup_borough.notna()).sum())
print('missing zone <=> missing borough (pickup):', ((raw.pickup_zone.isna()) == (raw.pickup_borough.isna())).all())
print('missing payment rows: mean fare', raw[raw.payment.isna()].fare.mean().round(2), 'vs others', raw[raw.payment.notna()].fare.mean().round(2))""")
md("## 3. Target: duration in minutes = (dropoff - pickup)")
code("""print(df[T].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(2).to_string())
print('skew:', round(df[T].skew(), 2))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(df[T], bins=80); ax[0].set(title='Duration (min)', xlabel='minutes')
ax[1].hist(np.log1p(df[T]), bins=80); ax[1].set(title='log1p(duration)', xlabel='log1p(min)')
plt.show()""")
md("## 4. Outliers and suspicious records")
code("""q1, q3 = df[T].quantile([.25, .75]); iqr = q3 - q1; hi = q3 + 1.5 * iqr
print(f'IQR rule upper fence: {hi:.1f} min; trips above: {(df[T] > hi).sum()} ({(df[T] > hi).mean()*100:.1f}%)')
print('duration <= 0:', (df[T] <= 0).sum(), '| duration < 1 min:', (df[T] < 1).sum(), '| > 180 min:', (df[T] > 180).sum())
print('distance == 0:', (df.distance == 0).sum(), '| of which duration > 5 min:', ((df.distance == 0) & (df[T] > 5)).sum())
df['speed_mph'] = df.distance / (df[T] / 60).replace(0, np.nan)
print(df.speed_mph.describe(percentiles=[.01, .5, .99]).round(1).to_string())
print('implied speed > 60 mph:', (df.speed_mph > 60).sum())
print('passengers == 0:', (df.passengers == 0).sum())
print('\\nSample suspicious rows (duration<=0 or speed>60):')
print(df[(df[T] <= 0) | (df.speed_mph > 60)][['pickup', 'dropoff', 'distance', 'fare', T, 'speed_mph']].head(10).to_string())""")
md("## 5. Numerical feature distributions")
code("""num = ['passengers', 'distance', 'fare', 'tip', 'tolls', 'total']
print(df[num].describe().round(2).to_string())
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for a, c in zip(axes.ravel(), num):
    a.hist(df[c], bins=50); a.set_title(c)
plt.tight_layout(); plt.show()""")
md("## 6. Categorical frequencies")
code("""for c in ['color', 'payment', 'pickup_borough', 'dropoff_borough']:
    print(df[c].value_counts(dropna=False).to_string(), '\\n')
print('distinct pickup zones:', df.pickup_zone.nunique(), '| dropoff zones:', df.dropoff_zone.nunique())
print(df.pickup_zone.value_counts().head(10).to_string())
pairs = df.dropna(subset=['pickup_zone','dropoff_zone']).groupby(['pickup_zone','dropoff_zone']).size()
print('\\ndistinct zone pairs:', len(pairs), '| pairs seen exactly once:', (pairs == 1).sum(), f'({(pairs == 1).mean()*100:.0f}%)')""")
md("## 7. Correlation (Pearson, numeric columns) — includes post-trip columns for diagnosis only")
code("""c = df[num + [T]].corr().round(2)
print(c[T].sort_values(ascending=False).to_string())
print('\\nNOTE: distance, fare, tip, tolls and total are only known after the trip and are NOT model inputs.')""")
md("## 8. Target vs. major features")
code("""print(df.groupby('color')[T].agg(['count', 'mean', 'median']).round(2).to_string())
print(df.groupby('pickup_borough')[T].agg(['count', 'mean', 'median']).round(2).to_string())
print(df.groupby('passengers')[T].agg(['count', 'mean', 'median']).round(2).to_string())
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(df.distance, df[T], s=4, alpha=.4); ax[0].set(xlabel='distance (mi) [post-trip, diagnostic only]', ylabel='duration (min)')
bp = df.dropna(subset=['pickup_borough', 'dropoff_borough']).assign(pair=lambda d: d.pickup_borough + '>' + d.dropoff_borough)
top = bp.pair.value_counts().loc[lambda s: s >= 30].index
bp[bp.pair.isin(top)].boxplot(column=T, by='pair', ax=ax[1], rot=60); ax[1].set(title='duration by borough pair (n>=30)', xlabel='')
plt.suptitle(''); plt.tight_layout(); plt.show()""")
md("## 9. Time patterns")
code("""df['hour'] = df.pickup.dt.hour; df['dow'] = df.pickup.dt.dayofweek; df['date'] = df.pickup.dt.date
print('pickup range:', df.pickup.min(), '->', df.pickup.max(), '| distinct dates:', df.date.nunique())
print('rows dated before 2019-03-01:', (df.pickup < '2019-03-01').sum())
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
df.groupby('hour')[T].median().plot(ax=ax[0], marker='o', title='median duration by pickup hour')
df.groupby('hour').size().plot.bar(ax=ax[1], title='trips by pickup hour')
df.groupby('dow')[T].median().plot.bar(ax=ax[2], title='median duration by day of week (0=Mon)')
plt.tight_layout(); plt.show()
print(df.groupby('dow')[T].agg(['count', 'median']).round(2).to_string())""")
md("## 10. Geographic view (zone names only; the file has no coordinates)")
code("""z = df.dropna(subset=['pickup_zone']).groupby('pickup_zone')[T].agg(['count', 'median']).query('count >= 30').sort_values('median')
print('pickup zones with >=30 trips:', len(z)); print(z.head(5).round(1).to_string()); print(z.tail(5).round(1).to_string())
print('\\nshare of trips staying within one borough:', round((df.pickup_borough == df.dropoff_borough).mean() * 100, 1), '%')""")
md("## 11. Leakage review of every column")
code("""review = pd.DataFrame([
 ('pickup', 'start timestamp', 'yes', 'INPUT (time features)'),
 ('dropoff', 'end timestamp', 'NO', 'REJECTED: defines the target'),
 ('passengers', 'driver-entered count', 'yes', 'INPUT'),
 ('distance', 'metered trip distance', 'NO', 'REJECTED: measured over the completed trip'),
 ('fare', 'metered fare', 'NO', 'REJECTED: function of distance and time travelled'),
 ('tip', 'tip', 'NO', 'REJECTED: paid at trip end'),
 ('tolls', 'tolls', 'NO', 'REJECTED: depends on route taken'),
 ('total', 'sum of charges', 'NO', 'REJECTED: post-trip'),
 ('color', 'taxi type', 'yes', 'INPUT'),
 ('payment', 'payment type', 'NO', 'REJECTED: recorded at trip end'),
 ('pickup_zone', 'start zone', 'yes', 'INPUT'),
 ('dropoff_zone', 'end zone', 'yes*', 'INPUT (*assumes destination is given when requesting an ETA)'),
 ('pickup_borough/dropoff_borough', 'derived from zones', 'yes', 'derived inside the model from zones'),
], columns=['column', 'meaning', 'known at prediction time', 'decision'])
print(review.to_string(index=False))
assert set(config.INPUT_COLUMNS) == {'pickup', 'passengers', 'color', 'pickup_zone', 'dropoff_zone'}""")
md("## 12. Data-quality findings and cleaning effect")
code("""cleaned, rep = clean(raw)
print(json.dumps(rep, indent=2))
print('target mean/median before cleaning:', df[T].mean().round(3), df[T].median().round(3))
print('target mean/median after  cleaning:', cleaned[T].mean().round(3), cleaned[T].median().round(3))
print('rows kept with duration < 1 min (not excluded):', (cleaned[T] < 1).sum())
print('rows kept with missing zone(s):', cleaned[['pickup_zone', 'dropoff_zone']].isna().any(axis=1).sum())""")

nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = ROOT / "notebooks" / "01_eda.ipynb"
NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
nbformat.write(nb, out)
print("wrote", out)

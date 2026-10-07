# DATASET.md — provenance, definitions, limits

Statements below are either (a) quoted from a source I could reach, (b) produced by code in this
repository (`notebooks/01_eda.ipynb`, `reports/metrics/*.json`), or (c) marked **Not verified**.

## Identity

| Item | Value |
|---|---|
| Dataset name | `taxis` (file `taxis.csv`) from the `seaborn-data` repository |
| File host / secondary source | GitHub repository `mwaskom/seaborn-data` |
| File URL used | https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv |
| Original publisher (as attributed by the host) | New York City Taxi & Limousine Commission (TLC), "TLC Trip Record Data" |
| Original URL (as attributed by the host) | https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page (also published at https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| Retrieval date | 2026-10-07 (UTC) |
| File size / SHA-256 | 869,349 bytes / `08d6d71784dbaa2651fee37fc03389754194c05d72d2d19cbc2c799dea6ac09d` |
| License | **Not verified.** The `seaborn-data` repository shows no license file for this dataset, and I could not reach the TLC site from the build environment to read its terms. For that reason the raw file is **not** committed to this repository; `scripts/download_data.py` fetches it. |

### How the source was verified, and what that does and does not establish

- The `seaborn-data` README lists `taxis` with the TLC page above as its source. It also states
  "Some of the datasets have also been modifed [sic] from their canonical sources", and documents **no**
  modifications and **no** sampling method for `taxis`. So how these 6,433 rows were selected or altered is
  **Not verified**.
- I could **not** download the original TLC files: this environment's network policy blocked
  `d37ci6vzurychx.cloudfront.net`, `nyc.gov`, `data.cityofnewyork.us`, UCI, Zenodo, Kaggle and others (HTTP 403 /
  connection failure). I did not work around the block. Consequently **no row was cross-checked against an
  original TLC file**.
- Consistency checks run on the file itself (all in `notebooks/01_eda.ipynb`): zone names and a one-to-one
  zone→borough mapping (zero zones map to more than one borough); timestamps parse and the dropoff is never earlier
  than the pickup; the metered `fare` correlates strongly with `distance` (Pearson 0.92 on the raw file); the
  data contains messy real-looking artefacts (66 trips under 1 minute, 11 trips with implied speed over 60 mph,
  zero-passenger records, missing zones) that a hand-built simulation is unlikely to contain. These checks are
  *consistent with* recorded trip data; they are **not proof** of provenance.
- Rejected candidate: Kaggle "Food Delivery Time Prediction", which Kaggle describes as simulated.
- Not used: any Kaggle re-upload, any generated data.

If you can reach the original TLC trip records, the better path is to rebuild the pipeline on them; the
schema here (pickup/dropoff timestamps, zones, passengers) is a subset of those records, but column mapping
would have to be redone and **has not been tested**.

## Target

`duration_min = (dropoff − pickup)` in minutes, `(dropoff − pickup).total_seconds() / 60`, derived from the
file's `pickup` and `dropoff` columns (no duration field is provided). The timestamps carry no timezone;
treated as naive New York wall-clock time (**Not verified**; the file does not state it). A difference of two
timestamps in the same zone is unaffected by the offset except across a DST change; no trip in the file spans the
2019-03-10 02:00 spring-forward gap (checked in code).

**This is taxi trip duration, not food/parcel delivery time.** No real delivery or Indian dataset could be
verified and downloaded, so this project demonstrates ETA regression on taxi trips as a stand-in.

## Columns

Meanings are taken from column names and common TLC conventions; the TLC data dictionary was **not** read here.

| Column | Type | Meaning | Use in model |
|---|---|---|---|
| `pickup` | timestamp | trip start | input (time features) |
| `dropoff` | timestamp | trip end | **rejected**: defines the target |
| `passengers` | int, 0–6 | passenger count | input |
| `distance` | float | trip distance (miles assumed; **Not verified**) | **rejected**: measured over the finished trip |
| `fare` | float | metered fare | **rejected**: derived from distance/time |
| `tip` | float | tip | **rejected**: paid at trip end |
| `tolls` | float | tolls | **rejected**: depends on route taken |
| `total` | float | total charge | **rejected**: post-trip |
| `color` | yellow / green | taxi type | input |
| `payment` | credit card / cash / missing | payment type | **rejected**: recorded at trip end |
| `pickup_zone`, `dropoff_zone` | text | TLC taxi zone names | inputs |
| `pickup_borough`, `dropoff_borough` | text | borough of the zone | not read from file; re-derived from zones inside the model |

`distance` is used only in a clearly labelled leaky reference experiment and in error diagnosis, never in the
shipped model. `dropoff_zone` is treated as known at prediction time, which assumes the destination is
given when an ETA is requested.

## Scope (all from the data)

- Rows: 6,433; 14 columns; no exact duplicate rows.
- Temporal: pickups from 2019-02-28 23:29:03 to 2019-03-31 23:43:45 (32 distinct dates; one row before March).
- Geographic: New York City only. Pickup boroughs: Manhattan 5,268, Queens 657, Brooklyn 383, Bronx 99, missing
  26. Dropoff boroughs additionally include Staten Island (2). 194 distinct pickup zones, 203 dropoff zones.
  **Nothing here covers India or any other region.**
- Taxi type: yellow 5,451, green 982.
- Target (all rows): median 10.9 min, mean 14.35, 99th percentile 57.6, max 107.7.

## Preprocessing performed (`src/preprocessing.py`)

1. Verify SHA-256 and schema (`src/data.py`).
2. Parse timestamps; compute the target.
3. Exclude exact duplicates (0 found), durations ≤ 0 or > 180 min (6 excluded, all ≤ 0), passengers outside 0–6 (0).
   Result: 6,427 rows.
4. Sort by pickup time; chronological split 70/15/15 → 4,498 train / 964 validation / 965 test.
5. Missing zones (44 rows remain with ≥1 missing zone) are **kept** and passed as missing values.
6. Features are built inside a scikit-learn transformer fitted on training data only (`src/features.py`).

## Known limitations

- Roughly one month of data; weekly/seasonal/holiday behaviour is unobserved. 56% of the zone pairs in the file
  occur exactly once, so most routes are rare.
- 66 retained trips last under 1 minute and 11 have implied speeds over 60 mph (see the notebook); these are
  likely recording artefacts but were kept because only non-positive durations are excluded.
- The sample's selection method is unknown, so it may not represent all TLC trips.
- No weather, traffic, or route data.

## Citation

No formal citation is provided by either source in what I could access. Suggested attribution:

- New York City Taxi & Limousine Commission, *TLC Trip Record Data*, https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page (original publisher as attributed by the host; accessed through the sample below).
- `mwaskom/seaborn-data`, file `taxis.csv`, https://github.com/mwaskom/seaborn-data (retrieved 2026-10-07).

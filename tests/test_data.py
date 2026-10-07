import pytest

from src import config
from src.data import DataValidationError, load_raw, validate_raw
from src.preprocessing import add_target, build_zone_borough_map, chronological_split, clean


def test_raw_file_matches_pinned_checksum(raw):
    assert len(raw) == 6433
    assert list(raw.columns) == config.RAW_COLUMNS


def test_load_raw_rejects_modified_file(tmp_path, raw):
    p = tmp_path / "taxis.csv"
    raw.head(50).to_csv(p, index=False)
    with pytest.raises(DataValidationError, match="SHA-256"):
        load_raw(p)


def test_load_raw_missing_file_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_raw(tmp_path / "nope.csv")


def test_validate_raw_report(raw):
    report = validate_raw(raw)
    assert report["n_rows"] == len(raw)
    assert report["n_exact_duplicate_rows"] == 0
    assert report["missing_values"]["pickup_zone"] == 26


def test_validate_raw_missing_column(raw):
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_raw(raw.drop(columns=["dropoff"]))


def test_validate_raw_bad_timestamp(raw):
    bad = raw.head(20).copy()
    bad.loc[0, "pickup"] = "not a date"
    with pytest.raises(DataValidationError, match="timestamps"):
        validate_raw(bad)


def test_target_is_dropoff_minus_pickup_in_minutes(raw):
    t = add_target(raw.iloc[[0]])[config.TARGET].iloc[0]
    assert t == pytest.approx(6.25)  # 20:21:09 -> 20:27:24


def test_clean_excludes_invalid_and_is_sorted(raw):
    report = validate_raw(raw)
    cleaned, rep = clean(raw)
    assert rep["excluded_invalid_duration"] == report["n_dropoff_not_after_pickup"]
    assert len(cleaned) == len(raw) - sum(v for k, v in rep.items() if k.startswith("excluded_"))
    assert (cleaned[config.TARGET] > 0).all()
    assert cleaned["pickup"].is_monotonic_increasing


def test_split_is_chronological_and_disjoint(splits, cleaned):
    train, val, test = splits
    assert len(train) + len(val) + len(test) == len(cleaned[0])
    assert train["pickup"].max() <= val["pickup"].min()
    assert val["pickup"].max() <= test["pickup"].min()
    assert not set(train.row_id) & set(test.row_id)


def test_zone_borough_map_is_a_function(cleaned):
    zb = build_zone_borough_map(cleaned[0])
    assert zb["JFK Airport"] == "Queens"
    assert len(zb) > 100


def test_leaky_columns_not_in_model_inputs():
    assert not set(config.INPUT_COLUMNS) & set(config.LEAKY_COLUMNS)
    for col in ("dropoff", "distance", "fare", "tip", "tolls", "total", "payment"):
        assert col not in config.INPUT_COLUMNS


def test_chronological_split_requires_sorted(raw):
    shuffled = add_target(raw).sample(frac=1.0, random_state=0)
    with pytest.raises(ValueError, match="sorted"):
        chronological_split(shuffled)

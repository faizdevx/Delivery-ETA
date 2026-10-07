import pytest

from src import config
from src.data import load_raw
from src.preprocessing import chronological_split, clean


@pytest.fixture(scope="session")
def raw():
    if not config.RAW_PATH.exists():
        pytest.fail(f"{config.RAW_PATH} missing: run `python scripts/download_data.py` first")
    return load_raw()


@pytest.fixture(scope="session")
def cleaned(raw):
    return clean(raw)


@pytest.fixture(scope="session")
def splits(cleaned):
    return chronological_split(cleaned[0])


@pytest.fixture(scope="session")
def bundle():
    if not config.MODEL_PATH.exists():
        pytest.fail("models/model.joblib missing: run `python scripts/train_model.py` first")
    from src.inference import load_bundle
    return load_bundle()

import pytest
from fastapi.testclient import TestClient

from app.main import app

VALID = {"pickup_datetime": "2019-03-15T18:30:00", "pickup_zone": "Midtown Center",
         "dropoff_zone": "JFK Airport", "passengers": 1, "taxi_color": "yellow"}


@pytest.fixture(scope="module")
def client(bundle):
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "gbt-v1"}


def test_root_json_and_html(client):
    assert client.get("/").json()["service"] == "Bharat Delivery ETA"
    html = client.get("/", headers={"accept": "text/html"})
    assert html.status_code == 200 and "<form" in html.text


def test_options(client):
    o = client.get("/options").json()
    assert "Midtown Center" in o["pickup_zones"] and o["taxi_colors"] == ["green", "yellow"]


def test_predict_success(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"predicted_trip_duration_minutes", "units", "model_version"}
    assert body["units"] == "minutes" and body["model_version"] == "gbt-v1"
    assert 0 < body["predicted_trip_duration_minutes"] < 200


@pytest.mark.parametrize("patch", [
    {"passengers": 9}, {"passengers": -1}, {"taxi_color": "blue"},
    {"pickup_datetime": "yesterday"}, {"pickup_datetime": "2019-03-15T18:30:00+05:30"},
    {"extra_field": 1}, {"pickup_zone": ""},
])
def test_predict_validation_errors(client, patch):
    assert client.post("/predict", json={**VALID, **patch}).status_code == 422


def test_predict_missing_field(client):
    body = {k: v for k, v in VALID.items() if k != "dropoff_zone"}
    assert client.post("/predict", json=body).status_code == 422


def test_predict_unknown_zone_is_422(client):
    r = client.post("/predict", json={**VALID, "pickup_zone": "Atlantis"})
    assert r.status_code == 422 and "unknown pickup_zone" in r.json()["detail"]


def test_model_missing_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=VALID).status_code == 503

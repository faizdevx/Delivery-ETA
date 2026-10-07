"""FastAPI service: GET /, GET /health, GET /options, POST /predict, web UI at / (browsers)."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src import config
from src.inference import InputError, load_bundle, predict_one

from .schemas import HealthResponse, PredictRequest, PredictResponse

log = logging.getLogger("bharat_eta")
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.environ.get("MODEL_PATH", str(config.MODEL_PATH))
    try:
        app.state.bundle = load_bundle(path)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        app.state.bundle = None
    yield


app = FastAPI(title="Bharat Delivery ETA", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _bundle(request: Request) -> dict:
    bundle = request.app.state.bundle
    if bundle is None:
        raise HTTPException(503, "model not loaded; run `python scripts/train_model.py`")
    return bundle


@app.get("/", include_in_schema=False)
def root(request: Request):
    """Browsers (Accept: text/html) get the web UI; API clients get JSON metadata."""
    if "text/html" in request.headers.get("accept", ""):
        return FileResponse(STATIC_DIR / "index.html")
    bundle = request.app.state.bundle
    return JSONResponse({
        "service": "Bharat Delivery ETA",
        "description": "Predicts NYC taxi trip duration in minutes (proxy for ETA).",
        "model_version": bundle["model_version"] if bundle else None,
        "endpoints": ["GET /health", "GET /options", "POST /predict", "GET /docs"],
    })


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    bundle = request.app.state.bundle
    if bundle is None:
        return JSONResponse(HealthResponse(status="model_not_loaded").model_dump(), status_code=503)
    return HealthResponse(status="ok", model_version=bundle["model_version"])


@app.get("/options")
def options(request: Request):
    """Valid categorical values and the training time range (used by the UI)."""
    b = _bundle(request)
    return {"pickup_zones": b["pickup_zones"], "dropoff_zones": b["dropoff_zones"],
            "taxi_colors": b["colors"], "passengers": list(config.PASSENGERS_RANGE),
            "training_pickup_range": [b["train_pickup_min"], b["train_pickup_max"]]}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    bundle = _bundle(request)
    try:
        minutes = predict_one(body.pickup_datetime, body.pickup_zone, body.dropoff_zone,
                              body.passengers, body.taxi_color, bundle)
    except InputError as exc:
        raise HTTPException(422, str(exc)) from exc
    return PredictResponse(predicted_trip_duration_minutes=round(minutes, 2),
                           model_version=bundle["model_version"])

"""Pydantic request/response models. Fields mirror the model's actual inputs."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "pickup_datetime": "2019-03-15T18:30:00",
        "pickup_zone": "Midtown Center",
        "dropoff_zone": "JFK Airport",
        "passengers": 1,
        "taxi_color": "yellow",
    }})

    pickup_datetime: datetime = Field(
        description="Pickup wall-clock time in New York local time, no timezone offset.")
    pickup_zone: str = Field(min_length=1, description="TLC taxi zone name seen in training data.")
    dropoff_zone: str = Field(min_length=1, description="TLC taxi zone name seen in training data.")
    passengers: int = Field(ge=0, le=6, description="Passenger count (0-6, as in training data).")
    taxi_color: Literal["yellow", "green"]

    @field_validator("pickup_datetime")
    @classmethod
    def _naive_only(cls, v: datetime) -> datetime:
        if v.tzinfo is not None:
            raise ValueError("send local New York time without a timezone offset")
        return v


class PredictResponse(BaseModel):
    predicted_trip_duration_minutes: float
    units: Literal["minutes"] = "minutes"
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "model_not_loaded"]
    model_version: str | None = None

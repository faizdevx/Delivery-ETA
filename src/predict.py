"""CLI: python -m src.predict --pickup "2019-03-15 18:30" --pickup-zone ... """
from __future__ import annotations

import argparse
import json
import sys

from .inference import InputError, load_bundle, predict_one


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Predict taxi trip duration (minutes).")
    ap.add_argument("--pickup", help="pickup local time, e.g. '2019-03-15 18:30'")
    ap.add_argument("--pickup-zone")
    ap.add_argument("--dropoff-zone")
    ap.add_argument("--passengers", type=int, default=1)
    ap.add_argument("--color", default="yellow")
    ap.add_argument("--list-zones", action="store_true", help="print valid zone names")
    args = ap.parse_args(argv)

    bundle = load_bundle()
    if args.list_zones:
        print("pickup zones:", *bundle["pickup_zones"], sep="\n  ")
        print("dropoff zones:", *bundle["dropoff_zones"], sep="\n  ")
        return 0
    if not (args.pickup and args.pickup_zone and args.dropoff_zone):
        ap.error("--pickup, --pickup-zone and --dropoff-zone are required")
    try:
        minutes = predict_one(args.pickup, args.pickup_zone, args.dropoff_zone,
                              args.passengers, args.color, bundle)
    except (InputError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"predicted_trip_duration_minutes": round(minutes, 2),
                      "units": "minutes", "model_version": bundle["model_version"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

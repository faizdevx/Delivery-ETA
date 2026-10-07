"""Download the raw dataset and verify its integrity.

Usage:
    python scripts/download_data.py              # download (or verify existing)
    python scripts/download_data.py --verify-only  # check a manually placed file

The raw file is saved byte-for-byte as received. The script never generates or
substitutes data: on any failure it exits non-zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

MANUAL_INSTRUCTIONS = f"""
Manual acquisition (if the download is blocked, e.g. by a network policy):
  1. Obtain the file from {config.DATASET_URL}
     using any machine that can reach it.
  2. Place it, unmodified, at {config.RAW_PATH}
  3. Run: python scripts/download_data.py --verify-only
The SHA-256 must equal {config.DATASET_SHA256}.
"""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str, code: int = 1) -> NoReturn:
    print(f"ERROR: {msg}", file=sys.stderr)
    print(MANUAL_INSTRUCTIONS, file=sys.stderr)
    sys.exit(code)


def report(path: Path, digest: str, status: str) -> None:
    print(f"status:   {status}")
    print(f"source:   {config.DATASET_URL}")
    print(f"file:     {path}")
    print(f"size:     {path.stat().st_size} bytes")
    print(f"sha256:   {digest}")


def verify_existing() -> str | None:
    if not config.RAW_PATH.exists():
        return None
    digest = sha256_of(config.RAW_PATH)
    if digest != config.DATASET_SHA256:
        fail(
            f"{config.RAW_PATH} exists but its SHA-256 ({digest}) does not match the "
            f"pinned {config.DATASET_SHA256}. Delete it or investigate before using it."
        , 2)
    return digest


def download() -> None:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    part = config.RAW_PATH.with_suffix(".csv.part")
    try:
        with urllib.request.urlopen(config.DATASET_URL, timeout=60) as resp:
            if resp.status != 200:
                fail(f"HTTP {resp.status} from {config.DATASET_URL}")
            data = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        fail(f"download failed: {exc}")
    part.write_bytes(data)
    digest = sha256_of(part)
    if digest != config.DATASET_SHA256:
        part.unlink()
        fail(
            f"checksum mismatch: got {digest}, expected {config.DATASET_SHA256}. "
            "Upstream file may have changed; the download was discarded.", 2
        )
    part.replace(config.RAW_PATH)
    meta = {
        "dataset": "taxis.csv (seaborn-data sample of NYC TLC trip records)",
        "url": config.DATASET_URL,
        "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "size_bytes": config.RAW_PATH.stat().st_size,
        "sha256": digest,
    }
    config.SOURCE_META_PATH.write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true",
                        help="only verify a file already placed in data/raw/")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    if args.verify_only:
        digest = verify_existing()
        if digest is None:
            fail(f"{config.RAW_PATH} not found")
            return
        report(config.RAW_PATH, digest, "verified (local file)")
        return

    if not args.force:
        digest = verify_existing()
        if digest is not None:
            report(config.RAW_PATH, digest, "already present, checksum verified")
            return
    download()
    report(config.RAW_PATH, sha256_of(config.RAW_PATH), "downloaded, checksum verified")


if __name__ == "__main__":
    main()

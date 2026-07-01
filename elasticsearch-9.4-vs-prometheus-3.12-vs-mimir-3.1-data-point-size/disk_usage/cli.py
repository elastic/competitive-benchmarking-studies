"""Standalone post-ingest storage measurement, independent of the ingest run.

Usage: uv run disk-usage   (reads ENGINE/RESULTS_FILE from the environment,
same as `uv run load`, and merges the measured size_bytes into the existing
result)
"""

import json
import os

from engine_config import ENGINE, RESULTS_FILE
from store.results import ResultStore

from .measure import measure_elasticsearch, measure_mimir, measure_prometheus


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.1f}gb"
    if size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.1f}mb"
    return f"{size_bytes}b"


def _bytes_per_dp_suffix(size_bytes: int, datapoints: int, unit: str) -> str:
    if not (size_bytes and datapoints):
        return ""
    return f"  ({size_bytes / datapoints:.2f} bytes/{unit})"


def _load_datapoints() -> int:
    """Best-effort read of `datapoints` from the ingest result, for the bytes/dp figure."""
    if not RESULTS_FILE:
        raise RuntimeError("RESULTS_FILE not set in environment")

    try:
        with open(RESULTS_FILE) as f:
            return int(json.load(f).get("datapoints", 0))
    except (FileNotFoundError, ValueError):
        raise RuntimeError(f"Could not read datapoints from {RESULTS_FILE}")


def main() -> None:
    datapoints = _load_datapoints()

    if ENGINE == "elasticsearch":
        count, size_bytes = measure_elasticsearch()
        bps = _bytes_per_dp_suffix(size_bytes, datapoints, "dp")
        print(f"\nElasticsearch: {count:,} docs  {_format_size(size_bytes)}{bps}")
    elif ENGINE == "prometheus":
        count, size_bytes = measure_prometheus()
        bps = _bytes_per_dp_suffix(size_bytes, datapoints, "sample")
        print(f"\nPrometheus: {count:,} series  {size_bytes / 1024**2:.1f} MB{bps}")
    else:
        count, size_bytes = measure_mimir()
        bps = _bytes_per_dp_suffix(size_bytes, datapoints, "dp")
        print(f"\nMimir: {count:,} series  {size_bytes / 1024**2:.1f} MB{bps}")

    if RESULTS_FILE:
        ResultStore(os.path.dirname(RESULTS_FILE)).save_storage_size(
            ENGINE,
            size_bytes,
            path=RESULTS_FILE,
        )


if __name__ == "__main__":
    main()

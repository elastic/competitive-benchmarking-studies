"""Standalone post-ingest storage measurement, independent of the ingest run.

Usage: uv run disk-usage   (reads the ENGINE and RESULTS_FILE environment
variables, same as `uv run load`, and merges the measured size_bytes into
the existing result)
"""

import argparse
import json
import os


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Accepted for CLI-invocation uniformity with `load`/`query` "
        "(run_engine.py forwards the same --benchmark to all three) — "
        "disk usage measurement doesn't depend on scenario parameters for "
        "most engines, except ClickHouse, which reads clickhouse.schema "
        "from the scenario to know which tables to measure.",
    )
    return parser.parse_args()


def _bytes_per_dp_suffix(size_bytes: int, datapoints: int, unit: str) -> str:
    if not (size_bytes and datapoints):
        return ""
    return f"  ({size_bytes / datapoints:.2f} bytes/{unit})"


def _load_datapoints(results_file: str) -> int:
    """Best-effort read of `datapoints` from the ingest result, for the bytes/dp figure."""
    try:
        with open(results_file) as f:
            return int(json.load(f).get("datapoints", 0))
    except (FileNotFoundError, ValueError):
        raise RuntimeError(f"Could not read datapoints from {results_file}")


def main() -> None:
    args = _parse_args()

    # Deferred past argument parsing (not at module level): engine_config
    # validates ENGINE at import time, which would otherwise make `--help`
    # crash instead of printing usage, before argparse gets a chance to
    # handle it. `.measure` also imports engine_config transitively, so it
    # has to move here too.
    from benchmark.engine_config import ENGINE, RESULTS_FILE
    from benchmark.scenarios import load_benchmark
    from benchmark.store.results import ResultStore
    from benchmark.utils.size import format_size

    from .measure import (
        measure_clickhouse,
        measure_elasticsearch,
        measure_mimir,
        measure_prometheus,
    )

    datapoints = _load_datapoints(RESULTS_FILE)

    if ENGINE == "elasticsearch":
        count, size_bytes = measure_elasticsearch()
        bps = _bytes_per_dp_suffix(size_bytes, datapoints, "dp")
        print(f"\nElasticsearch: {count:,} docs  {format_size(size_bytes)}{bps}")
    elif ENGINE == "prometheus":
        count, size_bytes = measure_prometheus()
        bps = _bytes_per_dp_suffix(size_bytes, datapoints, "sample")
        print(f"\nPrometheus: {count:,} series  {format_size(size_bytes)}{bps}")
    elif ENGINE == "clickhouse":
        count, size_bytes = measure_clickhouse(load_benchmark(args.benchmark))
        bps = _bytes_per_dp_suffix(size_bytes, datapoints, "row")
        print(f"\nClickHouse: {count:,} rows  {format_size(size_bytes)}{bps}")
    else:
        count, size_bytes = measure_mimir()
        bps = _bytes_per_dp_suffix(size_bytes, datapoints, "dp")
        print(f"\nMimir: {count:,} series  {format_size(size_bytes)}{bps}")

    ResultStore(os.path.dirname(RESULTS_FILE)).save_storage_size(
        ENGINE,
        size_bytes,
        path=RESULTS_FILE,
    )


if __name__ == "__main__":
    main()

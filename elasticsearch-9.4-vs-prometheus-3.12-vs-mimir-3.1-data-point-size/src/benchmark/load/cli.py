import argparse
import os


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark scenario to run (see scenarios/*.yml)",
    )
    parser.add_argument(
        "--wait-for-merges",
        action="store_true",
        help="After ingest, wait for background segment merges to complete "
        "before exiting (only applies to engines that support it; ignored "
        "otherwise).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Deferred past argument parsing (not at module level): engine_config
    # validates ENGINE at import time, which would otherwise make `--help`
    # crash instead of printing usage, before argparse gets a chance to
    # handle it. `.collector` also imports engine_config transitively, so
    # it has to move here too.
    from benchmark.engine_config import (
        DATA_DIR,
        DATA_STREAM,
        ENGINE,
        OTLP_ENDPOINT,
        RESULTS_FILE,
        VERSION,
    )
    from benchmark.scenarios import load_benchmark
    from benchmark.store.results import ResultStore
    from benchmark.utils.es import es_wait_for_merges
    from benchmark.utils.time import format_duration

    from .collector import run

    scenario = load_benchmark(args.benchmark)
    os.makedirs(DATA_DIR, exist_ok=True)

    print(
        f"engine={ENGINE}  scale={scenario.scale}  interval={scenario.interval}  "
        f"window={scenario.start_now_minus}"
    )
    print(f"target={OTLP_ENDPOINT}")

    datapoints, rate, elapsed, start_ts, end_ts = run(scenario)

    if datapoints:
        print(
            f"Ingested: {datapoints:,} data points ({rate:,.0f} dp/s) in {format_duration(elapsed)}"
        )
    else:
        print(f"metricsgenreceiver completed in {format_duration(elapsed)}")

    ResultStore(os.path.dirname(RESULTS_FILE)).save_ingest_result(
        ENGINE,
        VERSION,
        datapoints,
        start_ts=start_ts,
        end_ts=end_ts,
        elapsed_seconds=elapsed,
        benchmark=scenario.name,
        path=RESULTS_FILE,
    )

    if args.wait_for_merges:
        if ENGINE == "elasticsearch":
            es_wait_for_merges(DATA_STREAM)
        else:
            print(f"--wait-for-merges is not supported for engine={ENGINE}, ignoring")

    print("Data ingestion completed.")

"""OTel collector config rendering and metricsgenreceiver execution."""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import jinja2

from benchmark.engine_config import (
    CLICKHOUSE_DATABASE,
    DATA_DIR,
    ENGINE,
    EXPORT_ENDPOINT,
)
from benchmark.scenarios import BenchmarkScenario

from .config import parse_duration_seconds

_HERE = os.path.dirname(os.path.abspath(__file__))
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.path.join(_HERE, "templates")),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


def otel_config(scenario: BenchmarkScenario, debug: bool = False) -> str:
    # Prometheus 3.x does not reliably decompress gzip on OTLP requests; ES and
    # Mimir handle large gzip batches without issue.
    compression = ENGINE == "elasticsearch"
    return _jinja_env.get_template("otelcol.yaml.j2").render(
        engine=ENGINE,
        seed=scenario.seed,
        scale=scenario.scale,
        interval=scenario.interval,
        start_now_minus=scenario.start_now_minus,
        export_endpoint=EXPORT_ENDPOINT,
        compression=compression,
        debug=debug,
        clickhouse_database=CLICKHOUSE_DATABASE,
    )


def resolve_binary() -> str:
    # _HERE is <repo_root>/src/benchmark/load — .bin/ is a repo-root artifact
    # directory, three levels up from here.
    root = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
    found = shutil.which("metricsgenreceiver")
    if found:
        return found
    local = os.path.join(root, ".bin", "metricsgenreceiver")
    if os.path.isfile(local):
        return local
    sys.exit("metricsgenreceiver not found — run 'make setup' to install it")


def parse_datapoints(output: str) -> tuple[int, float]:
    datapoints, rate = 0, 0.0
    for line in output.splitlines():
        if '"datapoints"' in line:
            m = re.search(r'"datapoints"\s*:\s*(\d+)', line)
            if m:
                datapoints = int(m.group(1))
            m = re.search(r'"data_points_per_second"\s*:\s*([\d.]+)', line)
            if m:
                rate = float(m.group(1))
    return datapoints, rate


def run(scenario: BenchmarkScenario) -> tuple[int, float, float, int, int]:
    """Run metricsgenreceiver, return (datapoints, rate, elapsed_seconds, start_ts, end_ts)."""
    debug = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
    binary = resolve_binary()
    log_path = os.path.join(DATA_DIR, "metricsgenreceiver.log")

    if debug:
        print("Running metricsgenreceiver (debug/nop — no data will be ingested) ...")
    elif ENGINE == "clickhouse":
        print(f"Running metricsgenreceiver → clickhouse native {EXPORT_ENDPOINT} ...")
    else:
        print(f"Running metricsgenreceiver → {EXPORT_ENDPOINT}/v1/metrics ...")
    print(f"Logs → {log_path}")

    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "otelcol.yaml")
        with open(cfg, "w") as f:
            f.write(otel_config(scenario, debug=debug))

        t0 = time.time()
        with open(log_path, "w") as log_file:
            result = subprocess.run(
                [binary, "--config", cfg],
                stdout=log_file,
                stderr=log_file,
                text=True,
            )
        elapsed = time.time() - t0

    if result.returncode != 0:
        sys.exit(f"metricsgenreceiver exited with code {result.returncode}")

    with open(log_path) as f:
        log_contents = f.read()

    print(log_contents)
    datapoints, rate = parse_datapoints(log_contents)
    start_ts = int(t0) - parse_duration_seconds(scenario.start_now_minus)
    end_ts = int(t0)
    return datapoints, rate, elapsed, start_ts, end_ts

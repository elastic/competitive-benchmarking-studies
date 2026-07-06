#!/usr/bin/env python3
"""Bootstrap (Elasticsearch/ClickHouse only) + ingest/query/disk-usage
sequence for one engine.

Usage: uv run run-engine <elasticsearch|prometheus|mimir|clickhouse> [--benchmark <name>]

Derives ENGINE and RESULTS_FILE from the positional engine argument and sets
them in the environment for the load/query/disk-usage subprocesses below —
the caller only needs to supply the connection URLs (ELASTICSEARCH_URL/
PROMETHEUS_URL/MIMIR_URL/CLICKHOUSE_URL), which belong in .env since they're
static "how this machine is set up" details, not per-run parameters.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# This file lives at <repo_root>/src/benchmark/run_engine.py — deploy/config/
# is a repo-root directory, three levels up from here.
_ROOT = Path(__file__).parent.parent.parent


def _run(args: list[str]) -> None:
    result = subprocess.run(args)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "engine", choices=["elasticsearch", "prometheus", "mimir", "clickhouse"]
    )
    parser.add_argument("--benchmark", required=True)
    return parser.parse_args()


def _bootstrap_elasticsearch() -> None:
    # Imported here, not at module level: engine_config validates ENGINE (and
    # utils.es imports engine_config) at import time, so this must run after
    # main() has set ENGINE/RESULTS_FILE below, not before.
    from benchmark.engine_config import DATA_STREAM
    from benchmark.utils.es import (
        es_apply_component_template,
        es_apply_ilm_policy,
        es_recreate_data_stream,
        es_start_trial_license,
    )

    es_start_trial_license()
    es_apply_component_template(
        _ROOT / "deploy/config/elasticsearch/metrics-otel@custom.json",
        "metrics-otel@custom",
    )
    es_apply_ilm_policy(
        _ROOT / "deploy/config/elasticsearch/metrics-policy.json", "metrics-policy"
    )
    es_recreate_data_stream(DATA_STREAM)


def _bootstrap_clickhouse() -> None:
    # Imported here, not at module level: engine_config validates ENGINE at
    # import time, and utils.clickhouse imports engine_config transitively,
    # so this must run after main() has set ENGINE/RESULTS_FILE below.
    from benchmark.utils.clickhouse import ch_execute_sql_file

    ch_execute_sql_file(_ROOT / "deploy/config/clickhouse/schema.sql")
    print("✓ ClickHouse schema applied")


def main() -> None:
    args = _parse_args()

    # Single source of truth for which engine this run targets — propagated
    # to load/query/disk-usage via the environment they inherit as subprocesses.
    os.environ["ENGINE"] = args.engine
    os.environ["RESULTS_FILE"] = f"results/{args.engine}.json"

    load_extra_args = []
    if args.engine == "elasticsearch":
        _bootstrap_elasticsearch()
        load_extra_args.append("--wait-for-merges")
    elif args.engine == "clickhouse":
        _bootstrap_clickhouse()

    print("Ingesting data...")
    _run(["uv", "run", "load", "--benchmark", args.benchmark, *load_extra_args])
    print("Running queries...")
    _run(["uv", "run", "query", "--benchmark", args.benchmark])
    print("Calculating disk usage...")
    _run(["uv", "run", "disk-usage", "--benchmark", args.benchmark])


if __name__ == "__main__":
    main()

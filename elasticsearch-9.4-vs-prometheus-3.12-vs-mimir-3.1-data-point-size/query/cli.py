import argparse
import dataclasses
import os
import sys
import time

from store.results import ResultStore

from .executor import VegetaRunner
from .loader import QueryLoader
from .models import (
    AttackReport,
    BenchmarkResults,
    VegetaConfig,
    VegetaTarget,
    parse_time_arg,
    to_iso,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERIES_FILE = os.path.join(_ROOT, "queries.yml")


def main() -> None:
    results_file = os.environ.get("RESULTS_FILE")
    if not results_file:
        sys.exit("RESULTS_FILE environment variable is required")
    store = ResultStore(os.path.dirname(results_file))

    runner = VegetaRunner()

    parser = argparse.ArgumentParser(description="Query performance benchmark")
    parser.add_argument(
        "--from",
        dest="from_arg",
        default=None,
        metavar="TIME",
        help="Start time: Unix timestamp, duration from now (e.g. 270m, 4h), or 'now' (default: read from results/{engine}.json, fallback 270m)",
    )
    parser.add_argument(
        "--to",
        dest="to_arg",
        default=None,
        metavar="TIME",
        help="End time: Unix timestamp, duration from now, or 'now' (default: read from results/{engine}.json, fallback now)",
    )
    parser.add_argument(
        "--engine",
        dest="engine",
        help="Run only this engine (overrides QUERY_ENGINE env var)",
    )
    args = parser.parse_args()

    engine_filter = args.engine or os.environ.get("QUERY_ENGINE")

    from_arg = args.from_arg
    to_arg = args.to_arg

    if (from_arg is None or to_arg is None) and engine_filter:
        ts = store.load_time_range(engine_filter)
        if ts:
            start_ts, end_ts = ts
            from_arg = from_arg or str(start_ts)
            to_arg = to_arg or str(end_ts)
            print(
                f"Time range from results/{engine_filter}.json: "
                f"{to_iso(start_ts)} → {to_iso(end_ts)}"
            )

    from_arg = from_arg or "270m"
    to_arg = to_arg or "now"

    now = int(time.time())
    try:
        ctx = {
            "now": now,
            "from": parse_time_arg(from_arg, now),
            "to": parse_time_arg(to_arg, now),
        }
    except ValueError as e:
        sys.exit(f"Invalid time argument: {e}")

    try:
        defaults, groups = QueryLoader().load(QUERIES_FILE, ctx)
    except FileNotFoundError:
        sys.exit(f"queries.yml not found at {QUERIES_FILE}")

    if engine_filter and engine_filter not in groups:
        sys.exit(
            f"Engine {engine_filter!r} not found in queries.yml (known: {', '.join(groups)})"
        )

    all_results: list[AttackReport] = []

    for engine, group in groups.items():
        if engine_filter and engine != engine_filter:
            continue

        print(f"\n━━━ {engine} ━━━")
        engine_results: list[AttackReport] = []

        for i, query in enumerate(group.queries, start=1):
            vegeta_target = VegetaTarget.from_target_and_query(group.target, query)
            vegeta_cfg = VegetaConfig.from_defaults_and_query(defaults, query)

            has_warmup = (
                query.warmup_duration
                or defaults.warmup_duration
                or (query.warmup_count and query.warmup_count > 0)
            )
            if has_warmup:
                print(f"  warmup:  [{i}] {query.name} …", end="")
                runner.warmup(vegeta_target, query, defaults)
                print(" done")

            print(
                f"  running: [{i}] {query.name} "
                f"(rate={vegeta_cfg.effective_rate}, duration={vegeta_cfg.effective_duration}, "
                f"workers={vegeta_cfg.effective_workers}) …",
                end="",
            )
            report = runner.attack(vegeta_target, vegeta_cfg)
            report = dataclasses.replace(
                report, engine=engine, query_name=query.name, query_index=i
            )
            engine_results.append(report)
            all_results.append(report)
            print(
                f"  p50={report.p50_ms:.1f}ms  p99={report.p99_ms:.1f}ms  "
                f"rps={report.throughput:.1f}  ok={report.success_pct:.0f}%"
            )

        store.save_query_results(engine, engine_results)

    if all_results:
        print(
            BenchmarkResults(results=all_results, defaults=defaults).to_result_table()
        )
        return

    print("No queries ran — check --engine / QUERY_ENGINE filter.")

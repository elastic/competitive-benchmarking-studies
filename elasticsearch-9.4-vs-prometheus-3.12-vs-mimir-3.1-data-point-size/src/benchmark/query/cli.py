import argparse
import dataclasses
import os
import sys
import time

from benchmark.scenarios import load_benchmark
from benchmark.store.results import ResultStore

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query performance benchmark")
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark scenario providing the queryset to run (see scenarios/*.yml)",
    )
    parser.add_argument(
        "--from",
        dest="from_arg",
        default=None,
        metavar="TIME",
        help="Start time as a Unix timestamp, 'now', or a duration before now "
        "like '270m'/'4h'/'30s' (default: read from results/{engine}.json)",
    )
    parser.add_argument(
        "--to",
        dest="to_arg",
        default=None,
        metavar="TIME",
        help="End time as a Unix timestamp, 'now', or a duration before now "
        "like '270m'/'4h'/'30s' (default: read from results/{engine}.json)",
    )
    parser.add_argument(
        "--engine",
        dest="engine",
        help="Run only this engine (overrides ENGINE env var)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenario = load_benchmark(args.benchmark)

    engine_filter = args.engine or os.environ.get("ENGINE")
    if not engine_filter:
        sys.exit("No engine specified. Use --engine or ENGINE env var.")

    results_file = os.environ.get("RESULTS_FILE") or f"results/{engine_filter}.json"
    store = ResultStore(os.path.dirname(results_file))
    runner = VegetaRunner()

    ts = store.load_time_range(engine_filter)
    if not ts:
        sys.exit(
            f"results/{engine_filter}.json not found or missing start_ts/end_ts — run load first"
        )

    start_ts, end_ts = ts
    print(
        f"Time range from results/{engine_filter}.json: "
        f"{to_iso(start_ts)} → {to_iso(end_ts)}"
    )

    now = int(time.time())
    try:
        from_ts: int = parse_time_arg(args.from_arg, now) if args.from_arg else start_ts
        to_ts: int = parse_time_arg(args.to_arg, now) if args.to_arg else end_ts
    except ValueError as e:
        sys.exit(str(e))
    ctx = {
        "now": now,
        "from": from_ts,
        "to": to_ts,
    }

    try:
        defaults, groups = QueryLoader().load(str(scenario.queryset_path), ctx)
    except FileNotFoundError:
        sys.exit(f"Queryset not found at {scenario.queryset_path}")

    if engine_filter and engine_filter not in groups:
        sys.exit(f"Engine {engine_filter!r} not found in {scenario.queryset_path}")

    all_results: list[AttackReport] = []

    for engine, group in groups.items():
        if engine_filter and engine != engine_filter:
            continue

        print(f"\n━━━ {engine} ━━━")
        engine_results: list[AttackReport] = []

        for i, query in enumerate(group.queries, start=1):
            vegeta_target = VegetaTarget.from_target_and_query(group.target, query)
            vegeta_cfg = VegetaConfig.from_defaults_and_query(defaults, query)

            has_warmup = bool(query.warmup_duration or defaults.warmup_duration)
            if has_warmup:
                print(f"  warmup:  [{i}] {query.name} …", end="")
                runner.warmup(vegeta_target, query, defaults)
                print(" done")

            print(
                f"  running: [{i}] {query.name} \n"
                f"  query: {vegeta_target.body.decode('utf-8')} \n",
                f" (rate={vegeta_cfg.effective_rate}, duration={vegeta_cfg.effective_duration}, "
                f"workers={vegeta_cfg.effective_workers}, max_workers={vegeta_cfg.effective_max_workers}) \n",
                end="",
            )
            report = runner.attack(vegeta_target, vegeta_cfg)
            report = dataclasses.replace(
                report,
                engine=engine,
                query_name=query.name,
                query_index=i,
            )
            engine_results.append(report)
            all_results.append(report)
            print(
                f"  results: p50={report.p50_ms:.1f}ms  p99={report.p99_ms:.1f}ms  "
                f"rps={report.throughput:.1f}  ok={report.success_pct:.0f}% \n"
            )

        store.save_query_results(engine, engine_results)

    if all_results:
        print(
            BenchmarkResults(results=all_results, defaults=defaults).to_result_table()
        )
        return

    sys.exit("No queries ran — check --engine / ENGINE filter.")

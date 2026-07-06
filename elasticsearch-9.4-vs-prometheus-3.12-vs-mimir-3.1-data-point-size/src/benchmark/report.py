#!/usr/bin/env python3
"""Display bytes-per-data-point and query-latency comparison across engines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tabulate import tabulate

from benchmark.query.models import to_iso
from benchmark.utils.size import format_size
from benchmark.utils.time import format_duration

# This file lives at <repo_root>/src/benchmark/report.py — results/ is a
# repo-root artifact directory, three levels up from here.
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

ENGINES = ["elasticsearch", "prometheus", "mimir", "clickhouse"]

ENGINE_COLORS = {
    "elasticsearch": "#00BFB3",  # Elastic teal
    "prometheus": "#E6522C",  # Prometheus orange
    "mimir": "#5794F2",  # Grafana blue
    "clickhouse": "#FFCC01",  # ClickHouse yellow
}


def load_results(results_dir: Path, engines: list[str]) -> dict[str, dict]:
    results = {}
    for engine in engines:
        path = results_dir / f"{engine}.json"
        if not path.exists():
            continue
        try:
            with open(path) as f:
                results[engine] = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Skipping {path}: malformed result file ({e})", file=sys.stderr)
    return results


def render_storage_table(
    results: dict[str, dict], engines: list[str], tablefmt: str
) -> str:
    rows = []
    for engine in engines:
        r = results.get(engine)
        if r is None:
            continue
        dp = r.get("datapoints", 0)
        sb = r.get("size_bytes", 0)
        bpd = sb / dp if dp else 0
        elapsed = r.get("elapsed_seconds", 0)
        eps = r.get("eps", 0)
        rows.append(
            [
                engine,
                r.get("version", "?"),
                to_iso(r["run_at"]) if r.get("run_at") else "—",
                f"{dp:,}",
                format_size(sb),
                f"{bpd:.2f}",
                format_duration(elapsed) if elapsed else "—",
                f"{eps:,}" if eps else "—",
            ]
        )
    headers = [
        "Engine",
        "Version",
        "Run At",
        "Data Points",
        "Size",
        "Bytes/DP",
        "Elapsed",
        "EPS",
    ]
    return tabulate(rows, headers=headers, tablefmt=tablefmt)


def _latency_cell(q: dict | None) -> str:
    return f"{q['p50_ms']:.1f}/{q['p95_ms']:.1f}/{q['p99_ms']:.1f}" if q else "—"


def render_query_table(
    results: dict[str, dict], engines: list[str], tablefmt: str
) -> str | None:
    query_results = {
        engine: {q["name"]: q for q in r.get("queries", [])}
        for engine, r in results.items()
        if r.get("queries")
    }
    if not query_results:
        return None

    present_engines = [e for e in engines if e in query_results]
    query_names = list(next(iter(query_results.values())).keys())

    rows = [
        [name]
        + [_latency_cell(query_results[engine].get(name)) for engine in present_engines]
        for name in query_names
    ]
    headers = ["Query"] + [f"{e} (p50/p95/p99 ms)" for e in present_engines]
    return tabulate(rows, headers=headers, tablefmt=tablefmt)


def render_storage_chart(
    results: dict[str, dict], engines: list[str], out_path: Path
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping chart (pip install matplotlib)")
        return None

    present = [e for e in engines if e in results and results[e].get("datapoints")]
    if not present:
        print(
            "No storage data to chart yet — run the disk-usage step for at least one engine."
        )
        return None

    labels = [f"{e}\n{results[e].get('version', '?')}" for e in present]
    values = [results[e]["size_bytes"] / results[e]["datapoints"] for e in present]
    colors = [ENGINE_COLORS.get(e, "#888888") for e in present]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5, zorder=2)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.03,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Bytes per data point", fontsize=11)
    ax.set_title(
        "Storage Efficiency: Bytes per Data Point\n(lower is better)", fontsize=13
    )
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_query_chart(
    results: dict[str, dict], engines: list[str], out_path: Path
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return None

    query_results = {
        engine: {q["name"]: q for q in r.get("queries", [])}
        for engine, r in results.items()
        if r.get("queries")
    }
    if not query_results:
        return None

    present_engines = [e for e in engines if e in query_results]
    query_names = list(next(iter(query_results.values())).keys())
    query_labels = [f"Q{i + 1}" for i in range(len(query_names))]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(len(query_names)))
    n = len(present_engines)
    width = 0.8 / n

    for i, engine in enumerate(present_engines):
        p50s = [
            query_results[engine].get(name, {}).get("p50_ms", 0) for name in query_names
        ]
        offsets = [xi + (i - (n - 1) / 2) * width for xi in x]
        ax.bar(
            offsets,
            p50s,
            width=width,
            color=ENGINE_COLORS.get(engine, "#888888"),
            label=engine,
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(query_labels)
    ax.set_ylabel("p50 latency (ms)", fontsize=11)
    ax.set_title("Query Latency: p50 by Query and Engine", fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    engine_legend = ax.legend(loc="upper left", title="Datastore")
    ax.add_artist(engine_legend)

    query_handles = [
        Patch(facecolor="none", edgecolor="none", label=f"{label} = {name}")
        for label, name in zip(query_labels, query_names)
    ]
    ax.legend(
        handles=query_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=8,
        title="Queries",
        frameon=False,
        handlelength=0,
        handletextpad=0,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="emit GitHub-flavored markdown tables instead of plain text",
    )
    args = parser.parse_args()
    tablefmt = "github" if args.markdown else "simple"

    results = load_results(RESULTS_DIR, ENGINES)
    if not results:
        print(
            "No results yet. Run: make elasticsearch  make prometheus  make mimir  make clickhouse"
        )
        return

    print()
    print(render_storage_table(results, ENGINES, tablefmt))
    print()

    query_table = render_query_table(results, ENGINES, tablefmt)
    if query_table:
        print(query_table)
        print()

    out = render_storage_chart(results, ENGINES, RESULTS_DIR / "report.png")
    if out:
        print(f"Chart saved → {out}")

    out2 = render_query_chart(results, ENGINES, RESULTS_DIR / "query_latency.png")
    if out2:
        print(f"Chart saved → {out2}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Display bytes-per-data-point and query-latency comparison across engines."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from benchmark.query.models import to_iso
from benchmark.utils.size import format_size
from benchmark.utils.time import format_duration

# This file lives at <repo_root>/src/benchmark/report.py — results/ is a
# repo-root artifact directory, three levels up from here.
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

ENGINES = ["elasticsearch", "prometheus", "mimir"]

ENGINE_COLORS = {
    "elasticsearch": "#00BFB3",  # Elastic teal
    "prometheus": "#E6522C",  # Prometheus orange
    "mimir": "#5794F2",  # Grafana blue
}


@dataclass(frozen=True)
class QueryResult:
    name: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput_rps: float
    success_pct: float

    @classmethod
    def from_dict(cls, d: dict) -> "QueryResult":
        return cls(
            name=d.get("name", "?"),
            p50_ms=d.get("p50_ms", 0.0),
            p95_ms=d.get("p95_ms", 0.0),
            p99_ms=d.get("p99_ms", 0.0),
            mean_ms=d.get("mean_ms", 0.0),
            throughput_rps=d.get("throughput_rps", 0.0),
            success_pct=d.get("success_pct", 0.0),
        )


@dataclass(frozen=True)
class EngineResult:
    engine: str
    version: str = "?"
    run_at: int | None = None
    datapoints: int = 0
    size_bytes: int | None = None
    elapsed_seconds: float = 0.0
    eps: int = 0
    queries: tuple[QueryResult, ...] = field(default_factory=tuple)

    @classmethod
    def from_json(cls, engine: str, data: dict) -> "EngineResult":
        return cls(
            engine=engine,
            version=data.get("version", "?"),
            run_at=data.get("run_at"),
            datapoints=data.get("datapoints", 0),
            size_bytes=data.get("size_bytes"),
            elapsed_seconds=data.get("elapsed_seconds", 0.0),
            eps=data.get("eps", 0),
            queries=tuple(QueryResult.from_dict(q) for q in data.get("queries", [])),
        )

    @property
    def bytes_per_datapoint(self) -> float | None:
        if not self.datapoints or self.size_bytes is None:
            return None
        return self.size_bytes / self.datapoints


def load_results(results_dir: Path, engines: Sequence[str]) -> dict[str, EngineResult]:
    results = {}
    for engine in engines:
        path = results_dir / f"{engine}.json"
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Skipping {path}: malformed result file ({e})", file=sys.stderr)
            continue
        results[engine] = EngineResult.from_json(engine, data)
    return results


def render_storage_table(
    results: dict[str, EngineResult], engines: Sequence[str]
) -> str:
    w = 24
    lines = [
        f"{'Engine':<{w}} {'Version':<16} {'Run At':<20} {'Data Points':>15} {'Size':>10} "
        f"{'Bytes/DP':>10} {'Elapsed':>10} {'EPS':>10}",
        "─" * (w + 16 + 20 + 15 + 10 + 10 + 10 + 10 + 9),
    ]
    for engine in engines:
        r = results.get(engine)
        if r is None:
            continue
        run_at_str = to_iso(r.run_at) if r.run_at else "—"
        size_str = format_size(r.size_bytes) if r.size_bytes is not None else "—"
        bpd = r.bytes_per_datapoint
        bpd_str = f"{bpd:.2f}" if bpd is not None else "—"
        elapsed_str = format_duration(r.elapsed_seconds) if r.elapsed_seconds else "—"
        eps_str = f"{r.eps:,}" if r.eps else "—"
        lines.append(
            f"{engine:<{w}} {r.version:<16} {run_at_str:<20} {r.datapoints:>15,} {size_str:>10} "
            f"{bpd_str:>10} {elapsed_str:>10} {eps_str:>10}"
        )
    return "\n".join(lines)


def render_query_table(
    results: dict[str, EngineResult], engines: Sequence[str]
) -> str | None:
    engines_with_queries = [e for e in engines if results.get(e) and results[e].queries]
    if not engines_with_queries:
        return None

    blocks = []
    for engine in engines_with_queries:
        queries = results[engine].queries
        name_w = max(len("Query"), max(len(q.name) for q in queries))
        header = (
            f"{'Query':<{name_w}} {'p50 ms':>9} {'p95 ms':>9} {'p99 ms':>9} "
            f"{'Mean ms':>9} {'RPS':>9} {'OK%':>6}"
        )
        lines = [f"{engine} — query results", "─" * len(header), header]
        for q in queries:
            lines.append(
                f"{q.name:<{name_w}} {q.p50_ms:>9.2f} {q.p95_ms:>9.2f} {q.p99_ms:>9.2f} "
                f"{q.mean_ms:>9.2f} {q.throughput_rps:>9.1f} {q.success_pct:>6.1f}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_chart(
    results: dict[str, EngineResult], engines: Sequence[str], out_path: Path
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping chart (pip install matplotlib)")
        return None

    chartable = [
        e
        for e in engines
        if results.get(e) and results[e].bytes_per_datapoint is not None
    ]
    if not chartable:
        print(
            "No storage data to chart yet — run the disk-usage step for at least one engine."
        )
        return None

    labels = [f"{e}\n{results[e].version}" for e in chartable]
    values = [results[e].bytes_per_datapoint for e in chartable]
    colors = [ENGINE_COLORS.get(e, "#888888") for e in chartable]

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
    return out_path


def main() -> None:
    results = load_results(RESULTS_DIR, ENGINES)
    if not results:
        print("No results yet. Run: make elasticsearch  make prometheus  make mimir")
        return

    print()
    print(render_storage_table(results, ENGINES))
    print()

    query_table = render_query_table(results, ENGINES)
    if query_table:
        print(query_table)
        print()

    out = render_chart(results, ENGINES, RESULTS_DIR / "report.png")
    if out:
        print(f"Chart saved → {out}")


if __name__ == "__main__":
    main()

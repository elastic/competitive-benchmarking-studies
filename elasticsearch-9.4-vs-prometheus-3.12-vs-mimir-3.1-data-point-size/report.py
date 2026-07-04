#!/usr/bin/env python3
"""Display bytes-per-data-point comparison across all engines that have results."""

import argparse
import json
import os
import sys

from tabulate import tabulate


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--markdown",
    action="store_true",
    help="emit GitHub-flavored markdown tables instead of plain text",
)
args = parser.parse_args()
TABLEFMT = "github" if args.markdown else "simple"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

ENGINES = ["elasticsearch", "prometheus", "mimir"]

ENGINE_COLORS = {
    "elasticsearch": "#00BFB3",  # Elastic teal
    "prometheus": "#E6522C",  # Prometheus orange
    "mimir": "#5794F2",  # Grafana blue
}

results = {}
for engine in ENGINES:
    path = os.path.join(RESULTS_DIR, f"{engine}.json")
    if os.path.exists(path):
        with open(path) as f:
            results[engine] = json.load(f)

if not results:
    print("No results yet. Run: make elasticsearch  make prometheus  make mimir")
    sys.exit(0)

# ── Storage comparison table ─────────────────────────────────────────────────
storage_rows = []
for engine in ENGINES:
    if engine not in results:
        continue
    r = results[engine]
    dp = r.get("datapoints", 0)
    sb = r.get("size_bytes", 0)
    v = r.get("version", "?")
    bpd = sb / dp if dp else 0
    size_str = (
        f"{sb / 1024**3:.2f} GB"
        if sb >= 1024**3
        else f"{sb / 1024**2:.1f} MB"
        if sb >= 1024**2
        else f"{sb / 1024:.0f} KB"
    )
    elapsed = r.get("elapsed_seconds", 0)
    elapsed_str = _format_duration(elapsed) if elapsed else "—"
    eps = r.get("eps", 0)
    eps_str = f"{eps:,}" if eps else "—"
    storage_rows.append([engine, v, f"{dp:,}", size_str, f"{bpd:.2f}", elapsed_str, eps_str])

print()
print(
    tabulate(
        storage_rows,
        headers=["Engine", "Version", "Data Points", "Size", "Bytes/DP", "Elapsed", "EPS"],
        tablefmt=TABLEFMT,
    )
)
print()

# ── Query latency table ──────────────────────────────────────────────────────
query_results = {
    engine: {q["name"]: q for q in r.get("queries", [])}
    for engine, r in results.items()
    if r.get("queries")
}

def _latency_cell(q: dict | None) -> str:
    return f"{q['p50_ms']:.1f}/{q['p95_ms']:.1f}/{q['p99_ms']:.1f}" if q else "—"


if query_results:
    present_engines = [e for e in ENGINES if e in query_results]
    query_names = list(next(iter(query_results.values())).keys())

    query_rows = [
        [name] + [_latency_cell(query_results[engine].get(name)) for engine in present_engines]
        for name in query_names
    ]
    headers = ["Query"] + [f"{e} (p50/p95/p99 ms)" for e in present_engines]

    print(tabulate(query_rows, headers=headers, tablefmt=TABLEFMT))
    print()

# ── Bar chart ─────────────────────────────────────────────────────────────────
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed — skipping chart (pip install matplotlib)")
    sys.exit(0)

present = [e for e in ENGINES if e in results]
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
ax.set_title("Storage Efficiency: Bytes per Data Point\n(lower is better)", fontsize=13)
ax.set_ylim(0, max(values) * 1.25)
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "report.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Chart saved → {out}")

# ── Query latency bar chart ──────────────────────────────────────────────────
if query_results:
    from matplotlib.patches import Patch

    present_engines = [e for e in ENGINES if e in query_results]
    query_names = list(next(iter(query_results.values())).keys())
    query_labels = [f"Q{i + 1}" for i in range(len(query_names))]

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    x = list(range(len(query_names)))
    n = len(present_engines)
    width = 0.8 / n

    for i, engine in enumerate(present_engines):
        p50s = [query_results[engine].get(name, {}).get("p50_ms", 0) for name in query_names]
        offsets = [xi + (i - (n - 1) / 2) * width for xi in x]
        ax2.bar(
            offsets,
            p50s,
            width=width,
            color=ENGINE_COLORS.get(engine, "#888888"),
            label=engine,
            zorder=2,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(query_labels)
    ax2.set_ylabel("p50 latency (ms)", fontsize=11)
    ax2.set_title("Query Latency: p50 by Query and Engine", fontsize=13)
    ax2.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    engine_legend = ax2.legend(loc="upper left", title="Datastore")
    ax2.add_artist(engine_legend)

    query_handles = [
        Patch(facecolor="none", edgecolor="none", label=f"{label} = {name}")
        for label, name in zip(query_labels, query_names)
    ]
    ax2.legend(
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
    out2 = os.path.join(RESULTS_DIR, "query_latency.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Chart saved → {out2}")

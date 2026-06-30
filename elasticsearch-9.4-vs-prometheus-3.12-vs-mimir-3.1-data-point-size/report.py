#!/usr/bin/env python3
"""Display bytes-per-data-point comparison across all engines that have results."""

import json
import os
import sys


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"

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

# ── Text table ────────────────────────────────────────────────────────────────
W = 24
print()
print(
    f"{'Engine':<{W}} {'Version':<16} {'Data Points':>15} {'Size':>10} {'Bytes/DP':>10} {'Elapsed':>10} {'EPS':>10}"
)
print("─" * (W + 16 + 15 + 10 + 10 + 10 + 10 + 8))
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
    print(f"{engine:<{W}} {v:<16} {dp:>15,} {size_str:>10} {bpd:>9.2f} {elapsed_str:>10} {eps_str:>10}")
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

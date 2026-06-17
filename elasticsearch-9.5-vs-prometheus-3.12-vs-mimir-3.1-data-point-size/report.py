#!/usr/bin/env python3
"""Display bytes-per-data-point comparison across all engines that have results."""
import json, os, sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

engines = ["elasticsearch", "prometheus", "mimir"]
results = {}
for engine in engines:
    path = os.path.join(RESULTS_DIR, f"{engine}.json")
    if os.path.exists(path):
        with open(path) as f:
            results[engine] = json.load(f)

if not results:
    print("No results yet. Run: make elasticsearch  make prometheus  make mimir")
    sys.exit(0)

W = 24
print()
print(f"{'Engine':<{W}} {'Version':<16} {'Data Points':>15} {'Size':>10} {'Bytes/DP':>10}")
print("─" * (W + 16 + 15 + 10 + 10 + 6))
for engine in engines:
    if engine not in results:
        continue
    r  = results[engine]
    dp = r.get("datapoints", 0)
    sb = r.get("size_bytes", 0)
    v  = r.get("version", "?")
    bpd = sb / dp if dp else 0
    size_str = (f"{sb/1024**3:.2f} GB" if sb >= 1024**3
                else f"{sb/1024**2:.1f} MB" if sb >= 1024**2
                else f"{sb/1024:.0f} KB")
    print(f"{engine:<{W}} {v:<16} {dp:>15,} {size_str:>10} {bpd:>9.2f}")

print()

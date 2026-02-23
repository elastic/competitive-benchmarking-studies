from __future__ import annotations
import logging
import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_at_value(filename: str) -> Optional[str]:
    at = filename.find("@")
    if at == -1:
        return None
    i = at + 1
    j = i
    while j < len(filename) and filename[j].isdigit():
        j += 1
    return filename[i:j] or None


def parse_engine_variant(filename: str) -> tuple[str, str]:
    base = os.path.splitext(os.path.basename(filename))[0]
    base_lower = base.lower()
    if base_lower.startswith("es_"):
        return "Elasticsearch", base[3:]
    if base_lower.startswith("os_"):
        return "OpenSearch", base[3:]
    return "Other", base


def add_k_n_rescore_cols(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["s_n_r_value"].astype(str).str.split("_", expand=True)
    if parts.shape[1] < 3:
        raise ValueError("s_n_r_value must look like 'k_n_rescore' e.g. '10_12_2'")
    df = df.copy()
    df["k"] = parts[0]
    df["n"] = parts[1]
    df["rescore"] = parts[2]
    df["label"] = df["k"].astype(str) + "-" + df["n"].astype(str)
    return df


def prepare_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Prepare dataframe by converting throughput to numeric."""
    df = df.copy()
    if "throughput" not in df.columns:
        logger.warning("Missing throughput column")
        return None
    df["throughput"] = pd.to_numeric(df["throughput"], errors="coerce")
    df = df.dropna(subset=["throughput"])
    if df.empty:
        return None
    return df


def discover_at_values(folder_path: str) -> list[str]:
    at_vals: set[str] = set()
    for name in os.listdir(folder_path):
        if not name.endswith(".csv"):
            continue
        at_val = extract_at_value(name)
        if at_val is not None:
            at_vals.add(at_val)
    return sorted(at_vals, key=int)


def detect_rescore_values(folder_path: str, at_val: str) -> list[str]:
    values: set[str] = set()
    for name in os.listdir(folder_path):
        if not name.endswith(".csv"):
            continue
        if extract_at_value(name) != str(at_val):
            continue
        path = os.path.join(folder_path, name)
        try:
            df = pd.read_csv(path)
        except Exception:
            logger.exception("Error reading %s", name)
            continue
        if "s_n_r_value" not in df.columns:
            continue
        parts = df["s_n_r_value"].astype(str).str.split("_", expand=True)
        if parts.shape[1] >= 3:
            values.update(parts[2].dropna().astype(str).unique().tolist())
    return sorted(values, key=lambda v: (0, int(v)) if v.isdigit() else (1, v))


def build_pivot_for_at(
    folder_path: str,
    at_val: str,
    rescore_value: str,
    max_samples: int = 5,
) -> Optional[pd.DataFrame]:
    """Build pivot table for throughput comparison."""

    # --- Step 1: Read all relevant data ---
    es_dfs, os_dfs = [], []
    for name in sorted(os.listdir(folder_path)):
        if not name.endswith(".csv") or extract_at_value(name) != str(at_val):
            continue
        path = os.path.join(folder_path, name)
        try:
            df = pd.read_csv(path)
            if "s_n_r_value" in df.columns and "recall" in df.columns:
                df = add_k_n_rescore_cols(df)
                df_filtered = df[df["rescore"].astype(str) == str(rescore_value)]
                if not df_filtered.empty:
                    item = {"name": name, "df": df_filtered}
                    if name.startswith("es_"):
                        es_dfs.append(item)
                    elif name.startswith("os_"):
                        os_dfs.append(item)
        except Exception:
            logger.exception("Error during initial read of %s", name)
            continue

    if not es_dfs and not os_dfs:
        return None

    # --- Step 2: Combine all data and compute metrics ---
    all_dfs = es_dfs + os_dfs

    rows = []
    for item in all_dfs:
        name = item["name"]
        df = item["df"]
        engine, variant = parse_engine_variant(name)
        df = prepare_df(df)
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            rows.append(
                {
                    "label": r["label"],
                    "series": f"{engine}::{variant}",
                    "throughput": r["throughput"],
                }
            )

    if not rows:
        return None

    long_df = pd.DataFrame(rows)

    # --- Step 3: Find labels where both engines have data and compute throughput diff ---
    es_data = long_df[long_df["series"].str.startswith("Elasticsearch")].groupby("label")["throughput"].mean()
    os_data = long_df[long_df["series"].str.startswith("OpenSearch")].groupby("label")["throughput"].mean()
    common_labels = set(es_data.index) & set(os_data.index)

    def label_sort_key(label: str) -> tuple[int, int]:
        parts = str(label).split("-")
        k = int(parts[0]) if parts[0].isdigit() else 9999
        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 9999
        return (k, n)

    # --- Step 4: Select labels with a spectrum of relative throughput differences ---
    selected_labels: list[str] = []
    if common_labels:
        # Calculate relative throughput difference (ratio) for each label
        label_diffs = []
        for label in common_labels:
            es_val, os_val = es_data[label], os_data[label]
            min_val = min(es_val, os_val)
            if min_val > 0:
                ratio = max(es_val, os_val) / min_val
            else:
                ratio = float("inf") if max(es_val, os_val) > 0 else 1.0
            label_diffs.append((label, ratio))
        # Sort by ratio (largest first)
        label_diffs.sort(key=lambda x: x[1], reverse=True)

        if len(label_diffs) <= max_samples:
            selected_labels = [label for label, _ in label_diffs]
        else:
            # Always include the biggest relative difference
            selected_labels = [label_diffs[0][0]]
            # Sample evenly from the rest of the spectrum
            remaining = label_diffs[1:]
            step = max(1, len(remaining) // (max_samples - 1))
            selected_labels.extend([label for label, _ in remaining[::step][: max_samples - 1]])

    if not selected_labels:
        return None

    # Re-sort final selection by k-n for display
    selected_labels = sorted(selected_labels, key=label_sort_key)

    filtered_df = long_df[long_df["label"].isin(selected_labels)]

    # --- Step 6: Build pivot with only throughput ---
    pivot = filtered_df.pivot_table(
        index="label",
        columns="series",
        values="throughput",
        aggfunc="mean",
    )

    # Sort index by k-n values
    pivot = pivot.reindex(sorted(pivot.index, key=label_sort_key))

    return pivot


def sample_configs(pivot_df: pd.DataFrame, sample_n: int, seed: Optional[int]) -> pd.DataFrame:
    if sample_n <= 0 or sample_n >= len(pivot_df):
        return pivot_df
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pivot_df.index.to_numpy(), size=sample_n, replace=False)

    def label_key(s: object) -> list[object]:
        parts = str(s).split("-")
        out: list[object] = []
        for p in parts:
            out.append(int(p) if p.isdigit() else p)
        return out

    chosen_sorted = sorted(chosen, key=label_key)
    return pivot_df.loc[chosen_sorted]


def plot_grouped_bars(pivot_df: pd.DataFrame, title: str, out_png: str) -> None:
    """Plot throughput comparison as grouped bars."""
    labels = pivot_df.index.astype(str).tolist()
    x = np.arange(len(labels))
    series_list = list(pivot_df.columns)

    def series_key(series: str) -> tuple[int, str]:
        engine = series.split("::", 1)[0].lower()
        if engine == "elasticsearch":
            return (0, series)
        if engine == "opensearch":
            return (1, series)
        return (2, series)

    series_list = sorted(series_list, key=series_key)
    width = min(0.35, 0.8 / max(1, len(series_list)))
    n_bars = len(series_list)
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * width

    # Color palette for throughput only
    es_color = "#F04E98"  # Elastic pink
    os_color = "#005EB8"  # OpenSearch blue

    fig, ax = plt.subplots(figsize=(14, 6))
    for bar_i, series in enumerate(series_list):
        engine, variant = series.split("::", 1)
        engine_lower = engine.lower()
        if engine_lower == "elasticsearch":
            color = es_color
            engine_prefix = "es_"
        elif engine_lower == "opensearch":
            color = os_color
            engine_prefix = "os_"
        else:
            color = "#7A7A7A"
            engine_prefix = ""
        variant_clean = variant.split("@")[0]
        legend_name = f"{engine_prefix}{variant_clean}"
        ax.bar(
            x + offsets[bar_i],
            pivot_df[series],
            width,
            label=legend_name,
            color=color,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Throughput")
    ax.legend()
    plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


def generate_overview_plots(
    folder_path: str,
    output_dir: str,
    sample_n: int = 0,
    seed: Optional[int] = None,
) -> None:
    at_vals = discover_at_values(folder_path)
    if not at_vals:
        logger.warning("No @N found in filenames in %s", folder_path)
        return

    plot_count = 0
    for at_val in at_vals:
        rescores = detect_rescore_values(folder_path, at_val)
        if not rescores:
            continue
        for rescore in rescores:
            pivot = build_pivot_for_at(
                folder_path,
                at_val=at_val,
                rescore_value=rescore,
            )
            if pivot is None or pivot.empty:
                continue
            pivot = sample_configs(pivot, sample_n=sample_n, seed=seed)
            suffix = f"_sample{len(pivot)}" if sample_n > 0 else ""

            out_png = os.path.join(
                output_dir,
                f"overview_throughput@{at_val}_R{rescore}{suffix}.png",
            )
            title = f"Throughput Recall@{at_val} (Rescore {rescore})"
            plot_grouped_bars(pivot, title=title, out_png=out_png)
            logger.info("Saved %s", out_png)
            plot_count += 1
    if plot_count == 0:
        logger.warning("No plots generated for %s", folder_path)
    else:
        logger.info("Generated %d overview plots", plot_count)

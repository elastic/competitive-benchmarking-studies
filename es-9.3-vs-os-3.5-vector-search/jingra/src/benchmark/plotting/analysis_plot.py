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


def extract_r_value_from_row(s_n_r_value: str) -> str:
    return str(s_n_r_value).split("_")[-1]


def natural_sort_key(text: str) -> list[object]:
    out: list[object] = []
    buf = ""
    for ch in text:
        if ch.isdigit():
            buf += ch
            continue
        if buf:
            out.append(int(buf))
            buf = ""
        out.append(ch.lower())
    if buf:
        out.append(int(buf))
    return out


def plot_group(
    df: pd.DataFrame,
    x_axis: str,
    metric: str,
    color: str,
    label: str,
) -> None:
    """Plot scatter points connected by lines."""
    df_sorted = df.sort_values(by=x_axis)
    plt.scatter(df_sorted[x_axis], df_sorted[metric], color=color, s=15, alpha=0.7, label=label)
    plt.plot(df_sorted[x_axis], df_sorted[metric], color=color, linewidth=1.5, alpha=0.7)


def _read_csv_safe(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None
    except Exception:
        logger.exception("Error reading %s", path)
        return None


def _group_csvs_by_at(folder_path: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for name in os.listdir(folder_path):
        if not name.endswith(".csv"):
            continue
        at_val = extract_at_value(name)
        if at_val is None:
            continue
        grouped.setdefault(at_val, []).append(os.path.join(folder_path, name))
    return grouped


PINK_DARKEST = "#BF4173"
BLUE_DARKEST = "#2A64BF"


def _pick_color(
    label: str,
    at_val: str,
    pink_map: dict[str, str],
    blue_map: dict[str, str],
    use_darkest: bool = False,
) -> str:
    if label.startswith("es_"):
        return PINK_DARKEST if use_darkest else pink_map.get(at_val, "gray")
    if label.startswith("os_"):
        return BLUE_DARKEST if use_darkest else blue_map.get(at_val, "gray")
    return "gray"


def _axis_label(x_axis: str) -> tuple[str, str]:
    if x_axis == "server_latency_avg":
        return "Server Latency Avg (ms)", "Server Latency"
    if x_axis == "latency_avg":
        return "Latency Avg (ms)", "Latency"
    return x_axis.replace("_", " ").title(), x_axis.split("_", 1)[0].title()


def _format_max_label(at_val: str, y: float) -> str:
    y_str = f"{y:.6f}".rstrip("0").rstrip(".")
    return f"@{at_val} max {y_str}"


def _max_recall_per_at(
    grouped_csvs: dict[str, list[str]],
    sorted_at_vals: list[str],
    r_val: str,
    metric: str,
) -> dict[str, float]:
    """Return the maximum metric value achieved by any engine for each @N value."""
    result: dict[str, float] = {}
    for at_val in sorted_at_vals:
        max_val: Optional[float] = None
        for csv_path in grouped_csvs[at_val]:
            df = _read_csv_safe(csv_path)
            if df is None or "s_n_r_value" not in df.columns or metric not in df.columns:
                continue
            filtered = df[df["s_n_r_value"].map(extract_r_value_from_row) == r_val].copy()
            filtered[metric] = pd.to_numeric(filtered[metric], errors="coerce")
            filtered = filtered.dropna(subset=[metric])
            if filtered.empty:
                continue
            val = float(filtered[metric].max())
            max_val = val if max_val is None else max(max_val, val)
        if max_val is not None:
            result[at_val] = max_val
    return result


def _plot_one_figure(
    folder_path: str,
    grouped_csvs: dict[str, list[str]],
    sorted_at_vals: list[str],
    pink_map: dict[str, str],
    blue_map: dict[str, str],
    green_map: dict[str, str],
    *,
    x_axis: str,
    metric: str,
    r_val: str,
    at_filter: Optional[str],
    output_dir: str,
    use_log_scale: bool = True,
    scale_prefix: str = "log",
) -> bool:
    if at_filter is None:
        at_vals = sorted_at_vals
        title_suffix = " (All @N)"
        out_name = f"{scale_prefix}_{metric}_{x_axis}_R{r_val}_all_atN.png"
    else:
        at_vals = [at_filter]
        title_suffix = f" — @{at_filter} R{r_val}"
        out_name = f"{scale_prefix}_{metric}_{x_axis}_at{at_filter}_R{r_val}.png"
    max_recall_map = _max_recall_per_at(grouped_csvs, at_vals, r_val, metric)
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_any = False
    legend_items: list[tuple[str, object]] = []
    all_x: list[float] = []
    all_y: list[float] = []
    for at_val in at_vals:
        for csv_path in sorted(
            grouped_csvs[at_val], key=lambda p: natural_sort_key(os.path.basename(p))
        ):
            df = _read_csv_safe(csv_path)
            if df is None:
                continue
            if (
                "s_n_r_value" not in df.columns
                or x_axis not in df.columns
                or metric not in df.columns
            ):
                continue
            df = df[df["s_n_r_value"].map(extract_r_value_from_row) == r_val].copy()
            df[x_axis] = pd.to_numeric(df[x_axis], errors="coerce")
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
            df = df.dropna(subset=[x_axis, metric])
            if df.empty:
                continue
            df = df.sort_values(by=x_axis)
            label = os.path.splitext(os.path.basename(csv_path))[0]
            color = _pick_color(label, at_val, pink_map, blue_map, use_darkest=(at_filter is not None))
            plot_group(df, x_axis, metric, color, label)
            legend_items.append((label, plt.Line2D([], [], color=color, linewidth=2.5)))
            all_x.extend(df[x_axis].astype(float).tolist())
            all_y.extend(df[metric].astype(float).tolist())
            plotted_any = True
    if not plotted_any:
        plt.close(fig)
        return False
    x_label, title_x = _axis_label(x_axis)
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric.title())
    if at_filter is None:
        ax.set_title(f"{metric.title()} vs {title_x} — Rescore {r_val}{title_suffix}")
    else:
        ax.set_title(f"{metric.title()} vs {title_x}{title_suffix}")
    ax.grid(True)
    if use_log_scale:
        ax.set_xscale("log")
    if all_x and all_y:
        x_min = min(all_x) * 0.95 if use_log_scale else 0
        ax.set_xlim(x_min, max(all_x) * 1.05)
        # Ensure all points fit: use max of actual data and max_recall_map
        upper_limit = max(all_y) * 1.05
        if max_recall_map:
            upper_limit = max(upper_limit, max(max_recall_map.values()) * 1.05)
        ax.set_ylim(min(all_y) * 0.95, upper_limit)
    for at_val in at_vals:
        y = max_recall_map.get(at_val)
        if y is None:
            continue
        color = green_map.get(at_val, "green")
        ax.axhline(y=y, color=color, linestyle="--", linewidth=1, alpha=0.8)
        legend_items.append(
            (
                _format_max_label(at_val, y),
                plt.Line2D([], [], color=color, linestyle="--", linewidth=1),
            )
        )
    legend_items.sort(key=lambda t: natural_sort_key(t[0]))
    ax.legend(
        [ln for _, ln in legend_items],
        [lbl for lbl, _ in legend_items],
        loc="upper left" if at_filter is None else "best",
        bbox_to_anchor=(1.02, 1.0) if at_filter is None else None,
        frameon=True,
        fontsize=8,
    )
    fig.tight_layout()
    out_path = os.path.join(output_dir, out_name)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return True


def generate_analysis_plots(folder_path: str, output_dir: str) -> None:
    grouped_csvs = _group_csvs_by_at(folder_path)
    if not grouped_csvs:
        logger.warning("No valid CSV files with @N pattern found in %s", folder_path)
        return

    log_scale_dir = os.path.join(output_dir, "log_scale")
    linear_scale_dir = os.path.join(output_dir, "linear_scale")
    os.makedirs(log_scale_dir, exist_ok=True)
    os.makedirs(linear_scale_dir, exist_ok=True)

    sorted_at_vals = sorted(grouped_csvs.keys(), key=int)
    pink_shades = ["#F4A6B8", "#E68CA6", "#D97395", "#CC5A84", "#BF4173"]
    blue_shades = ["#8CB4F4", "#73A0E6", "#5A8CD9", "#4178CC", "#2A64BF"]
    green_shades = ["#66BB6A", "#4CAF50", "#43A047", "#388E3C", "#2E7D32"]
    pink_map = {at: pink_shades[i % len(pink_shades)] for i, at in enumerate(sorted_at_vals)}
    blue_map = {at: blue_shades[i % len(blue_shades)] for i, at in enumerate(sorted_at_vals)}
    green_map = {at: green_shades[i % len(green_shades)] for i, at in enumerate(sorted_at_vals)}

    metrics = ["recall"]
    x_axes = [
        "server_latency_avg",
        "latency_avg",
        "latency_median",
        "latency_p90",
        "latency_p95",
        "latency_p99",
        "server_latency_median",
        "server_latency_p90",
        "server_latency_p95",
        "server_latency_p99",
        "throughput",
    ]
    r_vals = ["1", "2", "3"]
    plot_count = 0

    scales = [
        {"use_log": True, "prefix": "log", "dir": log_scale_dir},
        {"use_log": False, "prefix": "linear", "dir": linear_scale_dir},
    ]

    for scale_info in scales:
        for metric in metrics:
            for x_axis in x_axes:
                use_log_scale = scale_info["use_log"] and x_axis != "throughput"

                if scale_info["prefix"] == "log" and x_axis == "throughput":
                    continue

                for r_val in r_vals:
                    if _plot_one_figure(
                        folder_path,
                        grouped_csvs,
                        sorted_at_vals,
                        pink_map,
                        blue_map,
                        green_map,
                        x_axis=x_axis,
                        metric=metric,
                        r_val=r_val,
                        at_filter=None,
                        output_dir=scale_info["dir"],
                        use_log_scale=use_log_scale,
                        scale_prefix=scale_info["prefix"] if x_axis != "throughput" else "linear",
                    ):
                        plot_count += 1
        for at_val in sorted_at_vals:
            for metric in metrics:
                for x_axis in x_axes:
                    use_log_scale = scale_info["use_log"] and x_axis != "throughput"
                    if scale_info["prefix"] == "log" and x_axis == "throughput":
                        continue

                    for r_val in r_vals:
                        if _plot_one_figure(
                            folder_path,
                            grouped_csvs,
                            sorted_at_vals,
                            pink_map,
                            blue_map,
                            green_map,
                            x_axis=x_axis,
                            metric=metric,
                            r_val=r_val,
                            at_filter=at_val,
                            output_dir=scale_info["dir"],
                            use_log_scale=use_log_scale,
                            scale_prefix=(
                                scale_info["prefix"] if x_axis != "throughput" else "linear"
                            ),
                        ):
                            plot_count += 1

    if plot_count == 0:
        logger.warning("No plots generated for %s", folder_path)
    else:
        logger.info("Generated %d analysis plots", plot_count)

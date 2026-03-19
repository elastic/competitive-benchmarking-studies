from __future__ import annotations
import logging
import os
from collections import defaultdict
from typing import Optional
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


def is_relevant_file(filename: str) -> bool:
    if not filename.endswith(".csv"):
        return False
    if not (filename.startswith("es_") or filename.startswith("os_")):
        return False
    return extract_at_value(filename) is not None


def _is_date_folder(name: str) -> bool:
    return len(name) == 8 and name.isdigit()


def _read_grouped_csvs(folder_path: str, prefix: str) -> dict[str, pd.DataFrame]:
    grouped: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for name in os.listdir(folder_path):
        if not name.startswith(prefix):
            continue
        if not is_relevant_file(name):
            continue
        at_val = extract_at_value(name)
        if not at_val:
            continue
        grouped[at_val].append(pd.read_csv(os.path.join(folder_path, name)))
    return {at_val: pd.concat(dfs, ignore_index=True) for at_val, dfs in grouped.items()}


def _recall_to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.rstrip("%"), errors="coerce")


def _latency_col_to_label(latency_col: str) -> str:
    """Convert latency column name to human-readable label."""
    col = latency_col.replace("server_", "Server ")
    if "_avg" in col:
        return col.replace("_avg", "").replace("latency", "Latency").replace("_", " ").title().replace("Latency", "Avg Latency")
    if "_median" in col:
        return col.replace("_median", "").replace("latency", "Latency").replace("_", " ").title().replace("Latency", "Median Latency")
    if "_p90" in col:
        return col.replace("_p90", "").replace("latency", "Latency").replace("_", " ").title().replace("Latency", "P90 Latency")
    if "_p95" in col:
        return col.replace("_p95", "").replace("latency", "Latency").replace("_", " ").title().replace("Latency", "P95 Latency")
    if "_p99" in col:
        return col.replace("_p99", "").replace("latency", "Latency").replace("_", " ").title().replace("Latency", "P99 Latency")
    return col.replace("_", " ").title()


def _require_columns(df: pd.DataFrame, cols: tuple[str, ...], *, context: str) -> bool:
    missing = [col for col in cols if col not in df.columns]
    if not missing:
        return True
    logger.warning("Missing columns %s (%s)", missing, context)
    return False


def generate_comparison(input_folder_path: str, output_folder_path: str, latency_col: str = "server_latency_avg") -> Optional[str]:
    try:
        es_by_at = _read_grouped_csvs(input_folder_path, "es_")
        os_by_at = _read_grouped_csvs(input_folder_path, "os_")
        if not es_by_at:
            logger.warning("No Elasticsearch files found in %s", input_folder_path)
            return None
        if not os_by_at:
            logger.warning("No OpenSearch files found in %s", input_folder_path)
            return None
        common_at_vals = sorted(set(es_by_at) & set(os_by_at), key=int)
        if not common_at_vals:
            logger.warning("No common @N values found between ES and OS in %s", input_folder_path)
            return None
        rows: list[dict] = []
        for at_val in common_at_vals:
            es_df = es_by_at[at_val].copy()
            os_df = os_by_at[at_val].copy()
            needed = ("param_key", "recall", latency_col, "throughput")
            if not _require_columns(es_df, needed, context=f"ES @{at_val}"):
                continue
            if not _require_columns(os_df, needed, context=f"OS @{at_val}"):
                continue
            merged = es_df.merge(os_df, on="param_key", suffixes=("_es", "_os"))
            if merged.empty:
                continue
            es_recall = _recall_to_float(merged["recall_es"])
            os_recall = _recall_to_float(merged["recall_os"])
            recall_pct = (es_recall - os_recall) / os_recall.replace({0: pd.NA}) * 100
            es_lat = pd.to_numeric(merged[f"{latency_col}_es"], errors="coerce")
            os_lat = pd.to_numeric(merged[f"{latency_col}_os"], errors="coerce")
            latency_xs = os_lat / es_lat.replace({0: pd.NA})

            es_throughput = pd.to_numeric(merged["throughput_es"], errors="coerce")
            os_throughput = pd.to_numeric(merged["throughput_os"], errors="coerce")
            throughput_xs = es_throughput / os_throughput.replace({0: pd.NA})

            latency_label = _latency_col_to_label(latency_col)
            for i in range(len(merged)):
                rp = recall_pct.iat[i]
                lx = latency_xs.iat[i]
                tx = throughput_xs.iat[i]
                rows.append(
                    {
                        "System": "Elasticsearch",
                        "param_key": merged.at[i, "param_key"],
                        "recall": es_recall.iat[i],
                        latency_label: es_lat.iat[i],
                        "throughput": es_throughput.iat[i],
                        "Recall %": "N/A" if pd.isna(rp) else f"{rp:.2f}%",
                        "Latency Xs": "N/A" if pd.isna(lx) else f"{lx:.2f}",
                        "Throughput Xs": "N/A" if pd.isna(tx) else f"{tx:.2f}",
                    }
                )
                rows.append(
                    {
                        "System": "OpenSearch",
                        "param_key": merged.at[i, "param_key"],
                        "recall": os_recall.iat[i],
                        latency_label: os_lat.iat[i],
                        "throughput": os_throughput.iat[i],
                        "Recall %": "N/A" if pd.isna(rp) else f"{rp:.2f}%",
                        "Latency Xs": "N/A" if pd.isna(lx) else f"{lx:.2f}",
                        "Throughput Xs": "N/A" if pd.isna(tx) else f"{tx:.2f}",
                    }
                )
        if not rows:
            logger.warning("No comparable data found in %s", input_folder_path)
            return None
        output_filename = f"{latency_col.replace('_avg', '')}_comparison.csv"
        out_path = os.path.join(output_folder_path, output_filename)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        logger.info("Saved %s", out_path)
        return out_path
    except Exception:
        logger.exception("Error generating comparison for %s", input_folder_path)
        return None


def generate_summary_comparison(input_folder_path: str, output_folder_path: str, latency_col: str = "server_latency_avg") -> Optional[str]:
    try:
        es_by_at = _read_grouped_csvs(input_folder_path, "es_")
        os_by_at = _read_grouped_csvs(input_folder_path, "os_")
        if not es_by_at or not os_by_at:
            logger.warning("Missing ES or OS files in %s", input_folder_path)
            return None
        common_at_vals = sorted(set(es_by_at) & set(os_by_at), key=int)
        if not common_at_vals:
            logger.warning("No common @N values found in %s", input_folder_path)
            return None
        rows: list[dict] = []
        for at_val in common_at_vals:
            es_df = es_by_at[at_val].copy()
            os_df = os_by_at[at_val].copy()
            needed = ("recall", latency_col)
            if not _require_columns(es_df, needed, context=f"ES @{at_val}"):
                continue
            if not _require_columns(os_df, needed, context=f"OS @{at_val}"):
                continue
            es_df["recall_float"] = _recall_to_float(es_df["recall"])
            os_df["recall_float"] = _recall_to_float(os_df["recall"])
            es_max = es_df["recall_float"].max(skipna=True)
            os_max = os_df["recall_float"].max(skipna=True)
            if pd.isna(es_max) or pd.isna(os_max):
                continue
            max_recall = min(es_max, os_max)
            tol = max_recall * 0.9999
            es_at = es_df.loc[es_df["recall_float"] >= tol]
            os_at = os_df.loc[os_df["recall_float"] >= tol]
            if es_at.empty or os_at.empty:
                continue
            es_min_lat = pd.to_numeric(es_at[latency_col], errors="coerce").min(skipna=True)
            os_min_lat = pd.to_numeric(os_at[latency_col], errors="coerce").min(skipna=True)
            if pd.isna(es_min_lat) or pd.isna(os_min_lat):
                continue
            speedup = (os_min_lat / es_min_lat) if es_min_lat > 0 else pd.NA
            latency_label = _latency_col_to_label(latency_col)
            rows.append(
                {
                    "Recall@N": f"@{at_val}",
                    "Max Recall": f"{max_recall:.4f}".rstrip("0").rstrip("."),
                    f"Elasticsearch {latency_label} (ms)": f"{es_min_lat:.2f}",
                    f"OpenSearch {latency_label} (ms)": f"{os_min_lat:.2f}",
                    "Speedup": "N/A" if pd.isna(speedup) else f"{speedup:.2f}",
                }
            )
        if not rows:
            logger.warning("No summary data generated for %s", input_folder_path)
            return None
        output_filename = f"{latency_col.replace('_avg', '')}_summary_comparison.csv"
        out_path = os.path.join(output_folder_path, output_filename)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        logger.info("Saved %s", out_path)
        return out_path
    except Exception:
        logger.exception("Error generating summary comparison for %s", input_folder_path)
        return None


def generate_throughput_summary_comparison(input_folder_path: str, output_folder_path: str) -> Optional[str]:
    try:
        es_by_at = _read_grouped_csvs(input_folder_path, "es_")
        os_by_at = _read_grouped_csvs(input_folder_path, "os_")
        if not es_by_at or not os_by_at:
            logger.warning("Missing ES or OS files for throughput summary in %s", input_folder_path)
            return None
        common_at_vals = sorted(set(es_by_at) & set(os_by_at), key=int)
        if not common_at_vals:
            logger.warning("No common @N values found for throughput summary in %s", input_folder_path)
            return None
        rows: list[dict] = []
        for at_val in common_at_vals:
            es_df = es_by_at[at_val].copy()
            os_df = os_by_at[at_val].copy()
            needed = ("recall", "throughput")
            if not _require_columns(es_df, needed, context=f"ES @{at_val}") or not _require_columns(os_df, needed, context=f"OS @{at_val}"):
                continue
            
            es_df["recall_float"] = _recall_to_float(es_df["recall"])
            os_df["recall_float"] = _recall_to_float(os_df["recall"])
            
            es_max_recall = es_df["recall_float"].max(skipna=True)
            os_max_recall = os_df["recall_float"].max(skipna=True)
            if pd.isna(es_max_recall) or pd.isna(os_max_recall):
                continue

            max_recall = min(es_max_recall, os_max_recall)
            tol = max_recall * 0.9999
            es_at = es_df.loc[es_df["recall_float"] >= tol]
            os_at = os_df.loc[os_df["recall_float"] >= tol]

            if es_at.empty or os_at.empty:
                continue

            es_max_throughput = pd.to_numeric(es_at["throughput"], errors="coerce").max(skipna=True)
            os_max_throughput = pd.to_numeric(os_at["throughput"], errors="coerce").max(skipna=True)
            
            if pd.isna(es_max_throughput) or pd.isna(os_max_throughput):
                continue

            throughput_speedup = es_max_throughput / os_max_throughput if os_max_throughput > 0 else pd.NA
            
            rows.append({
                "Recall@N": f"@{at_val}",
                "Max Recall": f"{max_recall:.4f}".rstrip("0").rstrip("."),
                "Elasticsearch Throughput": "N/A" if pd.isna(es_max_throughput) else f"{es_max_throughput:.2f}",
                "OpenSearch Throughput": "N/A" if pd.isna(os_max_throughput) else f"{os_max_throughput:.2f}",
                "Speedup": "N/A" if pd.isna(throughput_speedup) else f"{throughput_speedup:.2f}",
            })

        if not rows:
            logger.warning("No summary throughput data generated for %s", input_folder_path)
            return None
        
        out_path = os.path.join(output_folder_path, "throughput_summary_comparison.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        logger.info("Saved %s", out_path)
        return out_path
    except Exception:
        logger.exception("Error generating throughput summary comparison for %s", input_folder_path)
        return None



def run_comparison(results_dir: str, target_date: Optional[str] = None) -> int:
    folders: list[tuple[str, str]] = []
    if target_date is not None:
        date_folder = os.path.join(results_dir, target_date)
        if not os.path.isdir(date_folder):
            logger.warning("Date folder not found: %s", date_folder)
            return 0
        folders.append((target_date, date_folder))
    else:
        for name in os.listdir(results_dir):
            if not _is_date_folder(name):
                continue
            folder_path = os.path.join(results_dir, name)
            if os.path.isdir(folder_path):
                folders.append((name, folder_path))
    if not folders:
        logger.warning("No date folders found to compare")
        return 0
    folders.sort(key=lambda x: x[0])
    count = 0

    latency_cols = [
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
    ]

    for date_str, folder_path in folders:
        logger.info("Generating comparison for %s...", date_str)

        compare_output_dir = os.path.join(folder_path, "compare")
        os.makedirs(compare_output_dir, exist_ok=True)

        # Generate comparison and summary for all latency columns
        for latency_col in latency_cols:
            if generate_comparison(folder_path, compare_output_dir, latency_col=latency_col):
                count += 1
            if generate_summary_comparison(folder_path, compare_output_dir, latency_col=latency_col):
                count += 1

        # Generate throughput summary
        if generate_throughput_summary_comparison(folder_path, compare_output_dir):
            count += 1

    return count

from __future__ import annotations
import logging
import os
from typing import Optional
from .analysis_plot import generate_analysis_plots
from .overview_plot import generate_overview_plots

logger = logging.getLogger(__name__)


def _is_date_folder(name: str) -> bool:
    return len(name) == 8 and name.isdigit()


def run_plots(
    results_dir: str,
    target_date: Optional[str] = None,
    latency_col: str = "server_latency_avg",
    overview_sample: int = 0,
    overview_seed: Optional[int] = None,
) -> dict[str, bool]:
    results = {"overview": False, "log": False}
    folders_to_plot: list[tuple[str, str]] = []
    if target_date is not None:
        date_folder = os.path.join(results_dir, target_date)
        if not os.path.isdir(date_folder):
            logger.warning("Date folder not found: %s", date_folder)
            return results
        folders_to_plot.append((target_date, date_folder))
    else:
        for name in os.listdir(results_dir):
            if not _is_date_folder(name):
                continue
            folder_path = os.path.join(results_dir, name)
            if os.path.isdir(folder_path):
                folders_to_plot.append((name, folder_path))
    if not folders_to_plot:
        logger.warning("No date folders found to plot")
        return results
    folders_to_plot.sort(key=lambda x: x[0])
    for date_str, folder_path in folders_to_plot:
        logger.info("Generating plots for %s...", date_str)
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        has_es = any(f.startswith("es_") for f in csv_files)
        has_os = any(f.startswith("os_") for f in csv_files)
        if has_es and not has_os:
            logger.info("Note: %s has ES files but no OS files", date_str)
        elif has_os and not has_es:
            logger.info("Note: %s has OS files but no ES files", date_str)
        try:
            overview_output_dir = os.path.join(folder_path, "plots", "overview")
            os.makedirs(overview_output_dir, exist_ok=True)
            generate_overview_plots(
                folder_path,
                output_dir=overview_output_dir,
                sample_n=overview_sample,
                seed=overview_seed,
            )
            results["overview"] = True
        except Exception:
            logger.exception("Error generating overview plots for %s", date_str)
        try:
            log_output_dir = os.path.join(folder_path, "plots", "analysis")
            os.makedirs(log_output_dir, exist_ok=True)
            generate_analysis_plots(folder_path, output_dir=log_output_dir)
            results["analysis"] = True
        except Exception:
            logger.exception("Error generating analysis plots for %s", date_str)
    return results

from __future__ import annotations
import logging
import os
import shutil
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def extract_datetime_from_filename(filename: str) -> Optional[datetime]:
    base = os.path.basename(filename)
    if not base.endswith(".csv"):
        return None
    stem = base[:-4]
    if "_" not in stem:
        return None
    date_part, time_part = stem.rsplit("_", 1)
    if len(date_part) != 8 or len(time_part) != 6:
        return None
    if not (date_part.isdigit() and time_part.isdigit()):
        return None
    try:
        return datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def extract_date_from_filename(filename: str) -> Optional[str]:
    dt = extract_datetime_from_filename(filename)
    return dt.strftime("%Y%m%d") if dt is not None else None


def group_files_by_test_run(
    results_dir: str,
    target_dates: Optional[list[str]] = None,
) -> dict[str, list[str]]:
    csv_files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]
    if not csv_files:
        return {}
    file_times: list[tuple[datetime, str, str]] = []
    for name in csv_files:
        dt = extract_datetime_from_filename(name)
        if dt is None:
            continue
        date_str = dt.strftime("%Y%m%d")
        if target_dates is not None and date_str not in target_dates:
            continue
        file_times.append((dt, name, date_str))
    if not file_times:
        return {}
    file_times.sort(key=lambda x: x[0])
    groups: dict[str, list[str]] = defaultdict(list)
    current_group_start: Optional[datetime] = None
    current_group_id: Optional[str] = None
    for dt, filename, date_str in file_times:
        if current_group_start is None or (dt - current_group_start) > timedelta(hours=36):
            current_group_start = dt
            current_group_id = date_str
        groups[current_group_id].append(os.path.join(results_dir, filename))
    return dict(groups)


def _is_date_folder(name: str) -> bool:
    return len(name) == 8 and name.isdigit()


def organize_results_by_date(
    results_dir: str,
    target_date: Optional[str] = None,
) -> dict[str, str]:
    root_csv_files = [
        name
        for name in os.listdir(results_dir)
        if name.endswith(".csv") and os.path.isfile(os.path.join(results_dir, name))
    ]
    if not root_csv_files:
        organized_dirs: dict[str, str] = {}
        for name in os.listdir(results_dir):
            path = os.path.join(results_dir, name)
            if _is_date_folder(name) and os.path.isdir(path):
                if target_date is None or name == target_date:
                    organized_dirs[name] = path
        return organized_dirs
    target_dates = [target_date] if target_date is not None else None
    groups = group_files_by_test_run(results_dir, target_dates)
    if not groups:
        if target_date is None:
            logger.warning("No CSV files found to organize in %s", results_dir)
        else:
            logger.warning(
                "No CSV files found to organize in %s for date %s", results_dir, target_date
            )
        return {}
    organized_dirs: dict[str, str] = {}
    for date_id, file_paths in groups.items():
        date_dir = os.path.join(results_dir, date_id)
        os.makedirs(date_dir, exist_ok=True)
        copied = 0
        for src_path in file_paths:
            filename = os.path.basename(src_path)
            dst_path = os.path.join(date_dir, filename)
            if os.path.abspath(src_path) == os.path.abspath(dst_path):
                continue
            shutil.copy2(src_path, dst_path)
            copied += 1
        logger.info("Organized %d files into %s/", len(file_paths), date_id)
        if copied:
            logger.info("Copied %d files into %s/", copied, date_id)
        organized_dirs[date_id] = date_dir
    return organized_dirs


def get_available_dates(results_dir: str) -> list[str]:
    dates: set[str] = set()
    for name in os.listdir(results_dir):
        if not name.endswith(".csv"):
            continue
        date_str = extract_date_from_filename(name)
        if date_str is not None:
            dates.add(date_str)
    return sorted(dates)

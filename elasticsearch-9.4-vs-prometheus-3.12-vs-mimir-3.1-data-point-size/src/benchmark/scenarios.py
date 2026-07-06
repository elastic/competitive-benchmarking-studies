"""Resolves a named benchmark scenario (scenarios/<name>.yml) to a merged
config object, following its `queryset` reference into querysets/<name>.yml.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# This file lives at <repo_root>/src/benchmark/scenarios.py — scenarios/ and
# querysets/ are repo-root directories, three levels up from here.
_REPO_ROOT = Path(__file__).parent.parent.parent
_SCENARIOS_DIR = _REPO_ROOT / "scenarios"
_QUERYSETS_DIR = _REPO_ROOT / "querysets"


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    scale: int
    interval: str
    start_now_minus: str
    seed: int
    queryset_path: Path
    # ClickHouse only: inline `CREATE TABLE %s (...)` DDL per table (%s is
    # substituted with `<database>.<table>` at bootstrap time), kept in the
    # scenario file rather than fetched from an external URL at run time so
    # the schema a result was measured against is pinned and reviewable
    # alongside the scenario itself.
    clickhouse_database: str = "default"
    clickhouse_schema: dict[str, str] = field(default_factory=dict)


def load_benchmark(name: str) -> BenchmarkScenario:
    """Reads scenarios/<name>.yml, resolves its queryset path, returns a
    merged config object."""
    path = _SCENARIOS_DIR / f"{name}.yml"
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Benchmark {name!r} not found at {path}") from None

    ingest = data["ingest"]
    clickhouse = data.get("clickhouse", {})
    return BenchmarkScenario(
        name=data.get("name", name),
        scale=ingest["scale"],
        interval=ingest["interval"],
        start_now_minus=ingest["start_now_minus"],
        seed=ingest["seed"],
        queryset_path=_QUERYSETS_DIR / f"{data['queryset']}.yml",
        clickhouse_database=clickhouse.get("database", "default"),
        clickhouse_schema=clickhouse.get("schema", {}),
    )

"""Shared engine selection and connection config, used by load and disk_usage."""

import os
import sys

_ENGINES = {
    "elasticsearch": {
        "url_env": "ELASTICSEARCH_URL",
        "otlp_path": "/_otlp",
        "version_env": "ES_VERSION",
    },
    "prometheus": {
        "url_env": "PROMETHEUS_URL",
        "otlp_path": "/api/v1/otlp",
        "version_env": "PROMETHEUS_VERSION",
    },
    "mimir": {
        "url_env": "MIMIR_URL",
        "otlp_path": "/otlp",
        "version_env": "MIMIR_VERSION",
    },
    "clickhouse": {
        # ClickHouse has no OTLP endpoint — the OTel `clickhouse` exporter
        # writes over the native TCP protocol, not HTTP/OTLP. url_env is the
        # HTTP interface (schema bootstrap + queries); native_endpoint_env is
        # what the exporter actually connects to.
        "url_env": "CLICKHOUSE_URL",
        "native_endpoint_env": "CLICKHOUSE_NATIVE_ENDPOINT",
        "version_env": "CLICKHOUSE_VERSION",
    },
}


def _require(env_var: str) -> str:
    value = os.environ.get(env_var)
    if not value:
        sys.exit(
            f"{env_var} must be set (ENGINE={ENGINE!r}) — set it in the "
            "Makefile target's `export` block or the environment."
        )
    return value


ENGINE = os.environ.get("ENGINE")
if ENGINE not in _ENGINES:
    sys.exit(f"ENGINE must be set to one of {tuple(_ENGINES)} (got {ENGINE!r}).")

_info = _ENGINES[ENGINE]
RESULTS_FILE = _require("RESULTS_FILE")
BASE_URL = _require(_info["url_env"])
VERSION = os.environ.get(_info["version_env"], "?")
DATA_STREAM = _require("ES_DATA_STREAM") if ENGINE == "elasticsearch" else None

if ENGINE == "clickhouse":
    # The exporter connects over the native protocol, not BASE_URL (HTTP).
    # No auth: the benchmark's ClickHouse container runs with the `default`
    # user and no password (see docker-compose.yml).
    EXPORT_ENDPOINT = _require(_info["native_endpoint_env"])
    CLICKHOUSE_DATABASE = os.environ.get("CLICKHOUSE_DATABASE", "default")
else:
    EXPORT_ENDPOINT = BASE_URL + _info["otlp_path"]
    CLICKHOUSE_DATABASE = None

# This file lives at <repo_root>/src/benchmark/engine_config.py — data/ is a
# repo-root artifact directory, three levels up from here.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = os.path.join(_REPO_ROOT, "data", ENGINE)

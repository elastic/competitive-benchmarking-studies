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
OTLP_ENDPOINT = BASE_URL + _info["otlp_path"]
VERSION = os.environ.get(_info["version_env"], "?")
DATA_STREAM = _require("ES_DATA_STREAM") if ENGINE == "elasticsearch" else None

# This file lives at <repo_root>/src/benchmark/engine_config.py — data/ is a
# repo-root artifact directory, three levels up from here.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = os.path.join(_REPO_ROOT, "data", ENGINE)

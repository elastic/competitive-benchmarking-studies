"""Runtime configuration derived from environment variables."""

import os
import re

_DURATION_RE = re.compile(r"^(\d+)(s|m|h)$")


def parse_duration_seconds(value: str) -> int:
    m = _DURATION_RE.match(value.strip())
    if not m:
        raise ValueError(
            f"Cannot parse duration {value!r} — use e.g. '270m', '4h', '30s'"
        )
    amount, unit = int(m.group(1)), m.group(2)
    return amount * {"s": 1, "m": 60, "h": 3600}[unit]


SEED = 123
SCALE = int(os.environ.get("SCALE", "100"))
INTERVAL = os.environ.get("INTERVAL", "1s")
START_NOW_MINUS = os.environ.get("START_NOW_MINUS", "270m")
RESULTS_FILE = os.environ.get("RESULTS_FILE")  # optional: path to write JSON result

_ES_URL = os.environ.get("ELASTICSEARCH_URL")
_PROM_URL = os.environ.get("PROMETHEUS_URL")
_MIMIR_URL = os.environ.get("MIMIR_URL")

if _MIMIR_URL:
    ENGINE = "mimir"
    OTLP_ENDPOINT = f"{_MIMIR_URL}/otlp"
    DATA_STREAM = None
elif _PROM_URL:
    ENGINE = "prometheus"
    OTLP_ENDPOINT = f"{_PROM_URL}/api/v1/otlp"
    DATA_STREAM = None
elif _ES_URL:
    ENGINE = "elasticsearch"
    OTLP_ENDPOINT = f"{_ES_URL}/_otlp"
    DATA_STREAM = "metrics-hostmetricsreceiver.otel-default"
else:
    ENGINE = "elasticsearch"
    OTLP_ENDPOINT = "http://localhost:9200/_otlp"
    DATA_STREAM = "metrics-hostmetricsreceiver.otel-default"

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_HERE), "data", ENGINE)

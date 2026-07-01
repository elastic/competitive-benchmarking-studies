"""Shared engine selection and connection config, used by load and disk_usage."""

import os

ES_URL = os.environ.get("ELASTICSEARCH_URL")
PROM_URL = os.environ.get("PROMETHEUS_URL")
MIMIR_URL = os.environ.get("MIMIR_URL")
RESULTS_FILE = os.environ.get("RESULTS_FILE")  # optional: path to write JSON result

if MIMIR_URL:
    ENGINE = "mimir"
    OTLP_ENDPOINT = f"{MIMIR_URL}/otlp"
    DATA_STREAM = None
elif PROM_URL:
    ENGINE = "prometheus"
    OTLP_ENDPOINT = f"{PROM_URL}/api/v1/otlp"
    DATA_STREAM = None
elif ES_URL:
    ENGINE = "elasticsearch"
    OTLP_ENDPOINT = f"{ES_URL}/_otlp"
    DATA_STREAM = "metrics-hostmetricsreceiver.otel-default"
else:
    ENGINE = "elasticsearch"
    OTLP_ENDPOINT = "http://localhost:9200/_otlp"
    DATA_STREAM = "metrics-hostmetricsreceiver.otel-default"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", ENGINE)

_VERSION_ENV = {
    "elasticsearch": "ES_VERSION",
    "prometheus": "PROMETHEUS_VERSION",
    "mimir": "MIMIR_VERSION",
}[ENGINE]
VERSION = os.environ.get(_VERSION_ENV, "?")

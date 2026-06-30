import os

from .collector import run
from .config import DATA_DIR, ENGINE, INTERVAL, OTLP_ENDPOINT, SCALE, START_NOW_MINUS
from .report import report_elasticsearch, report_mimir, report_prometheus


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    print(
        f"engine={ENGINE}  scale={SCALE}  interval={INTERVAL}  window={START_NOW_MINUS}"
    )
    print(f"target={OTLP_ENDPOINT}")

    datapoints, rate, elapsed, start_ts, end_ts = run()

    if datapoints:
        print(
            f"Ingested: {datapoints:,} data points  ({rate:,.0f} dp/s)  in {elapsed:.1f}s"
        )
    else:
        print(f"metricsgenreceiver completed in {elapsed:.1f}s")

    if ENGINE == "elasticsearch":
        report_elasticsearch(datapoints, start_ts, end_ts)
    elif ENGINE == "prometheus":
        report_prometheus(datapoints, start_ts, end_ts)
    elif ENGINE == "mimir":
        report_mimir(datapoints, start_ts, end_ts)

    print("\nDone.")

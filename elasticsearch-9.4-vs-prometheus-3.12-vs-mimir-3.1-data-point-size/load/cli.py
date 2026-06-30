import os

from .collector import run
from .config import DATA_DIR, ENGINE, INTERVAL, OTLP_ENDPOINT, SCALE, START_NOW_MINUS
from .report import report_elasticsearch, report_mimir, report_prometheus


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    print(
        f"engine={ENGINE}  scale={SCALE}  interval={INTERVAL}  window={START_NOW_MINUS}"
    )
    print(f"target={OTLP_ENDPOINT}")

    datapoints, rate, elapsed, start_ts, end_ts = run()

    if datapoints:
        print(
            f"Ingested: {datapoints:,} data points  ({rate:,.0f} dp/s)  in {_format_duration(elapsed)}"
        )
    else:
        print(f"metricsgenreceiver completed in {_format_duration(elapsed)}")

    if ENGINE == "elasticsearch":
        report_elasticsearch(datapoints, start_ts, end_ts, elapsed)
    elif ENGINE == "prometheus":
        report_prometheus(datapoints, start_ts, end_ts, elapsed)
    elif ENGINE == "mimir":
        report_mimir(datapoints, start_ts, end_ts, elapsed)

    print("\nDone.")

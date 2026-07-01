import os

from engine_config import DATA_DIR, ENGINE, OTLP_ENDPOINT, RESULTS_FILE, VERSION
from store.results import ResultStore

from .collector import run
from .config import INTERVAL, SCALE, START_NOW_MINUS


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
            f"Ingested: {datapoints:,} data points ({rate:,.0f} dp/s) in {_format_duration(elapsed)}"
        )
    else:
        print(f"metricsgenreceiver completed in {_format_duration(elapsed)}")

    if RESULTS_FILE:
        ResultStore(os.path.dirname(RESULTS_FILE)).save_ingest_result(
            ENGINE,
            VERSION,
            datapoints,
            start_ts=start_ts,
            end_ts=end_ts,
            elapsed_seconds=elapsed,
            path=RESULTS_FILE,
        )

    print("\nDone.")

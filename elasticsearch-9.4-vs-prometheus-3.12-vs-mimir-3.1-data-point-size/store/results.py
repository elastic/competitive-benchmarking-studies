import json
from pathlib import Path


class ResultStore:
    def __init__(self, results_dir: str) -> None:
        self._dir = Path(results_dir)

    def save_ingest_result(
        self,
        engine: str,
        version: str,
        datapoints: int,
        start_ts: int,
        end_ts: int,
        elapsed_seconds: float = 0.0,
        path: str | None = None,
    ) -> None:
        dest = Path(path) if path else self._dir / f"{engine}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        eps = round(datapoints / elapsed_seconds) if elapsed_seconds > 0 else 0
        record = {
            "engine": engine,
            "version": version,
            "datapoints": datapoints,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "elapsed_seconds": round(elapsed_seconds, 1),
            "eps": eps,
        }
        with open(dest, "w") as f:
            json.dump(record, f, indent=2)
        print(f"Result saved to {dest}")

    def load_time_range(self, engine: str) -> tuple[int, int] | None:
        path = self._dir / f"{engine}.json"
        try:
            with open(path) as f:
                data = json.load(f)
            return data["start_ts"], data["end_ts"]
        except (FileNotFoundError, KeyError):
            return None

    def save_storage_size(
        self, engine: str, size_bytes: int, path: str | None = None
    ) -> None:
        dest = Path(path) if path else self._dir / f"{engine}.json"
        try:
            with open(dest) as f:
                record = json.load(f)
        except FileNotFoundError:
            return None

        record["size_bytes"] = size_bytes
        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, "w") as f:
            json.dump(record, f, indent=2)

        print(f"Storage size saved to {dest}")

    def save_query_results(self, engine: str, reports: list) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{engine}.json"
        try:
            with open(path) as f:
                record = json.load(f)
        except FileNotFoundError:
            record = {"engine": engine}
        record["queries"] = [r.to_result_dict() for r in reports]
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"Query results saved to {path}")

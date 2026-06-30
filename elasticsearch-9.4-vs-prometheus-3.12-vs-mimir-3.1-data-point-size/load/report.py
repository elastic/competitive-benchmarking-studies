"""Post-ingest storage measurement and result persistence for each engine."""

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from store.results import ResultStore

from .config import (
    DATA_DIR,
    DATA_STREAM,
    RESULTS_FILE,
    _ES_URL,
    _MIMIR_URL,
    _PROM_URL,
)


def _es_request(method: str, path: str) -> tuple[int, object]:
    base = (_ES_URL or "http://localhost:9200").rstrip("/")
    req = urllib.request.Request(base + path, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def _prom_metric(base_url: str, query: str) -> float:
    try:
        qs = urllib.parse.urlencode({"query": query})
        with urllib.request.urlopen(f"{base_url}/api/v1/query?{qs}") as r:
            result = json.loads(r.read()).get("data", {}).get("result", [])
        return float(result[0]["value"][1]) if result else 0.0
    except Exception:
        return 0.0


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except OSError:
                pass
    return total


def _dir_size_excl(path: str, excl: set) -> int:
    """Directory size excluding named subdirectories (e.g. WAL, WBL, snapshots)."""
    total = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in excl]
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except OSError:
                pass
    return total


def _save_result(
    engine: str,
    version: str,
    datapoints: int,
    size_bytes: int,
    start_ts: int,
    end_ts: int,
    elapsed_seconds: float = 0.0,
) -> None:
    if not RESULTS_FILE:
        return
    ResultStore(os.path.dirname(RESULTS_FILE)).save_ingest_result(
        engine, version, datapoints, size_bytes, start_ts, end_ts,
        elapsed_seconds=elapsed_seconds, path=RESULTS_FILE,
    )


def report_elasticsearch(datapoints: int, start_ts: int, end_ts: int, elapsed: float = 0.0) -> None:
    print(f"Force-merging {DATA_STREAM} to 1 segment per shard ...")
    t1 = time.time()
    status, _ = _es_request(
        "POST",
        f"/{DATA_STREAM}/_forcemerge?max_num_segments=1&wait_for_completion=true",
    )
    print(f"Force-merge complete in {time.time() - t1:.1f}s (HTTP {status})")

    # Flush so that all merged data is fully written to disk before we measure.
    # Without this, _cat/indices dataset.size can reflect pre-merge segment sizes.
    _es_request("POST", f"/{DATA_STREAM}/_flush")
    time.sleep(5)

    def _parse_size(s: str) -> int:
        for tok in s.split():
            m = re.match(r"([\d.]+)(kb|mb|gb|b)", tok.lower())
            if m:
                n, u = float(m.group(1)), m.group(2)
                return int(n * {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3}[u])
        return 0

    _, stats = _es_request(
        "GET", f"/_cat/indices/.ds-{DATA_STREAM}*?format=json&h=docs.count,dataset.size"
    )
    docs, size_bytes = 0, 0
    if stats:
        for idx in stats:
            docs += int(idx.get("docs.count", 0))
            size_bytes += _parse_size(idx.get("dataset.size", "0b"))

    size_str = (
        f"{size_bytes / 1024**3:.1f}gb"
        if size_bytes >= 1024**3
        else f"{size_bytes / 1024**2:.1f}mb"
        if size_bytes >= 1024**2
        else f"{size_bytes}b"
    )
    bps = (
        f"  ({size_bytes / datapoints:.2f} bytes/dp)"
        if (size_bytes and datapoints)
        else ""
    )
    print(f"\nElasticsearch: {docs:,} docs  {size_str}{bps}")
    _save_result(
        "elasticsearch",
        os.environ.get("ES_VERSION", "?"),
        datapoints,
        size_bytes,
        start_ts,
        end_ts,
        elapsed,
    )


def report_prometheus(datapoints: int, start_ts: int, end_ts: int, elapsed: float = 0.0) -> None:
    """Measure Prometheus storage via TSDB snapshot.

    POST /api/v1/admin/tsdb/snapshot atomically flushes WAL + head into a clean
    snapshot block. We then measure the snapshot directory size via
    `docker exec du -sb` inside the container — this sidesteps Docker Desktop
    mmap visibility issues that affect bind-mounted files read from the host.

    Matches the methodology from https://github.com/gouthamve/prom-elastic-benchmark/blob/main/scripts/measure-prom.sh
    """
    base = _PROM_URL or "http://localhost:9090"

    print("Triggering TSDB snapshot (flushes WAL + head into blocks) ...", flush=True)
    req = urllib.request.Request(
        f"{base}/api/v1/admin/tsdb/snapshot", method="POST", data=b""
    )
    try:
        with urllib.request.urlopen(req) as r:
            resp = json.loads(r.read())
        snap_name = resp["data"]["name"]
        print(f"  snapshot: {snap_name}")
    except Exception as e:
        sys.exit(f"Snapshot failed: {e}")

    # Find the container that is actually serving the Prometheus port — not the
    # compose service name, which may resolve to a container that lost the port
    # race when another Prometheus instance was already running.
    prom_port = urllib.parse.urlparse(base).port or 9090
    cp = subprocess.run(
        ["docker", "ps", "--filter", f"publish={prom_port}", "--format", "{{.ID}}"],
        capture_output=True,
        text=True,
    )
    container_id = cp.stdout.strip().splitlines()[0] if cp.stdout.strip() else ""
    if not container_id:
        sys.exit(f"Could not find a running container publishing port {prom_port}")

    du = subprocess.run(
        [
            "docker",
            "exec",
            container_id,
            "du",
            "-sb",
            f"/prometheus/snapshots/{snap_name}",
        ],
        capture_output=True,
        text=True,
    )
    if du.returncode != 0:
        sys.exit(f"docker exec du failed: {du.stderr.strip()}")
    size_bytes = int(du.stdout.split()[0])

    head_series = 0
    try:
        with urllib.request.urlopen(f"{base}/metrics") as r:
            for line in r.read().decode().splitlines():
                if line.startswith("prometheus_tsdb_head_series "):
                    head_series = int(float(line.split()[-1]))
    except Exception:
        pass

    bps = (
        f"  ({size_bytes / datapoints:.2f} bytes/sample)"
        if (size_bytes and datapoints)
        else ""
    )
    print(f"\nPrometheus: {head_series:,} series  {size_bytes / 1024**2:.1f} MB{bps}")
    _save_result(
        "prometheus",
        os.environ.get("PROMETHEUS_VERSION", "?"),
        datapoints,
        size_bytes,
        start_ts,
        end_ts,
        elapsed,
    )


def report_mimir(datapoints: int, start_ts: int, end_ts: int, elapsed: float = 0.0) -> None:
    """Flush Mimir ingester to blocks, then measure blocks directory on the host.

    Triggers POST /ingester/flush to force the in-memory TSDB head to write
    blocks to the mounted storage path, then waits for the blocks to appear
    and measures total size.
    """
    base = _MIMIR_URL or "http://localhost:8080"

    series = 0
    try:
        with urllib.request.urlopen(f"{base}/metrics") as r:
            for line in r.read().decode().splitlines():
                if line.startswith("cortex_ingester_memory_series "):
                    series = int(float(line.split()[-1]))
    except Exception:
        pass

    print("Flushing Mimir ingester to blocks...", flush=True)
    try:
        req = urllib.request.Request(f"{base}/ingester/flush", method="POST", data=b"")
        with urllib.request.urlopen(req) as r:
            print(f"  flush: HTTP {r.status}")
    except Exception as e:
        print(f"  flush failed: {e}")

    # Wait for blocks to appear in the mounted directory
    blocks_dir = os.path.join(DATA_DIR, "blocks")
    deadline = time.time() + 120
    while time.time() < deadline:
        if os.path.isdir(blocks_dir):
            tenant_dirs = [
                d
                for d in os.listdir(blocks_dir)
                if not d.startswith("_") and os.path.isdir(os.path.join(blocks_dir, d))
            ]
            if tenant_dirs:
                time.sleep(5)
                break
        time.sleep(2)

    size_bytes = _dir_size(DATA_DIR)
    bps = (
        f"  ({size_bytes / datapoints:.2f} bytes/dp)"
        if (size_bytes and datapoints)
        else ""
    )
    print(f"\nMimir: {series:,} series  {size_bytes / 1024**2:.1f} MB{bps}")
    _save_result(
        "mimir",
        os.environ.get("MIMIR_VERSION", "?"),
        datapoints,
        size_bytes,
        start_ts,
        end_ts,
        elapsed,
    )

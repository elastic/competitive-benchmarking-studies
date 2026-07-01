"""Per-engine on-disk storage measurement."""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from engine_config import DATA_DIR, DATA_STREAM, ES_URL, MIMIR_URL, PROM_URL


def _es_request(method: str, path: str) -> tuple[int, object]:
    base_url = ES_URL.rstrip("/")
    req = urllib.request.Request(base_url + path, method=method)

    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except OSError:
                pass
    return total


def measure_elasticsearch(num_segments=1) -> tuple[int, int]:
    """Force-merge, flush, then measure exact storage via _disk_usage.

    Returns (docs, size_bytes).
    """
    print(f"Force-merging {DATA_STREAM} to {num_segments} segment per shard ...")

    t1 = time.time()
    status, _ = _es_request(
        "POST",
        f"/{DATA_STREAM}/_forcemerge?max_num_segments={num_segments}&wait_for_completion=true",
    )
    print(f"Force-merge complete in {time.time() - t1:.1f}s (HTTP {status})")

    status, usage = _es_request(
        "POST",
        f"/{DATA_STREAM}/_disk_usage?expand_wildcards=all&run_expensive_tasks=true&flush=true",
    )
    if status >= 300 or not usage:
        sys.exit(f"_disk_usage failed (HTTP {status}): {usage}")

    size_bytes = sum(
        int(idx["all_fields"]["total_in_bytes"])
        for key, idx in usage.items()
        if key != "_shards"
    )

    _, doc_stats = _es_request(
        "GET", f"/_cat/indices/.ds-{DATA_STREAM}*?format=json&h=docs.count"
    )
    docs = sum(int(idx.get("docs.count", 0)) for idx in doc_stats or [])

    return docs, size_bytes


def measure_prometheus() -> tuple[int, int]:
    """Measure Prometheus storage via TSDB snapshot.

    POST /api/v1/admin/tsdb/snapshot atomically flushes WAL + head into a clean
    snapshot block. We then measure the snapshot directory size via
    `docker exec du -sb` inside the container — this sidesteps Docker Desktop
    mmap visibility issues that affect bind-mounted files read from the host.

    Matches the methodology from https://github.com/gouthamve/prom-elastic-benchmark/blob/main/scripts/measure-prom.sh

    Returns (head_series, size_bytes).
    """
    base_url = PROM_URL

    print("Triggering TSDB snapshot (flushes WAL + head into blocks) ...", flush=True)
    req = urllib.request.Request(
        f"{base_url}/api/v1/admin/tsdb/snapshot", method="POST", data=b""
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
    prom_port = urllib.parse.urlparse(base_url).port
    cp = subprocess.run(
        [
            "docker",
            "ps",
            "--filter",
            f"publish={prom_port}",
            "--format",
            "{{.ID}}",
        ],
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
        with urllib.request.urlopen(f"{base_url}/metrics") as r:
            for line in r.read().decode().splitlines():
                if line.startswith("prometheus_tsdb_head_series "):
                    head_series = int(float(line.split()[-1]))
    except Exception:
        pass

    return head_series, size_bytes


def measure_mimir() -> tuple[int, int]:
    """Flush Mimir ingester to blocks, then measure blocks directory on the host.

    Triggers POST /ingester/flush to force the in-memory TSDB head to write
    blocks to the mounted storage path, then waits for the blocks to appear
    and measures total size.

    Returns (series, size_bytes).
    """
    base = MIMIR_URL

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
    return series, size_bytes

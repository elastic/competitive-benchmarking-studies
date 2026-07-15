"""Single source of truth for raw Prometheus HTTP/snapshot requests."""

import json
import subprocess
import sys
import urllib.parse
import urllib.request

from benchmark.engine_config import BASE_URL


def prom_trigger_snapshot() -> str:
    """POST /api/v1/admin/tsdb/snapshot, atomically flushing WAL + head into a
    clean snapshot block. Returns the snapshot name."""
    print("Triggering TSDB snapshot (flushes WAL + head into blocks) ...", flush=True)
    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/admin/tsdb/snapshot", method="POST", data=b""
    )
    try:
        with urllib.request.urlopen(req) as r:
            resp = json.loads(r.read())
        snap_name = resp["data"]["name"]
        print(f"  snapshot: {snap_name}")
        return snap_name
    except Exception as e:
        sys.exit(f"Snapshot failed: {e}")


def prom_snapshot_size_bytes(snapshot_name: str) -> int:
    """Measure a snapshot's directory size via `docker exec du -sb` inside the
    container — this sidesteps Docker Desktop mmap visibility issues that
    affect bind-mounted files read from the host.

    Matches the methodology from https://github.com/gouthamve/prom-elastic-benchmark/blob/main/scripts/measure-prom.sh
    """
    # Find the container that is actually serving the Prometheus port — not the
    # compose service name, which may resolve to a container that lost the port
    # race when another Prometheus instance was already running.
    prom_port = urllib.parse.urlparse(BASE_URL).port
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
            f"/prometheus/snapshots/{snapshot_name}",
        ],
        capture_output=True,
        text=True,
    )
    if du.returncode != 0:
        sys.exit(f"docker exec du failed: {du.stderr.strip()}")
    return int(du.stdout.split()[0])


def prom_head_series() -> int:
    """Best-effort scrape of `prometheus_tsdb_head_series` from /metrics."""
    try:
        with urllib.request.urlopen(f"{BASE_URL}/metrics") as r:
            for line in r.read().decode().splitlines():
                if line.startswith("prometheus_tsdb_head_series "):
                    return int(float(line.split()[-1]))
    except Exception:
        pass
    return 0

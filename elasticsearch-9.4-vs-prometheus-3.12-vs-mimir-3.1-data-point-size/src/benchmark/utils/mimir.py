"""Single source of truth for raw Mimir HTTP/flush requests."""

import os
import time
import urllib.request

from benchmark.engine_config import BASE_URL
from benchmark.utils.fs import dir_size


def mimir_ingester_memory_series() -> int:
    """Best-effort scrape of `cortex_ingester_memory_series` from /metrics."""
    try:
        with urllib.request.urlopen(f"{BASE_URL}/metrics") as r:
            for line in r.read().decode().splitlines():
                if line.startswith("cortex_ingester_memory_series "):
                    return int(float(line.split()[-1]))
    except Exception:
        pass
    return 0


def mimir_flush(wait: bool = True) -> None:
    """POST /ingester/flush to force the in-memory TSDB head to write blocks
    to the mounted storage path.

    Raises on failure rather than swallowing it — a silently-skipped flush
    would let the caller go on to measure whatever's already on disk (stale
    or empty), producing a wrong storage number with no indication anything
    went wrong.
    """
    print("Flushing Mimir ingester to blocks...", flush=True)
    req = urllib.request.Request(
        f"{BASE_URL}/ingester/flush?wait={'true' if wait else 'false'}",
        method="POST",
        data=b"",
    )
    with urllib.request.urlopen(req) as r:
        print(f"  flush: HTTP {r.status}")


def _block_count(blocks_dir: str) -> int:
    count = 0
    for tenant in os.listdir(blocks_dir):
        tenant_path = os.path.join(blocks_dir, tenant)
        if tenant.startswith("_") or not os.path.isdir(tenant_path):
            continue
        count += sum(
            1
            for d in os.listdir(tenant_path)
            if os.path.isdir(os.path.join(tenant_path, d))
        )
    return count


def mimir_wait_for_stable_blocks(
    blocks_dir: str, interval: float = 15.0, timeout: float = 300.0
) -> None:
    """Poll block count and total size until two consecutive samples match.

    Guards against measuring mid-compaction: deletion_delay=0s only marks
    obsolete blocks for removal, the actual delete happens on the compactor's
    next cleanup_interval, so count/size can keep dropping after a flush.
    """
    deadline = time.time() + timeout
    prev = None
    while time.time() < deadline:
        if not os.path.isdir(blocks_dir):
            time.sleep(interval)
            continue
        current = (_block_count(blocks_dir), dir_size(blocks_dir))
        if current == prev:
            return
        prev = current
        print(f"  blocks={current[0]} size={current[1]:,}b (waiting for stability)")
        time.sleep(interval)
    print(
        f"  warning: blocks did not stabilize within {timeout:.0f}s, measuring anyway"
    )

"""Per-engine on-disk storage measurement."""

import os
import time

from benchmark.engine_config import DATA_DIR, DATA_STREAM
from benchmark.utils.es import es_disk_usage, es_doc_count, es_forcemerge
from benchmark.utils.fs import dir_size
from benchmark.utils.mimir import (
    mimir_flush,
    mimir_ingester_memory_series,
    mimir_wait_for_stable_blocks,
)
from benchmark.utils.prometheus import (
    prom_head_series,
    prom_snapshot_size_bytes,
    prom_trigger_snapshot,
)


def measure_elasticsearch(num_segments=1) -> tuple[int, int]:
    """Force-merge, flush, then measure exact storage via _disk_usage.

    Returns (docs, size_bytes).
    """
    es_forcemerge(DATA_STREAM, num_segments)
    docs = es_doc_count(DATA_STREAM)
    usage = es_disk_usage(DATA_STREAM)
    size_bytes = sum(
        int(idx["all_fields"]["total_in_bytes"])
        for key, idx in usage.items()
        if key != "_shards"
    )

    return docs, size_bytes


def measure_prometheus() -> tuple[int, int]:
    """Measure Prometheus storage via TSDB snapshot.

    Returns (head_series, size_bytes).
    """
    snapshot_name = prom_trigger_snapshot()
    size_bytes = prom_snapshot_size_bytes(snapshot_name)
    head_series = prom_head_series()
    return head_series, size_bytes


def measure_mimir() -> tuple[int, int]:
    """Flush Mimir ingester to blocks, then measure blocks directory on the host.

    Returns (series, size_bytes).
    """
    series = mimir_ingester_memory_series()
    mimir_flush(wait=True)

    blocks_dir = os.path.join(DATA_DIR, "blocks")

    # Wait for the compactor to finish merging fresh blocks and sweeping
    # obsolete ones (deletion_delay=0s, but removal only happens on the next
    # cleanup_interval) before trusting the size on disk.
    print("Waiting for blocks to stabilize (compaction + cleanup)...", flush=True)
    t0 = time.time()
    mimir_wait_for_stable_blocks(blocks_dir)
    print(f"  blocks stabilized after {time.time() - t0:.1f}s")

    # Measure only compacted blocks, not tsdb/ WAL or tsdb-sync/, so the figure
    # is comparable to the Prometheus snapshot (compacted blocks only).
    size_bytes = dir_size(blocks_dir)

    return series, size_bytes

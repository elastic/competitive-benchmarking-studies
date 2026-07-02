"""Single source of truth for raw Elasticsearch HTTP requests."""

import json
import time
import urllib.error
import urllib.request

from engine_config import ES_URL

MERGE_POLL_SECONDS = 10
MERGE_TIMEOUT_SECONDS = 3600


def _es_request(method: str, path: str) -> tuple[int, object]:
    base_url = ES_URL.rstrip("/")
    req = urllib.request.Request(base_url + path, method=method)

    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def _current_merges(data_stream: str) -> int:
    status, body = _es_request("GET", f"/{data_stream}/_stats/merge")
    if status != 200:
        raise RuntimeError(f"GET /{data_stream}/_stats/merge failed: {status} {body}")
    return body["_all"]["total"]["merges"]["current"]


def es_wait_for_merges(data_stream: str) -> None:
    print(f"Waiting for merges to complete on {data_stream}...")
    t1 = time.time()
    deadline = time.monotonic() + MERGE_TIMEOUT_SECONDS
    while True:
        current = _current_merges(data_stream)
        if current == 0:
            print(f"Merges complete in {time.time() - t1:.1f}s")
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Merges still in progress on {data_stream} after "
                f"{MERGE_TIMEOUT_SECONDS}s (current={current})"
            )
        time.sleep(MERGE_POLL_SECONDS)


def es_forcemerge(data_stream: str, num_segments: int) -> None:
    """Force-merge a data stream/index down to num_segments per shard."""
    print(f"Force-merging {data_stream} to {num_segments} segment per shard ...")
    t1 = time.time()
    path = f"/{data_stream}/_forcemerge?max_num_segments={num_segments}&wait_for_completion=true"
    status, body = _es_request("POST", path)
    if status != 200:
        raise RuntimeError(f"POST {path} failed: HTTP {status} {body}")
    print(f"Force-merge complete in {time.time() - t1:.1f}s")


def es_disk_usage(data_stream: str) -> object:
    """Run the _disk_usage API against a data stream/index. Returns the parsed usage body."""
    path = f"/{data_stream}/_disk_usage?expand_wildcards=all&run_expensive_tasks=true&flush=true"
    status, body = _es_request("POST", path)
    if status != 200 or not body:
        raise RuntimeError(f"POST {path} failed: HTTP {status} {body}")
    return body


def es_doc_count(data_stream: str) -> int:
    """Sum docs.count across all backing indices of a data stream via _cat/indices."""
    path = f"/_cat/indices/.ds-{data_stream}*?format=json&h=docs.count"
    status, doc_stats = _es_request("GET", path)
    if status != 200:
        raise RuntimeError(f"GET {path} failed: HTTP {status} {doc_stats}")
    return sum(int(idx.get("docs.count", 0)) for idx in doc_stats or [])

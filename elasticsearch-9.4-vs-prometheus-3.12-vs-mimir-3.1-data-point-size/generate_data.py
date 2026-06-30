#!/usr/bin/env python3
"""
Run metricsgenreceiver sending data directly to Elasticsearch, Prometheus, or Mimir
via OTLP (protobuf + gzip) — exactly matching the team's production benchmark setup.

Engine is auto-detected from environment variables:
  ELASTICSEARCH_URL → sends to ES  /_otlp
  PROMETHEUS_URL    → sends to Prometheus /api/v1/otlp
  MIMIR_URL         → sends to Mimir /otlp

Data flows:
  metricsgenreceiver → batch processor → OTLP/HTTP (proto + gzip) → engine

Post-ingest:
  Elasticsearch: force-merges to 1 segment per shard, reports bytes/dp via API
  Prometheus:    waits for compaction, reports via Prometheus metrics API
  Mimir:         waits for block compaction, reports directory size

Usage:
  ELASTICSEARCH_URL=http://localhost:9200 python3 generate_data.py
  PROMETHEUS_URL=http://localhost:9090    python3 generate_data.py
  MIMIR_URL=http://localhost:8080         python3 generate_data.py
  SCALE=100 START_NOW_MINUS=10m          python3 generate_data.py
"""

import json, os, platform, re, shutil, subprocess, sys, tarfile, tempfile, time, urllib.parse, urllib.request, urllib.error

# ── Configuration ─────────────────────────────────────────────────────────────

VERSION         = "1.0.7"
SEED            = 123
SCALE           = int(os.environ.get("SCALE", "10000"))
INTERVAL        = os.environ.get("INTERVAL", "1s")
START_NOW_MINUS = os.environ.get("START_NOW_MINUS", "270m")
RESULTS_FILE    = os.environ.get("RESULTS_FILE")  # optional: path to write JSON result

# Auto-detect engine from environment
_ES_URL   = os.environ.get("ELASTICSEARCH_URL")
_PROM_URL = os.environ.get("PROMETHEUS_URL")
_MIMIR_URL = os.environ.get("MIMIR_URL")

if _MIMIR_URL:
    ENGINE        = "mimir"
    OTLP_ENDPOINT = f"{_MIMIR_URL}/otlp"
    DATA_STREAM   = None
elif _PROM_URL:
    ENGINE        = "prometheus"
    OTLP_ENDPOINT = f"{_PROM_URL}/api/v1/otlp"
    DATA_STREAM   = None
elif _ES_URL:
    ENGINE        = "elasticsearch"
    OTLP_ENDPOINT = f"{_ES_URL}/_otlp"
    DATA_STREAM   = "metrics-demo.otel-default"
else:
    ENGINE        = "elasticsearch"
    OTLP_ENDPOINT = "http://localhost:9200/_otlp"
    DATA_STREAM   = "metrics-demo.otel-default"

_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(_HERE, "data", ENGINE)
BIN_DIR    = os.path.join(_HERE, ".bin")
BINARY_PATH = os.path.join(BIN_DIR, "metricsgenreceiver")


# ── Binary management ─────────────────────────────────────────────────────────

def _platform():
    system  = platform.system().lower()
    machine = platform.machine().lower()
    os_name = {"darwin": "darwin", "linux": "linux"}.get(system)
    arch    = {"x86_64":"amd64","amd64":"amd64","arm64":"arm64","aarch64":"arm64"}.get(machine)
    if not os_name or not arch:
        sys.exit(f"Unsupported platform: {system}/{machine}")
    return os_name, arch


def ensure_binary() -> str:
    found = shutil.which("metricsgenreceiver")
    if found:
        return found
    if os.path.exists(BINARY_PATH):
        return BINARY_PATH
    os_name, arch = _platform()
    url = (f"https://github.com/elastic/metricsgenreceiver/releases/"
           f"download/v{VERSION}/metricsgenreceiver_{os_name}_{arch}.tar.gz")
    print(f"Downloading metricsgenreceiver v{VERSION} ({os_name}/{arch})...")
    os.makedirs(BIN_DIR, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        urllib.request.urlretrieve(url, tmp.name)
        with tarfile.open(tmp.name, "r:gz") as tar:
            for m in tar.getmembers():
                if "metricsgenreceiver" in m.name and not m.isdir():
                    m.name = "metricsgenreceiver"
                    tar.extract(m, BIN_DIR, filter="data")
                    break
    os.chmod(BINARY_PATH, 0o755)
    return BINARY_PATH


# ── OTel Collector config ─────────────────────────────────────────────────────

def otel_config() -> str:
    if ENGINE in ("prometheus", "mimir"):
        # Merge resource attributes into datapoint attributes so that host.name
        # becomes a Prometheus/Mimir label; strip high-cardinality array fields
        transform_stmts = """\
  transform:
    metric_statements:
      - context: resource
        statements:
          - delete_key(attributes, "host.ip")
          - delete_key(attributes, "host.mac")
      - context: datapoint
        statements:
          - merge_maps(attributes, resource.attributes, "insert")"""
        exporter_name = f"otlphttp/{ENGINE}"
    else:
        # Elasticsearch: set data stream routing attributes; strip noise
        transform_stmts = """\
  transform:
    metric_statements:
      - context: resource
        statements:
          - set(attributes["data_stream.dataset"], "demo")
          - set(attributes["data_stream.namespace"], "default")
          - delete_key(attributes, "host.ip")
          - delete_key(attributes, "host.mac")"""
        exporter_name = "otlphttp/elasticsearch"

    # Mimir requires X-Scope-OrgID even with multitenancy disabled
    extra_headers = ('    headers:\n      X-Scope-OrgID: anonymous\n'
                     if ENGINE == "mimir" else "")

    # Prometheus's OTLP handler has a ~10MB request body limit. Binary protobuf is
    # ~75 bytes/sample so 20k samples ≈ 1.5MB — well under the limit and much faster
    # than JSON. No compression: Prometheus 3.x does not reliably decompress
    # Content-Encoding: gzip on OTLP requests. ES and Mimir handle large gzip
    # batches without issue.
    if ENGINE == "prometheus":
        encoding_line = ""
        compression_line = ""
        batch_size = 20000
        batch_timeout = "5s"
    elif ENGINE == "mimir":
        encoding_line = ""
        compression_line = ""
        batch_size = 100000
        batch_timeout = "10s"
    else:
        encoding_line = ""
        compression_line = "    compression: gzip\n"
        batch_size = 10000   # ES 9.4.x OTLP endpoint has an 8 MB body limit
        batch_timeout = "10s"

    return f"""\
receivers:
  metricsgen:
    seed: {SEED}
    start_now_minus: {START_NOW_MINUS}
    interval: {INTERVAL}
    real_time: false
    exit_after_end: true
    exit_after_end_timeout: 15s
    scenarios:
      - path: builtin/hostmetrics
        scale: {SCALE}

processors:
{transform_stmts}
  batch:
    send_batch_size: {batch_size}
    timeout: {batch_timeout}

exporters:
  {exporter_name}:
    endpoint: {OTLP_ENDPOINT}
{encoding_line}{compression_line}{extra_headers}    tls:
      insecure: true
    sending_queue:
      enabled: true
      block_on_overflow: true
      queue_size: 25
      num_consumers: 25

service:
  telemetry:
    metrics:
      level: none   # disable internal scrape endpoint to avoid port conflicts
  pipelines:
    metrics:
      receivers: [metricsgen]
      processors: [transform, batch]
      exporters: [{exporter_name}]
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _parse_datapoints(stderr: str) -> tuple[int, float]:
    datapoints, rate = 0, 0.0
    for line in stderr.splitlines():
        if '"datapoints"' in line:
            m = re.search(r'"datapoints"\s*:\s*(\d+)', line)
            if m:
                datapoints = int(m.group(1))
            m = re.search(r'"data_points_per_second"\s*:\s*([\d.]+)', line)
            if m:
                rate = float(m.group(1))
    return datapoints, rate


def _es_request(method, path):
    base = (_ES_URL or "http://localhost:9200").rstrip("/")
    req  = urllib.request.Request(base + path, method=method)
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


def _save_result(engine: str, version: str, datapoints: int, size_bytes: int,
                 elapsed_seconds: float = 0.0):
    if not RESULTS_FILE:
        return
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    eps = round(datapoints / elapsed_seconds) if elapsed_seconds > 0 else 0
    record = {"engine": engine, "version": version,
              "datapoints": datapoints, "size_bytes": size_bytes,
              "elapsed_seconds": round(elapsed_seconds, 1), "eps": eps}
    with open(RESULTS_FILE, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Result saved to {RESULTS_FILE}")


# ── Post-ingest reporting ──────────────────────────────────────────────────────

def report_elasticsearch(datapoints: int, elapsed: float = 0.0):
    print(f"Force-merging {DATA_STREAM} to 1 segment per shard ...")
    t1 = time.time()
    status, _ = _es_request("POST",
        f"/{DATA_STREAM}/_forcemerge?max_num_segments=1&wait_for_completion=true")
    print(f"Force-merge complete in {time.time()-t1:.1f}s (HTTP {status})")

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

    _, stats = _es_request("GET",
        f"/_cat/indices/.ds-{DATA_STREAM}*?format=json&h=docs.count,dataset.size")
    docs, size_bytes = 0, 0
    if stats:
        for idx in stats:
            docs += int(idx.get("docs.count", 0))
            size_bytes += _parse_size(idx.get("dataset.size", "0b"))

    size_str = (f"{size_bytes/1024**3:.1f}gb" if size_bytes >= 1024**3
                else f"{size_bytes/1024**2:.1f}mb" if size_bytes >= 1024**2
                else f"{size_bytes}b")
    bps = f"  ({size_bytes/datapoints:.2f} bytes/dp)" if (size_bytes and datapoints) else ""
    print(f"\nElasticsearch: {docs:,} docs  {size_str}{bps}")
    _save_result("elasticsearch", os.environ.get("ES_VERSION","?"), datapoints, size_bytes, elapsed)


def report_prometheus(datapoints: int, elapsed: float = 0.0):
    """Measure Prometheus storage via TSDB snapshot.

    POST /api/v1/admin/tsdb/snapshot atomically flushes WAL + head into a clean
    snapshot block. We then measure the snapshot directory size via
    `docker exec du -sb` inside the container — this sidesteps Docker Desktop
    mmap visibility issues that affect bind-mounted files read from the host.

    Matches the methodology from https://github.com/gouthamve/prom-elastic-benchmark/blob/main/scripts/measure-prom.sh
    """
    base = _PROM_URL or "http://localhost:9090"

    print("Triggering TSDB snapshot (flushes WAL + head into blocks) ...", flush=True)
    req = urllib.request.Request(f"{base}/api/v1/admin/tsdb/snapshot", method="POST", data=b"")
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
        capture_output=True, text=True,
    )
    container_id = cp.stdout.strip().splitlines()[0] if cp.stdout.strip() else ""
    if not container_id:
        sys.exit(f"Could not find a running container publishing port {prom_port}")

    du = subprocess.run(
        ["docker", "exec", container_id, "du", "-sb",
         f"/prometheus/snapshots/{snap_name}"],
        capture_output=True, text=True,
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

    bps = f"  ({size_bytes/datapoints:.2f} bytes/sample)" if (size_bytes and datapoints) else ""
    print(f"\nPrometheus: {head_series:,} series  {size_bytes/1024**2:.1f} MB{bps}")
    _save_result("prometheus", os.environ.get("PROMETHEUS_VERSION", "?"), datapoints, size_bytes, elapsed)


def report_mimir(datapoints: int, elapsed: float = 0.0):
    """Flush Mimir ingester to blocks, then measure blocks directory on the host.

    Triggers POST /ingester/flush to force the in-memory TSDB head to write
    blocks to the mounted storage path, then waits for the blocks to appear
    and measures total size.
    """
    base = _MIMIR_URL or "http://localhost:8080"

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
            tenant_dirs = [d for d in os.listdir(blocks_dir)
                           if not d.startswith("_") and os.path.isdir(os.path.join(blocks_dir, d))]
            if tenant_dirs:
                time.sleep(5)
                break
        time.sleep(2)

    size_bytes = _dir_size(DATA_DIR)
    # Subtract metadata-only dirs like __mimir_cluster
    bps    = f"  ({size_bytes/datapoints:.2f} bytes/dp)" if (size_bytes and datapoints) else ""
    series = 0
    try:
        with urllib.request.urlopen(f"{base}/metrics") as r:
            for line in r.read().decode().splitlines():
                if line.startswith("cortex_ingester_memory_series "):
                    series = int(float(line.split()[-1]))
    except Exception:
        pass
    print(f"\nMimir: {series:,} series  {size_bytes/1024**2:.1f} MB{bps}")
    _save_result("mimir", os.environ.get("MIMIR_VERSION","?"), datapoints, size_bytes, elapsed)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    binary = ensure_binary()

    print(f"engine={ENGINE}  scale={SCALE}  interval={INTERVAL}  window={START_NOW_MINUS}")
    print(f"target={OTLP_ENDPOINT}")

    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "otelcol.yaml")
        with open(cfg, "w") as f:
            f.write(otel_config())

        print(f"Running metricsgenreceiver → {OTLP_ENDPOINT}/v1/metrics ...")
        t0 = time.time()
        result = subprocess.run([binary, "--config", cfg], capture_output=True, text=True)
        elapsed = time.time() - t0

        if result.returncode != 0:
            sys.exit(f"metricsgenreceiver exited with code {result.returncode}")

    print(result.stderr)

    datapoints, rate = _parse_datapoints(result.stderr)

    if datapoints:
        print(f"Ingested: {datapoints:,} data points  ({rate:,.0f} dp/s)  in {_format_duration(elapsed)}")
    else:
        print(f"metricsgenreceiver completed in {_format_duration(elapsed)}")

    if ENGINE == "elasticsearch":
        report_elasticsearch(datapoints, elapsed)
    elif ENGINE == "prometheus":
        report_prometheus(datapoints, elapsed)
    elif ENGINE == "mimir":
        report_mimir(datapoints, elapsed)

    print("\nDone.")


if __name__ == "__main__":
    main()

# Elasticsearch 9.5 vs Prometheus 3.12 vs Mimir 3.1 — Data Point Storage Efficiency

Measures bytes per data point for three time-series engines after ingesting identical OTel hostmetrics data.

## Results

100 hosts × 270 minutes × 1s interval = 225,180,000 data points per engine. Measured after force-merge (ES) / compaction (Prometheus, Mimir).

| Engine        | Version        | Data Points | Size      | Bytes/DP |
|---------------|----------------|-------------|-----------|----------|
| Elasticsearch | 9.5.0-SNAPSHOT | 225,180,000 | 647.5 MB  | **3.02** |
| Mimir         | 3.1.0          | 225,180,000 | 832.5 MB  | 3.88     |
| Prometheus    | 3.12.0         | 225,180,000 | 1,011 MB  | 4.71     |

Elasticsearch's advantage comes from: TSDB columnar codec, synthetic `_source`, synthetic document IDs, doc value skippers (no inverted indices on dimensions), and trimmed sequence numbers — all applied automatically via the built-in OTel index template.

## Prerequisites

- Docker (with at least 6 GB memory allocated)
- Python 3.9+
- For ES 9.5.0-SNAPSHOT (full optimisations): `docker login docker.elastic.co` with an Elastic account
- For ES without Elastic account: edit `.env` → `ES_VERSION=9.4.3` (STORED _source, slightly higher bytes/DP)

## Quick Start

```bash
# Run all three engines sequentially (~45 min total)
make run

# Or run individually
make elasticsearch    # ~15 min
make prometheus       # ~15 min
make mimir            # ~15 min

# Display comparison
make report
```

## How It Works

1. **Data generation**: `metricsgenreceiver` generates synthetic OTel hostmetrics (100 hosts × 270 minutes × 1s interval = 225M data points) and streams directly to each engine via OTLP/HTTP with protobuf + gzip — no intermediate files.

2. **Elasticsearch setup**: The built-in `metrics-otel@template` index template activates automatically. `metrics-otel@custom` overrides replicas/shards/look_back_time. Trial license enables synthetic `_source`.

3. **Storage measurement**:
   - **Elasticsearch**: `_forcemerge` to 1 segment, then `_cat/indices` for `dataset.size`
   - **Prometheus/Mimir**: wait for compaction, then measure mounted data directory size

## Configuration

Edit `.env` to adjust:

| Variable | Default | Description |
|---|---|---|
| `ES_VERSION` | `9.5.0-SNAPSHOT` | ES image tag |
| `SCALE` | `100` | Simulated hosts |
| `INTERVAL` | `1s` | Metric emission interval |
| `START_NOW_MINUS` | `270m` | Data window (historical replay) |
| `ES_HEAP` | `4g` | Elasticsearch JVM heap |

## Methodology

Matches the team's [columnar metrics engine benchmark](https://www.elastic.co/search-labs/blog/elasticsearch-columnar-metrics-engine-30x-faster-prometheus) (low-cardinality setup: 100 hosts, 1s interval). Scale is configurable.

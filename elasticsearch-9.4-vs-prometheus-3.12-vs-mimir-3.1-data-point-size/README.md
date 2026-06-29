# Elasticsearch 9.4.2 vs Prometheus 3.12 vs Mimir 3.1 — Data Point Storage Efficiency

Measures bytes per data point for three time-series engines after ingesting identical OTel hostmetrics data.

## Results

| Engine        | Version         | Data Points | Size     | Bytes/DP |
|---------------|-----------------|-------------|----------|----------|
| Elasticsearch | 9.4.2           | 225,180,000 | 653.7 MB | 3.04     |
| Prometheus    | 3.12.0          | 225,180,000 | 828.2 MB | 3.86     |
| Mimir         | 3.1.0           | 225,180,000 | 829.3 MB | 3.86     |

Elasticsearch's advantage comes from: TSDB columnar codec, synthetic `_source`, synthetic document IDs, doc value skippers (no inverted indices on dimensions), and trimmed sequence numbers — all applied automatically via the built-in OTel index template.

## Prerequisites

- Docker (with at least 6 GB memory allocated)
- Python 3.9+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package and environment manager

## Quick Start

```bash
# Set up the Python environment
make setup

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

1. **Data generation**: `metricsgenreceiver` generates synthetic OTel hostmetrics (100 hosts × 270 minutes × ~139 metrics/host × 1s interval ≈ 225M data points) and streams directly to each engine via OTLP/HTTP with protobuf + gzip — no intermediate files.

2. **Elasticsearch setup**: The built-in `metrics-otel@template` index template activates automatically. `metrics-otel@custom` overrides replicas/shards/look_back_time. Trial license enables synthetic `_source`.

3. **Storage measurement**:
   - **Elasticsearch**: `_forcemerge` to 1 segment, then `_cat/indices` for `dataset.size`
   - **Prometheus**: `POST /api/v1/admin/tsdb/snapshot` → measure snapshot directory size (blocks only, excludes WAL) ÷ `prometheus_tsdb_head_samples_appended_total`. Matches the [methodology used by the Prometheus team](https://github.com/gouthamve/prom-elastic-benchmark/blob/632ae80262bf1bb6fc44aa89480307ef7576f51c/scripts/measure-prom.sh).
   - **Mimir**: `POST /ingester/flush` to force block compaction, then measure blocks directory size

## Configuration

Edit `.env` to adjust:

| Variable | Default | Description |
|---|---|---|
| `ES_VERSION` | `9.4.2` | ES image tag |
| `SCALE` | `100` | Simulated hosts |
| `INTERVAL` | `1s` | Metric emission interval |
| `START_NOW_MINUS` | `270m` | Data window (historical replay) |
| `ES_HEAP` | `4g` | Elasticsearch JVM heap |

## Methodology

Matches the team's [columnar metrics engine benchmark](https://www.elastic.co/search-labs/blog/elasticsearch-columnar-metrics-engine-30x-faster-prometheus) (low-cardinality setup: 100 hosts, 1s interval). Scale is configurable.

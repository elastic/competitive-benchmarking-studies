# Elasticsearch 9.4.2 vs Prometheus 3.12 vs Mimir 3.1 — Data Point Storage Efficiency

Measures bytes per data point for three time-series engines after ingesting identical OTel hostmetrics data.

## Results

| Engine        | Version         | Data Points | Size     | Bytes/DP |
|---------------|-----------------|-------------|----------|----------|
| Elasticsearch | 9.4.2           | 225,180,000 | 653.7 MB | 3.04     |
| Prometheus    | 3.12.0          | 225,180,000 | 828.2 MB | 3.86     |
| Mimir         | 3.1.0           | 225,180,000 | 829.3 MB | 3.86     |

Elasticsearch's advantage comes from: TSDB columnar codec, synthetic `_source`, synthetic document IDs, doc value skippers (no inverted indices on dimensions), and trimmed sequence numbers — all applied automatically via the built-in OTel index template.

`.env`'s default `MIMIR_VERSION` has since moved to `3.1.1` (see [Mimir's block-stability wait fix](#how-it-works)) — the Mimir row above is not yet re-measured against it; run `make mimir` and `make report` to refresh all three rows against current versions.

## Prerequisites

- Docker (memory allocated to Docker Desktop/engine must be at least `.env`'s `CONTAINER_MEMORY_LIMIT`, since one engine's container runs at a time)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- `curl` and `tar` (pre-installed on macOS and most Linux distros)

## Quick Start

```bash
# Install Python deps + download metricsgenreceiver and vegeta into .bin/
make setup

# Run all three engines sequentially — ingests data, runs queries, measures
# storage, then prints the comparison (~45 min total)
make run

# Or run one engine at a time — each target ingests, queries, and measures (~15 min)
make elasticsearch
make prometheus
make mimir

# Display the storage comparison table + chart
make report
```

Each `make <engine>` target starts the container, delegates to `run-engine`
(see [How It Works](#how-it-works)), then stops the container. To re-run just
one step against a container that's still up, call the underlying commands
directly:

```bash
# Re-run query benchmarks for one engine only
uv run query --engine elasticsearch --benchmark duration_240m-query_200m-scale_100

# Re-measure storage for one engine only — nothing is defaulted, so every
# required variable must be set explicitly (`make <engine>` sets all of these
# for you; this is what to export if calling disk-usage directly)
ENGINE=elasticsearch ELASTICSEARCH_URL=http://localhost:9200 \
  RESULTS_FILE=results/elasticsearch.json ES_DATA_STREAM=metrics-hostmetricsreceiver.otel-default \
  uv run disk-usage --benchmark duration_240m-query_200m-scale_100
```

## How It Works

1. **`make <engine>`** starts that engine's container, waits for it to
   accept traffic, then runs `uv run run-engine <engine> --benchmark <name>`,
   and stops the container afterward. Elasticsearch and Prometheus use
   `docker compose up --wait` (blocking on a `healthcheck:`); Mimir's image
   is a distroless single binary with no shell/wget/curl at all, so a
   container-side healthcheck is impossible there — its target instead
   polls `/ready` externally via `uv run wait-for` before proceeding.

2. **`run-engine`** ([`src/benchmark/run_engine.py`](src/benchmark/run_engine.py))
   is the single orchestrator for one engine's full cycle. It derives `ENGINE`
   and `RESULTS_FILE` from its own `<engine>` argument (so the caller never
   declares the engine twice), runs Elasticsearch-only bootstrap steps if
   applicable, then invokes `load` → `query` → `disk-usage` as subprocesses,
   forwarding `--benchmark` to each.

3. **Elasticsearch bootstrap** (skipped for Prometheus/Mimir): starts the
   30-day trial license (required for synthetic `_source`), applies the
   `metrics-otel@custom` component template and `metrics-policy` ILM policy
   from [`deploy/config/elasticsearch/`](deploy/config/elasticsearch/), and recreates the
   data stream so it picks up the new template settings.

4. **`load`** ([`src/benchmark/load/`](src/benchmark/load/)) runs
   `metricsgenreceiver` to generate synthetic OTel hostmetrics (scale × interval ×
   window from the selected benchmark scenario — the default is 100 hosts × 1s
   interval × 270 minutes × ~139 metrics/host ≈ 225M data points) and streams
   them directly to the engine via OTLP/HTTP with protobuf + gzip — no
   intermediate files.

5. **`query`** ([`src/benchmark/query/`](src/benchmark/query/)) runs the
   selected queryset's per-engine queries through `vegeta` and records
   p50/p95/p99 latency, throughput, and success rate.

6. **`disk-usage`** ([`src/benchmark/disk_usage/`](src/benchmark/disk_usage/))
   measures on-disk storage per engine:
   - **Elasticsearch**: `_forcemerge` to 1 segment, then `_disk_usage` for exact byte counts.
   - **Prometheus**: `POST /api/v1/admin/tsdb/snapshot` → measure snapshot directory size (blocks only, excludes WAL). Matches the [methodology used by the Prometheus team](https://github.com/gouthamve/prom-elastic-benchmark/blob/632ae80262bf1bb6fc44aa89480307ef7576f51c/scripts/measure-prom.sh).
   - **Mimir**: `POST /ingester/flush` to force block compaction, then polls block count + directory size until two consecutive samples match (`deletion_delay=0s` only marks obsolete blocks for removal — the actual delete happens on the compactor's next `cleanup_interval`, so measuring too early overcounts) before measuring the blocks directory.

7. **`report`** ([`src/benchmark/report.py`](src/benchmark/report.py)) reads
   `results/<engine>.json` for each engine and prints the comparison table +
   bar chart.

Every step reads the same required environment: `ENGINE` (one of
`elasticsearch`/`prometheus`/`mimir`), `RESULTS_FILE`, and the connection URL
matching the active engine — see [`src/benchmark/engine_config.py`](src/benchmark/engine_config.py).
Nothing is silently defaulted: if a required variable is missing, the command
aborts immediately with a clear message rather than guessing.

## Project Layout

```
src/benchmark/          # all Python code (installed as the "benchmark" package)
  engine_config.py      # ENGINE validation + per-engine connection config
  run_engine.py         # orchestrates bootstrap + load/query/disk-usage for one engine
  load/                 # metricsgenreceiver config rendering + execution
  query/                # queryset loading + vegeta execution
  disk_usage/           # per-engine storage measurement
  store/                # results/<engine>.json read/write
  utils/                # per-engine HTTP helpers (es.py, prometheus.py, mimir.py) + generic helpers (fs.py, size.py, time.py)
  scenarios.py          # resolves scenarios/<name>.yml + its queryset reference
  report.py             # cross-engine comparison table + chart
scenarios/               # scenario definitions (<name>.yml)
querysets/                # query definitions, referenced by name from a scenario
deploy/
  docker/                 # docker-compose.yml
  config/                 # per-engine runtime config (elasticsearch/, mimir.yaml, prometheus.yml)
```

## Benchmarks: scenarios & querysets

A **benchmark** (`scenarios/<name>.yml`) is the ingest scenario — how much
data to generate and which queryset to run against it:

```yaml
name: duration_240m-query_200m-scale_100
description: Bytes/datapoint comparison, 100 hosts, 1s interval, 270m window

ingest:
  scale: 100             # simulated hosts
  interval: 1s            # metric emission interval
  start_now_minus: 270m   # data window (historical replay)
  seed: 123               # reproducibility seed

queryset: default         # → querysets/default.yml
```

A **queryset** (`querysets/<name>.yml`) is the set of queries run against
each engine after ingest — global `defaults` (vegeta rate/duration/workers,
with per-query overrides) plus a `targets.<engine>` block per engine, each
with its own `target.base_url`/`method`/`path` and a list of `queries`
(Jinja2-templated bodies with `tend`/`tstart`/`now` and any custom
`queries_runtime_params` available). See [`querysets/default.yml`](querysets/default.yml)
for the full schema in practice.

Everything is resolved by [`src/benchmark/scenarios.py`](src/benchmark/scenarios.py)'s
`load_benchmark(name)`, which every command (`load`, `query`, `disk-usage`,
`run-engine`) takes as `--benchmark <name>` (required — there's no default
baked into the code; `Makefile`'s `BENCHMARK ?= duration_240m-query_200m-scale_100`
is the one place a convenience default lives).

**To run an existing benchmark under a different name:**

```bash
BENCHMARK=<name> make elasticsearch
# or directly:
uv run load --benchmark <name>
```

**To add a new benchmark scenario** (e.g. a higher-cardinality run):

1. Copy an existing `scenarios/*.yml`, give it a new `name` matching the filename, and adjust `ingest.*`.
2. Point `queryset:` at an existing queryset name to reuse its queries, or add a new `querysets/<name>.yml` if the new scenario needs different queries.
3. Run it: `BENCHMARK=<new-name> make elasticsearch` (or any engine).

**To add or change queries** for an existing queryset, edit its
`targets.<engine>.queries` list directly — no other file needs to change,
since every `scenarios/*.yml` referencing that queryset picks up the new
queries automatically.

## Adding a new engine

1. **`src/benchmark/engine_config.py`** — add an entry to `_ENGINES` (`url_env`, `otlp_path`, `version_env`).
2. **`src/benchmark/utils/<engine>.py`** — new module with the engine's raw HTTP calls (mirror `utils/prometheus.py`/`utils/mimir.py`'s shape); **`src/benchmark/disk_usage/measure.py`** — add `measure_<engine>()` and wire it into `disk_usage/cli.py`'s dispatch.
3. **`deploy/docker/docker-compose.yml`** — add the service, plus its config file under **`deploy/config/`**. Add a `healthcheck:` if the image has a shell and `wget`/`curl` (verify with `docker run --rm --entrypoint sh <image> -c 'wget --version'` before relying on it — don't assume; Mimir's image has neither). If it does, the Makefile target can use `docker compose up -d --wait <engine>`; if not, mirror Mimir's pattern instead (`docker compose up -d <engine>` + `uv run wait-for <ready-url>`).
4. **`Makefile`** — add a target mirroring `elasticsearch`/`prometheus`/`mimir`; **`.env`** — add the engine's version pin and connection URL.
5. **`querysets/*.yml`** — add a `targets.<engine>` block with equivalent queries.
6. **`src/benchmark/report.py`** — add the engine to `ENGINES` and `ENGINE_COLORS`.
7. If the engine needs bootstrap steps analogous to Elasticsearch's (license/template/ILM), add a `_bootstrap_<engine>()` to **`src/benchmark/run_engine.py`** and call it from `main()`.

## Configuration

Engine connection/version/resource details live in `.env` — see that file
directly for current values (deliberately not repeated here as a table,
since a hardcoded copy of `.env`'s values in this README would drift the
moment `.env` changes, which it does often as machine sizing gets tuned):

| Variable | Description |
|---|---|
| `ES_VERSION` / `PROMETHEUS_VERSION` / `MIMIR_VERSION` | Engine image tags |
| `ELASTICSEARCH_URL` / `PROMETHEUS_URL` / `MIMIR_URL` | Local docker compose ports |
| `ES_HEAP` | Elasticsearch JVM heap — should be roughly half of `CONTAINER_MEMORY_LIMIT` |
| `ES_DATA_STREAM` | Elasticsearch data stream/index name |
| `CONTAINER_MEMORY_LIMIT` / `CONTAINER_CPU_LIMIT` | Resource limits applied to each engine's container |

Scenario parameters (`scale`, `interval`, `start_now_minus`, `seed`, and which
queryset to run) live in `scenarios/*.yml` instead — see
[Benchmarks: scenarios & querysets](#benchmarks-scenarios--querysets) above.

## Methodology

Matches the team's [columnar metrics engine benchmark](https://www.elastic.co/search-labs/blog/elasticsearch-columnar-metrics-engine-30x-faster-prometheus) (low-cardinality setup: 100 hosts, 1s interval). Scale is configurable.

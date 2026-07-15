# Elasticsearch 9.4.2 vs Prometheus 3.12 vs Mimir 3.1 vs ClickHouse 26.5 — Data Point Storage Efficiency

Measures bytes per data point for four time-series engines after ingesting identical OTel hostmetrics data.

## Results

| Engine        | Version | Data Points | Size     | Bytes/DP | Elapsed | EPS     |
|---------------|---------|-------------|----------|----------|---------|---------|
| Elasticsearch | 9.4.2   | 225,180,000 | 801.5 MB | 3.73     | 14m17s  | 262,480 |
| Prometheus    | 3.12.0  | 225,180,000 | 828.9 MB | 3.86     | 10m27s  | 358,972 |
| Mimir         | 3.1.1   | 225,180,000 | 838.7 MB | 3.91     | 10m45s  | 348,854 |
| ClickHouse    | 26.5.1  | —           | —        | —        | —       | —       |

Elasticsearch's advantage comes from: TSDB columnar codec, synthetic `_source`, synthetic document IDs, doc value skippers (no inverted indices on dimensions), and trimmed sequence numbers — all applied automatically via the built-in OTel index template.

Elapsed/EPS measure the `metricsgenreceiver` ingestion run only (wall-clock time ÷ data points), not the storage-measurement step that follows it (force-merge/snapshot/flush).

### Query Benchmark Results

Each engine ran the same 12 PromQL/ES|QL/ClickHouse-SQL-equivalent queries (`queries.yml`) via `vegeta` at a fixed low rate, over the same time range as the ingested data. All queries returned 100% success across all engines.

| Query                                       | Elasticsearch (p50/p95/p99 ms) | Prometheus (p50/p95/p99 ms)   | Mimir (p50/p95/p99 ms)        |
|----------------------------------------------|---------------------------------|--------------------------------|---------------------------------|
| avg_avgot_memory_by_host_1h                   | 26.0 / 29.8 / 32.5               | 545.8 / 554.1 / 556.1           | 905.4 / 915.9 / 926.0            |
| avg_avgot_memory_by_host_5m                   | 42.8 / 46.6 / 48.2               | 508.9 / 513.3 / 514.7           | 837.0 / 847.2 / 853.4            |
| avg_avgot_memory_by_host_30m_window_90m       | 29.6 / 33.5 / 34.3               | 590.1 / 593.2 / 594.0           | 907.9 / 918.0 / 923.1            |
| avg_rate_cpu_by_host_1h                       | 143.8 / 147.0 / 148.3            | 3518.7 / 3557.6 / 3560.1        | 4251.7 / 4263.6 / 5493.0         |
| avg_rate_cpu_by_host_5m                       | 244.9 / 249.1 / 250.8            | 3262.1 / 3273.2 / 3279.7        | 3921.0 / 3929.1 / 3942.0         |
| avg_rate_cpu_by_host_30m_window_90m           | 171.3 / 174.6 / 176.0            | 3615.1 / 3658.3 / 3658.7        | 4381.7 / 4392.0 / 4393.6         |
| avg_avgot_cpu_load_1m_filtered_by_hosts_5m    | 7.0 / 7.4 / 8.1                  | 3.2 / 3.4 / 3.7                 | 5.3 / 5.4 / 5.6                  |
| avg_avgot_cpu_load_1m_prefix_by_hosts_5m      | 22.0 / 22.6 / 24.3               | 87.0 / 88.8 / 90.3              | 112.5 / 115.7 / 117.0            |
| sum_rate_sys_cpu_time_large_clause_5m         | 685.5 / 695.2 / 703.8            | 3379.1 / 3394.6 / 3402.4        | 4252.0 / 4262.8 / 4264.2         |
| sum_lot_filesystem_usage_top5                 | 25.7 / 27.9 / 29.6               | 440.8 / 444.8 / 445.9           | 547.0 / 551.6 / 552.6            |
| avgot_memory_by_tbucket_1h                    | 76.1 / 78.0 / 78.9               | 549.6 / 558.3 / 560.1           | 685.0 / 690.4 / 694.0            |
| rate_cpu_by_tbucket_1h                        | 271.6 / 278.2 / 285.7            | 3520.7 / 3576.7 / 3579.7        | 4344.1 / 4355.0 / 4355.8         |

Per-engine breakdown with RPS and success rate:

<details>
<summary>Elasticsearch</summary>

| Query                                          | p50 ms | p95 ms | p99 ms | RPS | OK% |
|-------------------------------------------------|-------:|-------:|-------:|----:|----:|
| avg_avgot_memory_by_host_1h                      |   26.0 |   29.8 |   32.5 | 3.0 | 100 |
| avg_avgot_memory_by_host_5m                      |   42.8 |   46.6 |   48.2 | 3.0 | 100 |
| avg_avgot_memory_by_host_30m_window_90m          |   29.6 |   33.5 |   34.3 | 3.0 | 100 |
| avg_rate_cpu_by_host_1h                          |  143.8 |  147.0 |  148.3 | 3.0 | 100 |
| avg_rate_cpu_by_host_5m                          |  244.9 |  249.1 |  250.8 | 3.0 | 100 |
| avg_rate_cpu_by_host_30m_window_90m              |  171.3 |  174.6 |  176.0 | 3.0 | 100 |
| avg_avgot_cpu_load_1m_filtered_by_hosts_5m       |    7.0 |    7.4 |    8.1 | 3.0 | 100 |
| avg_avgot_cpu_load_1m_prefix_by_hosts_5m         |   22.0 |   22.6 |   24.3 | 3.0 | 100 |
| sum_rate_sys_cpu_time_large_clause_5m            |  685.5 |  695.2 |  703.8 | 1.5 | 100 |
| sum_lot_filesystem_usage_top5                    |   25.7 |   27.9 |   29.6 | 3.0 | 100 |
| avgot_memory_by_tbucket_1h                       |   76.1 |   78.0 |   78.9 | 3.0 | 100 |
| rate_cpu_by_tbucket_1h                           |  271.6 |  278.2 |  285.7 | 3.0 | 100 |

</details>

<details>
<summary>Prometheus</summary>

| Query                                          | p50 ms | p95 ms | p99 ms | RPS | OK% |
|-------------------------------------------------|-------:|-------:|-------:|----:|----:|
| avg_avgot_memory_by_host_1h                      |  545.8 |  554.1 |  556.1 | 3.0 | 100 |
| avg_avgot_memory_by_host_5m                      |  508.9 |  513.3 |  514.7 | 3.0 | 100 |
| avg_avgot_memory_by_host_30m_window_90m          |  590.1 |  593.2 |  594.0 | 3.0 | 100 |
| avg_rate_cpu_by_host_1h                          | 3518.7 | 3557.6 | 3560.1 | 0.3 | 100 |
| avg_rate_cpu_by_host_5m                          | 3262.1 | 3273.2 | 3279.7 | 0.3 | 100 |
| avg_rate_cpu_by_host_30m_window_90m              | 3615.1 | 3658.3 | 3658.7 | 0.3 | 100 |
| avg_avgot_cpu_load_1m_filtered_by_hosts_5m       |    3.2 |    3.4 |    3.7 | 3.0 | 100 |
| avg_avgot_cpu_load_1m_prefix_by_hosts_5m         |   87.0 |   88.8 |   90.3 | 3.0 | 100 |
| sum_rate_sys_cpu_time_large_clause_5m            | 3379.1 | 3394.6 | 3402.4 | 0.3 | 100 |
| sum_lot_filesystem_usage_top5                    |  440.8 |  444.8 |  445.9 | 2.3 | 100 |
| avgot_memory_by_tbucket_1h                       |  549.6 |  558.3 |  560.1 | 3.0 | 100 |
| rate_cpu_by_tbucket_1h                           | 3520.7 | 3576.7 | 3579.7 | 0.3 | 100 |

</details>

<details>
<summary>Mimir</summary>

| Query                                          | p50 ms | p95 ms | p99 ms | RPS | OK% |
|-------------------------------------------------|-------:|-------:|-------:|----:|----:|
| avg_avgot_memory_by_host_1h                      |  905.4 |  915.9 |  926.0 | 3.0 | 100 |
| avg_avgot_memory_by_host_5m                      |  837.0 |  847.2 |  853.4 | 3.0 | 100 |
| avg_avgot_memory_by_host_30m_window_90m          |  907.9 |  918.0 |  923.1 | 3.0 | 100 |
| avg_rate_cpu_by_host_1h                          | 4251.7 | 4263.6 | 5493.0 | 0.2 | 100 |
| avg_rate_cpu_by_host_5m                          | 3921.0 | 3929.1 | 3942.0 | 0.3 | 100 |
| avg_rate_cpu_by_host_30m_window_90m              | 4381.7 | 4392.0 | 4393.6 | 0.2 | 100 |
| avg_avgot_cpu_load_1m_filtered_by_hosts_5m       |    5.3 |    5.4 |    5.6 | 3.0 | 100 |
| avg_avgot_cpu_load_1m_prefix_by_hosts_5m         |  112.5 |  115.7 |  117.0 | 3.0 | 100 |
| sum_rate_sys_cpu_time_large_clause_5m            | 4252.0 | 4262.8 | 4264.2 | 0.2 | 100 |
| sum_lot_filesystem_usage_top5                    |  547.0 |  551.6 |  552.6 | 1.8 | 100 |
| avgot_memory_by_tbucket_1h                       |  685.0 |  690.4 |  694.0 | 3.0 | 100 |
| rate_cpu_by_tbucket_1h                           | 4344.1 | 4355.0 | 4355.8 | 0.2 | 100 |

</details>

## Prerequisites

- Docker (memory allocated to Docker Desktop/engine must be at least `.env`'s `CONTAINER_MEMORY_LIMIT`, since one engine's container runs at a time)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- `curl` and `tar` (pre-installed on macOS and most Linux distros)

## Quick Start

```bash
# Install Python deps + download metricsgenreceiver and vegeta into .bin/
make setup

# Run all four engines sequentially — ingests data, runs query benchmarks, and
# measures on-disk storage for each engine before tearing it down
make run

# Or run individually — each target starts the engine, ingests data, runs
# queries, measures disk usage, then stops the engine before returning
make elasticsearch
make prometheus
make mimir
make clickhouse

# Display the storage comparison table + chart
make report
```

## How It Works

A full benchmark consists of the following steps for each engine sequentially:

   1. Start & wait for the engine's container to become available
   2. **`load`** ([`src/benchmark/load/`](src/benchmark/load/)) the data via `metricsgenreceiver`
   3. **`query`** ([`src/benchmark/query/`](src/benchmark/query/)) the data via `vegeta`
   4. **`disk-usage`** ([`src/benchmark/disk_usage/`](src/benchmark/disk_usage/)) measure on-disk storage

Then, **`report`** ([`src/benchmark/report.py`](src/benchmark/report.py)) prepares and prints the comparison table and bar chart for all datastore engines.

Each engine target stops its own container once it finishes, so only one datastore is ever running at a time — this keeps the host's CPU/memory budget dedicated to whichever engine is currently being measured.

## Project Layout

```
src/benchmark/          # all Python code (installed as the "benchmark" package)
  engine_config.py      # ENGINE validation + per-engine connection config
  run_engine.py         # orchestrates bootstrap + load/query/disk-usage for one engine
  load/                 # metricsgenreceiver config rendering + execution
  query/                # queryset loading + vegeta execution
  disk_usage/           # per-engine storage measurement
  store/                # results/<engine>.json read/write
  utils/                # per-engine HTTP helpers (es.py, prometheus.py, mimir.py, clickhouse.py) + generic helpers (fs.py, size.py, time.py)
  scenarios.py          # resolves scenarios/<name>.yml + its queryset reference
  report.py             # cross-engine comparison table + chart
scenarios/               # scenario definitions (<name>.yml)
querysets/                # query definitions, referenced by name from a scenario
deploy/
  docker/                 # docker-compose.yml
  config/                 # per-engine runtime config (elasticsearch/, mimir.yaml, prometheus.yml, clickhouse/)
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
| `ES_VERSION` / `PROMETHEUS_VERSION` / `MIMIR_VERSION` / `CLICKHOUSE_VERSION` | Engine image tags |
| `ELASTICSEARCH_URL` / `PROMETHEUS_URL` / `MIMIR_URL` / `CLICKHOUSE_URL` | Local docker compose ports (ClickHouse's is the HTTP interface — schema bootstrap + queries) |
| `ES_HEAP` | Elasticsearch JVM heap — should be roughly half of `CONTAINER_MEMORY_LIMIT` |
| `ES_DATA_STREAM` | Elasticsearch data stream/index name |
| `CLICKHOUSE_NATIVE_ENDPOINT` | Native TCP endpoint the OTel `clickhouse` exporter writes to (ClickHouse has no OTLP receiver) |
| `CLICKHOUSE_DATABASE` | ClickHouse database name (no auth — the container runs with the `default` user and no password) |
| `CONTAINER_MEMORY_LIMIT` / `CONTAINER_CPU_LIMIT` | Resource limits applied to each engine's container |

Scenario parameters (`scale`, `interval`, `start_now_minus`, `seed`, and which
queryset to run) live in `scenarios/*.yml` instead — see
[Benchmarks: scenarios & querysets](#benchmarks-scenarios--querysets) above.

## Methodology

Matches the team's [columnar metrics engine benchmark](https://www.elastic.co/search-labs/blog/elasticsearch-columnar-metrics-engine-30x-faster-prometheus) (low-cardinality setup: 100 hosts, 1s interval). Scale is configurable.

## Reproducible Cloud Runs (AWS)

For a clean, reproducible environment — or to run the benchmark on beefier hardware than a laptop — `deploy/terraform/aws/` provisions an EC2 instance that clones this repo and runs the benchmark unattended via cloud-init.

The instance type defaults to `c8gd.4xlarge`, which has local NVMe instance storage; the cloud-init script formats it (RAID-0 across multiple NVMe devices if the instance type has more than one) and mounts it as both the Docker `data-root` and the benchmark's working directory, so ingestion isn't bottlenecked by EBS.

```bash
cd deploy/terraform/aws
terraform init
terraform apply \
  -var github_token=<token>   # optional — omit for a public repo/branch
```

Key variables (see `variables.tf` for the full list and defaults):

| Variable | Default | Description |
|---|---|---|
| `repo` | `elastic/competitive-benchmarking-studies` | GitHub repo to clone |
| `branch` | `main` | Branch to check out |
| `github_token` | `""` | Optional — only needed for private repos/forks |
| `machine` | `c8gd.4xlarge` | EC2 instance type (must have local NVMe storage) |
| `nvme_mount` | `/data` | Mount point for the local NVMe device |
| `run_command` | `make setup run` | Full command executed on the instance after cloning — override to run a single engine, e.g. `make setup elasticsearch` |
| `shutdown` | `false` | Terminate the instance automatically once `run_command` finishes |
| `key_name` | `null` | EC2 key pair for SSH access (optional — see below if omitted) |
| `tags` | `{}` | Additional tags to apply to the instance |

Terraform prints `instance_id`, `public_ip`, and a ready-to-run `console_log_command` for tailing cloud-init progress via `aws ec2 get-console-output`.

### Retrieving results

The benchmark log is written to `<nvme_mount>/benchmark.log`, and the JSON results land in `<nvme_mount>/repo/otel-metrics/results/`.

If you didn't set `key_name`, you can still pull files off the instance via EC2 Instance Connect — push a short-lived key and `scp` with it:

```bash
INSTANCE_ID=$(terraform output -raw instance_id)
PUBLIC_IP=$(terraform output -raw public_ip)
AZ=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].Placement.AvailabilityZone' --output text)

ssh-keygen -t ed25519 -f /tmp/ec2-ic-key -N ""
aws ec2-instance-connect send-ssh-public-key \
  --instance-id "$INSTANCE_ID" --instance-os-user ubuntu \
  --availability-zone "$AZ" --ssh-public-key file:///tmp/ec2-ic-key.pub

scp -i /tmp/ec2-ic-key ubuntu@"$PUBLIC_IP":/data/benchmark.log .
scp -i /tmp/ec2-ic-key "ubuntu@$PUBLIC_IP:/data/repo/otel-metrics/results/*.json" ./results/
```

Remember to `terraform destroy` when you're done, unless `shutdown = true` was set (which terminates the instance automatically, since `instance_initiated_shutdown_behavior` is `terminate`).

# Elasticsearch 9.4.2 vs Prometheus 3.12 vs Mimir 3.1 — Data Point Storage Efficiency

Measures bytes per data point for three time-series engines after ingesting identical OTel hostmetrics data.

## Results

| Engine        | Version | Data Points | Size     | Bytes/DP | Elapsed | EPS     |
|---------------|---------|-------------|----------|----------|---------|---------|
| Elasticsearch | 9.4.2   | 225,180,000 | 801.5 MB | 3.73     | 14m17s  | 262,480 |
| Prometheus    | 3.12.0  | 225,180,000 | 828.9 MB | 3.86     | 10m27s  | 358,972 |
| Mimir         | 3.1.1   | 225,180,000 | 838.7 MB | 3.91     | 10m45s  | 348,854 |

Elasticsearch's advantage comes from: TSDB columnar codec, synthetic `_source`, synthetic document IDs, doc value skippers (no inverted indices on dimensions), and trimmed sequence numbers — all applied automatically via the built-in OTel index template.

Elapsed/EPS measure the `metricsgenreceiver` ingestion run only (wall-clock time ÷ data points), not the storage-measurement step that follows it (force-merge/snapshot/flush).

### Query Benchmark Results

Each engine ran the same 12 PromQL/ES|QL-equivalent queries (`queries.yml`) via `vegeta` at a fixed low rate, over the same time range as the ingested data. All queries returned 100% success across all three engines.

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

- Docker (with at least 6 GB memory allocated)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- `curl` and `tar` (pre-installed on macOS and most Linux distros)

## Quick Start

```bash
# Install Python deps + download metricsgenreceiver and vegeta into .bin/
make setup

# Run all three engines sequentially — ingests data, runs query benchmarks, and
# measures on-disk storage for each engine before tearing it down (~45 min total)
make run

# Or run individually — each target starts the engine, ingests data, runs
# queries, measures disk usage, then stops the engine before returning
make elasticsearch    # ~15 min
make prometheus       # ~15 min
make mimir            # ~15 min

# Display storage comparison
make report

# Re-run query benchmarks standalone (requires services to still be running)
make query            # all three engines
make query-es         # Elasticsearch only
make query-prometheus # Prometheus only
make query-mimir      # Mimir only

# Re-measure on-disk storage standalone, without re-ingesting
# (requires services to still be running)
make disk-usage            # all three engines
make disk-usage-es         # Elasticsearch only
make disk-usage-prometheus # Prometheus only
make disk-usage-mimir      # Mimir only
```

Each engine target stops its own container (`docker compose down <engine>`) once it finishes, so only one datastore is ever running at a time — this keeps the host's CPU/memory budget dedicated to whichever engine is currently being measured.

## How It Works

1. **Data generation**: `metricsgenreceiver` generates synthetic OTel hostmetrics (100 hosts × 270 minutes × ~139 metrics/host × 1s interval ≈ 225M data points) and streams directly to each engine via OTLP/HTTP with protobuf + gzip — no intermediate files.

2. **Elasticsearch setup**: The built-in `metrics-otel@template` index template activates automatically. `metrics-otel@custom` overrides replicas/shards/look_back_time. Trial license enables synthetic `_source`.

3. **Storage measurement** (`disk_usage/measure.py`, run via `make disk-usage*`):
   - **Elasticsearch**: `_forcemerge` to 1 segment, then `POST _disk_usage` (`run_expensive_tasks=true`) for an exact, per-shard on-disk size
   - **Prometheus**: `POST /api/v1/admin/tsdb/snapshot` → measure snapshot directory size (blocks only, excludes WAL) via `docker exec du -sb`. Matches the [methodology used by the Prometheus team](https://github.com/gouthamve/prom-elastic-benchmark/blob/632ae80262bf1bb6fc44aa89480307ef7576f51c/scripts/measure-prom.sh).
   - **Mimir**: `POST /ingester/flush` to force block compaction, then measure the `blocks/` directory size on the host (excludes `tsdb/`/`tsdb-sync/` WAL so it's comparable to the Prometheus snapshot)

## Configuration

Edit `.env` to adjust:

| Variable | Default | Description |
|---|---|---|
| `ES_VERSION` | `9.4.2` | Elasticsearch image tag |
| `PROMETHEUS_VERSION` | `3.12.0` | Prometheus image tag |
| `MIMIR_VERSION` | `3.1.0` | Mimir image tag |
| `SCALE` | `100` | Simulated hosts |
| `INTERVAL` | `1s` | Metric emission interval |
| `START_NOW_MINUS` | `270m` | Data window (historical replay) |
| `ES_HEAP` | `6g` | Elasticsearch JVM heap — set to half of `CONTAINER_MEMORY_LIMIT` |
| `CONTAINER_MEMORY_LIMIT` | `12g` | Memory limit applied to each datastore container (Elasticsearch, Prometheus, Mimir) |
| `CONTAINER_CPU_LIMIT` | `12` | CPU limit applied to each datastore container |
| `METRICSGENRECEIVER_VERSION` | `1.0.7` | Version of the synthetic data generator binary |
| `VEGETA_VERSION` | `12.12.0` | Version of the vegeta load-testing binary used for query benchmarks |

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

The benchmark log is written to `<nvme_mount>/benchmark.log`, and the JSON results land in `<nvme_mount>/repo/elasticsearch-9.4-vs-prometheus-3.12-vs-mimir-3.1-data-point-size/results/`.

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
scp -i /tmp/ec2-ic-key "ubuntu@$PUBLIC_IP:/data/repo/elasticsearch-9.4-vs-prometheus-3.12-vs-mimir-3.1-data-point-size/results/*.json" ./results/
```

Remember to `terraform destroy` when you're done, unless `shutdown = true` was set (which terminates the instance automatically, since `instance_initiated_shutdown_behavior` is `terminate`).

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

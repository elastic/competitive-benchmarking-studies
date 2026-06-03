# Elasticsearch 9.4 vs Qdrant 1.17: Vector Search Performance

This benchmark compares approximate nearest neighbor (ANN) vector search performance between **Elasticsearch 9.4** and **Qdrant 1.17** using their respective on-disk quantized vector strategies.

## Configuration

|                  | Elasticsearch 9.4                                  | Qdrant 1.17                                                      |
| ---------------- | -------------------------------------------------- | ---------------------------------------------------------------- |
| **Vector index** | `dense_vector` with `index_options.type: bbq_disk` | HNSW + binary quantization (`on_disk: true`, `always_ram: true`) |
| **Quantization** | BBQ (1-bit)                                        | Two-bits (`encoding: two_bits`)                                  |
| **HNSW**         | Server defaults                                    | `m` = 16, `ef_construct` = 100                                   |
| **Similarity**   | Cosine                                             | Cosine                                                           |
| **Sharding**     | 3 shards, 1 replica                                | 3 shards, replication factor 2                                   |
| **Post-load**    | Force merge                                        | —                                                                |

## Dataset

- **wiki-dpr-e5-768** — Wikipedia passages with **768-dimensional** E5 embeddings
- recall@100 sweep over engine tuning parameters (ES: `visit_percentage`; Qdrant: `hnsw_ef`)
- Pre-computed ground truth in the query parquet for recall calculation

## Infrastructure

- **3 data nodes** per cluster (`WORKER_COUNT` in `.env`; `n4-standard-8`)
- **320 Gi** persistent volume per data pod on **hyperdisk-balanced** (160 000 IOPS / 2 400 MB/s); StorageClass in `infra/k8s/storage-class.yml`

## Key Results (recall@100)

| ~Recall | Elasticsearch 9.4.0 params     | ES avg latency (ms) | Qdrant 1.17.1 params | QD avg latency (ms) |  Speedup |
| ------- | ------------------------------ | ------------------: | -------------------- | ------------------: | -------: |
| ~87%    | `visit_percentage=1` (88.7%)   |                  83 | `hnsw_ef=10` (87.2%) |                 172 |  **~2×** |
| ~92%    | `visit_percentage=1.5` (92.0%) |                  84 | `hnsw_ef=50` (93.3%) |                 607 |  **~7×** |
| ~94%    | `visit_percentage=2` (93.9%)   |                  91 | `hnsw_ef=60` (94.2%) |                 668 |  **~7×** |
| ~95%    | `visit_percentage=2.5` (95.2%) |                  91 | `hnsw_ef=70` (94.9%) |                 861 |  **~9×** |
| ~96%    | `visit_percentage=3` (96.0%)   |                  93 | `hnsw_ef=90` (95.9%) |                1177 | **~13×** |

Full per-parameter rows: `analyze/output/recall@100_full_results.csv`.

## Summary

- Elasticsearch 9.4 with BBQ disk delivers **much lower average latency** than Qdrant 1.17 with binary quantization on disk at similar recall@100
- The gap grows with recall: **~2× faster** at ~87% recall, **~13× faster** at ~96% recall
- ES latency is nearly flat across the recall range (~83–93 ms); Qdrant latency climbs steeply (~172–1177 ms)

## Prerequisites

The following tools must be available in your PATH:

| Tool | Notes |
| ---- | ----- |
| `gcloud` | Google Cloud SDK, authenticated (`gcloud auth login`) |
| `terraform` | >= 1.0 |
| `kubectl` | Configured after `make connect-k8s` |
| `helm` | >= 3, required for the OpenSearch operator |
| `docker` | With Buildx enabled for the multi-arch Jingra build |
| `make` | |
| `yq` | YAML processor |
| `envsubst` | Usually bundled with `gettext` |
| `jq` | Required to compact the GCP credentials JSON (see below) |

## Reproducing the Benchmark

### 1. Build the Jingra image

This benchmark requires **Jingra v0.2.2**. Clone the repo, check out the tag, and build a multi-arch image (requires Docker Buildx):

```bash
git clone https://github.com/elastic/jingra.git
cd jingra
git checkout v0.2.2
make build image=your-registry.example.com/your-namespace/jingra:0.2.2
```

`make build` runs all tests then pushes a `linux/amd64,linux/arm64/v8` image to your registry.

### 2. Configure secrets

Copy the example file and fill in each section:

```bash
cp .secrets.env.example .secrets.env
```

**GCP** (`PROJECT_ID`, `GOOGLE_CREDENTIALS`)

You need a GCP project with the following APIs enabled:

```bash
gcloud services enable container.googleapis.com compute.googleapis.com \
  --project=<your-project-id>
```

Create a service account and grant it the required roles:

```bash
gcloud iam service-accounts create benchmark-sa --project=<your-project-id>

for role in roles/container.admin roles/compute.admin roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding <your-project-id> \
    --member="serviceAccount:benchmark-sa@<your-project-id>.iam.gserviceaccount.com" \
    --role="$role"
done
```

Download the JSON key and compact it to a single line for `.secrets.env`:

```bash
gcloud iam service-accounts keys create key.json \
  --iam-account=benchmark-sa@<your-project-id>.iam.gserviceaccount.com
GOOGLE_CREDENTIALS=$(jq -c . key.json)
```

**Docker registry** (`DOCKER_*`)

These credentials are used to create a Kubernetes image pull secret so GKE can pull the Jingra image. Use the same registry you pushed to in step 1.

**Results cluster** (`RESULTS_ES_*`)

An Elasticsearch cluster used to store benchmark results. Any reachable Elasticsearch instance works — Elastic Cloud, self-hosted, or local. The user needs write access to create indices. Set `RESULTS_ES_URL`, `RESULTS_ES_USER`, and `RESULTS_ES_PASSWORD` accordingly.

### 3. Run the benchmark

Run from this directory (see `make help` for all targets and variables):

```bash
make terraform-apply ENGINE=qdrant
make k8s-apply       ENGINE=qdrant
make jingra-load     ENGINE=qdrant JINGRA_IMAGE=<image>
make jingra-eval     ENGINE=qdrant JINGRA_IMAGE=<image>

make terraform-apply ENGINE=elasticsearch
make k8s-apply       ENGINE=elasticsearch
make jingra-load     ENGINE=elasticsearch JINGRA_IMAGE=<image>
make jingra-eval     ENGINE=elasticsearch JINGRA_IMAGE=<image>

make analyze         ENGINES=elasticsearch,qdrant JINGRA_IMAGE=<image>
```

`make analyze` writes CSVs and plots under `analyze/output/`.

> **Note:** Elasticsearch `jingra-load` triggers a force merge after indexing. This is required for representative BBQ disk query performance and runs to completion before evaluation begins.

## Infrastructure

Terraform for provisioning each GKE cluster:

- `elasticsearch/terraform/` — Elasticsearch cluster
- `qdrant/terraform/` — Qdrant cluster

Kubernetes manifests for deploying each engine:

- `elasticsearch/k8s/` — ECK Elasticsearch + Kibana
- `qdrant/k8s/` — Qdrant StatefulSet + services

Shared infrastructure:

- `infra/k8s/` — StorageClass, Jingra job manifests, dataset PVC
- `infra/terraform/modules/gke-benchmark/` — shared GKE Terraform module

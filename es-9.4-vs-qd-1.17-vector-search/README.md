# Elasticsearch 9.4 vs Qdrant 1.17: Vector search performance

This benchmark compares approximate nearest neighbor (ANN) vector search between **Elasticsearch 9.4** (BBQ disk-backed vectors) and **Qdrant 1.17** (HNSW with binary quantization on disk and rescore oversampling) under matched hardware and the same query workload.

## Configuration

| | Elasticsearch 9.4 | Qdrant 1.17 |
| --- | --- | --- |
| **Vector index** | `dense_vector` with `index_options.type: bbq_disk` | HNSW + **binary** quantization (`on_disk: true`, `always_ram: true`) |
| **Quantization** | BBQ (1-bit) | Two-bits (`encoding: two_bits`) |
| **HNSW (graph)** | BBQ / server defaults for the template | `m` = 16, `ef_construct` = 100 |
| **Similarity** | Cosine | Cosine |
| **Sharding** | 3 shards, 1 replica | 3 shards, replication factor 2 |
| **Post-load** | Force merge (best-effort, `forcemerge: true`) | — |

## Dataset

- **wiki-dpr-e5-768** — DPR/Wikipedia-style passages with **768-dimensional** E5 embeddings (cosine)
- **recall@100** sweep over engine-specific tuning parameters (Elasticsearch: `visit_percentage`; Qdrant: `hnsw_ef` / `oversampling`)
- Pre-computed ground truth in the query parquet for recall calculation

## Infrastructure

- **3 data nodes** per engine cluster (`workerCount` in `shared/variables/k8s.yml`; pool `n4-standard-8` in `shared/variables/terraform.tfvars`)
- **320 Gi** persistent volume per data pod on **hyperdisk-balanced** (160 000 IOPS / 2 400 MB/s provisioned); StorageClass defined in `shared/infra/k8s/storage-class.yml`
- GKE on GCP (**us-central1-b** in `shared/variables/terraform.tfvars`)
- Separate clusters for Elasticsearch and Qdrant (`engines/<stack>/terraform/`)

## Key results (recall@100)

Matched **recall** tiers from `analyze/output/recall@100_summary.csv` (Jingra **average latency**, ms). Values are from the committed analysis run; rerun `make analyze` after new evaluations to refresh.

| ~Recall | Elasticsearch 9.4.0 params | ES avg latency (ms) | Qdrant 1.17.1 params | QD avg latency (ms) | Speedup |
| --- | --- | ---: | --- | ---: | ---: |
| ~87% | `visit_percentage=1` (88.7%) | 83 | `hnsw_ef=10` (87.2%) | 172 | **~2×** |
| ~92% | `visit_percentage=1.5` (92.0%) | 84 | `hnsw_ef=50` (93.3%) | 607 | **~7×** |
| ~94% | `visit_percentage=2` (93.9%) | 91 | `hnsw_ef=60` (94.2%) | 668 | **~7×** |
| ~95% | `visit_percentage=2.5` (95.2%) | 91 | `hnsw_ef=70` (94.9%) | 861 | **~9×** |
| ~96% | `visit_percentage=3` (96.0%) | 93 | `hnsw_ef=90` (95.9%) | 1177 | **~13×** |

Full per-parameter rows: `analyze/output/recall@100_full_results.csv`. Plots (if generated for that run): `analyze/output/`.

## Summary

- On this workload, Elasticsearch 9.4 with BBQ disk delivers **much lower average latency** than Qdrant 1.17 with binary quantization on disk + oversampling at similar recall@100.
- The gap grows with recall: **~2× faster** at ~87% recall, **~13× faster** at ~96% recall.
- ES latency is nearly flat across the recall range (~83–93 ms); Qdrant latency climbs steeply (~172–1177 ms).

## Reproducing the benchmark

### Building the Jingra image

This benchmark requires **Jingra v0.2.2**: https://github.com/elastic/jingra/releases/tag/v0.2.2

Clone the repo, check out the tag, and build a multi-arch image (requires Docker Buildx and a configured `buildx` builder):

```bash
git clone https://github.com/elastic/jingra.git
cd jingra
git checkout v0.2.2
make build image=<your-registry>/jingra:v0.2.2
```

`make build` runs all tests first (`mvn clean verify`), then pushes a `linux/amd64,linux/arm64/v8` image. Use `make build-no-cache` if you hit stale cache errors. Java version is read from `.java-version` in the repo root and passed as a build arg automatically.

### Running the benchmark

Run from this directory. Copy and fill these secret files first:

- `shared/secrets/.secrets.env.example` → `shared/secrets/.secrets.env`
- `shared/secrets/terraform.tfvars.example` → `shared/secrets/terraform.tfvars`
- `shared/secrets/credentials.json.example` → `shared/secrets/credentials.json` — GCP service account key (JSON) with permissions to create GKE clusters and manage GCS/Artifact Registry

Use the same `JINGRA_IMAGE` value everywhere you pass it below (the image tag you built above). Further targets and env vars: `make help`.

```bash
make qdrant terraform-apply
make qdrant k8s-apply
make qdrant jingra-load JINGRA_IMAGE=<jingra-image>
make qdrant jingra-eval JINGRA_IMAGE=<jingra-image>

make elasticsearch terraform-apply
make elasticsearch k8s-apply
make elasticsearch jingra-load JINGRA_IMAGE=<jingra-image>
make elasticsearch jingra-eval JINGRA_IMAGE=<jingra-image>

make analyze JINGRA_IMAGE=<jingra-image>
```

`make analyze` writes CSVs (and plots, if configured) under `analyze/output/`.

> **Note:** Elasticsearch `jingra-load` triggers a best-effort force merge after indexing (`forcemerge: true` in `engines/elasticsearch/jingra.yml`). This is required to reduce segment count and achieve representative query performance with BBQ disk vectors. The force merge runs asynchronously and is polled until complete before evaluation begins.

## Infrastructure setup

Terraform for provisioning each GKE cluster:

- `engines/elasticsearch/terraform/` — Elasticsearch cluster
- `engines/qdrant/terraform/` — Qdrant cluster

# Elasticsearch 9.3 vs OpenSearch 3.5: Vector Search Performance

This benchmark compares approximate nearest neighbor (ANN) vector search performance between **Elasticsearch 9.3** and **OpenSearch 3.5** using their respective quantized vector indexing strategies.

## Configuration

|                     | Elasticsearch 9.3         | OpenSearch 3.5               |
| ------------------- | ------------------------- | ---------------------------- |
| **Index Type**      | BBQ HNSW                  | FAISS (32x compression) |
| **HNSW Parameters** | m=16, ef_construction=100 | m=16, ef_construction=100    |
| **Similarity**      | Cosine                    | Cosine                       |

## Dataset

- **E-commerce search catalog** with 128-dimensional embeddings
- **10,000 queries** evaluated per configuration
- Pre-computed ground truth for recall calculation

## Infrastructure

- **6 data nodes** per cluster (e2-standard-16, 200GB disk)
- **3 shards**, 1 replica
- GKE on GCP (us-central1)

## Key Results (recall@100)

| num_candidates | ES 9.3 Recall | ES 9.3 Server Latency (ms) | OS 3.5 Recall | OS 3.5 Server Latency (ms) |
| -------------- | ------------- | -------------------------- | ------------- | -------------------------- |
| 250            | 77.0%         | 8                          | 70.2%         | 45                         |
| 1,000          | 91.6%         | 14                         | 87.5%         | 63                         |
| 5,000          | 97.5%         | 51                         | 96.8%         | 253                        |
| 10,000         | 98.5%         | 90                         | 98.4%         | 811                        |

## Summary

- Elasticsearch 9.3 with BBQ HNSW delivers **5-9x lower server latency** at comparable recall levels
- At 98%+ recall, Elasticsearch completes queries in ~90ms vs ~810ms for OpenSearch
- Elasticsearch achieves higher recall at lower `num_candidates` values, indicating more efficient graph traversal

## Reproducing the Benchmark

See [jingra/README.md](jingra/README.md) for instructions on running the benchmark tool.

## Infrastructure Setup

Terraform configurations for provisioning the GKE clusters are in:

- `elasticsearch-gcp/` - Elasticsearch cluster
- `opensearch-gcp/` - OpenSearch cluster

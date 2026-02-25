# Competitive Benchmarking Studies

A collection of performance benchmarks comparing search and database technologies under controlled, reproducible conditions.

## Benchmarks

### [Elasticsearch 9.3 vs OpenSearch 3.5: Vector Search](es-9.3-vs-os-3.5-vector-search/)

Compares ANN vector search performance using quantized HNSW indexes:

- **Elasticsearch 9.3** with BBQ HNSW
- **OpenSearch 3.5** with FAISS (32x compression)

**Key finding:** Elasticsearch delivers 5-9x lower server latency at comparable recall levels.

| Recall Target | ES 9.3 Server Latency | OS 3.5 Server Latency |
| ------------- | --------------------- | --------------------- |
| ~92%          | 14ms                  | 95ms                  |
| ~98%          | 90ms                  | 811ms                 |

See the [full benchmark details](es-9.3-vs-os-3.5-vector-search/README.md).

## Methodology

Each benchmark in this repository follows these principles:

- **Equivalent hardware** - Same instance types and cluster topology for all systems under test
- **Fair configuration** - Matching parameters where applicable (e.g., HNSW m, ef_construction)
- **Reproducible** - Infrastructure as code (Terraform) and benchmarking tools included
- **Transparent** - Raw results published

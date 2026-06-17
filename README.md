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
| ~93%          | 19ms                  | 95ms                  |
| ~98.5%        | 90ms                  | 811ms                 |

See the [full benchmark details](es-9.3-vs-os-3.5-vector-search/README.md).

### [Elasticsearch 9.4 vs Qdrant 1.18: Vector Search](es-9.4-vs-qd-1.18-vector-search/)

Compares ANN vector search performance using on-disk quantized indexes:

- **Elasticsearch 9.4** with DiskBBQ (2-bit)
- **Qdrant 1.18** with binary quantization (two-bit encoding, on disk)

**Key finding:** Elasticsearch delivers 2.3–7× lower latency at comparable recall levels on baseline network-attached storage, with the gap widening at higher recall targets. ES latency is insensitive to storage speed (page-cache bound); Qdrant latency scales with disk IOPS.

| Recall Target | ES 9.4 Avg Latency | QD 1.18 Avg Latency |
| ------------- | -----------------: | ------------------: |
| ~87%          | 135ms              | 316ms               |
| ~95%          | 123ms              | 885ms               |
| ~97%          | 123ms              | 882ms               |

See the [full benchmark details](es-9.4-vs-qd-1.18-vector-search/README.md).

### [Elasticsearch 9.5 vs Prometheus 3.12 vs Mimir 3.1: Metrics Storage Efficiency](elasticsearch-9.5-vs-prometheus-3.12-vs-mimir-3.1-data-point-size/)

Measures bytes per data point after ingesting 225M OTel hostmetrics samples (100 hosts × 270min × 1s) via OTLP. Runs entirely on local Docker — no cloud required.

| Engine        | Version        | Bytes/DP |
|---------------|----------------|----------|
| Elasticsearch | 9.5.0-SNAPSHOT | **3.02** |
| Mimir         | 3.1.0          | 3.88     |
| Prometheus    | 3.12.0         | 4.71     |

See the [full benchmark details](elasticsearch-9.5-vs-prometheus-3.12-vs-mimir-3.1-data-point-size/README.md).

## Methodology

Each benchmark in this repository follows these principles:

- **Equivalent hardware** - Same instance types and cluster topology for all systems under test
- **Fair configuration** - Matching parameters where applicable (e.g., HNSW m, ef_construction)
- **Reproducible** - Infrastructure as code (Terraform) and benchmarking tools included
- **Transparent** - Raw results published

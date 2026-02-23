# Jingra: A Comprehensive Benchmarking Tool

Jingra is a benchmarking tool for evaluating and comparing vector search performance across Elasticsearch and OpenSearch. It uses parquet-based datasets with pre-computed embeddings for testing.

## Getting Started

Follow these steps to set up and run the benchmark.

### 1. Installation

First, navigate to the project directory and install the dependencies.

```bash
cd jingra
uv pip install -e .
```

### 2. Configuration

Next, create your environment configuration file and fill in the necessary credentials.

```bash
cp src/benchmark/.env.sample src/benchmark/.env
```

You will need to edit the newly created `.env` file to include the following:

- `ELASTIC_URL`, `ELASTIC_USER`, `ELASTIC_PASSWORD`: Connection details for your Elasticsearch instance.
- `OPENSEARCH_URL`, `OPENSEARCH_USER`, `OPENSEARCH_PASSWORD`: Connection details for your OpenSearch instance.

## Usage (CLI)

All commands must be run from within the `jingra/` directory.

### Common Examples

- **Run a full benchmark (Load > Ingest > Evaluate):**

  ```bash
  uv run python -m src.benchmark.main --engine elasticsearch --load-kb --ingest-data --evaluate-retrieval
  ```

- **Run evaluation only (if data is already ingested):**

  ```bash
  uv run python -m src.benchmark.main --engine elasticsearch --evaluate-retrieval
  ```

- **Delete the index and re-ingest data:**

  ```bash
  uv run python -m src.benchmark.main --engine elasticsearch --delete-index --ingest-data
  ```

- **Generate plots and comparison reports from results:**

  ```bash
  # Generate plots for a specific date's results
  uv run python -m src.benchmark.main --plot-results 20260127

  # Generate a side-by-side comparison CSV
  uv run python -m src.benchmark.main --compare-results 20260127
  ```

### CLI Flag Reference

| Flag                       | Purpose                                                                      |
| :------------------------- | :--------------------------------------------------------------------------- |
| `--engine`                 | `elasticsearch` or `opensearch` (overrides `config.yaml`).                   |
| `--dataset`                | A dataset key from `config.yaml` (e.g., `ecommerce-search-128`).             |
| `--config`                 | Path to an alternate `config.yaml` file.                                     |
| `--load-kb`                | Display parquet dataset info (record and query counts).                      |
| `--ingest-data`            | Create the index and bulk-ingest documents.                                  |
| `--delete-index`           | Delete the existing index before ingestion.                                  |
| `--evaluate-retrieval`     | Run the full evaluation using `s_n_r_groups` from the config.                |
| `--quick-eval S_N_R,...`   | Evaluate a specific, comma-separated list of `s_n_r` values.                 |
| `--exact-match [N]`        | Perform a brute-force exact search for validation at recall@N (default: 10). |
| `--dump-engine-config`     | Dump engine cluster/index settings to the results `.config/` folder.         |
| `--plot-results [DATE]`    | Generate plots from results, with an optional `YYYYMMDD` date filter.        |
| `--compare-results [DATE]` | Generate an ES-vs-OS comparison CSV, with an optional date filter.           |

## Architecture Overview

- **Engines** (`engines/`): Contains the `VectorSearchEngine` abstract base class (`base.py`) and concrete implementations for each search engine. New engines can be added by implementing this interface and registering them in the `ENGINES` dictionary in `engines/__init__.py`.
- **Datasets** (`datasets/`): Parquet-based dataset loader for pre-computed embeddings. Dataset configuration is driven by `config/config.yaml`.
- **Evaluation** (`evaluation/`): This module contains the core logic for performance measurement, including search execution (`parquet_search.py`), metric calculation (`metrics.py`), and reporting (`reporting.py`).
- **Plotting & Comparison** (`plotting/`, `comparison/`): Modules responsible for generating visualizations and side-by-side performance reports from the raw benchmark results.
- **Configuration** (`config/config.yaml`): The central configuration file for datasets, engine parameters, evaluation settings, and output paths.

### Understanding `s_n_r_groups`

The benchmark uses parameter triplets called `s_n_r_groups` to control search behavior. The name is a shorthand for **s**ize\_**n**umCandidates\_**r**escore, configured in `config.yaml` as a string: `"size_numCandidates_rescore"`.

- **`size`**: The number of top-k results to return.
- **`numCandidates`**: The number of candidates to explore in the HNSW graph.
- **`rescore`**: An oversampling factor for a potential rescoring phase.

Results for each run are saved to a date-stamped directory in `results/` (e.g., `results/20260127/`).

## Contributing

To ensure the project remains consistent and easy to maintain, please follow these guidelines when contributing:

- **Follow existing patterns.** Do not invent new result formats or output structures.
- **Prefer configuration over code.** When adding datasets or parameters, modify `config.yaml` rather than hardcoding new logic.
- **Isolate engine-specific code.** Keep query logic contained within the respective engine's Python file (e.g., `elasticsearch.py`).
- **Preserve config comments.** The `s_n_r_groups` in `config.yaml` are commented for users to enable as needed. Do not alter this structure.

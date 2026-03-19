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

Create your configuration files:

```bash
# Copy the main config template
cp jingra.yaml.template jingra.yaml

# Copy the environment file
cp src/benchmark/.env.sample src/benchmark/.env
```

Edit the `.env` file to include your connection details:

- `ELASTIC_URL`, `ELASTIC_USER`, `ELASTIC_PASSWORD`: Connection details for your Elasticsearch instance.
- `OPENSEARCH_URL`, `OPENSEARCH_USER`, `OPENSEARCH_PASSWORD`: Connection details for your OpenSearch instance.

Edit `jingra.yaml` to configure your datasets, queries, and benchmark parameters.

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
| `--engine`                 | `elasticsearch` or `opensearch` (overrides `jingra.yaml`).                   |
| `--dataset`                | A dataset key from `jingra.yaml` (e.g., `ecommerce-search-128`).             |
| `--config`                 | Path to an alternate config file.                                            |
| `--load-kb`                | Display parquet dataset info (record and query counts).                      |
| `--ingest-data`            | Create the index and bulk-ingest documents.                                  |
| `--delete-index`           | Delete the existing index before ingestion.                                  |
| `--evaluate-retrieval`     | Run the full evaluation using `param_groups` from the dataset config.        |
| `--quick-eval JSON`        | Evaluate with a JSON array of param objects (see below).                     |
| `--exact-match [N]`        | Perform a brute-force exact search for validation at recall@N (default: 10). |
| `--dump-engine-config`     | Dump engine cluster/index settings to the results `.config/` folder.         |
| `--plot-results [DATE]`    | Generate plots from results, with an optional `YYYYMMDD` date filter.        |
| `--compare-results [DATE]` | Generate an ES-vs-OS comparison CSV, with an optional date filter.           |

### Quick Eval Example

```bash
uv run python -m src.benchmark.main --engine elasticsearch --quick-eval '[{"size":100,"k":100,"num_candidates":500,"rescore":1}]'
```

## Architecture Overview

- **Engines** (`engines/`): Contains the `VectorSearchEngine` abstract base class (`base.py`) and concrete implementations for each search engine. New engines can be added by implementing this interface and registering them in the `ENGINES` dictionary in `engines/__init__.py`.
- **Queries** (`queries/`): JSON query templates for each engine. Each query defines the search DSL with placeholders for runtime parameters.
- **Schemas** (`schemas/`): JSON index mapping templates for each engine. Each schema defines the index structure required for its corresponding query.
- **Datasets** (`datasets/`): Parquet-based dataset loader for pre-computed embeddings. Dataset configuration is driven by `jingra.yaml`.
- **Evaluation** (`evaluation/`): This module contains the core logic for performance measurement, including search execution (`parquet_search.py`), metric calculation (`metrics.py`), and reporting (`reporting.py`).
- **Plotting & Comparison** (`plotting/`, `comparison/`): Modules responsible for generating visualizations and side-by-side performance reports from the raw benchmark results.
- **Configuration** (`jingra.yaml`): The central configuration file for datasets, engine parameters, evaluation settings, and output paths.

### Understanding `param_groups`

Each dataset in `jingra.yaml` defines `param_groups` - collections of parameter sets used for benchmarking. Parameters are passed directly to the query template.

For vector search queries, typical parameters include:
- **`size`**: The number of top-k results to return.
- **`k`**: The k value for k-NN search.
- **`num_candidates`**: The number of candidates to explore in the HNSW graph.
- **`rescore`**: An oversampling factor for rescoring.

Example configuration:
```yaml
param_groups:
  recall@100:
    - {size: 100, k: 100, num_candidates: 250, rescore: 1}
    - {size: 100, k: 100, num_candidates: 500, rescore: 1}
    - {size: 100, k: 100, num_candidates: 1000, rescore: 1}
```

Results for each run are saved to a date-stamped directory in `results/` (e.g., `results/20260127/`).

## Contributing

To ensure the project remains consistent and easy to maintain, please follow these guidelines when contributing:

- **Follow existing patterns.** Do not invent new result formats or output structures.
- **Prefer configuration over code.** When adding datasets or parameters, modify `jingra.yaml` rather than hardcoding new logic.
- **Isolate engine-specific code.** Keep query logic contained within the query template JSON files in `queries/`.
- **Add matching schema and query files.** When adding a new query type, create both `schemas/<engine>/<name>.json` and `queries/<engine>/<name>.json`.

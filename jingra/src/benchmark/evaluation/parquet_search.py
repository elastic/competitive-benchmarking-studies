from __future__ import annotations
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset

from ..datasets.parquet_loader import ParquetDatasetLoader
from ..datasets.types import Query
from ..engines.base import VectorSearchEngine
from .results_uploader import get_uploader

logger = logging.getLogger(__name__)


class ProgressLogger:
    """K8s-friendly progress logger that outputs new lines at intervals."""

    def __init__(self, total: int, desc: str, log_interval_seconds: float = 10.0):
        self.total = total
        self.desc = desc
        self.log_interval = log_interval_seconds
        self.completed = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def update(self, n: int = 1) -> None:
        self.completed += n
        now = time.time()
        if now - self.last_log_time >= self.log_interval or self.completed == self.total:
            elapsed = now - self.start_time
            rate = self.completed / elapsed if elapsed > 0 else 0
            pct = self.completed / self.total * 100
            remaining = (self.total - self.completed) / rate if rate > 0 else 0
            logger.info(
                "%s: %d/%d (%.1f%%) | %.1f query/s | elapsed: %.0fs | remaining: ~%.0fs",
                self.desc, self.completed, self.total, pct, rate, elapsed, remaining
            )
            self.last_log_time = now

    def close(self) -> None:
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        logger.info(
            "%s: completed %d queries in %.1fs (%.1f query/s)",
            self.desc, self.completed, elapsed, rate
        )


def _params_to_key(params: Dict[str, Any]) -> str:
    """Convert params dict to a string key for column naming."""
    parts = [f"{k}={v}" for k, v in sorted(params.items()) if k != "query_vector"]
    return "_".join(parts)


def _warmup_search(
    engine: VectorSearchEngine,
    queries: List[Query],
    index_name: str,
    query_name: str,
    params: Dict[str, Any],
    warmup_workers: int = 8,
) -> None:
    """Run warmup queries to prime caches."""

    def execute_warmup(query: Query) -> None:
        if not query.vector:
            return
        engine.search(
            index_name=index_name,
            query_name=query_name,
            query_vector=query.vector,
            **params,
        )

    param_key = _params_to_key(params)
    with ThreadPoolExecutor(max_workers=warmup_workers) as executor:
        futures = [executor.submit(execute_warmup, q) for q in queries]
        progress = ProgressLogger(len(futures), f"Warmup {param_key}")
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.warning(f"Warmup query failed: {e}")
            progress.update()
        progress.close()


def _measure_parallel(
    queries: List[Query],
    engine: VectorSearchEngine,
    index_name: str,
    query_name: str,
    params: Dict[str, Any],
    measurement_workers: int = 8,
    upload_results: bool = False,
) -> Tuple[List[Dict[str, Any]], float]:
    """Run measurement queries and return results with elapsed time."""
    param_key = _params_to_key(params)

    def execute_single(query: Query, idx: int) -> Tuple[int, Dict[str, Any]]:
        output: Dict[str, Any] = {
            f"dense_retrieval_response_at_{param_key}": [],
            f"dense_retrieval_client_latency_at_{param_key}": None,
            f"dense_retrieval_server_latency_at_{param_key}": None,
            "expected_result": query.expected_result,
        }

        if not query.vector:
            return idx, output

        try:
            response = engine.search(
                index_name=index_name,
                query_name=query_name,
                query_vector=query.vector,
                **params,
            )
            doc_ids, client_latency, server_latency = engine.parse_search_response(response)
            output[f"dense_retrieval_response_at_{param_key}"] = doc_ids
            output[f"dense_retrieval_client_latency_at_{param_key}"] = client_latency
            output[f"dense_retrieval_server_latency_at_{param_key}"] = server_latency
        except Exception as e:
            logger.warning(f"Search failed for query {idx}, params={param_key}: {e}")

        return idx, output

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=measurement_workers) as executor:
        futures = [executor.submit(execute_single, q, idx) for idx, q in enumerate(queries)]
        indexed_results = []
        progress = ProgressLogger(len(futures), f"Measure {param_key}")

        for future in as_completed(futures):
            try:
                indexed_results.append(future.result())
            except Exception as e:
                logger.warning(f"Measurement query failed: {e}")
            progress.update()

        progress.close()
        indexed_results.sort(key=lambda x: x[0])
        results = [r[1] for r in indexed_results]

    elapsed_seconds = time.time() - start_time

    # Upload results AFTER measurement is complete (doesn't affect timing)
    if upload_results:
        uploader = get_uploader()
        if uploader and uploader.enabled:
            logger.info("Uploading %d query results to metrics index...", len(results))
            upload_docs = []
            for idx, (query, result) in enumerate(zip(queries, results)):
                doc_ids = result.get(f"dense_retrieval_response_at_{param_key}", [])
                client_latency = result.get(f"dense_retrieval_client_latency_at_{param_key}")
                server_latency = result.get(f"dense_retrieval_server_latency_at_{param_key}")
                ground_truth = query.expected_result
                if ground_truth and not isinstance(ground_truth, list):
                    ground_truth = [ground_truth]

                upload_docs.append({
                    "param_key": param_key,
                    "params": params,
                    "query_index": idx,
                    "result_ids": [str(r) for r in doc_ids] if doc_ids else [],
                    "result_count": len(doc_ids) if doc_ids else 0,
                    "client_latency_ms": client_latency,
                    "server_latency_ms": server_latency,
                    "ground_truth_ids": [str(g) for g in ground_truth] if ground_truth else [],
                    "ground_truth_count": len(ground_truth) if ground_truth else 0,
                })

            uploaded = uploader.upload_query_results_bulk(upload_docs)
            logger.info("Uploaded %d query results", uploaded)

    return results, elapsed_seconds


def create_parquet_evaluation_dataset(
    engine: VectorSearchEngine,
    parquet_loader: ParquetDatasetLoader,
    query_name: str,
    param_list: List[Dict[str, Any]],
    warmup_rounds: int = 3,
    warmup_workers: int = 8,
    measurement_rounds: int = 3,
    measurement_workers: int = 8,
) -> Dataset:
    """
    Create an evaluation dataset from a parquet-based dataset.

    Uses pre-computed embeddings from the parquet file.
    """
    index_name = parquet_loader.get_index_name()
    ground_truth_field = parquet_loader.get_ground_truth_field()

    # Load all queries into memory for repeated iterations
    logger.info("Loading queries from parquet...")
    queries = list(parquet_loader.load_queries())
    logger.info("Loaded %d queries", len(queries))

    # Build initial dataset with ground truth
    records: List[Dict[str, Any]] = []
    for query in queries:
        records.append(
            {
                ground_truth_field: query.expected_result,
                "_meta_conditions": query.meta_conditions,
            }
        )
    merged = Dataset.from_list(records)

    # Deduplicate params while preserving order
    seen = set()
    unique_params = []
    for p in param_list:
        key = _params_to_key(p)
        if key not in seen:
            seen.add(key)
            unique_params.append(p)

    for params in unique_params:
        param_key = _params_to_key(params)

        # Warmup phase
        for round_num in range(warmup_rounds):
            logger.info(
                f"Warm-up round {round_num + 1}/{warmup_rounds} for {param_key} ({warmup_workers} workers)..."
            )
            _warmup_search(
                engine,
                queries,
                index_name,
                query_name,
                params,
                warmup_workers=warmup_workers,
            )

        # Measurement phase
        for round_num in range(measurement_rounds):
            is_final_round = round_num == measurement_rounds - 1
            logger.info(
                f"Measurement round {round_num + 1}/{measurement_rounds} for {param_key} ({measurement_workers} workers)..."
            )
            results, elapsed_seconds = _measure_parallel(
                queries,
                engine,
                index_name,
                query_name,
                params,
                measurement_workers=measurement_workers,
                upload_results=is_final_round,  # Only upload on final round
            )

        # Add columns from the final measurement round
        for col_suffix in ["response", "client_latency", "server_latency"]:
            col = f"dense_retrieval_{col_suffix}_at_{param_key}"
            merged = merged.add_column(col, [r.get(col) for r in results])

        elapsed_col = f"elapsed_time_at_{param_key}"
        merged = merged.add_column(elapsed_col, [elapsed_seconds] * len(results))

    return merged

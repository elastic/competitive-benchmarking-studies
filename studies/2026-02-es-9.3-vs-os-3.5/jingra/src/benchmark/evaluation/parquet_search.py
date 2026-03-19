from __future__ import annotations
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset
from tqdm import tqdm

from ..datasets.parquet_loader import ParquetDatasetLoader
from ..datasets.types import Query
from ..engines.base import VectorSearchEngine

logger = logging.getLogger(__name__)


def _warmup_search_with_filters(
    engine: VectorSearchEngine,
    queries: List[Query],
    index_name: str,
    vector_field: str,
    s_n_r_value: str,
    warmup_workers: int = 8,
) -> None:
    """Run warmup queries with filters to prime caches."""
    size, num_candidates, rescore = map(int, s_n_r_value.split("_"))

    def execute_warmup(query: Query) -> None:
        if not query.vector:
            return
        filter_query = None
        if query.meta_conditions:
            filter_query = engine.condition_parser.parse(query.meta_conditions)
        engine.vector_search(
            index_name=index_name,
            query_vector=query.vector,
            vector_field=vector_field,
            size=size,
            num_candidates=num_candidates,
            rescore=rescore,
            filter_query=filter_query,
        )

    with ThreadPoolExecutor(max_workers=warmup_workers) as executor:
        futures = [executor.submit(execute_warmup, q) for q in queries]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Warmup {s_n_r_value}",
            unit="query",
        ):
            try:
                future.result()
            except Exception as e:
                logger.warning(f"Warmup query failed: {e}")


def _measure_parallel_with_filters(
    queries: List[Query],
    engine: VectorSearchEngine,
    index_name: str,
    vector_field: str,
    s_n_r_value: str,
    measurement_workers: int = 8,
) -> Tuple[List[Dict[str, Any]], float]:
    """Run measurement queries with filters and return results with elapsed time."""
    size, num_candidates, rescore = map(int, s_n_r_value.split("_"))

    def execute_single(query: Query, idx: int) -> Tuple[int, Dict[str, Any]]:
        output: Dict[str, Any] = {
            f"dense_retrieval_response_at_{s_n_r_value}": [],
            f"dense_retrieval_client_latency_at_{s_n_r_value}": None,
            f"dense_retrieval_server_latency_at_{s_n_r_value}": None,
            "expected_result": query.expected_result,
        }

        if not query.vector:
            return idx, output

        try:
            filter_query = None
            if query.meta_conditions:
                filter_query = engine.condition_parser.parse(query.meta_conditions)

            response = engine.vector_search(
                index_name=index_name,
                query_vector=query.vector,
                vector_field=vector_field,
                size=size,
                num_candidates=num_candidates,
                rescore=rescore,
                filter_query=filter_query,
            )
            doc_ids, client_latency, server_latency = engine.parse_search_response(response)
            output[f"dense_retrieval_response_at_{s_n_r_value}"] = doc_ids
            output[f"dense_retrieval_client_latency_at_{s_n_r_value}"] = client_latency
            output[f"dense_retrieval_server_latency_at_{s_n_r_value}"] = server_latency
        except Exception as e:
            logger.warning(f"Search failed for query {idx}, s_n_r={s_n_r_value}: {e}")

        return idx, output

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=measurement_workers) as executor:
        futures = [executor.submit(execute_single, q, idx) for idx, q in enumerate(queries)]
        indexed_results = [
            f.result()
            for f in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Measuring {s_n_r_value}",
                unit="query",
            )
        ]
        indexed_results.sort(key=lambda x: x[0])
        results = [r[1] for r in indexed_results]
    elapsed_seconds = time.time() - start_time

    return results, elapsed_seconds


def create_parquet_evaluation_dataset(
    engine: VectorSearchEngine,
    parquet_loader: ParquetDatasetLoader,
    s_n_r_values: List[str],
    warmup_rounds: int = 3,
    warmup_workers: int = 8,
    measurement_rounds: int = 3,
    measurement_workers: int = 8,
) -> Dataset:
    """
    Create an evaluation dataset from a parquet-based dataset.

    Uses pre-computed embeddings from the parquet file.
    Supports filtered kNN search via meta_conditions in queries.
    """
    index_name = parquet_loader.get_index_name()
    vector_field = parquet_loader.get_vector_field_name()
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

    unique_s_n_r = list(dict.fromkeys(s_n_r_values))

    # Warmup phase
    for s_n_r_value in unique_s_n_r:
        for round_num in range(warmup_rounds):
            logger.info(
                f"Warm-up round {round_num + 1}/{warmup_rounds} for s_n_r={s_n_r_value} ({warmup_workers} workers)..."
            )
            _warmup_search_with_filters(
                engine,
                queries,
                index_name,
                vector_field,
                s_n_r_value,
                warmup_workers=warmup_workers,
            )

        # Measurement phase
        # for s_n_r_value in unique_s_n_r:
        for round_num in range(measurement_rounds):
            logger.info(
                f"Measurement round {round_num + 1}/{measurement_rounds} for s_n_r={s_n_r_value} ({measurement_workers} workers)..."
            )
            results, elapsed_seconds = _measure_parallel_with_filters(
                queries,
                engine,
                index_name,
                vector_field,
                s_n_r_value,
                measurement_workers=measurement_workers,
            )

        # Add columns from the final measurement round
        for col_suffix in ["response", "client_latency", "server_latency"]:
            col = f"dense_retrieval_{col_suffix}_at_{s_n_r_value}"
            merged = merged.add_column(col, [r.get(col) for r in results])

        elapsed_col = f"elapsed_time_at_{s_n_r_value}"
        merged = merged.add_column(elapsed_col, [elapsed_seconds] * len(results))

    return merged

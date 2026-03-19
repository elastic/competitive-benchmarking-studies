import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from ..engines.base import VectorSearchEngine
from ..datasets.parquet_loader import ParquetDatasetLoader

logger = logging.getLogger(__name__)


def verify_parquet_exact_match(
    engine: VectorSearchEngine,
    parquet_loader: ParquetDatasetLoader,
    index_name: str,
    vector_field: str,
    ground_truth_field: str,
    size: int = 10,
    workers: int = 4,
) -> Dict[str, Any]:
    """Run exact match similarity searches for parquet datasets using script_score with dotProduct."""
    logger.info(f"Loading queries from parquet for exact match verification (size={size})...")

    # Load all queries into memory
    queries: List[Dict[str, Any]] = []
    for idx, query in enumerate(parquet_loader.load_queries()):
        queries.append({
            "idx": idx,
            "vector": query.vector,
            "ground_truth": query.expected_result,
        })

    logger.info(f"Loaded {len(queries)} queries for exact match verification")

    def execute_exact_search(query_data: Dict, idx: int) -> Tuple[int, Dict]:
        output = {
            "query_idx": idx,
            "ground_truth": query_data.get("ground_truth", []),
            "retrieved_ids": [],
            "recall": 0.0,
            "error": None,
        }
        embedding = query_data.get("vector", [])

        if not embedding:
            output["error"] = "No embedding"
            return idx, output

        # Convert numpy array to list if needed
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        try:
            query_dsl = {
                "query": {
                    "function_score": {
                        "query": {"match_all": {}},
                        "script_score": {
                            "script": {
                                "source": f"dotProduct(params.queryVector, '{vector_field}') + 1.0",
                                "params": {"queryVector": embedding},
                            }
                        },
                        "boost_mode": "replace",
                    }
                },
                "size": size,
            }

            response = engine._timed_search(index_name, query_dsl)
            hits = response.get("hits", {}).get("hits", [])
            doc_ids = [hit.get("_id") for hit in hits]
            output["retrieved_ids"] = doc_ids

            gt = query_data.get("ground_truth", [])
            # Handle numpy arrays
            if hasattr(gt, "tolist"):
                gt = gt.tolist()
            gt_ids = [str(g) for g in gt] if isinstance(gt, list) else [str(gt)] if gt else []
            if gt_ids:
                output["recall"] = len(set(gt_ids[:size]) & set(doc_ids[:size])) / min(len(gt_ids), size)

        except Exception as e:
            output["error"] = str(e)
            logger.warning(f"Exact match failed for query {idx}: {e}")

        return idx, output

    logger.info(f"Running exact similarity searches with {workers} workers...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(execute_exact_search, query, query["idx"])
            for query in queries
        ]
        indexed_results = [
            f.result()
            for f in tqdm(
                as_completed(futures), total=len(queries), desc="Exact searches"
            )
        ]
        indexed_results.sort(key=lambda x: x[0])
        results = [r[1] for r in indexed_results]

    elapsed_seconds = time.time() - start_time

    valid_results = [r for r in results if r["error"] is None]
    failed_results = [r for r in results if r["error"] is not None]

    recalls = [r["recall"] for r in valid_results]
    avg_recall = float(np.mean(recalls)) if recalls else 0.0

    recall_zero = sum(1 for r in recalls if r == 0)
    recall_low = sum(1 for r in recalls if 0 < r < 0.5)
    recall_medium = sum(1 for r in recalls if 0.5 <= r < 0.8)
    recall_high = sum(1 for r in recalls if 0.8 <= r < 1.0)
    recall_perfect = sum(1 for r in recalls if r == 1.0)

    print("\n" + "=" * 80)
    print(" " * 15 + "EXACT MATCH SIMILARITY SEARCH RESULTS (PARQUET)")
    print(f" " * 12 + f"(script_score with dotProduct, size={size})")
    print("=" * 80 + "\n")

    print(f"Total queries:                    {len(queries)}")
    print(f"Successful searches:              {len(valid_results)}")
    print(f"Failed searches:                  {len(failed_results)}")
    print(f"Search time:                      {elapsed_seconds:.2f}s")
    print(f"Throughput:                       {len(valid_results)/elapsed_seconds:.2f} qps")
    print()
    print(f"Average Recall:                   {avg_recall:.4f} ({avg_recall*100:.2f}%)")
    print()
    print("Recall Distribution:")
    print(f"  Recall = 0.0:                   {recall_zero} ({recall_zero/len(results)*100:.1f}%)")
    print(f"  0.0 < Recall < 0.5:             {recall_low} ({recall_low/len(results)*100:.1f}%)")
    print(
        f"  0.5 <= Recall < 0.8:            {recall_medium} ({recall_medium/len(results)*100:.1f}%)"
    )
    print(f"  0.8 <= Recall < 1.0:            {recall_high} ({recall_high/len(results)*100:.1f}%)")
    print(
        f"  Recall = 1.0:                   {recall_perfect} ({recall_perfect/len(results)*100:.1f}%)"
    )

    if failed_results:
        print("\n" + "-" * 80)
        print("FAILED QUERIES:")
        print("-" * 80)
        for r in failed_results[:20]:
            print(f"  Query {r['query_idx']}: error")
            print(f"    Error: {r['error']}")
        if len(failed_results) > 20:
            print(f"  ... and {len(failed_results) - 20} more failures")

    print("\n" + "-" * 80)
    print("SAMPLE QUERY RESULTS:")
    print("-" * 80)

    examples_by_level = {
        "perfect": next((r for r in results if r["recall"] == 1.0 and not r["error"]), None),
        "high": next((r for r in results if 0.8 <= r["recall"] < 1.0 and not r["error"]), None),
        "medium": next((r for r in results if 0.5 <= r["recall"] < 0.8 and not r["error"]), None),
        "low": next((r for r in results if 0.0 < r["recall"] < 0.5 and not r["error"]), None),
        "zero": next((r for r in results if r["recall"] == 0.0 and not r["error"]), None),
    }

    for level, example in examples_by_level.items():
        if example:
            gt_preview = example['ground_truth'][:3] if len(example['ground_truth']) > 3 else example['ground_truth']
            print(f"\n{level.upper()} Recall Example:")
            print(f"  Query {example['query_idx']}")
            print(f"  Ground Truth (first 3): {gt_preview}")
            print(f"  Retrieved (top 3): {example['retrieved_ids'][:3]}")
            print(f"  Recall: {example['recall']:.4f}")

    print("\n" + "=" * 80 + "\n")

    return {
        "size": size,
        "total_queries": len(queries),
        "successful_searches": len(valid_results),
        "failed_searches": len(failed_results),
        "avg_recall": avg_recall,
        "recall_zero": recall_zero,
        "recall_low": recall_low,
        "recall_medium": recall_medium,
        "recall_high": recall_high,
        "recall_perfect": recall_perfect,
        "throughput": len(valid_results) / elapsed_seconds,
        "results": results,
    }

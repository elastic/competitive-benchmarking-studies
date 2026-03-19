from typing import List

import numpy as np


def precision_at_k(ground_truth_ids: List[str], retrieved_ids: List[str], k: int) -> float:
    retrieved_at_k = retrieved_ids[:k]
    if not retrieved_at_k:
        return 0.0
    return len(set(ground_truth_ids) & set(retrieved_at_k)) / len(retrieved_at_k)


def recall_at_k(ground_truth_ids: List[str], retrieved_ids: List[str], k: int) -> float:
    if not ground_truth_ids:
        return 0.0
    return len(set(ground_truth_ids) & set(retrieved_ids[:k])) / len(ground_truth_ids)


def f1_at_k(ground_truth_ids: List[str], retrieved_ids: List[str], k: int) -> float:
    p = precision_at_k(ground_truth_ids, retrieved_ids, k)
    r = recall_at_k(ground_truth_ids, retrieved_ids, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def mrr_at_k(ground_truth_ids: List[str], retrieved_ids: List[str], k: int) -> float:
    if not ground_truth_ids:
        return 0.0
    ground_truth_set = set(ground_truth_ids)
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in ground_truth_set:
            return 1.0 / rank
    return 0.0


def latency_stats(values: list) -> dict:
    if not values:
        return {"avg": 0, "median": 0, "p90": 0, "p95": 0, "p99": 0}
    return {
        "avg": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset

from .metrics import precision_at_k, recall_at_k, f1_at_k, mrr_at_k, latency_stats


def calculate_retrieval_metrics(
    evaluation_dataset: Dataset,
    ground_truth_field: str,
    retrieval_methods: List[str],
    s_n_r_values: List[str],
    recall_label: str,
    save_results: bool = True,
    output_dir: str = "results",
    engine_info: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, float]]:
    results = {}

    for base_method in retrieval_methods:
        for s_n_r_value in s_n_r_values:
            method_column = f"{base_method}_at_{s_n_r_value}"
            precisions, recalls, f1s, mrrs = [], [], [], []
            client_latencies, server_latencies = [], []
            elapsed_time = None
            k = int(s_n_r_value.split("_")[0])

            for row in evaluation_dataset:
                ground_truth = row.get(ground_truth_field)
                retrieved = row.get(method_column)
                if ground_truth is None or retrieved is None:
                    continue

                gt_ids = (
                    ground_truth
                    if isinstance(ground_truth, list)
                    else [ground_truth] if ground_truth else []
                )
                ret_ids = retrieved if isinstance(retrieved, list) else []

                precisions.append(precision_at_k(gt_ids, ret_ids, k))
                recalls.append(recall_at_k(gt_ids, ret_ids, k))
                f1s.append(f1_at_k(gt_ids, ret_ids, k))
                mrrs.append(mrr_at_k(gt_ids, ret_ids, k))

                client_lat = row.get(method_column.replace("response", "client_latency"))
                if client_lat is not None:
                    client_latencies.append(client_lat)

                server_lat = row.get(method_column.replace("response", "server_latency"))
                if server_lat is not None:
                    server_latencies.append(server_lat)

                elapsed_time = row.get(f"elapsed_time_at_{s_n_r_value}")

            num_samples = len(precisions)
            client_stats = latency_stats(client_latencies)
            server_stats = latency_stats(server_latencies)
            throughput = (num_samples / elapsed_time) if elapsed_time and elapsed_time > 0 else 0.0

            results[method_column] = {
                "precision": float(np.mean(precisions)) if precisions else 0.0,
                "recall": float(np.mean(recalls)) if recalls else 0.0,
                "f1": float(np.mean(f1s)) if f1s else 0.0,
                "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
                "latency_avg": client_stats["avg"],
                "latency_median": client_stats["median"],
                "latency_p90": client_stats["p90"],
                "latency_p95": client_stats["p95"],
                "latency_p99": client_stats["p99"],
                "server_latency_avg": server_stats["avg"],
                "server_latency_median": server_stats["median"],
                "server_latency_p90": server_stats["p90"],
                "server_latency_p95": server_stats["p95"],
                "server_latency_p99": server_stats["p99"],
                "throughput": throughput,
                "num_samples": num_samples,
            }

    _print_formatted_results(results, retrieval_methods, s_n_r_values)

    if save_results:
        csv_path, json_path = _save_metrics_to_files(
            results, retrieval_methods, s_n_r_values, recall_label, output_dir, engine_info
        )
        print(f"\nResults saved to:\n  - CSV:  {csv_path}\n  - JSON: {json_path}\n")

    return results


def _print_formatted_results(
    results: Dict[str, Dict[str, float]], base_methods: List[str], s_n_r_values: List[str]
) -> None:
    grouped = {}
    for val in s_n_r_values:
        size, num_candidates, rescore = map(int, val.split("_"))
        grouped.setdefault((size, num_candidates), []).append(rescore)

    print("\n" + "=" * 150)
    print(" " * 38 + "RETRIEVAL EVALUATION RESULTS")
    print("=" * 150 + "\n")

    for base_method in base_methods:
        method_display_name = base_method.replace("_", " ").title()

        for (size, num_candidates), rescores in grouped.items():
            rescores_sorted = sorted(rescores)
            config_label = f"s={size}  n={num_candidates}"

            header = f"{'':40} {'Metric':<28} " + " ".join(
                [f"{'Rescore_' + str(r):<13}" for r in rescores_sorted]
            )
            print(header)
            print("-" * 150)

            metrics = [
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1 Score", "f1"),
                ("MRR", "mrr"),
                ("", None),
                ("Client Avg (ms)", "latency_avg"),
                ("Client Median (ms)", "latency_median"),
                ("Client P90 (ms)", "latency_p90"),
                ("Client P95 (ms)", "latency_p95"),
                ("Client P99 (ms)", "latency_p99"),
                ("", None),
                ("Server Avg (ms)", "server_latency_avg"),
                ("Server Median (ms)", "server_latency_median"),
                ("Server P90 (ms)", "server_latency_p90"),
                ("Server P95 (ms)", "server_latency_p95"),
                ("Server P99 (ms)", "server_latency_p99"),
                ("", None),
                ("Throughput (qps)", "throughput"),
            ]

            for i, (name, key) in enumerate(metrics):
                if key is None:
                    print()
                    continue
                values = []
                for r in rescores_sorted:
                    method_key = f"{base_method}_at_{size}_{num_candidates}_{r}"
                    value = results.get(method_key, {}).get(key, 0.0)
                    values.append(f"{value:<13.4f}")
                if i == 0:
                    label = method_display_name
                elif i == 1:
                    label = config_label
                else:
                    label = ""
                print(f"{label:<40} {name:<28} " + " ".join(values))

            print("-" * 150 + "\n")


def _save_metrics_to_files(
    results: Dict[str, Dict[str, float]],
    base_methods: List[str],
    s_n_r_values: List[str],
    recall_label: str,
    output_dir: str,
    engine_info: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    date_str = datetime.now().strftime("%Y%m%d")
    output_path = Path(output_dir) / date_str
    output_path.mkdir(parents=True, exist_ok=True)

    if engine_info:
        recall_suffix = recall_label.replace("recall", "")
        version = engine_info.get("version", "unknown").replace("-SNAPSHOT", "")
        file_prefix = f"{engine_info.get('short_name', 'unknown')}_{version}_{engine_info.get('vector_type', 'unknown')}{recall_suffix}"
    else:
        file_prefix = f"retrieval_metrics_{recall_label}"

    csv_path = output_path / f"{file_prefix}.csv"
    json_path = output_path / f"{file_prefix}.json"

    with open(json_path, "w") as f:
        json.dump(
            {
                "test_date": date_str,
                "evaluation_datetime": datetime.now().isoformat(),
                "engine_info": engine_info or {},
                "metrics": results,
            },
            f,
            indent=2,
        )

    fieldnames = [
        "method",
        "s_n_r_value",
        "precision",
        "recall",
        "f1_score",
        "mrr",
        "latency_avg",
        "latency_median",
        "latency_p90",
        "latency_p95",
        "latency_p99",
        "server_latency_avg",
        "server_latency_median",
        "server_latency_p90",
        "server_latency_p95",
        "server_latency_p99",
        "throughput",
        "num_samples",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for base_method in base_methods:
            for s_n_r_value in s_n_r_values:
                m = results.get(f"{base_method}_at_{s_n_r_value}", {})
                writer.writerow(
                    {
                        "method": base_method,
                        "s_n_r_value": s_n_r_value,
                        "precision": f"{m.get('precision', 0):.4f}".rstrip("0").rstrip("."),
                        "recall": f"{m.get('recall', 0):.4f}".rstrip("0").rstrip("."),
                        "f1_score": f"{m.get('f1', 0):.4f}".rstrip("0").rstrip("."),
                        "mrr": f"{m.get('mrr', 0):.4f}".rstrip("0").rstrip("."),
                        "latency_avg": f"{m.get('latency_avg', 0):.2f}",
                        "latency_median": f"{m.get('latency_median', 0):.2f}",
                        "latency_p90": f"{m.get('latency_p90', 0):.2f}",
                        "latency_p95": f"{m.get('latency_p95', 0):.2f}",
                        "latency_p99": f"{m.get('latency_p99', 0):.2f}",
                        "server_latency_avg": f"{m.get('server_latency_avg', 0):.2f}",
                        "server_latency_median": f"{m.get('server_latency_median', 0):.2f}",
                        "server_latency_p90": f"{m.get('server_latency_p90', 0):.2f}",
                        "server_latency_p95": f"{m.get('server_latency_p95', 0):.2f}",
                        "server_latency_p99": f"{m.get('server_latency_p99', 0):.2f}",
                        "throughput": f"{m.get('throughput', 0):.2f}",
                        "num_samples": m.get("num_samples", 0),
                    }
                )

    return str(csv_path), str(json_path)

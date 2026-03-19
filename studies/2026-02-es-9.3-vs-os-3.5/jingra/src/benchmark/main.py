#!/usr/bin/env python3
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from .comparison.compare import run_comparison
from .config import load_config
from .datasets import ParquetDatasetLoader
from .engines import get_engine
from .evaluation import calculate_retrieval_metrics, create_parquet_evaluation_dataset
from .plotting import get_available_dates, organize_results_by_date, run_plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _resolve_target_date(arg_value: Optional[str], is_auto: bool) -> Optional[str]:
    """Resolve target date from CLI argument. Returns YYYYMMDD string or None."""
    if arg_value and arg_value != "auto":
        return arg_value
    if is_auto:
        return datetime.now().strftime("%Y%m%d")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Vector Search Benchmark Tool")

    parser.add_argument(
        "--engine",
        type=str,
        choices=["elasticsearch", "opensearch"],
        help="Vector search engine to use (overrides config.yaml)",
    )
    parser.add_argument("--dataset", type=str, help="Dataset to use (overrides config.yaml)")
    parser.add_argument("--config", type=str, help="Path to config.yaml file")
    parser.add_argument(
        "--load-kb",
        action="store_true",
        help="Load and display parquet dataset info",
    )
    parser.add_argument(
        "--ingest-data",
        action="store_true",
        help="Create index and ingest data into the search engine",
    )
    parser.add_argument(
        "--evaluate-retrieval", action="store_true", help="Run retrieval evaluation"
    )
    parser.add_argument(
        "--delete-index",
        action="store_true",
        help="Delete the index. Can be used standalone or with --ingest-data to recreate.",
    )
    parser.add_argument(
        "--plot-results",
        nargs="?",
        const="auto",
        metavar="DATE",
        help="Generate plots for results. If DATE (YYYYMMDD) provided, plot only that date. "
        "If 'auto', organize and plot after evaluation. Can be used standalone.",
    )
    parser.add_argument(
        "--compare-results",
        nargs="?",
        const="auto",
        metavar="DATE",
        help="Generate ES vs OS comparison CSV. If DATE (YYYYMMDD) provided, compare only that date. "
        "If 'auto', compare after evaluation. Can be used standalone.",
    )
    parser.add_argument(
        "--quick-eval",
        type=str,
        metavar="S_N_R,...",
        help="Run evaluation with specific s_n_r values (comma-separated). "
        "Example: --quick-eval '10_10_1,10_2000_2'",
    )
    parser.add_argument(
        "--exact-match",
        nargs="?",
        const=10,
        type=int,
        metavar="N",
        help="Run exact match verification with recall@N (default: 10). Example: --exact-match 20",
    )
    parser.add_argument(
        "--dump-engine-config",
        action="store_true",
        help="Dump engine configuration and settings to the results folder.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.engine:
        config.engine = args.engine

    if args.dataset:
        if args.dataset not in config.datasets:
            logger.error(
                f"Dataset '{args.dataset}' not found. Available: {list(config.datasets.keys())}"
            )
            sys.exit(1)
        config.dataset = args.dataset

    logger.info(f"Using engine: {config.engine}, dataset: {config.dataset}")

    dataset_config = config.get_current_dataset()

    # Create parquet loader
    logger.info("Using parquet dataset loader")
    dataset_loader = ParquetDatasetLoader(dataset_config.to_dict())

    engine = None
    if (
        args.delete_index
        or args.ingest_data
        or args.evaluate_retrieval
        or args.exact_match
        or args.quick_eval
        or args.dump_engine_config
    ):
        logger.info(f"Initializing {config.engine} connection...")
        engine = get_engine(config.engine, {config.engine: config.get_engine_config()})
        if not engine.connect():
            logger.error(f"Failed to connect to {config.engine}")
            sys.exit(1)

    # Determine the target date for results to ensure consistent directory naming
    current_results_date = _resolve_target_date(
        None, True
    )  # Use the existing helper for current date
    if not current_results_date:  # _resolve_target_date can return None
        logger.error("Could not resolve current results date.")
        sys.exit(1)
    output_base_dir = Path(config.output.results_dir) / current_results_date

    if args.dump_engine_config:
        index_name = dataset_loader.get_index_name()
        if not engine.index_exists(index_name):
            logger.warning(
                "Index '%s' does not exist, skipping dumping engine config and settings.",
                index_name,
            )
        else:
            dump_engine_config_info(engine, index_name, output_base_dir)
        # If only dumping config, we can exit here or proceed with other actions if any
        if not (
            args.load_kb
            or args.ingest_data
            or args.evaluate_retrieval
            or args.quick_eval
            or args.exact_match
        ):
            logger.info("Engine configuration dumped. Exiting as no other actions requested.")
            sys.exit(0)

    if args.delete_index and not args.ingest_data:
        # Standalone delete-index (when not combined with --ingest-data)
        index_name = dataset_loader.get_index_name()
        if engine.index_exists(index_name):
            logger.info(f"Deleting index: {index_name}")
            engine.delete_index(index_name)
        else:
            logger.warning(f"Index '{index_name}' does not exist")

    if args.load_kb:
        logger.info("Loading parquet dataset info...")
        logger.info(
            f"Parquet dataset: {dataset_loader.count_data()} data records, "
            f"{dataset_loader.count_queries()} queries"
        )

    if args.ingest_data:
        index_name = dataset_loader.get_index_name()

        if args.delete_index and engine.index_exists(index_name):
            logger.info(f"Deleting existing index: {index_name}")
            engine.delete_index(index_name)

        embedding_dimensions = dataset_config.vector_size

        logger.info(f"Creating index: {index_name}")
        engine.create_index(
            index_name=index_name,
            dataset_config=dataset_config.to_dict(),
            embedding_dimensions=embedding_dimensions,
        )

        total_docs = dataset_loader.count_data()
        logger.info(f"Streaming {total_docs} parquet records into {index_name}...")
        success_count, error_count = engine.ingest_streaming(
            index_name=index_name,
            action_generator=dataset_loader.stream_bulk_actions(),
            total=total_docs,
            chunk_size=2000,
            thread_count=4,
        )
        logger.info(f"Ingestion complete. Success: {success_count}, Errors: {error_count}")

    if args.evaluate_retrieval:
        index_name = dataset_loader.get_index_name()
        engine_info = {
            "short_name": engine.get_short_name(),
            "version": engine.get_version(),
            "vector_type": engine.get_vector_type(index_name),
        }
        logger.info(
            f"Engine: {engine_info['short_name']} v{engine_info['version']} ({engine_info['vector_type']})"
        )

        ground_truth_field = dataset_loader.get_ground_truth_field()

        for recall_label, s_n_r_values in config.s_n_r_groups.items():
            logger.info(f"Running evaluation for {recall_label}...")
            evaluation_dataset = create_parquet_evaluation_dataset(
                engine=engine,
                parquet_loader=dataset_loader,
                s_n_r_values=s_n_r_values,
                warmup_rounds=config.evaluation.warmup_rounds,
                warmup_workers=config.evaluation.warmup_workers,
                measurement_rounds=config.evaluation.measurement_rounds,
                measurement_workers=config.evaluation.measurement_workers,
            )
            calculate_retrieval_metrics(
                evaluation_dataset=evaluation_dataset,
                ground_truth_field=ground_truth_field,
                retrieval_methods=config.evaluation.retrieval_methods,
                s_n_r_values=s_n_r_values,
                recall_label=recall_label,
                save_results=True,
                output_dir=config.output.results_dir,
                engine_info=engine_info,
            )

        logger.info("Evaluation complete!")

    if args.quick_eval:
        index_name = dataset_loader.get_index_name()
        engine_info = {
            "short_name": engine.get_short_name(),
            "version": engine.get_version(),
            "vector_type": engine.get_vector_type(index_name),
        }
        logger.info(
            f"Engine: {engine_info['short_name']} v{engine_info['version']} ({engine_info['vector_type']})"
        )

        ground_truth_field = dataset_loader.get_ground_truth_field()

        s_n_r_values = [v.strip() for v in args.quick_eval.split(",")]

        # Group by size (first number) to derive recall labels
        groups: Dict[str, List[str]] = {}
        for val in s_n_r_values:
            size = val.split("_")[0]
            groups.setdefault(f"recall@{size}", []).append(val)

        for recall_label, group_values in groups.items():
            logger.info(f"Quick eval for {recall_label}: {group_values}")
            evaluation_dataset = create_parquet_evaluation_dataset(
                engine=engine,
                parquet_loader=dataset_loader,
                s_n_r_values=group_values,
                warmup_rounds=config.evaluation.warmup_rounds,
                warmup_workers=config.evaluation.warmup_workers,
                measurement_rounds=config.evaluation.measurement_rounds,
                measurement_workers=config.evaluation.measurement_workers,
            )
            calculate_retrieval_metrics(
                evaluation_dataset=evaluation_dataset,
                ground_truth_field=ground_truth_field,
                retrieval_methods=config.evaluation.retrieval_methods,
                s_n_r_values=group_values,
                recall_label=recall_label,
                save_results=True,
                output_dir=config.output.results_dir,
                engine_info=engine_info,
            )

        logger.info("Quick evaluation complete!")

    if args.exact_match:
        from .evaluation import verify_parquet_exact_match

        index_name = dataset_loader.get_index_name()
        vector_field = dataset_loader.get_vector_field_name()

        logger.info(f"Running exact match verification for parquet (recall@{args.exact_match})...")
        verify_parquet_exact_match(
            engine=engine,
            parquet_loader=dataset_loader,
            index_name=index_name,
            vector_field=vector_field,
            ground_truth_field=dataset_loader.get_ground_truth_field(),
            size=args.exact_match,
        )

    # Plotting
    if args.plot_results is not None:
        results_dir = config.output.results_dir
        is_auto = args.plot_results == "auto" and args.evaluate_retrieval
        target_date = _resolve_target_date(args.plot_results, is_auto)

        if target_date:
            logger.info(f"Plotting results for date: {target_date}")

        try:
            organized_dirs = organize_results_by_date(results_dir, target_date=target_date)
            if organized_dirs:
                logger.info(f"Results organized in {len(organized_dirs)} date folder(s)")
        except Exception as e:
            logger.error(f"Error organizing results: {e}", exc_info=True)

        try:
            logger.info("Generating plots...")
            plot_results = run_plots(
                results_dir=results_dir, target_date=target_date, latency_col="server_latency_avg"
            )
            if any(plot_results.values()):
                logger.info("Plot generation complete!")
            else:
                logger.warning("No plots were generated")
        except Exception as e:
            logger.error(f"Error generating plots: {e}", exc_info=True)

    # Comparison
    if args.compare_results is not None:
        results_dir = config.output.results_dir
        is_auto = args.compare_results == "auto" and args.evaluate_retrieval
        target_date = _resolve_target_date(args.compare_results, is_auto)

        if target_date:
            logger.info(f"Generating comparison for date: {target_date}")

        try:
            count = run_comparison(results_dir, target_date=target_date)
            if count > 0:
                logger.info(f"Generated {count} comparison file(s)")
            else:
                logger.warning("No comparisons were generated")
        except Exception as e:
            logger.error(f"Error generating comparison: {e}", exc_info=True)

    has_action = any(
        [
            args.delete_index,
            args.load_kb,
            args.ingest_data,
            args.evaluate_retrieval,
            args.quick_eval,
            args.plot_results is not None,
            args.compare_results is not None,
            args.exact_match,
        ]
    )
    if not has_action:
        parser.print_help()


def dump_engine_config_info(engine: Any, index_name: str, output_base_dir: Path) -> None:
    logger.info("Dumping engine configuration and settings...")
    config_dir = output_base_dir / ".config"
    config_dir.mkdir(parents=True, exist_ok=True)

    api_calls = {
        "cluster_settings": (engine._client.cluster.get_settings, {"include_defaults": True}),
        "index_stats": (engine._client.indices.stats, {"index": index_name, "human": True}),
        "index_settings": (
            engine._client.indices.get,
            {"index": index_name, "include_defaults": True},
        ),
        "root_info": (engine._client.info, {}),
    }

    for name, (method, kwargs) in api_calls.items():
        try:
            response = method(**kwargs)
            # Convert ObjectApiResponse to a dictionary if applicable
            if hasattr(response, "body") and isinstance(response.body, (dict, list)):
                response_to_dump = response.body
            elif isinstance(response, (dict, list)):
                response_to_dump = response
            else:
                logger.warning(
                    f"Response for {name} is of unexpected type {type(response)}. Attempting direct serialization."
                )
                response_to_dump = response

            if not response_to_dump or (
                isinstance(response_to_dump, dict) and not response_to_dump
            ):
                logger.warning(
                    f"API call for {name} returned empty or no data for {engine.get_short_name()}. Not writing file."
                )
                continue

            file_path = config_dir / f"{engine.get_short_name()}_{name}.json"
            with open(file_path, "w") as f:
                json.dump(response_to_dump, f, indent=4)
            logger.info(f"Successfully dumped {name} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to dump {name} for {engine.get_short_name()}: {e}", exc_info=True)


if __name__ == "__main__":
    main()

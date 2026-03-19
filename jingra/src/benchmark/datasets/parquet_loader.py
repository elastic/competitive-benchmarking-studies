from __future__ import annotations
import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

import pyarrow as pa
import pyarrow.dataset as ds

from .types import Query, Record

logger = logging.getLogger(__name__)

# Project root directory (jingra/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class ParquetDatasetLoader:
    """Loader for parquet-based datasets with pre-computed embeddings."""

    def __init__(
        self,
        dataset_config: Dict[str, Any],
    ):
        self.config = dataset_config
        self.index_name = dataset_config.get("index_name", "parquet_index")
        self.vector_size = dataset_config.get("vector_size", 128)
        self.distance = dataset_config.get("distance", "cosine")

        # Path configuration (optional env names for download-if-missing)
        path_config = dataset_config.get("path", {})
        self.data_path = path_config.get("data_path")
        self.queries_path = path_config.get("queries_path")
        self.data_url_env = path_config.get("data_url_env", "")
        self.queries_url_env = path_config.get("queries_url_env", "")

        # Data mapping configuration
        data_mapping = dataset_config.get("data_mapping", {})
        self._id_field = data_mapping.get("id_field")
        self._vector_field = data_mapping.get("vector_field", "vector")

        # Queries mapping configuration
        queries_mapping = dataset_config.get("queries_mapping", {})
        self._query_vector_field = queries_mapping.get("query_vector_field", "vector")
        self._ground_truth_field = queries_mapping.get("ground_truth_field", "closest_ids")
        self._conditions_field = queries_mapping.get("conditions_field", "conditions")

        # Schema for metadata fields
        self.schema = dataset_config.get("schema", {})

        # Cached datasets
        self._data_dataset: Optional[ds.Dataset] = None
        self._queries_dataset: Optional[ds.Dataset] = None
        self._metadata_fields: Optional[List[str]] = None

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path, treating relative paths as relative to PROJECT_ROOT."""
        p = Path(path)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p

    def _download_if_missing(self, resolved_path: Path, url_env: str) -> None:
        """If the file does not exist and url_env is set, download from that URL to the path."""
        if resolved_path.exists():
            return
        url = os.environ.get(url_env) if url_env else None
        if not url:
            raise FileNotFoundError(
                f"Dataset file not found: {resolved_path}. "
                f"Set {url_env} to download from a URL, or place the file at this path."
            )
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s to %s", url, resolved_path)
        req = urllib.request.Request(url, headers={"User-Agent": "jingra-benchmark/1.0"})
        with urllib.request.urlopen(req) as resp:
            with open(resolved_path, "wb") as f:
                while chunk := resp.read(8192):
                    f.write(chunk)
        logger.info("Downloaded %s", resolved_path)

    def _load_data_dataset(self) -> ds.Dataset:
        """Load the data parquet dataset lazily."""
        if self._data_dataset is None:
            if not self.data_path:
                raise ValueError("data_path not configured")
            resolved_path = self._resolve_path(self.data_path)
            self._download_if_missing(resolved_path, self.data_url_env)
            logger.info("Loading data parquet from %s", resolved_path)
            self._data_dataset = ds.dataset(str(resolved_path), format="parquet")
            self._compute_metadata_fields()
        return self._data_dataset

    def _load_queries_dataset(self) -> ds.Dataset:
        """Load the queries parquet dataset lazily."""
        if self._queries_dataset is None:
            if not self.queries_path:
                raise ValueError("queries_path not configured")
            resolved_path = self._resolve_path(self.queries_path)
            self._download_if_missing(resolved_path, self.queries_url_env)
            logger.info("Loading queries parquet from %s", resolved_path)
            self._queries_dataset = ds.dataset(str(resolved_path), format="parquet")
        return self._queries_dataset

    def _compute_metadata_fields(self) -> None:
        """Compute metadata fields by excluding vector and id fields."""
        dataset = self._load_data_dataset()
        all_columns = set(dataset.schema.names)
        core_fields: Set[str] = {self._vector_field}
        if self._id_field:
            core_fields.add(self._id_field)
        self._metadata_fields = list(all_columns - core_fields)

    def _validate_fields(self, dataset: ds.Dataset, required_fields: Set[str], operation: str) -> None:
        """Validate that required fields exist in the dataset."""
        available = set(dataset.schema.names)
        missing = required_fields - available
        if missing:
            raise ValueError(
                f"Fields required for '{operation}' missing from parquet: {list(missing)}"
            )

    def load_data(self) -> Iterator[Record]:
        """Yield Record objects from the data parquet file."""
        dataset = self._load_data_dataset()
        self._validate_fields(dataset, {self._vector_field}, "load_data")

        # Enable prefetching for faster reading
        for batch in dataset.to_batches(
            batch_readahead=8,
            fragment_scan_options=ds.ParquetFragmentScanOptions(
                use_buffered_stream=True,
                pre_buffer=True,
            ),
        ):
            for row in batch.to_pylist():
                record_id = row.get(self._id_field) if self._id_field else None
                metadata = {
                    field: row.get(field)
                    for field in (self._metadata_fields or [])
                    if field in row
                }
                yield Record(
                    id=record_id,
                    vector=row.get(self._vector_field),
                    metadata=metadata,
                )

    def load_queries(self) -> Iterator[Query]:
        """Yield Query objects from the queries parquet file."""
        dataset = self._load_queries_dataset()
        self._validate_fields(dataset, {self._query_vector_field}, "load_queries")

        for batch in dataset.to_batches():
            for row in batch.to_pylist():
                # Parse conditions - can be JSON string or already parsed dict
                conditions = None
                if self._conditions_field in row:
                    conditions_value = row[self._conditions_field]
                    if isinstance(conditions_value, str):
                        conditions = json.loads(conditions_value)
                    elif isinstance(conditions_value, dict):
                        conditions = conditions_value

                yield Query(
                    vector=row.get(self._query_vector_field),
                    meta_conditions=conditions,
                    expected_result=row.get(self._ground_truth_field),
                )

    def create_bulk_actions(self) -> List[Dict[str, Any]]:
        """Create bulk actions for ingestion from the data parquet file."""
        logger.info("Creating bulk actions from parquet data")
        actions: List[Dict[str, Any]] = []

        for record in self.load_data():
            source: Dict[str, Any] = {
                self._vector_field: record.vector,
            }
            if record.metadata:
                source.update(record.metadata)

            action: Dict[str, Any] = {"_source": source}
            if record.id is not None:
                action["_id"] = str(record.id)

            actions.append(action)

        logger.info("Created %d bulk actions", len(actions))
        return actions

    def stream_bulk_actions(self) -> Iterator[Dict[str, Any]]:
        """Stream bulk actions for memory-efficient ingestion of large datasets.

        Optimized to process parquet batches directly without Record object overhead.
        """
        logger.info("Streaming bulk actions from parquet data")
        dataset = self._load_data_dataset()

        # Process batches directly for better performance
        for batch in dataset.to_batches(
            batch_readahead=8,
            fragment_scan_options=ds.ParquetFragmentScanOptions(
                use_buffered_stream=True,
                pre_buffer=True,
            ),
        ):
            # Convert entire batch to Python list at once
            rows = batch.to_pylist()
            for row in rows:
                source: Dict[str, Any] = {self._vector_field: row.get(self._vector_field)}

                # Add metadata fields directly
                for field in self._metadata_fields or []:
                    if field in row:
                        source[field] = row[field]

                action: Dict[str, Any] = {"_source": source}
                if self._id_field and row.get(self._id_field) is not None:
                    action["_id"] = str(row[self._id_field])

                yield action

    def get_index_name(self) -> str:
        """Return the configured index name."""
        return self.index_name

    def get_vector_field_name(self) -> str:
        """Return the vector field name."""
        return self._vector_field

    def get_ground_truth_field(self) -> str:
        """Return the ground truth field name."""
        return self._ground_truth_field

    def count_data(self) -> int:
        """Return the number of records in the data file."""
        dataset = self._load_data_dataset()
        return dataset.count_rows()

    def count_queries(self) -> int:
        """Return the number of queries in the queries file."""
        dataset = self._load_queries_dataset()
        return dataset.count_rows()

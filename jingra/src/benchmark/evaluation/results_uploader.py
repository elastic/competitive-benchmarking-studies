"""Upload benchmark results to Elasticsearch for centralized storage."""
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

logger = logging.getLogger(__name__)


def _get_index_suffix() -> str:
    """Return current year-month suffix for index names."""
    return datetime.now().strftime("%Y-%m")


def get_results_index() -> str:
    """Get the index name for summary results/metrics."""
    return f"jingra-results-{_get_index_suffix()}"


def get_metrics_index() -> str:
    """Get the index name for individual query metrics."""
    return f"jingra-metrics-{_get_index_suffix()}"


class ResultsUploader:
    """Handles uploading benchmark results to Elasticsearch."""

    def __init__(self):
        self._client: Optional[Elasticsearch] = None
        self._enabled = False
        self._run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._engine_info: Dict[str, str] = {}
        self._dataset_name: str = "unknown"

    def initialize(
        self,
        engine_info: Optional[Dict[str, str]] = None,
        dataset_name: Optional[str] = None,
    ) -> bool:
        """Initialize the uploader with connection and metadata."""
        url = os.environ.get("RESULTS_ES_URL")
        user = os.environ.get("RESULTS_ES_USER")
        password = os.environ.get("RESULTS_ES_PASSWORD")

        if not url:
            logger.info("RESULTS_ES_URL not set, results upload disabled")
            return False

        try:
            kwargs: Dict[str, Any] = {
                "verify_certs": True,
                "max_retries": 3,
                "retry_on_timeout": True,
            }
            if user and password:
                kwargs["basic_auth"] = (user, password)

            self._client = Elasticsearch(hosts=[url], **kwargs)
            if not self._client.ping():
                logger.warning("Failed to ping results Elasticsearch cluster")
                return False

            logger.info("Connected to results Elasticsearch at %s", url)
            self._enabled = True
            self._engine_info = engine_info or {}
            self._dataset_name = dataset_name or "unknown"

            self._ensure_indices()
            return True
        except Exception:
            logger.exception("Failed to connect to results Elasticsearch")
            return False

    def _ensure_indices(self) -> None:
        """Create indices with mappings if they don't exist."""
        if not self._client:
            return

        results_index = get_results_index()
        metrics_index = get_metrics_index()

        # Results index for summary statistics
        if not self._client.indices.exists(index=results_index):
            self._client.indices.create(
                index=results_index,
                body={
                    "mappings": {
                        "properties": {
                            "@timestamp": {"type": "date"},
                            "run_id": {"type": "keyword"},
                            "engine": {"type": "keyword"},
                            "engine_version": {"type": "keyword"},
                            "vector_type": {"type": "keyword"},
                            "dataset": {"type": "keyword"},
                            "recall_label": {"type": "keyword"},
                            "method": {"type": "keyword"},
                            "param_key": {"type": "keyword"},
                            "params": {"type": "object", "enabled": False},
                            "precision": {"type": "float"},
                            "recall": {"type": "float"},
                            "f1": {"type": "float"},
                            "mrr": {"type": "float"},
                            "latency_avg": {"type": "float"},
                            "latency_median": {"type": "float"},
                            "latency_p90": {"type": "float"},
                            "latency_p95": {"type": "float"},
                            "latency_p99": {"type": "float"},
                            "server_latency_avg": {"type": "float"},
                            "server_latency_median": {"type": "float"},
                            "server_latency_p90": {"type": "float"},
                            "server_latency_p95": {"type": "float"},
                            "server_latency_p99": {"type": "float"},
                            "throughput": {"type": "float"},
                            "num_samples": {"type": "integer"},
                        }
                    }
                },
            )
            logger.info("Created results index: %s", results_index)

        # Metrics index for individual query metrics
        if not self._client.indices.exists(index=metrics_index):
            self._client.indices.create(
                index=metrics_index,
                body={
                    "mappings": {
                        "properties": {
                            "@timestamp": {"type": "date"},
                            "run_id": {"type": "keyword"},
                            "engine": {"type": "keyword"},
                            "engine_version": {"type": "keyword"},
                            "vector_type": {"type": "keyword"},
                            "dataset": {"type": "keyword"},
                            "param_key": {"type": "keyword"},
                            "params": {"type": "object", "enabled": False},
                            "query_index": {"type": "integer"},
                            "result_ids": {"type": "keyword"},
                            "result_count": {"type": "integer"},
                            "client_latency_ms": {"type": "float"},
                            "server_latency_ms": {"type": "float"},
                            "ground_truth_ids": {"type": "keyword"},
                            "ground_truth_count": {"type": "integer"},
                        }
                    }
                },
            )
            logger.info("Created metrics index: %s", metrics_index)

    @property
    def enabled(self) -> bool:
        """Return whether results upload is enabled."""
        return self._enabled

    @property
    def run_id(self) -> str:
        """Return the unique run ID for this benchmark session."""
        return self._run_id

    def upload_query_result(
        self,
        query_index: int,
        param_key: str,
        params: Dict[str, Any],
        result_ids: List[Any],
        client_latency_ms: Optional[float],
        server_latency_ms: Optional[float],
        ground_truth_ids: Optional[List[Any]] = None,
    ) -> None:
        """Upload a single query result."""
        if not self._enabled or not self._client:
            return

        doc = {
            "@timestamp": datetime.utcnow().isoformat(),
            "run_id": self._run_id,
            "engine": self._engine_info.get("short_name", "unknown"),
            "engine_version": self._engine_info.get("version", "unknown"),
            "vector_type": self._engine_info.get("vector_type", "unknown"),
            "dataset": self._dataset_name,
            "param_key": param_key,
            "params": params,
            "query_index": query_index,
            "result_ids": [str(r) for r in result_ids] if result_ids else [],
            "result_count": len(result_ids) if result_ids else 0,
            "client_latency_ms": client_latency_ms,
            "server_latency_ms": server_latency_ms,
            "ground_truth_ids": [str(g) for g in ground_truth_ids] if ground_truth_ids else [],
            "ground_truth_count": len(ground_truth_ids) if ground_truth_ids else 0,
        }

        try:
            self._client.index(index=get_metrics_index(), body=doc)
        except Exception as e:
            logger.warning("Failed to upload query result %d: %s", query_index, e)

    def upload_query_results_bulk(
        self,
        results: List[Dict[str, Any]],
    ) -> int:
        """Upload multiple query results in bulk. Returns count of successful uploads."""
        if not self._enabled or not self._client:
            return 0

        actions = []
        for r in results:
            doc = {
                "@timestamp": datetime.utcnow().isoformat(),
                "run_id": self._run_id,
                "engine": self._engine_info.get("short_name", "unknown"),
                "engine_version": self._engine_info.get("version", "unknown"),
                "vector_type": self._engine_info.get("vector_type", "unknown"),
                "dataset": self._dataset_name,
                **r,
            }
            actions.append({"_index": get_metrics_index(), "_source": doc})

        try:
            success, errors = bulk(self._client, actions, raise_on_error=False)
            if errors:
                logger.warning("Bulk upload had %d errors", len(errors))
            return success
        except Exception as e:
            logger.warning("Failed to bulk upload results: %s", e)
            return 0

    def upload_metrics(
        self,
        results: Dict[str, Dict[str, Any]],
        base_methods: List[str],
        param_list: List[Dict[str, Any]],
        recall_label: str,
    ) -> bool:
        """Upload summary metrics."""
        if not self._enabled or not self._client:
            return False

        timestamp = datetime.utcnow().isoformat()
        docs = []

        for base_method in base_methods:
            for params in param_list:
                param_key = _params_to_key(params)
                method_key = f"{base_method}_at_{param_key}"
                m = results.get(method_key, {})

                doc = {
                    "@timestamp": timestamp,
                    "run_id": self._run_id,
                    "engine": self._engine_info.get("short_name", "unknown"),
                    "engine_version": self._engine_info.get("version", "unknown"),
                    "vector_type": self._engine_info.get("vector_type", "unknown"),
                    "dataset": self._dataset_name,
                    "recall_label": recall_label,
                    "method": base_method,
                    "param_key": param_key,
                    "params": params,
                    "precision": m.get("precision", 0),
                    "recall": m.get("recall", 0),
                    "f1": m.get("f1", 0),
                    "mrr": m.get("mrr", 0),
                    "latency_avg": m.get("latency_avg", 0),
                    "latency_median": m.get("latency_median", 0),
                    "latency_p90": m.get("latency_p90", 0),
                    "latency_p95": m.get("latency_p95", 0),
                    "latency_p99": m.get("latency_p99", 0),
                    "server_latency_avg": m.get("server_latency_avg", 0),
                    "server_latency_median": m.get("server_latency_median", 0),
                    "server_latency_p90": m.get("server_latency_p90", 0),
                    "server_latency_p95": m.get("server_latency_p95", 0),
                    "server_latency_p99": m.get("server_latency_p99", 0),
                    "throughput": m.get("throughput", 0),
                    "num_samples": m.get("num_samples", 0),
                }
                docs.append(doc)

        try:
            for doc in docs:
                self._client.index(index=get_results_index(), body=doc)
            logger.info("Uploaded %d summary results to %s", len(docs), get_results_index())
            return True
        except Exception:
            logger.exception("Failed to upload metrics to Elasticsearch")
            return False


def _params_to_key(params: Dict[str, Any]) -> str:
    """Convert params dict to a string key for column naming."""
    parts = [f"{k}={v}" for k, v in sorted(params.items()) if k != "query_vector"]
    return "_".join(parts)


# Global uploader instance
_uploader: Optional[ResultsUploader] = None


def get_uploader() -> ResultsUploader:
    """Get or create the global results uploader."""
    global _uploader
    if _uploader is None:
        _uploader = ResultsUploader()
    return _uploader

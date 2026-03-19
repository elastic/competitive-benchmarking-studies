from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VectorSearchEngine(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._client = None

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def create_index(
        self,
        index_name: str,
        dataset_config: dict[str, Any],
        embedding_dimensions: int = 128,
    ) -> bool:
        pass

    def delete_index(self, index_name: str) -> bool:
        if self._client is None:
            logger.error("%s client not initialized", self.__class__.__name__)
            return False
        try:
            if not self.index_exists(index_name):
                logger.warning("Index '%s' does not exist", index_name)
                return False
            self._client.indices.delete(index=index_name)
            logger.info("Deleted index '%s'", index_name)
            return True
        except Exception:
            logger.exception("Failed to delete index '%s'", index_name)
            return False

    def index_exists(self, index_name: str) -> bool:
        if self._client is None:
            return False
        return bool(self._client.indices.exists(index=index_name))

    def ingest_data(
        self,
        index_name: str,
        documents: list[dict[str, Any]],
        mode: str = "bulk",
        chunk_size: int = 500,
    ) -> tuple[int, list[Any]]:
        if self._client is None:
            logger.error("%s client not initialized", self.__class__.__name__)
            return 0, [{"error": "Client not initialized"}]
        if not documents:
            logger.info("No documents provided for ingestion")
            return 0, []
        actions = [
            {
                "_op_type": "index",
                "_index": index_name,
                "_id": doc.get("_id"),
                "_source": doc.get("_source", doc),
            }
            for doc in documents
        ]
        successes = 0
        errors_list: list[Any] = []
        bulk_fn, parallel_bulk_fn = self._get_bulk_helpers()
        try:
            if mode == "bulk":
                logger.info("Starting bulk ingestion of %s documents", len(actions))
                progress = tqdm(
                    actions,
                    total=len(actions),
                    desc=f"Ingesting to '{index_name}'",
                    unit="doc",
                )
                successes, errors_list = bulk_fn(
                    client=self._client,
                    actions=progress,
                    chunk_size=chunk_size,
                    raise_on_error=False,
                    raise_on_exception=False,
                    max_retries=3,
                    request_timeout=300,
                )
            elif mode == "parallel_bulk":
                logger.info("Starting parallel bulk ingestion of %s documents", len(actions))
                for success, info in tqdm(
                    parallel_bulk_fn(
                        client=self._client,
                        actions=actions,
                        chunk_size=chunk_size,
                        thread_count=4,
                        raise_on_error=False,
                        raise_on_exception=False,
                    ),
                    total=len(actions),
                    desc=f"Parallel ingesting to '{index_name}'",
                    unit="doc",
                ):
                    if success:
                        successes += 1
                    else:
                        errors_list.append(info)
            logger.info(
                "Ingestion complete: %s successes, %s errors",
                successes,
                len(errors_list),
            )
            return successes, errors_list
        except Exception as exc:
            logger.exception("Error during ingestion")
            errors_list.append({"error": str(exc)})
            return successes, errors_list

    def ingest_streaming(
        self,
        index_name: str,
        action_generator: Iterator[dict[str, Any]],
        total: Optional[int] = None,
        chunk_size: int = 2000,
        thread_count: int = 4,
    ) -> tuple[int, int]:
        """
        Streaming ingestion using parallel_bulk with a generator.
        Memory-efficient for large datasets.

        Returns (success_count, error_count).
        """
        if self._client is None:
            logger.error("%s client not initialized", self.__class__.__name__)
            return 0, 1

        def _action_wrapper():
            for doc in action_generator:
                yield {
                    "_op_type": "index",
                    "_index": index_name,
                    "_id": doc.get("_id"),
                    "_source": doc.get("_source", doc),
                }

        _, parallel_bulk_fn = self._get_bulk_helpers()
        successes = 0
        errors = 0

        logger.info(
            "Starting streaming ingestion (chunk_size=%d, threads=%d)",
            chunk_size,
            thread_count,
        )

        try:
            for success, info in tqdm(
                parallel_bulk_fn(
                    client=self._client,
                    actions=_action_wrapper(),
                    chunk_size=chunk_size,
                    thread_count=thread_count,
                    raise_on_error=False,
                    raise_on_exception=False,
                ),
                total=total,
                desc=f"Ingesting to '{index_name}'",
                unit="doc",
            ):
                if success:
                    successes += 1
                else:
                    errors += 1
                    if errors <= 5:
                        logger.warning("Ingest error: %s", info)

            logger.info("Ingestion complete: %d successes, %d errors", successes, errors)
            return successes, errors
        except Exception as exc:
            logger.exception("Error during streaming ingestion")
            return successes, errors + 1

    @abstractmethod
    def _get_bulk_helpers(self) -> tuple:
        """Return (bulk, parallel_bulk) functions for the engine's client library."""
        pass

    @abstractmethod
    def vector_search(
        self,
        index_name: str,
        query_vector: list[float],
        vector_field: str,
        size: int,
        num_candidates: int,
        rescore: int,
        filter_query: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        pass

    def parse_search_response(
        self,
        response: dict[str, Any],
    ) -> tuple[list[str], Optional[float], Optional[int]]:
        hits = response.get("hits", {}).get("hits", [])
        doc_ids = [hit.get("_id") for hit in hits]
        return doc_ids, response.get("_client_latency_ms"), response.get("took")

    def _timed_search(self, index_name: str, query_dsl: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            logger.error("%s client not initialized", self.__class__.__name__)
            return {"hits": {"hits": []}, "took": None, "_client_latency_ms": None}
        try:
            start = time.time()
            response = self._client.search(index=index_name, body=query_dsl)
            result = dict(response)
            result["_client_latency_ms"] = (time.time() - start) * 1000
            return result
        except Exception:
            logger.exception("Search failed")
            return {"hits": {"hits": []}, "took": None, "_client_latency_ms": None}

    @abstractmethod
    def get_version(self) -> str:
        pass

    @abstractmethod
    def get_short_name(self) -> str:
        pass

    @abstractmethod
    def get_vector_type(self, index_name: str) -> str:
        pass

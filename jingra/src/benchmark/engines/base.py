from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional
from tqdm import tqdm

from ..queries import QueryLoader, QueryTemplate
from ..schemas import SchemaLoader, SchemaTemplate

logger = logging.getLogger(__name__)


class VectorSearchEngine(ABC):
    _query_loader: Optional[QueryLoader] = None
    _schema_loader: Optional[SchemaLoader] = None

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._client = None
        self._query_templates: dict[str, QueryTemplate] = {}
        self._schema_templates: dict[str, SchemaTemplate] = {}

    @classmethod
    def get_query_loader(cls) -> QueryLoader:
        """Get the shared query loader instance."""
        if cls._query_loader is None:
            cls._query_loader = QueryLoader()
        return cls._query_loader

    def load_query_template(self, query_name: str) -> Optional[QueryTemplate]:
        """Load and cache a query template for this engine."""
        if query_name in self._query_templates:
            return self._query_templates[query_name]

        engine_name = self.get_engine_name()
        loader = self.get_query_loader()
        template = loader.load(engine_name, query_name)
        if template:
            self._query_templates[query_name] = template
        return template

    def list_available_queries(self) -> list[str]:
        """List available query templates for this engine."""
        return self.get_query_loader().list_queries(self.get_engine_name())

    @classmethod
    def get_schema_loader(cls) -> SchemaLoader:
        """Get the shared schema loader instance."""
        if cls._schema_loader is None:
            cls._schema_loader = SchemaLoader()
        return cls._schema_loader

    def load_schema_template(self, schema_name: str) -> Optional[SchemaTemplate]:
        """Load and cache a schema template for this engine."""
        if schema_name in self._schema_templates:
            return self._schema_templates[schema_name]

        engine_name = self.get_engine_name()
        loader = self.get_schema_loader()
        template = loader.load(engine_name, schema_name)
        if template:
            self._schema_templates[schema_name] = template
        return template

    def list_available_schemas(self) -> list[str]:
        """List available schema templates for this engine."""
        return self.get_schema_loader().list_schemas(self.get_engine_name())

    def execute_query(
        self,
        index_name: str,
        query_name: str,
        **params: Any,
    ) -> dict[str, Any]:
        """Execute a query using a template."""
        template = self.load_query_template(query_name)
        if template is None:
            logger.error("Query template '%s' not found for engine '%s'", query_name, self.get_engine_name())
            return {"hits": {"hits": []}, "took": None, "_client_latency_ms": None}

        errors = template.validate_params(**params)
        if errors:
            logger.error("Query parameter validation failed: %s", errors)
            return {"hits": {"hits": []}, "took": None, "_client_latency_ms": None}

        query_dsl = template.render(**params)
        return self._timed_search(index_name, query_dsl)

    @abstractmethod
    def get_engine_name(self) -> str:
        """Return the engine name used for loading query templates."""
        pass

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def create_index(
        self,
        index_name: str,
        schema_name: str,
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
        chunk_size: int = 5000,
        thread_count: int = 8,
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

        # Disable refresh during bulk ingestion for better performance
        try:
            self._client.indices.put_settings(
                index=index_name,
                body={"index": {"refresh_interval": "-1"}},
            )
            logger.info("Disabled refresh_interval for faster ingestion")
        except Exception as e:
            logger.warning("Could not disable refresh_interval: %s", e)

        logger.info(
            "Starting streaming ingestion (chunk_size=%d, threads=%d)",
            chunk_size,
            thread_count,
        )

        log_interval = 100_000  # Log every 100K docs
        last_logged = 0
        start_time = time.time()

        try:
            for success, info in parallel_bulk_fn(
                client=self._client,
                actions=_action_wrapper(),
                chunk_size=chunk_size,
                thread_count=thread_count,
                queue_size=thread_count * 2,  # Buffer chunks for smoother throughput
                raise_on_error=False,
                raise_on_exception=False,
                request_timeout=120,  # 2 min timeout per bulk request
            ):
                if success:
                    successes += 1
                else:
                    errors += 1
                    if errors <= 5:
                        logger.warning("Ingest error: %s", info)

                processed = successes + errors
                if processed - last_logged >= log_interval:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    pct = (processed / total * 100) if total else 0
                    logger.info(
                        "Ingested %d / %d docs (%.1f%%) - %.0f docs/sec",
                        processed, total or 0, pct, rate
                    )
                    last_logged = processed

            elapsed = time.time() - start_time
            rate = successes / elapsed if elapsed > 0 else 0
            logger.info(
                "Ingestion complete: %d successes, %d errors in %.1fs (%.0f docs/sec)",
                successes, errors, elapsed, rate
            )
            return successes, errors
        except Exception as exc:
            logger.exception("Error during streaming ingestion")
            return successes, errors + 1
        finally:
            # Re-enable refresh and force a refresh to make docs searchable
            try:
                self._client.indices.put_settings(
                    index=index_name,
                    body={"index": {"refresh_interval": "1s"}},
                )
                self._client.indices.refresh(index=index_name)
                logger.info("Re-enabled refresh_interval and refreshed index")
            except Exception as e:
                logger.warning("Could not re-enable refresh_interval: %s", e)

    @abstractmethod
    def _get_bulk_helpers(self) -> tuple:
        """Return (bulk, parallel_bulk) functions for the engine's client library."""
        pass

    def search(
        self,
        index_name: str,
        query_name: str,
        **params: Any,
    ) -> dict[str, Any]:
        """Execute a search using the named query template with given parameters."""
        return self.execute_query(index_name=index_name, query_name=query_name, **params)

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

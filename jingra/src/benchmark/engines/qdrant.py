from __future__ import annotations
import logging
import os
import time
from typing import Any, Iterator, Optional
import requests

from qdrant_client import QdrantClient
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PointStruct,
    Range,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SearchParams,
    VectorParams,
)

from .base import VectorSearchEngine

logger = logging.getLogger(__name__)


class QdrantEngine(VectorSearchEngine):
    """Qdrant vector search engine implementation."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._client: Optional[QdrantClient] = None

    def connect(self) -> bool:
        url = os.getenv(self.config.get("url_env", "QDRANT_URL"))
        api_key = os.getenv(self.config.get("api_key_env", "QDRANT_API_KEY"))

        if not url:
            logger.error("Qdrant URL not set in environment")
            return False

        try:
            kwargs: dict[str, Any] = {
                "timeout": 120,
                "prefer_grpc": True,
            }
            if api_key:
                kwargs["api_key"] = api_key

            self._client = QdrantClient(url=url, **kwargs)

            # Test connection by getting collections
            self._client.get_collections()
            logger.info("Connected to Qdrant at %s", url)
            return True
        except Exception:
            logger.exception("Failed to connect to Qdrant")
            return False

    def create_index(self, index_name: str, schema_name: str) -> bool:
        """Create a Qdrant collection (index equivalent)."""
        if self._client is None:
            logger.error("Qdrant client not initialized")
            return False

        try:
            collections = self._client.get_collections().collections
            if any(c.name == index_name for c in collections):
                logger.warning("Collection '%s' already exists", index_name)
                return False

            # Get vector config from schema or use defaults
            template = self.load_schema_template(schema_name)
            if template:
                schema = template.render()
                vector_size = schema.get("vector_size", 128)
                distance_str = schema.get("distance", "cosine").lower()
                shard_number = schema.get("shard_number", 1)
                replication_factor = schema.get("replication_factor", 1)
                hnsw_config = schema.get("hnsw_config", {})
                quantization_config = schema.get("quantization_config", {})
            else:
                vector_size = self.config.get("vector_size", 128)
                distance_str = self.config.get("distance", "cosine").lower()
                shard_number = self.config.get("shard_number", 1)
                replication_factor = self.config.get("replication_factor", 1)
                hnsw_config = {}
                quantization_config = {}

            distance_map = {
                "cosine": Distance.COSINE,
                "euclidean": Distance.EUCLID,
                "dot": Distance.DOT,
                "l2": Distance.EUCLID,
            }
            distance = distance_map.get(distance_str, Distance.COSINE)

            # Build HNSW config if specified
            hnsw_config_obj = None
            if hnsw_config:
                hnsw_config_obj = HnswConfigDiff(
                    m=hnsw_config.get("m", 16),
                    ef_construct=hnsw_config.get("ef_construct", 100),
                )

            # Build quantization config if specified
            quantization_config_obj = None
            quantization_type = "none"
            if "binary" in quantization_config:
                binary_cfg = quantization_config["binary"]
                quantization_config_obj = BinaryQuantization(
                    binary=BinaryQuantizationConfig(
                        always_ram=binary_cfg.get("always_ram", True),
                    )
                )
                quantization_type = "binary"
            elif "scalar" in quantization_config:
                scalar_cfg = quantization_config["scalar"]
                quantization_config_obj = ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=scalar_cfg.get("quantile", 0.99),
                        always_ram=scalar_cfg.get("always_ram", True),
                    )
                )
                quantization_type = "scalar"

            self._client.create_collection(
                collection_name=index_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
                shard_number=shard_number,
                replication_factor=replication_factor,
                hnsw_config=hnsw_config_obj,
                quantization_config=quantization_config_obj,
            )
            logger.info(
                "Created Qdrant collection '%s' (size=%d, distance=%s, shards=%d, replicas=%d, quantization=%s)",
                index_name, vector_size, distance_str, shard_number, replication_factor - 1, quantization_type
            )
            return True
        except Exception:
            logger.exception("Failed to create collection '%s'", index_name)
            return False

    def delete_index(self, index_name: str) -> bool:
        """Delete a Qdrant collection."""
        if self._client is None:
            logger.error("Qdrant client not initialized")
            return False

        try:
            self._client.delete_collection(collection_name=index_name)
            logger.info("Deleted Qdrant collection '%s'", index_name)
            return True
        except Exception:
            logger.exception("Failed to delete collection '%s'", index_name)
            return False

    def index_exists(self, index_name: str) -> bool:
        """Check if a collection exists."""
        if self._client is None:
            return False
        try:
            collections = self._client.get_collections().collections
            return any(c.name == index_name for c in collections)
        except Exception:
            return False

    def ingest_streaming(
        self,
        index_name: str,
        action_generator: Iterator[dict[str, Any]],
        total: Optional[int] = None,
        chunk_size: int = 1000,
        thread_count: int = 8,
    ) -> tuple[int, int]:
        """Stream ingest documents into Qdrant collection."""
        if self._client is None:
            logger.error("Qdrant client not initialized")
            return 0, 1

        vector_field = self.config.get("vector_field", "search_catalog_embedding")
        successes = 0
        errors = 0
        batch: list[PointStruct] = []

        log_interval = 100_000
        last_logged = 0
        start_time = time.time()

        logger.info("Starting Qdrant streaming ingestion (batch_size=%d)", chunk_size)

        try:
            for doc in action_generator:
                source = doc.get("_source", doc)
                doc_id = doc.get("_id")
                vector = source.get(vector_field)

                if vector is None:
                    errors += 1
                    continue

                # Build payload (all fields except vector)
                payload = {k: v for k, v in source.items() if k != vector_field}

                # Qdrant requires point IDs to be integers or UUIDs
                # IDs come as integers or strings like "140997621" - parse to int
                if doc_id is not None:
                    point_id = int(doc_id)
                else:
                    point_id = successes + errors

                batch.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

                if len(batch) >= chunk_size:
                    try:
                        self._client.upsert(
                            collection_name=index_name,
                            points=batch,
                            wait=False,  # Don't wait for indexing
                        )
                        successes += len(batch)
                    except Exception as e:
                        errors += len(batch)
                        if errors <= 5:
                            logger.warning("Batch upsert error: %s", e)
                    batch = []

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

            # Final batch
            if batch:
                try:
                    self._client.upsert(
                        collection_name=index_name,
                        points=batch,
                        wait=True,  # Wait for final batch
                    )
                    successes += len(batch)
                except Exception as e:
                    errors += len(batch)
                    logger.warning("Final batch upsert error: %s", e)

            elapsed = time.time() - start_time
            rate = successes / elapsed if elapsed > 0 else 0
            logger.info(
                "Ingestion complete: %d successes, %d errors in %.1fs (%.0f docs/sec)",
                successes, errors, elapsed, rate
            )
            return successes, errors

        except Exception as exc:
            logger.exception("Error during Qdrant streaming ingestion")
            return successes, errors + 1

    def search(
        self,
        index_name: str,
        query_name: str,
        **params: Any,
    ) -> dict[str, Any]:
        """Execute a vector search on Qdrant."""
        if self._client is None:
            logger.error("Qdrant client not initialized")
            return {"results": [], "_client_latency_ms": None}

        query_vector = params.get("query_vector")
        if query_vector is None:
            logger.error("query_vector is required for Qdrant search")
            return {"results": [], "_client_latency_ms": None}

        limit = params.get("size", params.get("k", 10))
        num_candidates = params.get("num_candidates")
        meta_conditions = params.get("meta_conditions")

        # Build filter from meta_conditions (runtime conditions from query data)
        query_filter = self._build_filter(meta_conditions) if meta_conditions else None

        # If no runtime filter, try to get static filter from query template
        if query_filter is None:
            template = self.load_query_template(query_name)
            if template:
                query_def = template.render()
                static_filter = query_def.get("filter")
                if static_filter:
                    query_filter = self._build_filter_from_template(static_filter)

        # Build search params with hnsw_ef if num_candidates specified
        search_params = None
        if num_candidates:
            search_params = SearchParams(hnsw_ef=num_candidates)

        try:
            # Build request payload
            payload = {
                "query": query_vector,
                "limit": limit,
                "with_payload": False,
            }

            if query_filter:
                # Convert Filter object to dict
                filter_dict = query_filter.dict() if hasattr(query_filter, 'dict') else query_filter.model_dump()
                payload["filter"] = filter_dict

            if search_params:
                # Convert SearchParams to dict
                params_dict = search_params.dict() if hasattr(search_params, 'dict') else search_params.model_dump()
                payload["params"] = params_dict

            # Get Qdrant URL from client (parse from the connection string)
            qdrant_url = os.getenv(self.config.get("url_env", "QDRANT_URL"))
            api_key = os.getenv(self.config.get("api_key_env", "QDRANT_API_KEY"))

            # Build request URL
            url = f"{qdrant_url}/collections/{index_name}/points/query"

            # Set up headers
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["api-key"] = api_key

            # Make HTTP request
            start = time.time()
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            client_latency = (time.time() - start) * 1000

            response.raise_for_status()
            result = response.json()

            # Extract server latency from top level
            server_latency = None
            if 'time' in result and result['time'] is not None:
                server_latency = result['time'] * 1000  # Convert seconds to ms

            # Log once for debugging (with full response structure)
            if not hasattr(self, '_logged_time_extraction'):
                logger.info(f"Qdrant response keys: {result.keys()}")
                logger.info(f"Qdrant server latency: {server_latency}ms (from HTTP response.time={result.get('time')}s)")
                self._logged_time_extraction = True

            # Parse points from result - the structure is {"result": [...], "time": 0.01}
            from qdrant_client.http.models import ScoredPoint
            points_data = result.get('result', [])

            # Check if points_data is a list or if we need to go deeper
            if isinstance(points_data, dict):
                # Some responses might have {"result": {"points": [...]}}
                points_data = points_data.get('points', [])

            points = [ScoredPoint(**p) for p in points_data] if points_data else []

            return {
                "results": points,
                "_client_latency_ms": client_latency,
                "_server_latency_ms": server_latency,
            }
        except Exception:
            logger.exception("Qdrant search failed")
            return {"results": [], "_client_latency_ms": None, "_server_latency_ms": None}

    def _build_filter(self, meta_conditions: dict[str, Any]) -> Optional[Filter]:
        """Convert meta_conditions to Qdrant filter."""
        if not meta_conditions:
            return None

        must_conditions = []

        for field, condition in meta_conditions.items():
            if isinstance(condition, dict):
                # Range conditions
                if "gte" in condition or "lte" in condition or "gt" in condition or "lt" in condition:
                    must_conditions.append(
                        FieldCondition(
                            key=field,
                            range=Range(
                                gte=condition.get("gte"),
                                lte=condition.get("lte"),
                                gt=condition.get("gt"),
                                lt=condition.get("lt"),
                            ),
                        )
                    )
            else:
                # Exact match
                must_conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=condition))
                )

        if must_conditions:
            return Filter(must=must_conditions)
        return None

    def _build_filter_from_template(self, filter_def: dict[str, Any]) -> Optional[Filter]:
        """Convert query template filter definition to Qdrant filter."""
        if not filter_def:
            return None

        must_conditions = []

        # Handle "must" array format from template
        must_list = filter_def.get("must", [])
        for cond in must_list:
            key = cond.get("key")
            if not key:
                continue

            # Match condition
            match_cond = cond.get("match")
            if match_cond:
                value = match_cond.get("value")
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
                continue

            # Range condition
            range_cond = cond.get("range")
            if range_cond:
                must_conditions.append(
                    FieldCondition(
                        key=key,
                        range=Range(
                            gte=range_cond.get("gte"),
                            lte=range_cond.get("lte"),
                            gt=range_cond.get("gt"),
                            lt=range_cond.get("lt"),
                        ),
                    )
                )

        if must_conditions:
            return Filter(must=must_conditions)
        return None

    def parse_search_response(
        self,
        response: dict[str, Any],
    ) -> tuple[list[str], Optional[float], Optional[int]]:
        """Parse Qdrant search response."""
        results = response.get("results", [])
        # Convert integer IDs back to strings for recall comparison
        doc_ids = [str(hit.id) for hit in results]
        client_latency = response.get("_client_latency_ms")
        server_latency = response.get("_server_latency_ms")
        return doc_ids, client_latency, server_latency

    def _get_bulk_helpers(self) -> tuple:
        """Not used for Qdrant - returns None."""
        return None, None

    def get_engine_name(self) -> str:
        return "qdrant"

    def get_version(self) -> str:
        if self._client is None:
            return "unknown"
        try:
            # Get cluster info
            info = self._client.get_collections()
            # Qdrant client doesn't expose server version easily
            # Return client version instead
            import qdrant_client
            return qdrant_client.__version__
        except Exception:
            return "unknown"

    def get_short_name(self) -> str:
        return "qd"

    def get_vector_type(self, index_name: str) -> str:
        if self._client is None:
            return "unknown"
        try:
            info = self._client.get_collection(collection_name=index_name)
            # Check quantization config
            quant_type = "hnsw"
            if info.config.quantization_config:
                quant_cfg = info.config.quantization_config
                if hasattr(quant_cfg, "binary") and quant_cfg.binary:
                    quant_type = "binary_hnsw"
                elif hasattr(quant_cfg, "scalar") and quant_cfg.scalar:
                    quant_type = "scalar_hnsw"
            return quant_type
        except Exception:
            return "unknown"

from __future__ import annotations
import logging
import os
from typing import Any, Optional
import urllib3
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk, parallel_bulk
from .base import VectorSearchEngine
from .condition_parser import OpenSearchConditionParser

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("opensearch").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class OpenSearchEngine(VectorSearchEngine):
    condition_parser = OpenSearchConditionParser()

    def connect(self) -> bool:
        load_dotenv(override=True)
        url = os.getenv(self.config.get("url_env", "OPENSEARCH_URL"))
        user = os.getenv(self.config.get("user_env", "OPENSEARCH_USER"))
        password = os.getenv(self.config.get("password_env", "OPENSEARCH_PASSWORD"))
        if not url:
            logger.error("OpenSearch URL not set in environment")
            return False
        try:
            kwargs: dict[str, Any] = {
                "connection_class": RequestsHttpConnection,
                "use_ssl": url.startswith("https"),
                "verify_certs": False,
                "max_retries": 5,
                "retry_on_timeout": True,
                "timeout": 30,
                "http_compress": True,
                "pool_maxsize": 100,
            }
            if user and password:
                kwargs["http_auth"] = (user, password)
            self._client = OpenSearch(hosts=[url], **kwargs)
            if not self._client.ping():
                raise ConnectionError("Failed to ping OpenSearch cluster")
            logger.info("Connected to OpenSearch at %s", url)
            return True
        except Exception:
            logger.exception("Failed to connect to OpenSearch")
            return False

    def create_index(
        self,
        index_name: str,
        dataset_config: dict[str, Any],
        embedding_dimensions: int = 128,
    ) -> bool:
        if self._client is None:
            logger.error("OpenSearch client not initialized")
            return False
        try:
            if self._client.indices.exists(index=index_name):
                logger.warning("Index '%s' already exists", index_name)
                return False

            vector_field_name = dataset_config.get("data_mapping", {}).get("vector_field", "vector")

            # Get distance metric from dataset config
            distance = dataset_config.get("distance", "cosine")
            space_type_map = {
                "cosine": "cosinesimil",
                "dot_product": "innerproduct",
                "l2": "l2",
            }
            space_type = space_type_map.get(distance, "innerproduct")

            properties: dict[str, Any] = {
                vector_field_name: {
                    "type": "knn_vector",
                    "dimension": embedding_dimensions,
                    "space_type": space_type,
                    "data_type": "float",
                    "compression_level": "32x",
                    "mode": "in_memory",
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "parameters": {"ef_construction": 100, "m": 16},
                    },
                },
            }

            # Add schema-defined fields
            schema = dataset_config.get("schema", {})
            for field_name, field_type in schema.items():
                properties[field_name] = self._map_schema_type(field_type)

            self._client.indices.create(
                index=index_name,
                body={
                    "mappings": {"properties": properties},
                    "settings": {
                        "number_of_shards": "3",
                        "number_of_replicas": "1",
                        "index.knn": True,
                    },
                },
            )
            logger.info("Created OpenSearch index '%s'", index_name)
            return True
        except Exception:
            logger.exception("Failed to create index '%s'", index_name)
            return False

    def _map_schema_type(self, field_type: str) -> dict[str, Any]:
        """Map schema type strings to OpenSearch field mappings."""
        type_map = {
            "bool": {"type": "boolean"},
            "boolean": {"type": "boolean"},
            "int": {"type": "integer"},
            "integer": {"type": "integer"},
            "long": {"type": "long"},
            "float": {"type": "float"},
            "double": {"type": "double"},
            "keyword": {"type": "keyword"},
            "text": {"type": "text"},
            "date": {"type": "date"},
            "geo_point": {"type": "geo_point"},
        }
        return type_map.get(field_type, {"type": "keyword"})

    def _get_bulk_helpers(self) -> tuple:
        return bulk, parallel_bulk

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
        if self._client is None:
            logger.error("OpenSearch client not initialized")
            return {"hits": {"hits": []}}
        knn_inner: dict[str, Any] = {
            "vector": query_vector,
            "k": num_candidates,
            "method_parameters": {"ef_search": num_candidates},
            "rescore": {"oversample_factor": rescore},
        }
        if filter_query is not None:
            knn_inner["filter"] = filter_query
        query_dsl = {
            "query": {"knn": {vector_field: knn_inner}},
            "size": size,
            "_source": {"excludes": [vector_field]},
        }
        return self._timed_search(index_name, query_dsl)

    def get_version(self) -> str:
        if self._client is None:
            return "unknown"
        try:
            return self._client.info().get("version", {}).get("number", "unknown")
        except Exception:
            return "unknown"

    def get_short_name(self) -> str:
        return "os"

    def get_vector_type(self, index_name: str) -> str:
        if self._client is None:
            return "unknown"
        try:
            mappings = self._client.indices.get_mapping(index=index_name)
            # Handle both direct index name and aliased/shrunk indices
            if index_name in mappings:
                props = mappings[index_name].get("mappings", {}).get("properties", {})
            elif mappings:
                # Take the first (and likely only) index in the response
                first_index = next(iter(mappings.values()))
                props = first_index.get("mappings", {}).get("properties", {})
            else:
                return "unknown"
            for field_config in props.values():
                if field_config.get("type") != "knn_vector":
                    continue
                method = field_config.get("method", {})
                engine = method.get("engine", "nmslib")
                encoder = method.get("parameters", {}).get("encoder", {})
                if (
                    encoder.get("name") == "sq"
                    and encoder.get("parameters", {}).get("type") == "fp16"
                ):
                    return f"{engine}_fp16"
                return engine
            return "unknown"
        except Exception:
            return "unknown"

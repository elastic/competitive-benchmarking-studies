from __future__ import annotations
import logging
import os
from typing import Any, Optional
import urllib3
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, parallel_bulk
from .base import VectorSearchEngine
from .condition_parser import ElasticsearchConditionParser

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class ElasticsearchEngine(VectorSearchEngine):
    condition_parser = ElasticsearchConditionParser()

    def connect(self) -> bool:
        load_dotenv(override=True)
        url = os.getenv(self.config.get("url_env", "ELASTIC_URL"))
        user = os.getenv(self.config.get("user_env", "ELASTIC_USER"))
        password = os.getenv(self.config.get("password_env", "ELASTIC_PASSWORD"))
        if not url:
            logger.error("Elasticsearch URL not set in environment")
            return False
        try:
            kwargs: dict[str, Any] = {
                "verify_certs": False,
                "max_retries": 5,
                "retry_on_timeout": True,
                "http_compress": True,
                "connections_per_node": 100,
            }
            if user and password:
                kwargs["http_auth"] = (user, password)
            self._client = Elasticsearch(hosts=[url], **kwargs)
            if not self._client.ping():
                raise ConnectionError("Failed to ping Elasticsearch cluster")
            logger.info("Connected to Elasticsearch at %s", url)
            return True
        except Exception:
            logger.exception("Failed to connect to Elasticsearch")
            return False

    def create_index(
        self,
        index_name: str,
        dataset_config: dict[str, Any],
        embedding_dimensions: int = 128,
    ) -> bool:
        if self._client is None:
            logger.error("Elasticsearch client not initialized")
            return False
        try:
            if self._client.indices.exists(index=index_name):
                logger.warning("Index '%s' already exists", index_name)
                return False

            vector_field_name = dataset_config.get("data_mapping", {}).get("vector_field", "vector")

            # Get distance metric from dataset config
            distance = dataset_config.get("distance", "cosine")
            similarity_map = {
                "cosine": "cosine",
                "dot_product": "dot_product",
                "l2": "l2_norm",
            }
            similarity = similarity_map.get(distance, "dot_product")

            properties: dict[str, Any] = {
                vector_field_name: {
                    "type": "dense_vector",
                    "element_type": "float",
                    "dims": embedding_dimensions,
                    "index": True,
                    "similarity": similarity,
                    "index_options": {
                        "type": "bbq_hnsw",
                        "ef_construction": 100,
                        "m": 16,
                        "rescore_vector": {"oversample": 3.0},
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
                    },
                },
            )
            logger.info("Created Elasticsearch index '%s'", index_name)
            return True
        except Exception:
            logger.exception("Failed to create index '%s'", index_name)
            return False

    def _map_schema_type(self, field_type: str) -> dict[str, Any]:
        """Map schema type strings to Elasticsearch field mappings."""
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
            logger.error("Elasticsearch client not initialized")
            return {"hits": {"hits": []}}
        knn_clause: dict[str, Any] = {
            "field": vector_field,
            "query_vector": query_vector,
            "k": num_candidates,
            "num_candidates": num_candidates,
            "rescore_vector": {"oversample": rescore},
        }
        if filter_query is not None:
            knn_clause["filter"] = filter_query
        query_dsl = {
            "query": {"knn": knn_clause},
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
        return "es"

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
                if field_config.get("type") == "dense_vector":
                    # Check index_options.type first, fall back to checking if indexed
                    index_opts = field_config.get("index_options", {})
                    if index_opts:
                        return index_opts.get("type", "hnsw")
                    # If no index_options but index=true, it's using default hnsw
                    if field_config.get("index", True):
                        return "hnsw"
                    return "flat"
            return "unknown"
        except Exception as e:
            logger.debug(f"Failed to get vector type for {index_name}: {e}")
            return "unknown"

from __future__ import annotations
import logging
import os
from typing import Any, Optional
import urllib3
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
                "sniff_on_start": False,
                "sniff_on_node_failure": False,
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
        schema_name: str,
    ) -> bool:
        if self._client is None:
            logger.error("Elasticsearch client not initialized")
            return False
        try:
            if self._client.indices.exists(index=index_name):
                logger.warning("Index '%s' already exists", index_name)
                return False

            template = self.load_schema_template(schema_name)
            if template is None:
                logger.error("Schema template '%s' not found", schema_name)
                return False

            body = template.render()
            self._client.indices.create(index=index_name, body=body)
            logger.info("Created Elasticsearch index '%s' with schema '%s'", index_name, schema_name)
            return True
        except Exception:
            logger.exception("Failed to create index '%s'", index_name)
            return False

    def _get_bulk_helpers(self) -> tuple:
        return bulk, parallel_bulk

    def get_engine_name(self) -> str:
        return "elasticsearch"

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

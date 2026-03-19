from __future__ import annotations
import logging
import os
from typing import Any, Optional
import urllib3
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
        schema_name: str,
    ) -> bool:
        if self._client is None:
            logger.error("OpenSearch client not initialized")
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
            logger.info("Created OpenSearch index '%s' with schema '%s'", index_name, schema_name)
            return True
        except Exception:
            logger.exception("Failed to create index '%s'", index_name)
            return False

    def _get_bulk_helpers(self) -> tuple:
        return bulk, parallel_bulk

    def get_engine_name(self) -> str:
        return "opensearch"

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

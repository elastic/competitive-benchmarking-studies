from __future__ import annotations
import copy
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

QUERIES_DIR = Path(__file__).parent.parent.parent.parent / "queries"


class QueryTemplate:
    """Represents a loaded query template with metadata."""

    def __init__(
        self,
        name: str,
        description: str,
        template: dict[str, Any],
        parameters: dict[str, Any],
        engine: str,
    ):
        self.name = name
        self.description = description
        self.template = template
        self.parameters = parameters
        self.engine = engine

    def render(self, **kwargs: Any) -> dict[str, Any]:
        """
        Render the template with provided values.

        Handles special cases:
        - Removes filter clause if filter is None
        - Substitutes placeholders with actual values
        - For OpenSearch, handles dynamic field name in knn clause
        """
        result = self._substitute(copy.deepcopy(self.template), kwargs)
        result = self._clean_nulls(result)
        return result

    def _substitute(self, obj: Any, values: dict[str, Any]) -> Any:
        """Recursively substitute placeholders in the template."""
        if isinstance(obj, str):
            # Check if this is a placeholder string like "{{field_name}}"
            if obj.startswith("{{") and obj.endswith("}}"):
                key = obj[2:-2]
                return values.get(key, obj)
            return obj
        elif isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                # Handle dynamic keys (e.g., "{{vector_field}}" as a key)
                if k.startswith("{{") and k.endswith("}}"):
                    key_name = k[2:-2]
                    new_key = values.get(key_name, k)
                    result[new_key] = self._substitute(v, values)
                else:
                    result[k] = self._substitute(v, values)
            return result
        elif isinstance(obj, list):
            return [self._substitute(item, values) for item in obj]
        return obj

    def _clean_nulls(self, obj: Any) -> Any:
        """Remove keys with None values (e.g., filter when not provided)."""
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                cleaned_v = self._clean_nulls(v)
                # Remove None values and empty dicts
                if cleaned_v is not None and cleaned_v != {}:
                    cleaned[k] = cleaned_v
            return cleaned
        elif isinstance(obj, list):
            return [self._clean_nulls(item) for item in obj if item is not None]
        return obj

    def validate_params(self, **kwargs: Any) -> list[str]:
        """Validate that required parameters are provided."""
        errors = []
        for param_name, param_config in self.parameters.items():
            if param_config.get("required", False) and param_name not in kwargs:
                errors.append(f"Missing required parameter: {param_name}")
        return errors


class QueryLoader:
    """Loads and manages query templates for search engines."""

    def __init__(self, queries_dir: Optional[Path] = None):
        self.queries_dir = queries_dir or QUERIES_DIR
        self._cache: dict[str, dict[str, QueryTemplate]] = {}

    def list_queries(self, engine: str) -> list[str]:
        """List available query templates for an engine."""
        engine_dir = self.queries_dir / engine
        if not engine_dir.exists():
            logger.warning("No queries directory found for engine: %s", engine)
            return []
        return [f.stem for f in engine_dir.glob("*.json")]

    def load(self, engine: str, query_name: str) -> Optional[QueryTemplate]:
        """Load a query template by engine and name."""
        cache_key = f"{engine}/{query_name}"
        if cache_key in self._cache.get(engine, {}):
            return self._cache[engine][cache_key]

        query_path = self.queries_dir / engine / f"{query_name}.json"
        if not query_path.exists():
            logger.error("Query template not found: %s", query_path)
            return None

        try:
            with open(query_path) as f:
                data = json.load(f)

            template = QueryTemplate(
                name=data.get("name", query_name),
                description=data.get("description", ""),
                template=data["template"],
                parameters=data.get("parameters", {}),
                engine=engine,
            )

            # Cache the template
            if engine not in self._cache:
                self._cache[engine] = {}
            self._cache[engine][cache_key] = template

            return template
        except Exception:
            logger.exception("Failed to load query template: %s", query_path)
            return None

    def load_all(self, engine: str) -> dict[str, QueryTemplate]:
        """Load all query templates for an engine."""
        templates = {}
        for query_name in self.list_queries(engine):
            template = self.load(engine, query_name)
            if template:
                templates[query_name] = template
        return templates

    def get_queries_dir(self) -> Path:
        """Return the queries directory path."""
        return self.queries_dir

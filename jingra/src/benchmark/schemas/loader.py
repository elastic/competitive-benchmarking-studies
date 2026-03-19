from __future__ import annotations
import copy
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SCHEMAS_DIR = Path(__file__).parent.parent.parent.parent / "schemas"


class SchemaTemplate:
    """Represents a loaded schema template."""

    def __init__(
        self,
        name: str,
        description: str,
        template: dict[str, Any],
        engine: str,
    ):
        self.name = name
        self.description = description
        self.template = template
        self.engine = engine

    def render(self) -> dict[str, Any]:
        """Return a copy of the template."""
        return copy.deepcopy(self.template)


class SchemaLoader:
    """Loads and manages schema templates for search engines."""

    def __init__(self, schemas_dir: Optional[Path] = None):
        self.schemas_dir = schemas_dir or SCHEMAS_DIR
        self._cache: dict[str, dict[str, SchemaTemplate]] = {}

    def list_schemas(self, engine: str) -> list[str]:
        """List available schema templates for an engine."""
        engine_dir = self.schemas_dir / engine
        if not engine_dir.exists():
            logger.warning("No schemas directory found for engine: %s", engine)
            return []
        return [f.stem for f in engine_dir.glob("*.json")]

    def load(self, engine: str, schema_name: str) -> Optional[SchemaTemplate]:
        """Load a schema template by engine and name."""
        cache_key = f"{engine}/{schema_name}"
        if cache_key in self._cache.get(engine, {}):
            return self._cache[engine][cache_key]

        schema_path = self.schemas_dir / engine / f"{schema_name}.json"
        if not schema_path.exists():
            logger.error("Schema template not found: %s", schema_path)
            return None

        try:
            with open(schema_path) as f:
                data = json.load(f)

            template = SchemaTemplate(
                name=data.get("name", schema_name),
                description=data.get("description", ""),
                template=data["template"],
                engine=engine,
            )

            if engine not in self._cache:
                self._cache[engine] = {}
            self._cache[engine][cache_key] = template

            return template
        except Exception:
            logger.exception("Failed to load schema template: %s", schema_path)
            return None

    def get_schemas_dir(self) -> Path:
        """Return the schemas directory path."""
        return self.schemas_dir

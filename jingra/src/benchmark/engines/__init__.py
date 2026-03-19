from .base import VectorSearchEngine
from .elasticsearch import ElasticsearchEngine
from .opensearch import OpenSearchEngine
from .qdrant import QdrantEngine

ENGINES = {
    "elasticsearch": ElasticsearchEngine,
    "opensearch": OpenSearchEngine,
    "qdrant": QdrantEngine,
}


def get_engine(engine_name: str, config: dict) -> VectorSearchEngine:
    engine_name = engine_name.lower()
    if engine_name not in ENGINES:
        raise ValueError(f"Unsupported engine: {engine_name}. Supported: {list(ENGINES.keys())}")
    return ENGINES[engine_name](config.get(engine_name, {}))

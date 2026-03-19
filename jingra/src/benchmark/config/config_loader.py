import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    warmup_workers: int = 8
    measurement_workers: int = 8
    warmup_rounds: int = 3
    measurement_rounds: int = 3
    retrieval_methods: List[str] = field(default_factory=lambda: ["dense_retrieval_response"])


@dataclass
class OutputConfig:
    results_dir: str = "results"


@dataclass
class ParquetPathConfig:
    data_path: str = ""
    queries_path: str = ""
    data_url_env: str = ""
    queries_url_env: str = ""


@dataclass
class ParquetDataMappingConfig:
    id_field: str = ""
    vector_field: str = "vector"


@dataclass
class ParquetQueriesMappingConfig:
    query_vector_field: str = "vector"
    ground_truth_field: str = "closest_ids"
    conditions_field: str = "conditions"


@dataclass
class DatasetConfig:
    index_name: str = ""
    vector_size: int = 128
    distance: str = "cosine"
    schema_name: str = ""
    query_name: str = ""
    path: Optional[ParquetPathConfig] = None
    data_mapping: Optional[ParquetDataMappingConfig] = None
    queries_mapping: Optional[ParquetQueriesMappingConfig] = None
    schema: Dict[str, str] = field(default_factory=dict)
    param_groups: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_name": self.index_name,
            "type": "parquet",
            "vector_size": self.vector_size,
            "distance": self.distance,
            "schema_name": self.schema_name,
            "query_name": self.query_name,
            "path": {
                "data_path": self.path.data_path if self.path else "",
                "queries_path": self.path.queries_path if self.path else "",
                "data_url_env": self.path.data_url_env if self.path else "",
                "queries_url_env": self.path.queries_url_env if self.path else "",
            },
            "data_mapping": {
                "id_field": self.data_mapping.id_field if self.data_mapping else "",
                "vector_field": self.data_mapping.vector_field if self.data_mapping else "vector",
            },
            "queries_mapping": {
                "query_vector_field": self.queries_mapping.query_vector_field if self.queries_mapping else "vector",
                "ground_truth_field": self.queries_mapping.ground_truth_field if self.queries_mapping else "closest_ids",
                "conditions_field": self.queries_mapping.conditions_field if self.queries_mapping else "conditions",
            },
            "schema": self.schema,
            "param_groups": self.param_groups,
        }

    def is_parquet(self) -> bool:
        return True


@dataclass
class AppConfig:
    engine: str = "elasticsearch"
    elasticsearch: Dict[str, str] = field(default_factory=dict)
    opensearch: Dict[str, str] = field(default_factory=dict)
    qdrant: Dict[str, str] = field(default_factory=dict)
    dataset: str = "ecommerce-search-128"
    datasets: Dict[str, DatasetConfig] = field(default_factory=dict)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def get_current_dataset(self) -> DatasetConfig:
        if self.dataset not in self.datasets:
            raise ValueError(
                f"Dataset '{self.dataset}' not found. Available: {list(self.datasets.keys())}"
            )
        return self.datasets[self.dataset]

    def get_engine_config(self) -> Dict[str, str]:
        engine_configs = {
            "elasticsearch": self.elasticsearch,
            "opensearch": self.opensearch,
            "qdrant": self.qdrant,
        }
        if self.engine not in engine_configs:
            raise ValueError(f"Unknown engine: {self.engine}")
        return engine_configs[self.engine]


def _parse_parquet_path_config(path_dict: Optional[Dict[str, Any]]) -> Optional[ParquetPathConfig]:
    if not path_dict:
        return None
    return ParquetPathConfig(
        data_path=path_dict.get("data_path", ""),
        queries_path=path_dict.get("queries_path", ""),
        data_url_env=path_dict.get("data_url_env", ""),
        queries_url_env=path_dict.get("queries_url_env", ""),
    )


def _parse_parquet_data_mapping(mapping_dict: Optional[Dict[str, Any]]) -> Optional[ParquetDataMappingConfig]:
    if not mapping_dict:
        return None
    return ParquetDataMappingConfig(
        id_field=mapping_dict.get("id_field", ""),
        vector_field=mapping_dict.get("vector_field", "vector"),
    )


def _parse_parquet_queries_mapping(mapping_dict: Optional[Dict[str, Any]]) -> Optional[ParquetQueriesMappingConfig]:
    if not mapping_dict:
        return None
    return ParquetQueriesMappingConfig(
        query_vector_field=mapping_dict.get("query_vector_field", "vector"),
        ground_truth_field=mapping_dict.get("ground_truth_field", "closest_ids"),
        conditions_field=mapping_dict.get("conditions_field", "conditions"),
    )


def _parse_dataset_config(name: str, ds_dict: Dict[str, Any]) -> DatasetConfig:
    return DatasetConfig(
        index_name=ds_dict.get("index_name", name),
        vector_size=ds_dict.get("vector_size", 128),
        distance=ds_dict.get("distance", "cosine"),
        schema_name=ds_dict.get("schema_name", ""),
        query_name=ds_dict.get("query_name", ""),
        path=_parse_parquet_path_config(ds_dict.get("path")),
        data_mapping=_parse_parquet_data_mapping(ds_dict.get("data_mapping")),
        queries_mapping=_parse_parquet_queries_mapping(ds_dict.get("queries_mapping")),
        schema=ds_dict.get("schema", {}),
        param_groups=ds_dict.get("param_groups", {}),
    )


def load_config(config_path: Optional[str] = None) -> AppConfig:
    if config_path is None:
        # Look for jingra.yaml at project root first, fall back to config.yaml
        project_root = Path(__file__).parent.parent.parent.parent
        jingra_path = project_root / "jingra.yaml"
        if jingra_path.exists():
            config_path = jingra_path
        else:
            config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    datasets = {
        name: _parse_dataset_config(name, ds) for name, ds in raw.get("datasets", {}).items()
    }

    eval_cfg = raw.get("evaluation", {})
    evaluation = EvaluationConfig(
        warmup_workers=eval_cfg.get("warmup_workers", 8),
        measurement_workers=eval_cfg.get("measurement_workers", 8),
        warmup_rounds=eval_cfg.get("warmup_rounds", 3),
        measurement_rounds=eval_cfg.get("measurement_rounds", 3),
        retrieval_methods=eval_cfg.get("retrieval_methods", ["dense_retrieval_response"]),
    )

    output = OutputConfig(results_dir=raw.get("output", {}).get("results_dir", "results"))

    config = AppConfig(
        engine=raw.get("engine", "elasticsearch"),
        elasticsearch=raw.get("elasticsearch", {}),
        opensearch=raw.get("opensearch", {}),
        qdrant=raw.get("qdrant", {}),
        dataset=raw.get("dataset", "ecommerce-search-128"),
        datasets=datasets,
        evaluation=evaluation,
        output=output,
    )

    logger.info(f"Loaded configuration: engine={config.engine}, dataset={config.dataset}")
    return config

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Record:
    """Represents a single record from a dataset for indexing."""

    id: Any
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class Query:
    """Represents a query with optional filter conditions and ground truth."""

    vector: List[float]
    meta_conditions: Optional[Dict[str, Any]] = None
    expected_result: Optional[List[Any]] = None

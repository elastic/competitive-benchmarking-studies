from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class FilterType(str, Enum):
    FULL_MATCH = "match"
    RANGE = "range"
    GEO = "geo"


FieldValue = Union[str, int, float, bool]
MetaConditions = Dict[str, List[Any]]


class BaseConditionParser(ABC):
    """Base class for parsing filter conditions into engine-specific DSL."""

    def parse(self, meta_conditions: Optional[MetaConditions]) -> Optional[Any]:
        """
        Parse meta conditions into engine-specific filter format.

        The internal representation has the following structure:
        {
            "and": [
                {"field_name": {"match": {"value": 80}}},
                {"field_name": {"range": {"gte": 10, "lte": 100}}}
            ]
        }

        There is always an operator ("and" / "or") and a list of operands.
        """
        if meta_conditions is None or len(meta_conditions) == 0:
            return None
        return self.build_condition(
            and_subfilters=self._create_condition_subfilters(meta_conditions.get("and")),
            or_subfilters=self._create_condition_subfilters(meta_conditions.get("or")),
        )

    @abstractmethod
    def build_condition(
        self,
        and_subfilters: Optional[List[Any]],
        or_subfilters: Optional[List[Any]],
    ) -> Optional[Any]:
        """Build the final condition from AND/OR subfilters."""
        pass

    def _create_condition_subfilters(self, entries: Optional[List[Any]]) -> Optional[List[Any]]:
        """Convert a list of condition entries into engine-specific filters."""
        if entries is None:
            return None

        output_filters = []
        for entry in entries:
            for field_name, field_filters in entry.items():
                for condition_type, value in field_filters.items():
                    condition = self._build_filter(field_name, FilterType(condition_type), value)
                    output_filters.append(condition)
        return output_filters

    def _build_filter(
        self,
        field_name: str,
        filter_type: FilterType,
        criteria: Dict[str, Any],
    ) -> Any:
        """Build a single filter based on its type."""
        if filter_type == FilterType.FULL_MATCH:
            return self.build_exact_match_filter(field_name, value=criteria.get("value"))
        if filter_type == FilterType.RANGE:
            return self.build_range_filter(
                field_name,
                lt=criteria.get("lt"),
                gt=criteria.get("gt"),
                lte=criteria.get("lte"),
                gte=criteria.get("gte"),
            )
        if filter_type == FilterType.GEO:
            return self.build_geo_filter(
                field_name,
                lon=criteria.get("lon"),
                lat=criteria.get("lat"),
                radius=criteria.get("radius"),
            )
        raise NotImplementedError(f"Filter type {filter_type} not implemented")

    @abstractmethod
    def build_exact_match_filter(self, field_name: str, value: FieldValue) -> Any:
        """Build an exact match filter."""
        pass

    @abstractmethod
    def build_range_filter(
        self,
        field_name: str,
        lt: Optional[FieldValue],
        gt: Optional[FieldValue],
        lte: Optional[FieldValue],
        gte: Optional[FieldValue],
    ) -> Any:
        """Build a range filter."""
        pass

    @abstractmethod
    def build_geo_filter(
        self,
        field_name: str,
        lat: float,
        lon: float,
        radius: float,
    ) -> Any:
        """Build a geo distance filter."""
        pass


class ElasticsearchConditionParser(BaseConditionParser):
    """Condition parser for Elasticsearch DSL."""

    def build_condition(
        self,
        and_subfilters: Optional[List[Any]],
        or_subfilters: Optional[List[Any]],
    ) -> Optional[Any]:
        bool_clause: Dict[str, Any] = {}
        if and_subfilters:
            bool_clause["must"] = and_subfilters
        if or_subfilters:
            bool_clause["should"] = or_subfilters
            if not and_subfilters:
                bool_clause["minimum_should_match"] = 1
        return {"bool": bool_clause} if bool_clause else None

    def build_exact_match_filter(self, field_name: str, value: FieldValue) -> Any:
        # Use term query for boolean and keyword fields, match for text
        if isinstance(value, bool):
            return {"term": {field_name: value}}
        return {"match": {field_name: value}}

    def build_range_filter(
        self,
        field_name: str,
        lt: Optional[FieldValue],
        gt: Optional[FieldValue],
        lte: Optional[FieldValue],
        gte: Optional[FieldValue],
    ) -> Any:
        range_clause: Dict[str, Any] = {}
        if lt is not None:
            range_clause["lt"] = lt
        if gt is not None:
            range_clause["gt"] = gt
        if lte is not None:
            range_clause["lte"] = lte
        if gte is not None:
            range_clause["gte"] = gte
        return {"range": {field_name: range_clause}}

    def build_geo_filter(
        self,
        field_name: str,
        lat: float,
        lon: float,
        radius: float,
    ) -> Any:
        return {
            "geo_distance": {
                "distance": f"{radius}m",
                field_name: {"lat": lat, "lon": lon},
            }
        }


class OpenSearchConditionParser(BaseConditionParser):
    """Condition parser for OpenSearch DSL (same as Elasticsearch)."""

    def build_condition(
        self,
        and_subfilters: Optional[List[Any]],
        or_subfilters: Optional[List[Any]],
    ) -> Optional[Any]:
        bool_clause: Dict[str, Any] = {}
        if and_subfilters:
            bool_clause["must"] = and_subfilters
        if or_subfilters:
            bool_clause["should"] = or_subfilters
            if not and_subfilters:
                bool_clause["minimum_should_match"] = 1
        return {"bool": bool_clause} if bool_clause else None

    def build_exact_match_filter(self, field_name: str, value: FieldValue) -> Any:
        if isinstance(value, bool):
            return {"term": {field_name: value}}
        return {"match": {field_name: value}}

    def build_range_filter(
        self,
        field_name: str,
        lt: Optional[FieldValue],
        gt: Optional[FieldValue],
        lte: Optional[FieldValue],
        gte: Optional[FieldValue],
    ) -> Any:
        range_clause: Dict[str, Any] = {}
        if lt is not None:
            range_clause["lt"] = lt
        if gt is not None:
            range_clause["gt"] = gt
        if lte is not None:
            range_clause["lte"] = lte
        if gte is not None:
            range_clause["gte"] = gte
        return {"range": {field_name: range_clause}}

    def build_geo_filter(
        self,
        field_name: str,
        lat: float,
        lon: float,
        radius: float,
    ) -> Any:
        return {
            "geo_distance": {
                "distance": f"{radius}m",
                field_name: {"lat": lat, "lon": lon},
            }
        }

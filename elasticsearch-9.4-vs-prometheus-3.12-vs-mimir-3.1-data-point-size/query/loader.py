import yaml
from jinja2 import Environment, StrictUndefined

from .models import Defaults, QueryDefinition, Target, TargetGroup, to_iso


class QueryLoader:
    """Loads and renders query definitions from a YAML file."""

    def __init__(self) -> None:
        self._env = Environment(undefined=StrictUndefined)
        self._env.filters["to_iso"] = to_iso

    def _render(self, template_str: str, ctx: dict) -> str:
        return self._env.from_string(str(template_str)).render(**ctx)

    @staticmethod
    def _coerce(value: str) -> int | float | str:
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value

    def load(self, path: str, ctx: dict) -> tuple[Defaults, dict[str, TargetGroup]]:
        with open(path) as f:
            data = yaml.safe_load(f)

        enriched = dict(ctx)
        for key, val in data.get("queries_runtime_params", {}).items():
            enriched[key] = self._coerce(self._render(val, ctx))

        defaults = Defaults(**data.get("defaults", {}))
        groups: dict[str, TargetGroup] = {}
        for engine_name, engine_cfg in data["targets"].items():
            target = Target(**engine_cfg["target"])
            queries: list[QueryDefinition] = []
            for q in engine_cfg["queries"]:
                rendered_q = dict(q)
                if "body" in rendered_q:
                    rendered_q["body"] = self._render(rendered_q["body"], enriched)
                if "params" in rendered_q:
                    rendered_q["params"] = {
                        k: self._render(v, enriched)
                        for k, v in rendered_q["params"].items()
                    }
                queries.append(QueryDefinition(**rendered_q))
            groups[engine_name] = TargetGroup(target=target, queries=queries)

        return defaults, groups

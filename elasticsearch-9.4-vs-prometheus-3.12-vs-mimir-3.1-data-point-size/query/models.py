import base64
import datetime
import json
import re
import urllib.parse
from dataclasses import dataclass, field

from jinja2 import Environment, PackageLoader, Template

_DURATION_RE = re.compile(r"^(\d+)(s|m|h)$")


def to_iso(ts: int | float) -> str:
    return datetime.datetime.fromtimestamp(int(ts), tz=datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def parse_time_arg(value: str, now: int) -> int:
    """Accept a Unix timestamp, 'now', or a duration like '270m' / '4h' / '30s'."""
    if value == "now":
        return now

    m = _DURATION_RE.match(value)
    if m:
        amount, unit = int(m.group(1)), m.group(2)
        return now - amount * {"s": 1, "m": 60, "h": 3600}[unit]
    try:
        return int(value)
    except ValueError:
        raise ValueError(
            f"Cannot parse time {value!r} — use a Unix timestamp, 'now', or a duration like '270m'"
        ) from None


@dataclass
class Target:
    base_url: str
    headers: dict[str, str] = field(default_factory=dict)
    method: str | None = None
    path: str | None = None


@dataclass
class QueryDefinition:
    name: str
    method: str | None = None
    path: str | None = None
    body: str | None = None
    params: dict[str, str] = field(default_factory=dict)
    # per-query vegeta overrides (fall back to Defaults when absent)
    rate: str | None = None
    duration: str | None = None
    workers: int | None = None
    max_workers: int | None = None
    timeout: str | None = None
    # warmup phase — runs a separate attack before the measured one; results discarded
    warmup_duration: str | None = None
    warmup_rate: str | None = None


@dataclass
class TargetGroup:
    target: Target
    queries: list[QueryDefinition]


@dataclass
class Defaults:
    rate: str
    duration: str
    workers: int
    max_workers: int
    timeout: str
    warmup_duration: str | None = None


@dataclass
class VegetaTarget:
    method: str
    url: str
    headers: dict[str, list[str]]
    body: bytes | None = None

    @classmethod
    def from_target_and_query(
        cls, target: "Target", query: "QueryDefinition"
    ) -> "VegetaTarget":
        method = query.method or target.method
        path = query.path or target.path
        if not method:
            raise ValueError(
                f"Query {query.name!r} has no method and the target defines no default"
            )
        if not path:
            raise ValueError(
                f"Query {query.name!r} has no path and the target defines no default"
            )
        headers = {k: [v] for k, v in target.headers.items()}
        url = target.base_url.rstrip("/") + path
        if query.params:
            url += "?" + urllib.parse.urlencode(query.params)
        body = query.body.strip().encode() if query.body else None
        return cls(method=method, url=url, headers=headers, body=body)

    def serialize(self) -> str:
        obj: dict = {"method": self.method, "url": self.url, "header": self.headers}
        if self.body is not None:
            obj["body"] = base64.b64encode(self.body).decode()
        return json.dumps(obj)


@dataclass
class AttackReport:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput: float
    success_pct: float
    engine: str = ""
    query_name: str = ""
    query_index: int = 0

    def to_result_dict(self) -> dict[str, str | float]:
        return {
            "name": self.query_name,
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "throughput_rps": round(self.throughput, 3),
            "success_pct": round(self.success_pct, 1),
        }


def _make_result_template() -> Template:
    env = Environment(
        loader=PackageLoader("query", "templates"),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["lj"] = lambda s, w: str(s).ljust(w)
    env.filters["rj"] = lambda v, w, fmt="": format(v, fmt).rjust(w)
    return env.get_template("result_table.j2")


_RESULT_TMPL = _make_result_template()


@dataclass
class BenchmarkResults:
    results: list[AttackReport]
    defaults: Defaults

    def to_result_table(self) -> str:
        engine_w = max(len("Engine"), max(len(r.engine) for r in self.results))
        query_w = max(
            len("Query"),
            max(len(f"[{r.query_index}] {r.query_name}") for r in self.results),
        )
        header = (
            f"{'Engine':<{engine_w}} {'Query':<{query_w}}"
            f" {'p50 ms':>9} {'p95 ms':>9} {'p99 ms':>9} {'RPS':>7} {'OK%':>6}"
        )
        sep = "─" * len(header)
        return _RESULT_TMPL.render(
            results=self.results,
            defaults=self.defaults,
            sep=sep,
            header=header,
            engine_w=engine_w,
            query_w=query_w,
        )


@dataclass
class VegetaConfig:
    effective_rate: str
    effective_duration: str
    effective_workers: int
    effective_timeout: str
    effective_max_workers: int

    @classmethod
    def from_defaults_and_query(
        cls, defaults: Defaults, query: QueryDefinition
    ) -> "VegetaConfig":
        return cls(
            effective_rate=query.rate or defaults.rate,
            effective_duration=query.duration or defaults.duration,
            effective_workers=query.workers or defaults.workers,
            effective_timeout=query.timeout or defaults.timeout,
            effective_max_workers=query.max_workers or defaults.max_workers,
        )

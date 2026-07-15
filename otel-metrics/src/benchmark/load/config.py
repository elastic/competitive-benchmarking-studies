"""Duration parsing helper for ingest scenario config."""

import re

_DURATION_RE = re.compile(r"^(\d+)(s|m|h)$")


def parse_duration_seconds(value: str) -> int:
    m = _DURATION_RE.match(value.strip())
    if not m:
        raise ValueError(
            f"Cannot parse duration {value!r} — use e.g. '270m', '4h', '30s'"
        )
    amount, unit = int(m.group(1)), m.group(2)
    return amount * {"s": 1, "m": 60, "h": 3600}[unit]

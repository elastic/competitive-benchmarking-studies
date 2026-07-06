"""Single source of truth for raw ClickHouse HTTP requests."""

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from benchmark.engine_config import BASE_URL


def _ch_request(sql: str) -> tuple[int, bytes]:
    """POST a raw SQL statement to the ClickHouse HTTP interface.

    No auth headers — the benchmark's `default` user runs with no password
    (see docker-compose.yml/users.d.xml), so this is a plain unauthenticated
    request.
    """
    req = urllib.request.Request(
        BASE_URL,
        method="POST",
        data=sql.encode(),
        headers={"Content-Type": "text/plain"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def ch_execute(sql: str) -> None:
    """Run a DDL/utility statement (no result rows expected)."""
    status, body = _ch_request(sql)
    if status != 200:
        raise RuntimeError(
            f"ClickHouse query failed: HTTP {status} {body.decode(errors='replace')}"
        )


def ch_query_json(sql: str) -> list[dict]:
    """Run a SELECT and return its rows, via ClickHouse's JSONEachRow format."""
    status, body = _ch_request(sql + " FORMAT JSONEachRow")
    if status != 200:
        raise RuntimeError(
            f"ClickHouse query failed: HTTP {status} {body.decode(errors='replace')}"
        )
    text = body.decode().strip()
    return [json.loads(line) for line in text.splitlines() if line]


def ch_execute_sql_file(path: Path) -> None:
    """Split path's contents on `;`-terminated statements and execute each
    one in file order."""
    for stmt in path.read_text().split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue

        ch_execute(stmt + ";")


def ch_optimize_table(table: str) -> None:
    """OPTIMIZE TABLE ... FINAL — merges all parts down to one per partition,
    the ClickHouse analogue of Elasticsearch's forcemerge, so the on-disk
    size reflects fully-compacted storage rather than mid-merge parts."""
    ch_execute(f"OPTIMIZE TABLE {table} FINAL")


def ch_table_stats(database: str, tables: list[str]) -> tuple[int, int]:
    """Sum (rows, bytes_on_disk) across system.parts for the given tables.

    Returns (rows, size_bytes).
    """
    table_list = ", ".join(f"'{t}'" for t in tables)
    rows = ch_query_json(
        "SELECT sum(rows) AS rows, sum(bytes_on_disk) AS bytes_on_disk "
        "FROM system.parts "
        f"WHERE active AND database = '{database}' AND table IN ({table_list})"
    )
    if not rows:
        return 0, 0
    row = rows[0]
    return int(row.get("rows") or 0), int(row.get("bytes_on_disk") or 0)

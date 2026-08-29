import asyncio
import re
from typing import Any

import psycopg
from langchain_core.tools import tool
from psycopg.rows import dict_row

from backend.app.config import get_settings

_ALLOWED_LEADING = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CREATE|EXEC|EXECUTE|CALL)\b",
    re.IGNORECASE,
)


class UnsafeQueryError(Exception):
    """A query failed the read-only safety check (§5.2's "parameterised SQL")."""


def _validate_read_only(query: str) -> None:
    """Single SELECT/WITH statement, no mutating keywords, no statement stacking.

    This is a *tool-level* check, not a database-level guarantee. The connection
    this tool uses (`settings.postgres_url`, the `nexus` role from
    `docker-compose.yml`) has full read/write privileges — it was provisioned for
    Phase 2's never-built document registry, not scoped down for this tool. A
    dedicated read-only Postgres role is the stronger, deployment-level control;
    this check is what's actually enforced today, and is not a substitute for that
    role existing before this tool is exposed to anything beyond local development.
    """
    stripped = query.strip().rstrip(";")

    if ";" in stripped:
        raise UnsafeQueryError("Multiple statements are not allowed.")

    if not _ALLOWED_LEADING.match(stripped):
        raise UnsafeQueryError("Only SELECT/WITH statements are allowed.")

    if _FORBIDDEN_KEYWORDS.search(stripped):
        raise UnsafeQueryError("Query contains a disallowed keyword.")


def _run_sync(
    query: str, params: list[Any] | None, postgres_url: str, max_rows: int
) -> list[dict[str, Any]]:
    """The actual query, via psycopg's *sync* API run in a worker thread.

    Not `psycopg.AsyncConnection`: verified directly that it raises
    `psycopg.InterfaceError` on Windows under the default `ProactorEventLoop` —
    psycopg's async mode requires `SelectorEventLoop`. Switching the process-wide
    event loop policy to Selector was considered and rejected: this project's MCP
    stdio servers (§5.1, verified working end to end) need Proactor for subprocess
    support, which `SelectorEventLoop` does not provide on Windows at all. The two
    requirements conflict, so the sync driver run via `asyncio.to_thread` sidesteps
    the conflict entirely rather than picking a loop policy that breaks the other
    feature.
    """
    with psycopg.connect(postgres_url, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchmany(max_rows)


async def run_readonly_query(
    query: str, params: list[Any] | None = None
) -> list[dict[str, Any]]:
    """Execute a read-only, parameterized SQL query against the project database.

    Parameterized via psycopg's placeholder binding (`%s`), never string
    interpolation — `params` are bound by the driver, not formatted into `query`.
    Capped at `settings.sql_tool_max_rows` regardless of what the query itself
    requests, via `fetchmany()` rather than trusting a `LIMIT` clause to be present.
    """
    _validate_read_only(query)
    settings = get_settings()
    return await asyncio.to_thread(
        _run_sync, query, params, settings.postgres_url, settings.sql_tool_max_rows
    )


@tool
async def sql_query(query: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    """Run a read-only SQL query (SELECT or WITH...SELECT only) against the project
    database. INSERT/UPDATE/DELETE/DDL statements and multi-statement queries are
    rejected. Use `params` for any user-supplied values — never build them into
    the query string. Results are capped at a fixed row limit.
    """
    return await run_readonly_query(query, params)

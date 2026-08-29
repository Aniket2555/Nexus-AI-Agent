"""Document registry: a Postgres-backed record of every document NEXUS has ever
been asked to ingest.

Before this module, the only trace of an uploaded document was its chunks inside
Qdrant/Elasticsearch — there was no record of what was uploaded, by whom, when, or
of its original content, and `_JOBS` in `api/v1/documents.py` is an in-memory dict
that does not survive a restart. `scripts/reindex.py` could therefore only rebuild
a collection from a `--source-dir` the operator manually maintained in lockstep
with every past upload (see that script's module docstring, and Phases.md §2.7's
closing paragraph). This module closes that gap.

Plain SQL DDL run directly via psycopg's sync API in a worker thread — no ORM, no
migration framework. This project has exactly one table here; Alembic/SQLAlchemy
would be more machinery than the problem warrants, and it already has a working
precedent for "talk to Postgres without an ORM" in
`backend/app/mcp/tools/custom/sql.py`. The sync-driver-via-`asyncio.to_thread`
choice below is copied from that file for the same reason documented there:
`psycopg.AsyncConnection` raises `psycopg.InterfaceError` under Windows's default
`ProactorEventLoop`, and switching the process-wide loop policy to Selector would
break the MCP stdio servers' subprocess support (§5.1), which requires Proactor.
Running the sync driver in a thread sidesteps the conflict instead of trading one
requirement for the other.

Storing original bytes vs. metadata-only:
    NEXUS has no blob store (S3/MinIO/etc.) anywhere in this stack. The only place
    an uploaded file's original bytes exist today is the request handler's memory,
    for the lifetime of one background task (`documents.py`'s `_ingest`) — once
    that task returns, they are gone. A metadata-only registry ("doc_id, filename,
    when, status") would tell you *that* something was uploaded but not let
    `reindex.py` actually recover and re-chunk it, which is the entire point of
    closing this gap — the operator would still need to have kept the original
    file somewhere themselves, exactly the status quo this module exists to fix.

    So: the original bytes are stored as `content BYTEA`, subject to the same
    `settings.max_upload_bytes` cap the upload endpoint already enforces (25 MB by
    default) — nothing new is uploadable that couldn't already reach the server.
    This is a real tradeoff, not a free win: bytea rows at that size bloat the
    table and its WAL, and a production deployment at real document volume should
    move this to object storage with only a path/key left in this column. For a
    local/dev-scale registry, though, it is the only option that doesn't stand up
    a second storage system — consistent with this project's existing bias toward
    the simplest thing that actually works today over the "correct at scale"
    answer (see the redis-stack-over-separate-search-engine and
    sync-psycopg-over-asyncpg calls elsewhere in this codebase).
"""

import asyncio
from typing import Any

import psycopg
from psycopg.rows import dict_row

from backend.app.config import get_settings

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS document_registry (
    doc_id              TEXT PRIMARY KEY,
    tenant_id           TEXT NOT NULL,
    filename            TEXT NOT NULL,
    suffix              TEXT NOT NULL,
    content_type        TEXT,
    uploaded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    status              TEXT NOT NULL,
    chunk_count         INTEGER NOT NULL DEFAULT 0,
    vector_count        INTEGER NOT NULL DEFAULT 0,
    keyword_count       INTEGER NOT NULL DEFAULT 0,
    error               TEXT,
    -- Original upload bytes. NULL for rows backfilled without content, or for a
    -- future "metadata-only" ingest path — every read path below tolerates NULL.
    content             BYTEA,
    content_size_bytes  INTEGER
)
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS document_registry_tenant_id_idx
    ON document_registry (tenant_id)
"""

_METADATA_COLUMNS = (
    "doc_id, tenant_id, filename, suffix, content_type, uploaded_at, status, "
    "chunk_count, vector_count, keyword_count, error, content_size_bytes"
)


def _connect(postgres_url: str) -> psycopg.Connection:
    return psycopg.connect(postgres_url, row_factory=dict_row)


def _ensure_schema_sync(postgres_url: str) -> None:
    """Idempotent DDL — safe to call on every startup, same as
    `DenseRetriever.ensure_collection()` / `SparseRetriever.ensure_index()` in
    `main.py`'s lifespan, which this is meant to sit alongside.
    """
    with _connect(postgres_url) as conn, conn.cursor() as cur:
        cur.execute(_CREATE_TABLE_SQL)
        cur.execute(_CREATE_INDEX_SQL)
        conn.commit()


async def ensure_schema() -> None:
    settings = get_settings()
    await asyncio.to_thread(_ensure_schema_sync, settings.postgres_url)


def _record_document_sync(postgres_url: str, values: tuple[Any, ...]) -> None:
    with _connect(postgres_url) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO document_registry (
                doc_id, tenant_id, filename, suffix, content_type, status,
                chunk_count, vector_count, keyword_count, error,
                content, content_size_bytes
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                filename           = EXCLUDED.filename,
                suffix             = EXCLUDED.suffix,
                content_type       = EXCLUDED.content_type,
                uploaded_at        = now(),
                status             = EXCLUDED.status,
                chunk_count        = EXCLUDED.chunk_count,
                vector_count       = EXCLUDED.vector_count,
                keyword_count      = EXCLUDED.keyword_count,
                error              = EXCLUDED.error,
                content            = EXCLUDED.content,
                content_size_bytes = EXCLUDED.content_size_bytes
            """,
            values,
        )
        conn.commit()


async def record_document(
    doc_id: str,
    tenant_id: str,
    filename: str,
    suffix: str,
    status: str,
    content_type: str | None = None,
    chunk_count: int = 0,
    vector_count: int = 0,
    keyword_count: int = 0,
    error: str | None = None,
    content: bytes | None = None,
) -> None:
    """Insert or replace one document's registry row.

    Upsert on `doc_id`, matching the "re-uploading a corrected version replaces
    the previous one" convention `documents.py`'s `_ingest` already applies to
    Qdrant/Elasticsearch (`delete_document` + re-upsert) — the registry row for a
    given `{tenant_id}:{filename}` should never fork into two.
    """
    settings = get_settings()
    values = (
        doc_id,
        tenant_id,
        filename,
        suffix,
        content_type,
        status,
        chunk_count,
        vector_count,
        keyword_count,
        error,
        content,
        len(content) if content is not None else None,
    )
    await asyncio.to_thread(_record_document_sync, settings.postgres_url, values)


def _list_documents_sync(postgres_url: str, tenant_id: str) -> list[dict[str, Any]]:
    with _connect(postgres_url) as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {_METADATA_COLUMNS} FROM document_registry "
            f"WHERE tenant_id = %s ORDER BY uploaded_at DESC",
            (tenant_id,),
        )
        return cur.fetchall()


async def list_documents(tenant_id: str) -> list[dict[str, Any]]:
    """Every registry row for one tenant, newest first. Powers `GET /documents`."""
    settings = get_settings()
    return await asyncio.to_thread(_list_documents_sync, settings.postgres_url, tenant_id)


def _get_document_sync(postgres_url: str, doc_id: str) -> dict[str, Any] | None:
    with _connect(postgres_url) as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {_METADATA_COLUMNS} FROM document_registry WHERE doc_id = %s",
            (doc_id,),
        )
        return cur.fetchone()


async def get_document(doc_id: str) -> dict[str, Any] | None:
    """One registry row, or None if `doc_id` was never ingested. Powers
    `GET /documents/{doc_id}`.
    """
    settings = get_settings()
    return await asyncio.to_thread(_get_document_sync, settings.postgres_url, doc_id)


def _list_documents_with_content_sync(
    postgres_url: str, tenant_id: str
) -> list[dict[str, Any]]:
    with _connect(postgres_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT doc_id, filename, content FROM document_registry "
            "WHERE tenant_id = %s AND status = 'succeeded' AND content IS NOT NULL "
            "ORDER BY filename",
            (tenant_id,),
        )
        rows = cur.fetchall()
    for row in rows:
        if row["content"] is not None and not isinstance(row["content"], bytes):
            row["content"] = bytes(row["content"])
    return rows


async def list_documents_with_content(tenant_id: str) -> list[dict[str, Any]]:
    """Every *successfully* ingested document for a tenant, with its original
    bytes attached. This is what closes the §2.7 gap: `scripts/reindex.py
    --from-registry` calls this instead of requiring a `--source-dir` the operator
    maintains by hand. Rows with `status != "succeeded"` or `content IS NULL`
    (failed ingests, or any row ever written without bytes) are excluded — there
    is nothing a reindex could do with either.
    """
    settings = get_settings()
    return await asyncio.to_thread(
        _list_documents_with_content_sync, settings.postgres_url, tenant_id
    )

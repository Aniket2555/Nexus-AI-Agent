"""Live-Postgres tests for the document registry (`backend/app/models/database.py`).

Not mocked, on purpose: `backend/app/mcp/tools/custom/sql.py`'s tests only exercise
the pure regex validator, but this module's whole job is talking to a real
database correctly (upsert semantics, tenant scoping, BYTEA round-tripping as
`bytes` rather than `memoryview`) — none of that is meaningfully testable against
a mock. Per Phases.md's Phase 1 banner ("everything gets run, not just read"),
these run against the actual `nexus-postgres` container `docker-compose.yml`
brings up (`POSTGRES_URL` in `.env`, defaulting to
`postgresql://nexus:nexus@localhost:5432/nexus`) — the same instance
`scripts/seed_data.py` was verified against directly in this session. If that
container isn't running, every test here fails with a real connection error
rather than silently skipping, which is the intended signal.

Each test uses a random, disposable tenant_id and cleans its own rows up in a
`finally` block, so re-running this file (or running it alongside a developer's
own manual testing against the same database) never accumulates junk rows or
collides with another test's data.
"""

import uuid

import psycopg
import pytest

from backend.app.config import get_settings
from backend.app.models.database import (
    ensure_schema,
    get_document,
    list_documents,
    list_documents_with_content,
    record_document,
)


def _fresh_tenant_id() -> str:
    return f"test-registry-{uuid.uuid4().hex[:12]}"


def _delete_tenant_rows(tenant_id: str) -> None:
    settings = get_settings()
    with psycopg.connect(settings.postgres_url) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM document_registry WHERE tenant_id = %s", (tenant_id,))
        conn.commit()


@pytest.fixture
def tenant_id():
    tid = _fresh_tenant_id()
    try:
        yield tid
    finally:
        _delete_tenant_rows(tid)


@pytest.fixture(autouse=True)
async def _schema():
    """Makes each test independent of whether the real FastAPI app's lifespan
    (main.py) has ever started in this environment — `ensure_schema()` is
    idempotent (`CREATE TABLE/INDEX IF NOT EXISTS`), so calling it before every
    test is cheap and guarantees the table exists.
    """
    await ensure_schema()


async def test_ensure_schema_is_idempotent():
    await ensure_schema()
    await ensure_schema()


async def test_record_and_get_document_round_trips(tenant_id):
    doc_id = f"{tenant_id}:handbook.md"
    await record_document(
        doc_id,
        tenant_id,
        filename="handbook.md",
        suffix=".md",
        content_type="text/markdown",
        status="succeeded",
        chunk_count=3,
        vector_count=3,
        keyword_count=3,
        content=b"# Handbook\n\nSome content.",
    )

    row = await get_document(doc_id)

    assert row is not None
    assert row["doc_id"] == doc_id
    assert row["tenant_id"] == tenant_id
    assert row["filename"] == "handbook.md"
    assert row["suffix"] == ".md"
    assert row["status"] == "succeeded"
    assert row["chunk_count"] == 3
    assert row["vector_count"] == 3
    assert row["keyword_count"] == 3
    assert row["error"] is None
    assert row["content_size_bytes"] == len(b"# Handbook\n\nSome content.")
    assert "content" not in row


async def test_get_document_returns_none_for_unknown_id():
    assert await get_document("nonexistent-tenant:nonexistent-file.md") is None


async def test_record_document_upserts_rather_than_duplicates(tenant_id):
    doc_id = f"{tenant_id}:report.txt"
    await record_document(
        doc_id,
        tenant_id,
        filename="report.txt",
        suffix=".txt",
        status="failed",
        error="BoomError: something went wrong",
        content=b"original bytes",
    )
    await record_document(
        doc_id,
        tenant_id,
        filename="report.txt",
        suffix=".txt",
        status="succeeded",
        chunk_count=2,
        vector_count=2,
        keyword_count=2,
        content=b"corrected bytes",
    )

    rows = await list_documents(tenant_id)
    matching = [r for r in rows if r["doc_id"] == doc_id]

    assert len(matching) == 1, "re-recording the same doc_id must update, not duplicate"
    assert matching[0]["status"] == "succeeded"
    assert matching[0]["error"] is None
    assert matching[0]["content_size_bytes"] == len(b"corrected bytes")


async def test_list_documents_only_returns_the_requested_tenant(tenant_id):
    other_tenant_id = _fresh_tenant_id()
    try:
        await record_document(
            f"{tenant_id}:mine.md",
            tenant_id,
            filename="mine.md",
            suffix=".md",
            status="succeeded",
        )
        await record_document(
            f"{other_tenant_id}:not-mine.md",
            other_tenant_id,
            filename="not-mine.md",
            suffix=".md",
            status="succeeded",
        )

        rows = await list_documents(tenant_id)

        assert {r["doc_id"] for r in rows} == {f"{tenant_id}:mine.md"}
    finally:
        _delete_tenant_rows(other_tenant_id)


async def test_list_documents_with_content_excludes_failed_and_contentless_rows(tenant_id):
    await record_document(
        f"{tenant_id}:ok.md",
        tenant_id,
        filename="ok.md",
        suffix=".md",
        status="succeeded",
        content=b"has content",
    )
    await record_document(
        f"{tenant_id}:failed.md",
        tenant_id,
        filename="failed.md",
        suffix=".md",
        status="failed",
        error="boom",
        content=b"content that shouldn't matter, ingest failed",
    )
    await record_document(
        f"{tenant_id}:no-content.md",
        tenant_id,
        filename="no-content.md",
        suffix=".md",
        status="succeeded",
        content=None,
    )

    rows = await list_documents_with_content(tenant_id)

    assert [r["filename"] for r in rows] == ["ok.md"]
    assert isinstance(rows[0]["content"], bytes), "BYTEA must come back as bytes, not memoryview"
    assert rows[0]["content"] == b"has content"

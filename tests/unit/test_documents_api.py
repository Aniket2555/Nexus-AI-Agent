"""TestClient tests for the document registry endpoints (`GET /documents`,
`GET /documents/{doc_id}`) added to `backend/app/api/v1/documents.py`.

Standalone app around just the documents router, same reasoning as
`test_workflows_api.py`: the real app's lifespan (main.py) provisions
Qdrant/Elasticsearch/memory collections this test doesn't need. It *does* still
call `ensure_schema()` directly (see the module-scoped fixture below), since the
registry endpoints do need `document_registry` to exist.

Also same `with TestClient(app) as client:` requirement `test_workflows_api.py`
documents: outside that `with` block, TestClient opens a fresh event loop per
request and tears it down afterward, which is wrong for anything that shares
state across requests via the process-wide `asyncio.to_thread` calls this
module's endpoints make. Entered once, module-wide, for the same reason.

Auth: Phase 8's `enforce_api_rate_limit` dependency (`backend/app/api/middleware/
rate_limit.py`) now guards these endpoints. Rather than depend on whatever
`NEXUS_API_KEYS` happens to be configured in this machine's `.env` (fragile, and
not this test's concern), the dependency is overridden with a fake principal via
FastAPI's `app.dependency_overrides` — the standard way to swap out auth in a
test client without touching real config or Redis.
"""

import uuid

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.middleware.auth import Principal
from backend.app.api.middleware.rate_limit import enforce_api_rate_limit
from backend.app.api.v1.documents import router
from backend.app.config import get_settings
from backend.app.models.database import ensure_schema, record_document

app = FastAPI()
app.include_router(router, prefix="/api/v1")

TENANT_A = f"test-docs-api-a-{uuid.uuid4().hex[:8]}"
TENANT_B = f"test-docs-api-b-{uuid.uuid4().hex[:8]}"


def _principal_a() -> Principal:
    return Principal(tenant_id=TENANT_A, user_id="test-user-a", api_key="test-key-a")


app.dependency_overrides[enforce_api_rate_limit] = _principal_a

_client_cm = TestClient(app)
client = _client_cm.__enter__()


def _delete_tenant_rows(*tenant_ids: str) -> None:
    settings = get_settings()
    with psycopg.connect(settings.postgres_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM document_registry WHERE tenant_id = ANY(%s)", (list(tenant_ids),)
            )
        conn.commit()


@pytest.fixture(scope="module", autouse=True)
def _seed_registry():
    import asyncio

    async def _seed() -> None:
        await ensure_schema()

        await record_document(
            f"{TENANT_A}:alpha.md",
            TENANT_A,
            "alpha.md",
            ".md",
            content_type="text/markdown",
            status="succeeded",
            chunk_count=2,
            vector_count=2,
            keyword_count=2,
            content=b"alpha content",
        )
        await record_document(
            f"{TENANT_A}:beta.md",
            TENANT_A,
            "beta.md",
            ".md",
            status="failed",
            error="ParseError: could not parse file",
        )
        await record_document(
            f"{TENANT_B}:secret.md",
            TENANT_B,
            "secret.md",
            ".md",
            status="succeeded",
            content=b"tenant B's content",
        )

    asyncio.run(_seed())
    try:
        yield
    finally:
        _delete_tenant_rows(TENANT_A, TENANT_B)


def test_list_documents_returns_only_the_authenticated_tenants_documents():
    response = client.get("/api/v1/documents")
    assert response.status_code == 200

    body = response.json()
    assert body["tenant_id"] == TENANT_A

    filenames = {d["filename"] for d in body["documents"]}
    assert filenames == {"alpha.md", "beta.md"}


def test_list_documents_includes_status_and_counts():
    response = client.get("/api/v1/documents")
    by_filename = {doc["filename"]: doc for doc in response.json()["documents"]}

    assert by_filename["alpha.md"]["status"] == "succeeded"
    assert by_filename["alpha.md"]["chunk_count"] == 2
    assert by_filename["beta.md"]["status"] == "failed"
    assert by_filename["beta.md"]["error"] == "ParseError: could not parse file"
    assert "content" not in by_filename["alpha.md"]


def test_get_document_returns_the_record_for_the_authenticated_tenant():
    response = client.get(f"/api/v1/documents/{TENANT_A}:alpha.md")
    assert response.status_code == 200

    body = response.json()
    assert body["doc_id"] == f"{TENANT_A}:alpha.md"
    assert body["status"] == "succeeded"
    assert body["content_size_bytes"] == len(b"alpha content")


def test_get_document_404s_for_unknown_doc_id():
    response = client.get(f"/api/v1/documents/{TENANT_A}:does-not-exist.md")
    assert response.status_code == 404


def test_get_document_404s_for_another_tenants_document():
    """TENANT_A's principal must not be able to read TENANT_B's document just by
    knowing (or guessing) its doc_id — the endpoint checks ownership, not just
    existence."""
    response = client.get(f"/api/v1/documents/{TENANT_B}:secret.md")
    assert response.status_code == 404

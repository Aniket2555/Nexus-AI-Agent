"""Pydantic request/response models for the document registry (§1.6, §2.7).

`backend/app/api/v1/documents.py`'s existing endpoints (`POST /upload`,
`GET /jobs/{job_id}`) were built before this module existed and return
hand-built `dict[str, Any]` bodies. They are left exactly as they are here —
retyping them risks changing a response shape a test or client already depends
on, for no functional gain. `workflows.py`'s `SandboxWorkflowRequest` /
`DemoWorkflowRequest` were also considered for promotion here and left alone:
they are workflow-specific (no registry/database code needs them), and moving
them would only add an import indirection with nothing to reuse.

What *is* new below are the shapes for the registry itself — `DocumentRecord`
(one row of `document_registry`, minus the raw `content` bytes; see
`models/database.py`'s module docstring for why those bytes are stored at all)
and the list response that wraps it, used by the new `GET /documents` and
`GET /documents/{doc_id}` endpoints.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class DocumentStatus(StrEnum):
    """Mirrors the string values `documents.py`'s in-memory `_JOBS` dict has used
    since Phase 1 (`"pending"`, `"running"`, `"succeeded"`, `"failed"`) — this is a
    typed view of the same vocabulary, not a new one, so a registry row's `status`
    always matches what `GET /jobs/{job_id}` would have reported for the same job.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    """One row of the `document_registry` table, as returned to API callers.

    Deliberately excludes the `content` column (original file bytes) — nothing
    outside `scripts/reindex.py` needs the raw bytes back over HTTP, and a
    multi-megabyte document would make every list/get response huge for no
    reason. `content_size_bytes` is included instead, so a caller can tell
    whether bytes were retained at all without fetching them.
    """

    doc_id: str
    tenant_id: str
    filename: str
    suffix: str
    content_type: str | None = None
    uploaded_at: datetime
    status: DocumentStatus
    chunk_count: int
    vector_count: int
    keyword_count: int
    error: str | None = None
    content_size_bytes: int | None = None


class DocumentListResponse(BaseModel):
    tenant_id: str
    documents: list[DocumentRecord]

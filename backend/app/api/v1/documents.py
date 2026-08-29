import mimetypes
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status

from backend.app.api.middleware.auth import Principal
from backend.app.api.middleware.rate_limit import enforce_api_rate_limit
from backend.app.config import get_settings
from backend.app.models import database as registry
from backend.app.models.schemas import DocumentListResponse, DocumentRecord
from backend.app.rag.ingestion.dispatch import SUPPORTED_SUFFIXES, extract_pages
from backend.app.rag.processing.chunker import chunk_pages
from backend.app.rag.retrieval.dense import get_retriever
from backend.app.rag.retrieval.sparse import get_sparse_retriever

router = APIRouter(prefix="/documents", tags=["documents"])

READ_CHUNK_BYTES = 1048576

_JOBS: dict[str, dict[str, Any]] = {}


async def _ingest(job_id: str, filename: str, payload: bytes, tenant_id: str) -> None:
    suffix = Path(filename).suffix.lower()
    content_type = mimetypes.guess_type(filename)[0]

    _JOBS[job_id]["status"] = "running"
    doc_id = f"{tenant_id}:{filename}"

    try:
        chunks = chunk_pages(
            extract_pages(filename, payload), doc_id=doc_id, source=filename, tenant_id=tenant_id
        )

        dense = get_retriever()
        sparse = get_sparse_retriever()

        await dense.delete_document(doc_id)
        await sparse.delete_document(doc_id)

        vectors = await dense.upsert_chunks(chunks)
        keywords = await sparse.upsert_chunks(chunks)

        _JOBS[job_id] |= {
            "status": "succeeded",
            "chunks": len(chunks),
            "vectors": vectors,
            "keyword_docs": keywords,
        }

        try:
            await registry.record_document(
                doc_id,
                tenant_id,
                filename,
                suffix,
                content_type=content_type,
                status="succeeded",
                chunk_count=len(chunks),
                vector_count=vectors,
                keyword_count=keywords,
                content=payload,
            )
        except Exception as registry_exc:
            _JOBS[job_id]["registry_error"] = f"{type(registry_exc).__name__}: {registry_exc}"
    except Exception as exc:
        _JOBS[job_id] |= {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        try:
            await registry.record_document(
                doc_id,
                tenant_id,
                filename,
                suffix,
                content_type=content_type,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                content=payload,
            )
        except Exception:
            pass


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    principal: Annotated[Principal, Depends(enforce_api_rate_limit)],
) -> dict[str, Any]:
    settings = get_settings()
    tenant_id = principal["tenant_id"]

    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported extension {suffix!r}. Supported: {sorted(SUPPORTED_SUFFIXES)}.",
        )

    payload = bytearray()
    chunk = await file.read(READ_CHUNK_BYTES)
    while chunk:
        payload.extend(chunk)
        if len(payload) > settings.max_upload_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"File exceeds the {settings.max_upload_bytes} byte limit.",
            )
        chunk = await file.read(READ_CHUNK_BYTES)

    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {"status": "pending", "filename": filename}
    background_tasks.add_task(_ingest, job_id, filename, bytes(payload), tenant_id)

    return {"job_id": job_id, "status": "pending", "filename": filename}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    if job_id not in _JOBS:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown job id.")
    return {"job_id": job_id, **_JOBS[job_id]}


@router.get("")
async def list_documents(
    principal: Annotated[Principal, Depends(enforce_api_rate_limit)],
) -> DocumentListResponse:
    """Every document ever ingested for a tenant, per the Postgres registry —
    independent of `_JOBS`, so this survives a restart and lists documents
    uploaded by any worker process, not just this one.
    """
    tenant_id = principal["tenant_id"]
    rows = await registry.list_documents(tenant_id)
    return DocumentListResponse(
        tenant_id=tenant_id, documents=[DocumentRecord.model_validate(row) for row in rows]
    )


@router.get("/{doc_id}")
async def get_document(
    doc_id: str, principal: Annotated[Principal, Depends(enforce_api_rate_limit)]
) -> DocumentRecord:
    """One document's registry record, keyed by the same `doc_id` convention
    (`"{tenant_id}:{filename}"`) used throughout ingestion and retrieval."""
    row = await registry.get_document(doc_id)
    if row is None or row["tenant_id"] != principal["tenant_id"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown document id.")
    return DocumentRecord.model_validate(row)

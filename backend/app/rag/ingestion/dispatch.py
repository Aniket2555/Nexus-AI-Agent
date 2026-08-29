import io
from pathlib import Path

from backend.app.rag.ingestion.excel_processor import extract_excel_pages
from backend.app.rag.ingestion.image_processor import extract_image_pages
from backend.app.rag.ingestion.pdf_parser import extract_pdf_pages
from backend.app.rag.ingestion.pptx_processor import extract_pptx_pages

SUPPORTED_SUFFIXES = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".xls",
    ".csv",
    ".txt",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
}


def extract_pages(filename: str, payload: bytes) -> list[tuple[int, str]]:
    """Return (page_number, text) pairs. Non-paginated formats are a single page 1.

    Shared by both the upload endpoint (documents.py) and scripts/reindex.py — the
    two real callers that need "bytes in, pages out" for the same set of formats.
    §2.5's format table maps one-to-one onto these branches, each delegating to its
    own ingestion module rather than growing this function.
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_pages(payload)
    if suffix == ".docx":
        import docx

        document = docx.Document(io.BytesIO(payload))
        return [(1, "\n\n".join(p.text for p in document.paragraphs))]
    if suffix in (".xlsx", ".xls", ".csv"):
        return extract_excel_pages(filename, payload)
    if suffix == ".pptx":
        return extract_pptx_pages(payload)
    if suffix in (".png", ".jpg", ".jpeg"):
        return extract_image_pages(filename, payload)
    return [(1, payload.decode("utf-8", errors="replace"))]

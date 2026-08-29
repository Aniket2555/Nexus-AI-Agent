import io

import pdfplumber


def extract_pdf_pages(payload: bytes) -> list[tuple[int, str]]:
    """PDF text + tables, per page.

    Phase 1 used PyMuPDF's plain get_text(), which flattens tables into unstructured
    inline text — a question answerable only from a table cell has no reliable chunk
    to retrieve from. pdfplumber gives structured cells instead, rendered here as a
    markdown table appended after the page's prose so both are in the same chunk-able
    text and a table-only question still matches.
    """
    pages: list[tuple[int, str]] = []
    with pdfplumber.open(io.BytesIO(payload)) as doc:
        for number, page in enumerate(doc.pages, 1):
            parts = [page.extract_text() or ""]
            for table in page.extract_tables():
                rendered = _render_table(table)
                if rendered:
                    parts.append(rendered)
            pages.append((number, "\n\n".join(p for p in parts if p.strip())))
    return pages


def _render_table(rows: list[list[str | None]]) -> str:
    """Render extracted table cells as a GitHub-flavoured markdown table.

    Not every extracted "table" is well-formed — pdfplumber can return a single-row
    or empty table on dense layouts — so this degrades to nothing rather than
    emitting a broken table block.
    """
    cleaned = [[(cell or "").strip() for cell in row] for row in rows if any(row)]
    if len(cleaned) < 2:
        return ""

    header, *body = cleaned
    width = len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in body:
        row = (row + [""] * width)[:width]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

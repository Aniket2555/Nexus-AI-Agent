import io

from PIL import Image

from backend.app.rag.ingestion.image_processor import extract_image_pages


def _make_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 40), "white").save(buf, "PNG")
    return buf.getvalue()


def test_missing_tesseract_binary_degrades_gracefully_instead_of_crashing():
    """This dev environment has no Tesseract binary installed (verified: `tesseract
    --version` -> command not found) — pytesseract.TesseractNotFoundError is the
    realistic path this test exercises, not a mock standing in for it.
    """
    pages = extract_image_pages("scan.png", _make_png())

    assert len(pages) == 1
    page_number, text = pages[0]
    assert page_number == 1
    assert "scan.png" in text
    assert "no OCR text extracted" in text


def test_returns_single_page():
    pages = extract_image_pages("photo.jpg", _make_png())
    assert len(pages) == 1

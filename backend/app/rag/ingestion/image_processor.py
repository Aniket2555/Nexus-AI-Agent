import io
import logging

import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


def extract_image_pages(filename: str, payload: bytes) -> list[tuple[int, str]]:
    """OCR a single image into one "page" of text.

    §2.5 lists GPT-4o Vision as an alternative to Tesseract OCR; that option is
    dropped here for the same reason OpenAI embeddings were in Phase 1 (DECISIONS.md
    D7) — it would reintroduce a paid, keyed dependency into a stack that is
    otherwise free. Tesseract is the OCR *binary* pytesseract wraps; it is a system
    package, not something `pip install` can provide, so this degrades to a plain
    placeholder instead of failing the whole upload if it isn't on PATH.
    """
    image = Image.open(io.BytesIO(payload))
    try:
        text = pytesseract.image_to_string(image).strip()
    except pytesseract.TesseractNotFoundError:
        logger.warning(
            "Tesseract binary not found on PATH; ingesting %r without OCR text. "
            "Install Tesseract (https://github.com/tesseract-ocr/tesseract) to "
            "enable it.",
            filename,
        )
        text = ""

    if not text:
        text = f"[Image: {filename} — no OCR text extracted]"
    return [(1, text)]

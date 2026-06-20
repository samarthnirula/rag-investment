"""OCR fallback for scanned PDFs using Tesseract via pytesseract.

Used when a PDF page has no extractable text layer (char count < threshold)
and the vision API is unavailable or disabled.

Requires:
  pip install pytesseract Pillow
  # macOS:  brew install tesseract
  # Ubuntu: apt-get install tesseract-ocr
  # Docker: see Dockerfile (tesseract-ocr package)
"""
from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

# Minimum characters after OCR before we trust the result.
# Very low char counts are usually blank pages or borders.
_MIN_OCR_CHARS = 30


def extract_text_with_ocr(page) -> str | None:
    """Run Tesseract OCR on a PyMuPDF page object.

    Renders the page at 2× zoom (~144 DPI) for reasonable accuracy without
    excessive memory use. Returns None if pytesseract is not installed,
    Tesseract is not on PATH, or the page yields too little text.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        _log.debug("pytesseract or Pillow not installed — OCR skipped")
        return None

    try:
        import fitz  # PyMuPDF already imported in pdf_parser; re-import for isolation
        mat = fitz.Matrix(2, 2)   # 2× zoom in both axes
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
        cleaned = text.strip()
        if len(cleaned) < _MIN_OCR_CHARS:
            return None
        return cleaned
    except pytesseract.TesseractNotFoundError:
        _log.warning(
            "Tesseract binary not found. Install it: "
            "macOS→ brew install tesseract | Ubuntu→ apt-get install tesseract-ocr"
        )
        return None
    except Exception as exc:
        _log.debug("OCR failed for page: %s", exc)
        return None

"""PDF text extraction with page-level metadata and table detection."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz
import pdfplumber

from insightlens.ingestion.ocr_extractor import extract_text_with_ocr
from insightlens.ingestion.vision_extractor import extract_visual_content

# Pages with fewer than this many characters are treated as visual-dominant
# (image, logo, or chart fills the slide with minimal text).
# Raised from 50 → 120 to catch pages with a short caption above an image.
_VISUAL_CHAR_THRESHOLD = 120

# Pages below this character count trigger a vision API call to extract
# chart / map / logo content that text extraction cannot read.
_VISION_CALL_THRESHOLD = 300

# Lines that look like footnote markers or standalone page numbers should never
# become a slide title (e.g. "(1)", "6", "Page 3").
_TITLE_EXCLUDE_RE = re.compile(
    r"^\(?[\d]+\)?[\.\)]?\s*$|^page\s*\d+$", re.IGNORECASE
)

# Matches the start of a footnote line. Patterns covered (B.1):
#   (1) text  — parenthesised number
#   1. text   — number + period
#   1) text   — number + closing paren
#   1 Capital — bare number + capital letter (common in REIT decks)
#   (a) text  — letter footnote
#   Note: / Notes: — table footnote block header
#   * text / † text — symbol footnotes
_FOOTNOTE_START_RE = re.compile(
    r"^[\(\*†‡§¶]?\d+[\.\)]\s+\S"   # (1), 1., 1)
    r"|^\d+\s+[A-Z]"                 # 1 Capital
    r"|^\([a-z]\)\s+\S"              # (a) text
    r"|^[Nn]otes?:?\s+\S"           # Note: / Notes:
    r"|^\*\s+\S|^†\s+\S"            # * text / † text
)


class PDFParsingError(Exception):
    """Raised when a PDF cannot be opened or parsed."""


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str
    char_count: int
    is_likely_visual: bool
    slide_title: str | None = None
    tables: tuple = field(default_factory=tuple)  # tuple[tuple[tuple[str]]] — extracted tables
    vision_text: str | None = None               # Claude vision extraction for chart/map/logo pages


@dataclass(frozen=True)
class ParsedDocument:
    file_path: Path
    pages: list[ParsedPage]

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


def _tag_footnotes(text: str) -> str:
    """Prefix footnote lines with [FOOTNOTE] so retrieval can surface them precisely.

    Footnotes in investment slide decks appear in the bottom ~25% of a slide's
    text and often contain the authoritative qualifier for a figure shown in the
    chart above (e.g., "actual market yield was 5.47%" contradicting the
    normalized figure in the headline).  Tagging them allows BM25 to find them
    when a query asks for a specific date-qualified or scope-qualified number, and
    allows the LLM prompt to treat them as authoritative overrides.
    """
    lines = text.splitlines()
    if len(lines) < 4:
        return text
    # Only scan the bottom quarter (at least the last 5 lines) for footnotes.
    boundary = max(len(lines) * 3 // 4, len(lines) - 5)
    result = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i >= boundary and stripped and _FOOTNOTE_START_RE.match(stripped):
            result.append(f"[FOOTNOTE] {line}")
        else:
            result.append(line)
    return "\n".join(result)


def parse_pdf(path: Path, *, fast_mode: bool = False) -> ParsedDocument:
    """Extract text and tables from a PDF, one entry per page."""
    if not path.exists():
        raise PDFParsingError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise PDFParsingError(f"Expected .pdf extension, got: {path.suffix}")

    try:
        document = fitz.open(path)
    except fitz.FileDataError as exc:
        raise PDFParsingError(f"Corrupt or unreadable PDF: {path.name}. Error: {exc}") from exc

    pages: list[ParsedPage] = []
    try:
        with pdfplumber.open(path) as plumber_doc:
            for index, page in enumerate(document):
                text = page.get_text("text") or ""
                stripped = _tag_footnotes(text.strip())
                slide_title = _extract_slide_title(stripped)

                # Extract structured tables via pdfplumber (preserves row/column layout).
                raw_tables = (
                    []
                    if fast_mode
                    else (plumber_doc.pages[index].extract_tables() or [])
                )
                tables = tuple(
                    tuple(
                        tuple(str(cell or "") for cell in row)
                        for row in table
                        if row
                    )
                    for table in raw_tables
                    if table
                )

                # Vision extraction: for sparse pages (likely chart/map/logo content)
                # pass the rendered page image to Claude vision to read visual elements
                # that text extraction cannot access (bar charts, geographic maps,
                # tenant logo tables, etc.).
                vision_text: str | None = None
                if not fast_mode and len(stripped) < _VISION_CALL_THRESHOLD:
                    vision_text = extract_visual_content(page) or None

                # OCR fallback: for truly blank text layers (scanned PDFs) where
                # neither native text extraction nor vision produced anything useful,
                # run Tesseract OCR directly on the rendered page image.
                # Only activates when pytesseract + tesseract binary are installed.
                if not fast_mode and len(stripped) < 50 and not vision_text:
                    ocr_result = extract_text_with_ocr(page)
                    if ocr_result:
                        stripped = _tag_footnotes(ocr_result)

                pages.append(
                    ParsedPage(
                        page_number=index + 1,
                        text=stripped,
                        char_count=len(stripped),
                        is_likely_visual=len(stripped) < _VISUAL_CHAR_THRESHOLD,
                        slide_title=slide_title,
                        tables=tables,
                        vision_text=vision_text,
                    )
                )
    finally:
        document.close()

    if not any(page.text for page in pages):
        raise PDFParsingError(
            f"No extractable text found in {path.name}. "
            f"This appears to be a fully scanned PDF. "
            f"Install pytesseract + Tesseract (brew install tesseract) to enable OCR."
        )

    return ParsedDocument(file_path=path, pages=pages)


def _extract_slide_title(text: str) -> str | None:
    """Return the first short line as the likely slide title."""
    if not text:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if (
            stripped
            and len(stripped) <= 80
            and len(stripped.split()) <= 12
            and not stripped.isdigit()
            and not _TITLE_EXCLUDE_RE.match(stripped)
        ):
            return stripped
    return None

"""PDF text extraction with page-level metadata and table detection."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz
import pdfplumber

# Lines that look like footnote markers or standalone page numbers should never
# become a slide title (e.g. "(1)", "6", "Page 3").
_TITLE_EXCLUDE_RE = re.compile(
    r"^\(?[\d]+\)?[\.\)]?\s*$|^page\s*\d+$", re.IGNORECASE
)

# Matches the start of a footnote line: "(1) text", "1. text", "1) text",
# "* text", "† text".  Used to detect footnote lines in the bottom portion
# of a slide's extracted text.
_FOOTNOTE_START_RE = re.compile(
    r"^[\(\*†‡§¶]?\d+[\.\)]\s+\S|^\*\s+\S|^†\s+\S"
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


def parse_pdf(path: Path) -> ParsedDocument:
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
                plumber_page = plumber_doc.pages[index]
                raw_tables = plumber_page.extract_tables() or []
                tables = tuple(
                    tuple(
                        tuple(str(cell or "") for cell in row)
                        for row in table
                        if row
                    )
                    for table in raw_tables
                    if table
                )

                pages.append(
                    ParsedPage(
                        page_number=index + 1,
                        text=stripped,
                        char_count=len(stripped),
                        is_likely_visual=len(stripped) < 50,
                        slide_title=slide_title,
                        tables=tables,
                    )
                )
    finally:
        document.close()

    if not any(page.text for page in pages):
        raise PDFParsingError(
            f"No extractable text found in {path.name}. "
            f"This may be a scanned PDF requiring OCR."
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

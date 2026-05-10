"""PDF text extraction with page-level metadata."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz


class PDFParsingError(Exception):
    """Raised when a PDF cannot be opened or parsed."""


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str
    char_count: int
    is_likely_visual: bool


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


def parse_pdf(path: Path) -> ParsedDocument:
    """Extract text from a PDF, one entry per page."""
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
        for index, page in enumerate(document):
            text = page.get_text("text") or ""
            stripped = text.strip()
            pages.append(
                ParsedPage(
                    page_number=index + 1,
                    text=stripped,
                    char_count=len(stripped),
                    is_likely_visual=len(stripped) < 50,
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

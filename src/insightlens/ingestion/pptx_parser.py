"""PPTX text extraction with slide-level metadata."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from insightlens.ingestion.pdf_parser import ParsedDocument, ParsedPage


class PPTXParsingError(Exception):
    """Raised when a PPTX cannot be opened or parsed."""


_SLIDE_RE = re.compile(r"ppt/slides/slide(\d+)\.xml$")
_NOTES_RE = re.compile(r"ppt/notesSlides/notesSlide(\d+)\.xml$")


def parse_pptx(path: Path) -> ParsedDocument:
    """Extract text from a PowerPoint deck, one entry per slide."""
    if not path.exists():
        raise PPTXParsingError(f"PPTX not found: {path}")
    if path.suffix.lower() != ".pptx":
        raise PPTXParsingError(f"Expected .pptx extension, got: {path.suffix}")

    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            slide_names = sorted(
                (name for name in names if _SLIDE_RE.match(name)),
                key=lambda name: int(_SLIDE_RE.match(name).group(1)),  # type: ignore[union-attr]
            )
            if not slide_names:
                raise PPTXParsingError(f"No slides found in {path.name}.")

            notes_by_slide = {
                int(match.group(1)): name
                for name in names
                if (match := _NOTES_RE.match(name))
            }

            pages: list[ParsedPage] = []
            for slide_index, slide_name in enumerate(slide_names, start=1):
                text = _extract_xml_text(archive.read(slide_name))
                slide_match = _SLIDE_RE.match(slide_name)
                slide_number = int(slide_match.group(1)) if slide_match else slide_index
                notes_name = notes_by_slide.get(slide_number)
                if notes_name:
                    notes = _extract_xml_text(archive.read(notes_name))
                    if notes:
                        text = f"{text}\n\nSpeaker notes:\n{notes}".strip()

                pages.append(
                    ParsedPage(
                        page_number=slide_index,
                        text=text,
                        char_count=len(text),
                        is_likely_visual=len(text) < 120,
                        slide_title=_extract_slide_title(text),
                    )
                )
    except zipfile.BadZipFile as exc:
        raise PPTXParsingError(f"Corrupt or unreadable PPTX: {path.name}. Error: {exc}") from exc
    except ElementTree.ParseError as exc:
        raise PPTXParsingError(f"Corrupt slide XML in {path.name}. Error: {exc}") from exc

    if not any(page.text for page in pages):
        raise PPTXParsingError(
            f"No extractable text found in {path.name}. "
            "Image-only slides are not OCR-supported yet."
        )

    return ParsedDocument(file_path=path, pages=pages)


def _extract_xml_text(xml_bytes: bytes) -> str:
    root = ElementTree.fromstring(xml_bytes)
    parts: list[str] = []
    for node in root.iter():
        if node.tag.endswith("}t") or node.tag == "t":
            if node.text and node.text.strip():
                parts.append(node.text.strip())
    return "\n".join(parts)


def _extract_slide_title(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) <= 100 and len(stripped.split()) <= 14:
            return stripped
    return None

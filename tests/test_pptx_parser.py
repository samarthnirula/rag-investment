from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from insightlens.ingestion.pptx_parser import PPTXParsingError, parse_pptx


def _write_pptx(path: Path) -> None:
    slide_xml = """<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:sp><p:txBody><a:p><a:r><a:t>Opening Statement</a:t></a:r></a:p></p:txBody></p:sp>
      <p:sp><p:txBody><a:p><a:r><a:t>Witness timeline and exhibit list</a:t></a:r></a:p></p:txBody></p:sp>
    </p:spTree>
  </p:cSld>
</p:sld>
"""
    notes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<p:notes xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
         xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Ask about Exhibit 4.</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
</p:notes>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("ppt/slides/slide1.xml", slide_xml)
        archive.writestr("ppt/notesSlides/notesSlide1.xml", notes_xml)


def test_parse_pptx_extracts_slide_text_and_notes(tmp_path: Path) -> None:
    deck = tmp_path / "case_deck.pptx"
    _write_pptx(deck)

    parsed = parse_pptx(deck)

    assert parsed.total_pages == 1
    assert parsed.pages[0].page_number == 1
    assert parsed.pages[0].slide_title == "Opening Statement"
    assert "Witness timeline and exhibit list" in parsed.pages[0].text
    assert "Speaker notes:" in parsed.pages[0].text
    assert "Ask about Exhibit 4." in parsed.pages[0].text


def test_parse_pptx_rejects_non_pptx_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "deck.txt"
    file_path.write_text("not a deck")

    with pytest.raises(PPTXParsingError):
        parse_pptx(file_path)

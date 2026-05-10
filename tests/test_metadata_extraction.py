from pathlib import Path

from insightlens.ingestion.document_metadata import extract_metadata


def test_extracts_version_from_filename():
    metadata = extract_metadata(Path("Acme_InvestorDeck_v2.pdf"), "Acme Corp\nInvestor Update")
    assert metadata.company == "Acme"
    assert metadata.version_label == "2"
    assert metadata.document_type == "investor_presentation"


def test_extracts_quarter_year():
    metadata = extract_metadata(Path("Acme_2024_Q3.pdf"), "Acme quarterly report")
    assert metadata.version_label is not None
    assert "2024" in metadata.version_label or "Q3" in metadata.version_label.upper()


def test_unknown_filename_falls_back_to_first_page():
    metadata = extract_metadata(Path("file.pdf"), "Initech Holdings\nStrategy Review 2023")
    assert metadata.company is not None

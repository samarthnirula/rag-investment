from insightlens.analysis.case_insights import extract_case_insights
from insightlens.storage.chunk_repository import RetrievedChunk


def _chunk(text: str, chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc1",
        file_name="complaint.pdf",
        company="Matter",
        version_label=None,
        page_number=1,
        chunk_text=text,
        similarity=0.0,
    )


def test_extracts_timeline_entities_and_conflict_candidates():
    chunks = [
        _chunk(
            "On January 5, 2024, Jane Smith signed the agreement. "
            "The settlement amount was $25,000.", "c1"
        ),
        _chunk(
            "On January 5, 2024, Jane Smith signed the agreement. "
            "The settlement amount was $30,000.", "c2"
        ),
    ]

    result = extract_case_insights(chunks)

    assert result.timeline
    assert any(item["title"] == "Jane Smith" for item in result.entities)
    assert result.contradictions
    assert "Client Summary" in result.client_summary

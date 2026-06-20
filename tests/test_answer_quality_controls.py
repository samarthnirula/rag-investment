import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from insightlens.storage.chunk_repository import RetrievedChunk


def _source(
    *,
    document_id: str = "doc1",
    source_type: str = "document",
    file_name: str = "source.pdf",
    page: int = 1,
    text: str = "Relevant legal issue text.",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{document_id}-{page}-{source_type}",
        document_id=document_id,
        file_name=file_name,
        company="Matter",
        version_label=None,
        page_number=page,
        chunk_text=text,
        similarity=1.0,
        source_type=source_type,
    )


def test_broad_legal_question_caps_high_confidence_for_single_primary_source():
    import backend.demo_router as quality

    confidence = {"score": 5, "rating": "High", "rationale": "Directly supported."}
    capped = quality._cap_confidence_for_coverage(
        "What are the key legal issues in this matter?",
        [_source()],
        confidence,
    )

    assert capped is not None
    assert capped["score"] == 3
    assert "one or fewer primary documents" in capped["rationale"]


def test_broad_legal_question_gets_scope_note_for_thin_retrieval():
    import backend.demo_router as quality

    note = quality._scope_note_for_coverage(
        "What are the key legal issues in this matter?",
        [_source()],
    )

    assert note is not None
    assert "limited synthesis" in note
    assert "complete issue list" in note


def test_secondary_only_confidence_caps_at_three():
    import backend.demo_router as quality

    confidence = {"score": 5, "rating": "High", "rationale": "Supported."}
    capped = quality._cap_confidence_for_coverage(
        "Who is Virginia Giuffre?",
        [_source(source_type="demo_summary", file_name="Shared demo overview")],
        confidence,
    )

    assert capped is not None
    assert capped["score"] == 3
    assert "secondary demo summary context" in capped["rationale"]


def test_source_excerpt_uses_meaningful_sentence_not_generic_terms():
    import backend.demo_router as quality

    text = (
        "This matter includes many public records. "
        "The contemporaneous records show that USAO managers had concerns about legal issues, "
        "witness credibility, and the impact of a trial on victims. "
        "Other background material mentions Epstein repeatedly."
    )

    excerpt = quality._best_source_excerpt(
        text,
        "What are the key legal issues in the Epstein matter?",
    )

    assert "USAO managers had concerns" in excerpt
    assert excerpt != "key"
    assert excerpt != "Epstein"


def test_dedupes_url_encoded_copy_suffix_sources():
    import backend.demo_router as quality

    sources = [
        _source(
            document_id="doc-a",
            file_name="2020.11%20DOJ%20Office%20of%20Professional%20Responsibility%20Report%20Executive%20Summary.pdf",
            page=1,
        ),
        _source(
            document_id="doc-b",
            file_name="2020.11%20DOJ%20Office%20of%20Professional%20Responsibility%20Report%20Executive%20Summary (1).pdf",
            page=1,
        ),
        _source(
            document_id="doc-c",
            file_name="2020.11 DOJ Office of Professional Responsibility Report Executive Summary.pdf",
            page=1,
        ),
    ]

    deduped = quality._dedupe_comparable_sources(sources)

    assert len(deduped) == 1
    assert deduped[0].document_id == "doc-a"


def test_demo_source_citation_label_is_stable_legal_style():
    import backend.demo_router as quality

    label = quality._citation_label(
        _source(
            source_type="demo_summary",
            file_name="Shared demo overview and timeline",
            page=1,
        )
    )

    assert label == "Atticus demo Epstein matter summary, at 1"
    assert "DEMO SUMMARY" not in label


def test_workspace_note_reports_retrieved_source_universe_and_jurisdiction():
    import backend.demo_router as quality

    note = quality._workspace_note(
        "What are the key legal issues?",
        [
            _source(
                file_name="2020.11 DOJ Office of Professional Responsibility Report Executive Summary.pdf",
                text="The non-prosecution agreement and Acosta decisions were reviewed.",
            ),
            _source(source_type="demo_summary", file_name="Shared demo overview and timeline"),
        ],
    )

    assert "1 primary source chunk" in note
    assert "1 secondary/generated context source" in note
    assert "S.D. Fla." in note


def test_lawyer_followups_are_actionable_searches():
    import backend.demo_router as quality

    answer = quality._ensure_lawyer_followups(
        "## Direct answer\nThe record supports limited issues.",
        "What are the key legal issues in the Epstein matter?",
        [_source()],
    )

    assert "## Actionable follow-up searches" in answer
    assert "Search for: Giuffre v. Maxwell" in answer
    assert "Pull:" in answer

from __future__ import annotations

from insightlens.analysis.case_intelligence import build_case_overview, build_case_timeline
from insightlens.retrieval.hybrid_search import _expand_query
from insightlens.storage.chunk_repository import RetrievedChunk


def _chunk(text: str, page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"doc-{page}",
        document_id="doc",
        file_name="madeline-case.pdf",
        company=None,
        version_label=None,
        page_number=page,
        chunk_text=text,
        similarity=0.0,
    )


def test_case_overview_extracts_roles_and_issues() -> None:
    chunks = [
        _chunk(
            "Victim Madeline Deparde was killed in a bicycle collision on 08/17/06. "
            "Defendant Scott Zion was charged with vehicular manslaughter and reckless driving. "
            "Sunnyview Police Department prepared an accident report with photographs."
        )
    ]

    overview = build_case_overview(chunks, "Sample Case")

    party_text = " ".join(f"{p['name']} {p['role']}" for p in overview["parties"])
    assert "Madeline Deparde" in party_text
    assert "Scott Zion" in party_text
    assert overview["matter_type"] == "Criminal / investigative matter"
    assert any("charged" in issue.lower() or "offense" in issue.lower() for issue in overview["key_issues"])


def test_case_timeline_uses_dates_from_chunks() -> None:
    chunks = [
        _chunk("The accident occurred on 08/17/06 near Sunnyview Grocery Depot and killed Madeline Deparde.")
    ]

    events = build_case_timeline(chunks)

    assert events
    assert events[0]["date"] == "08/17/06"
    assert "accident" in events[0]["description"].lower()


def test_query_expansion_maps_lawyer_language_to_document_language() -> None:
    expanded = _expand_query("who is the suspect in the case?")

    assert "defendant" in expanded
    assert "accused" in expanded
    assert "charged" in expanded

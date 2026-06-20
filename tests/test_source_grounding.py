from insightlens.generation.prompts import build_user_prompt
from insightlens.demo.epstein_people import epstein_people_context_text
from insightlens.storage.chunk_repository import RetrievedChunk


def _chunk(
    text: str,
    *,
    source_type: str = "document",
    file_name: str = "evidence.pdf",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{source_type}-1",
        document_id=f"{source_type}-doc",
        file_name=file_name,
        company="Matter",
        version_label=None,
        page_number=1,
        chunk_text=text,
        similarity=1.0,
        document_type="test",
        source_type=source_type,
    )


def test_prompt_labels_document_evidence_source_type():
    prompt = build_user_prompt(
        "What happened?",
        [_chunk("Jane Smith signed the agreement in the Southern District of New York on January 5, 2024.")],
    )

    assert "source type: DOCUMENT EVIDENCE" in prompt
    assert "citation label: evidence, at 1" in prompt
    assert "jurisdiction: S.D.N.Y." in prompt
    assert "Jane Smith signed the agreement" in prompt


def test_prompt_labels_secondary_demo_summary_and_caps_confidence_instruction():
    prompt = build_user_prompt(
        "Who is Virginia Giuffre?",
        [
            _chunk(
                "Virginia Giuffre: Survivor. Primary survivor-witness in the public Epstein record.",
                source_type="demo_summary",
                file_name="Shared demo overview and timeline",
            )
        ],
    )

    assert "source type: DEMO SUMMARY" in prompt
    assert "secondary context" in prompt
    assert "confidence no higher than 3" in prompt
    assert "Virginia Giuffre" in prompt


def test_prompt_labels_public_people_context_and_contains_bill_clinton():
    prompt = build_user_prompt(
        "Who is Bill Clinton in the Epstein matter?",
        [
            _chunk(
                epstein_people_context_text(),
                source_type="public_context",
                file_name="Public reference people index for Epstein matter",
            )
        ],
    )

    assert "source type: PUBLIC REFERENCE CONTEXT" in prompt
    assert "Bill Clinton" in prompt
    assert "A person being named" in prompt
    assert "committed a crime" in prompt
    assert "secondary context" in prompt


def test_prompt_requires_lawyer_format_and_evidence_profile():
    prompt = build_user_prompt(
        "What are the key legal issues?",
        [_chunk("The report discusses prosecutorial discretion and the NPA.")],
    )

    assert "Evidence coverage profile" in prompt
    assert "## Gaps and uncertainties" in prompt
    assert prompt.index("## Gaps and uncertainties") < prompt.index("## Direct answer")
    assert "## Evidence by jurisdiction" in prompt
    assert "## Actionable follow-up searches" in prompt
    assert "## Workspace note" in prompt
    assert "broad legal synthesis based on only one primary document is maximum 3" in prompt

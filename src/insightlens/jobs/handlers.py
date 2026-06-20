"""Job handler implementations registered with BackgroundJobRunner."""
from __future__ import annotations

import logging

from insightlens.analysis.case_insights import extract_case_insights
from insightlens.config import AppConfig
from insightlens.storage.chunk_repository import ChunkRepository
from insightlens.storage.insights_repository import InsightsRepository
from insightlens.storage.snowflake_client import open_connection

_log = logging.getLogger(__name__)

JOB_TYPE_EXTRACT_INSIGHTS = "extract_case_insights"
JOB_TYPE_INGEST_PDF = "ingest_pdf"


def make_extract_insights_handler(cfg: AppConfig):
    """Return a handler for the extract_case_insights job type.

    Tries to use the LLM client for verification; falls back to deterministic
    extraction if the API key is missing or invalid.
    """
    def handler(payload: dict, job_id: str) -> None:
        case_id = payload["case_id"]
        doc_ids = payload["doc_ids"]
        user_id = payload.get("user_id")

        _log.info("Extracting insights for case %s (%d docs)", case_id, len(doc_ids))

        # Try to build an LLM client for verification — graceful fallback if key invalid
        llm = None
        try:
            from insightlens.generation.llm_client import ClaudeClient
            llm = ClaudeClient(
                api_key=cfg.anthropic_api_key,
                model=cfg.generation_model,
                max_tokens=1024,
            )
        except Exception as exc:
            _log.warning("LLM client unavailable, using deterministic-only insights: %s", exc)

        with open_connection(cfg.db) as conn:
            chunks = ChunkRepository(conn).get_chunks_for_documents(doc_ids)
            result = extract_case_insights(chunks, llm_client=llm)

            repo = InsightsRepository(conn)
            repo.replace_case_insights(case_id, "timeline", result.timeline)
            repo.replace_case_insights(case_id, "entity", result.entities)
            repo.replace_case_insights(case_id, "contradiction", result.contradictions)
            repo.create_artifact(
                user_id=user_id or "",
                case_id=case_id,
                artifact_type="client_summary",
                title="Client Summary",
                content=result.client_summary,
            )

        _log.info(
            "Insights done for case %s: %d timeline, %d entities, %d contradictions",
            case_id, len(result.timeline), len(result.entities), len(result.contradictions),
        )

    return handler


def make_ingest_pdf_handler(cfg: AppConfig):
    """Return a handler that dispatches ingest_pdf jobs to the Celery worker.

    Kept for backward compatibility with the old BackgroundJobRunner interface
    (used by the Streamlit UI). The FastAPI bulk-upload endpoint now dispatches
    directly via ``ingest_pdf_task.delay()``.
    """
    def handler(payload: dict, job_id: str) -> None:
        from insightlens.jobs.tasks import ingest_pdf_task

        file_path = payload["file_path"]
        case_id = payload.get("case_id")
        user_id = payload.get("user_id")
        filename = payload.get("file_name", "")

        ingest_pdf_task.delay(job_id, file_path, case_id, user_id, filename)
        _log.info("ingest_pdf job %s dispatched to Celery worker", job_id)

    return handler


def register_all_handlers(runner, cfg: AppConfig) -> None:
    """Register all known job handlers on the given runner."""
    runner.register(JOB_TYPE_EXTRACT_INSIGHTS, make_extract_insights_handler(cfg))
    runner.register(JOB_TYPE_INGEST_PDF, make_ingest_pdf_handler(cfg))

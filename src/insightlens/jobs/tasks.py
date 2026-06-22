"""Celery task definitions for background processing."""
from __future__ import annotations

import logging
import random
import time
import uuid
from pathlib import Path

from insightlens.jobs.celery_app import celery_app

_log = logging.getLogger(__name__)

# How long to wait after an ingest before regenerating the case overview/
# timeline. Bulk uploads enqueue one ingest_pdf_task per file with no signal
# for "this was the last file in the batch" — so each successful ingest
# schedules generation after this delay, and a Redis token lets a later
# ingest cancel an earlier still-pending one. This collapses an N-file batch
# into a single regeneration shortly after the last file finishes, instead
# of N redundant Claude calls.
_CASE_REGEN_DEBOUNCE_SECONDS = 30


def _schedule_case_regeneration(case_id: str, user_id: str) -> None:
    """Debounce overview+timeline regeneration across a burst of ingests."""
    token = uuid.uuid4().hex
    try:
        import redis as _redis_mod
        import os
        rc = _redis_mod.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            socket_connect_timeout=1, socket_timeout=1, decode_responses=True,
        )
        rc.set(f"case_regen_token:{case_id}", token, ex=_CASE_REGEN_DEBOUNCE_SECONDS + 60)
    except Exception as exc:
        _log.warning("Could not set debounce token for case %s, regenerating immediately: %s", case_id, exc)
        token = None

    generate_case_overview_task.apply_async(
        args=[case_id, user_id, token], countdown=_CASE_REGEN_DEBOUNCE_SECONDS,
    )
    generate_case_timeline_task.apply_async(
        args=[case_id, user_id, token], countdown=_CASE_REGEN_DEBOUNCE_SECONDS,
    )


def _is_latest_regen_token(case_id: str, token: str | None) -> bool:
    """Return True if `token` is still the most recently scheduled one (or
    debouncing is unavailable, in which case we fail open and proceed)."""
    if token is None:
        return True
    try:
        import redis as _redis_mod
        import os
        rc = _redis_mod.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            socket_connect_timeout=1, socket_timeout=1, decode_responses=True,
        )
        current = rc.get(f"case_regen_token:{case_id}")
        return current is None or current == token
    except Exception:
        return True


@celery_app.task(
    bind=True,
    name="insightlens.jobs.tasks.ingest_pdf_task",
    max_retries=3,
    default_retry_delay=60,
    queue="ingest",
)
def ingest_pdf_task(
    self,
    job_id: str,
    file_path: str,
    case_id: str | None,
    user_id: str | None,
    filename: str,
) -> dict:
    """Ingest a single PDF into the database and link it to a case.

    Args:
        job_id:    DB record ID in ``background_jobs``.
        file_path: Absolute path to the temporary PDF on disk.
        case_id:   Case to link the ingested document to (may be None).
        user_id:   Owning Firebase UID (may be None for system docs).
        filename:  Original file name (used only for logging / error messages).

    Retries up to 3 times with a 60-second delay on any exception.
    The temp file is deleted after the final attempt (success or failure).
    """
    # Import everything inside the task so the module can be imported cheaply
    # by the Celery worker at startup.
    from insightlens.config import load_config
    from insightlens.embeddings.embedder import Embedder
    from insightlens.ingestion.ingest_service import IngestService
    from insightlens.storage.cases_repository import CasesRepository
    from insightlens.storage.jobs_repository import JobsRepository
    from insightlens.storage.snowflake_client import open_connection

    cfg = load_config()
    path = Path(file_path)

    # Mark as running in the DB on the first attempt only.
    if self.request.retries == 0:
        try:
            with open_connection(cfg.db) as conn:
                JobsRepository(conn).mark_running(job_id)
        except Exception as exc:
            _log.warning("ingest_pdf_task: could not mark job %s running: %s", job_id, exc)

    try:
        embedder = Embedder(model=cfg.embedding_model)
        svc = IngestService(cfg=cfg, embedder=embedder)
        result = svc.ingest(path, user_id=user_id, original_file_name=filename)

        if result.error and not result.skipped:
            raise RuntimeError(result.error)

        if case_id and result.document_id and not result.error:
            with open_connection(cfg.db) as conn:
                CasesRepository(conn).add_document_to_case(case_id, result.document_id)

        with open_connection(cfg.db) as conn:
            JobsRepository(conn).mark_completed(job_id)

        path.unlink(missing_ok=True)

        _log.info(
            "ingest_pdf_task completed: job_id=%s file=%s chunks=%d",
            job_id, filename, result.chunks_inserted,
        )

        # After every successful ingest, (re)schedule overview/timeline
        # generation so they stay current as documents are added to the
        # case. Debounced so a multi-file bulk upload collapses into one
        # regeneration shortly after the last file finishes.
        if case_id and user_id and result.chunks_inserted > 0:
            try:
                _schedule_case_regeneration(case_id, user_id)
            except Exception as exc:
                _log.warning("ingest_pdf_task: could not schedule AI tasks for case %s: %s", case_id, exc)

        return {"job_id": job_id, "chunks_inserted": result.chunks_inserted}

    except Exception as exc:
        _log.error(
            "ingest_pdf_task failed (attempt %d/%d): job_id=%s file=%s err=%s",
            self.request.retries + 1, self.max_retries + 1, job_id, filename, exc,
        )
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            # All retries exhausted — permanently fail the DB record.
            try:
                with open_connection(cfg.db) as conn:
                    JobsRepository(conn).mark_failed(job_id, str(exc)[:2000])
            except Exception as db_exc:
                _log.error("ingest_pdf_task: could not mark job %s failed: %s", job_id, db_exc)
            path.unlink(missing_ok=True)
            raise


@celery_app.task(
    bind=True,
    name="insightlens.jobs.tasks.generate_case_overview_task",
    max_retries=2,
    default_retry_delay=30,
    queue="ingest",
)
def generate_case_overview_task(self, case_id: str, user_id: str, _debounce_token: str | None = None) -> dict:
    """Generate an AI case overview (summary, parties, key issues) and store it."""
    import json
    import re
    from insightlens.config import load_config
    from insightlens.generation.llm_client import ClaudeClient
    from insightlens.storage.chunk_repository import ChunkRepository
    from insightlens.storage.snowflake_client import open_connection

    if not _is_latest_regen_token(case_id, _debounce_token):
        return {"skipped": True, "reason": "superseded by a later ingest"}

    cfg = load_config()
    try:
        with open_connection(cfg.db) as conn:
            cur = conn.cursor()
            cur.execute("SELECT case_name FROM cases WHERE case_id = %s", (case_id,))
            row = cur.fetchone()
            cur.close()
            if not row:
                return {"skipped": True, "reason": "case not found"}
            case_name = row[0]
            chunks = ChunkRepository(conn).get_chunks_for_case(case_id, user_id, limit=80)

        if not chunks:
            return {"skipped": True, "reason": "no chunks yet"}

        combined = "\n\n---\n\n".join(
            f"[{c.file_name} p.{c.page_number}]\n{c.chunk_text[:1200]}"
            for c in chunks[:40]
        )

        llm = ClaudeClient(api_key=cfg.anthropic_api_key, model=cfg.generation_model)
        prompt = (
            f'Analyze these legal case documents for case "{case_name}".\n\n'
            f"{combined}\n\n"
            "Return ONLY a JSON object (no markdown) with these keys:\n"
            '{"summary":"2-3 paragraph case summary","parties":[{"role":"Plaintiff","name":"..."}],'
            '"key_issues":["issue 1","issue 2"],"jurisdiction":"court/state or null","matter_type":"e.g. Civil Litigation"}'
        )
        response = llm.generate(
            "You are a legal AI. Extract structured information from legal documents. Reply with raw JSON only.",
            prompt,
        )

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise RuntimeError("No JSON in overview response")
        data = json.loads(json_match.group())

        with open_connection(cfg.db) as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO case_overviews
                       (case_id, user_id, summary, parties, key_issues, jurisdiction, matter_type)
                   VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
                   ON CONFLICT (case_id) DO UPDATE SET
                       summary=EXCLUDED.summary, parties=EXCLUDED.parties,
                       key_issues=EXCLUDED.key_issues, jurisdiction=EXCLUDED.jurisdiction,
                       matter_type=EXCLUDED.matter_type, generated_at=NOW()""",
                (
                    case_id, user_id,
                    data.get("summary", ""),
                    json.dumps(data.get("parties", [])),
                    json.dumps(data.get("key_issues", [])),
                    data.get("jurisdiction"),
                    data.get("matter_type"),
                ),
            )
            cur.execute(
                "UPDATE cases SET overview_generated = TRUE WHERE case_id = %s", (case_id,)
            )
            cur.close()
            conn.commit()

        _log.info("generate_case_overview_task completed case_id=%s", case_id)
        return {"case_id": case_id, "status": "overview_generated"}

    except Exception as exc:
        _log.error("generate_case_overview_task failed case_id=%s: %s", case_id, exc)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            raise


@celery_app.task(
    bind=True,
    name="insightlens.jobs.tasks.generate_case_timeline_task",
    max_retries=2,
    default_retry_delay=30,
    queue="ingest",
)
def generate_case_timeline_task(self, case_id: str, user_id: str, _debounce_token: str | None = None) -> dict:
    """Generate a chronological event timeline from case documents and store it."""
    import json
    import re
    from insightlens.config import load_config
    from insightlens.generation.llm_client import ClaudeClient
    from insightlens.storage.chunk_repository import ChunkRepository
    from insightlens.storage.snowflake_client import open_connection

    if not _is_latest_regen_token(case_id, _debounce_token):
        return {"skipped": True, "reason": "superseded by a later ingest"}

    cfg = load_config()
    try:
        with open_connection(cfg.db) as conn:
            cur = conn.cursor()
            cur.execute("SELECT case_name FROM cases WHERE case_id = %s", (case_id,))
            row = cur.fetchone()
            cur.close()
            if not row:
                return {"skipped": True, "reason": "case not found"}
            case_name = row[0]
            chunks = ChunkRepository(conn).get_chunks_for_case(case_id, user_id, limit=80)

        if not chunks:
            return {"skipped": True, "reason": "no chunks yet"}

        combined = "\n\n---\n\n".join(
            f"[{c.file_name} p.{c.page_number}]\n{c.chunk_text[:1200]}"
            for c in chunks[:40]
        )

        llm = ClaudeClient(api_key=cfg.anthropic_api_key, model=cfg.generation_model)
        prompt = (
            f'Extract a chronological timeline of key events from the case "{case_name}".\n\n'
            f"{combined}\n\n"
            "Return ONLY a JSON array (no markdown) of events sorted by date:\n"
            '[{"date":"YYYY-MM-DD or month/year","title":"short event title",'
            '"description":"1-2 sentence description","source_doc":"filename",'
            '"page":1}]'
        )
        response = llm.generate(
            "You are a legal AI. Extract a chronological event timeline from legal documents. Reply with a raw JSON array only.",
            prompt,
        )

        arr_match = re.search(r"\[.*\]", response, re.DOTALL)
        if not arr_match:
            raise RuntimeError("No JSON array in timeline response")
        events = json.loads(arr_match.group())

        with open_connection(cfg.db) as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO case_timelines (case_id, user_id, events)
                   VALUES (%s, %s, %s::jsonb)
                   ON CONFLICT (case_id) DO UPDATE SET events=EXCLUDED.events, generated_at=NOW()""",
                (case_id, user_id, json.dumps(events)),
            )
            cur.execute(
                "UPDATE cases SET timeline_generated = TRUE WHERE case_id = %s", (case_id,)
            )
            cur.close()
            conn.commit()

        _log.info("generate_case_timeline_task completed case_id=%s events=%d", case_id, len(events))
        return {"case_id": case_id, "status": "timeline_generated", "events": len(events)}

    except Exception as exc:
        _log.error("generate_case_timeline_task failed case_id=%s: %s", case_id, exc)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            raise

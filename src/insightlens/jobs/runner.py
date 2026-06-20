"""Thread-based background job runner.

Starts a single daemon thread that polls the background_jobs table and
dispatches work to registered handlers.  Designed to run inside the
Streamlit process — no separate worker needed.

Usage:
    from insightlens.jobs.runner import get_runner
    runner = get_runner(cfg)   # idempotent — returns same instance per process
    runner.start()             # no-op if already running

Handlers are registered with:
    runner.register("job_type", handler_fn)

where handler_fn(payload: dict, job_id: str) -> None.
Any exception raised by a handler marks the job as 'failed'.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Callable

from insightlens.config import AppConfig
from insightlens.storage.jobs_repository import JobsRepository
from insightlens.storage.snowflake_client import open_connection

_log = logging.getLogger(__name__)

_POLL_INTERVAL = 3   # seconds between DB polls when queue may have work
_IDLE_INTERVAL = 10  # seconds between polls when queue was empty

Handler = Callable[[dict, str], None]


class BackgroundJobRunner:
    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._handlers: dict[str, Handler] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def register(self, job_type: str, handler: Handler) -> None:
        self._handlers[job_type] = handler

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="job-runner", daemon=True
        )
        self._thread.start()
        _log.info("BackgroundJobRunner started")

    def stop(self) -> None:
        self._stop.set()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                did_work = self._tick()
            except Exception:
                _log.exception("Job runner tick error")
                did_work = False
            wait = _POLL_INTERVAL if did_work else _IDLE_INTERVAL
            self._stop.wait(wait)

    def _tick(self) -> bool:
        """Claim and run one job. Returns True if a job was processed."""
        job_types = list(self._handlers.keys()) or None
        try:
            with open_connection(self._cfg.db) as conn:
                repo = JobsRepository(conn)
                job = repo.claim_next(job_types)
                if not job:
                    return False
                _log.info("Running job %s type=%s", job.job_id, job.job_type)
                handler = self._handlers.get(job.job_type)
                if not handler:
                    repo.mark_failed(job.job_id, f"No handler for job_type={job.job_type}")
                    return True
                try:
                    handler(job.payload, job.job_id)
                    repo.mark_completed(job.job_id)
                    _log.info("Job %s completed", job.job_id)
                except Exception as exc:
                    err = traceback.format_exc()
                    repo.mark_failed(job.job_id, err[:2000])
                    _log.error("Job %s failed: %s", job.job_id, exc)
        except Exception:
            _log.exception("DB error in job runner tick")
        return True


# ── Process-level singleton ────────────────────────────────────────────────────

_runner_lock = threading.Lock()
_runner: BackgroundJobRunner | None = None


def get_runner(cfg: AppConfig) -> BackgroundJobRunner:
    """Return the process-level runner, creating it if needed."""
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = BackgroundJobRunner(cfg)
        return _runner

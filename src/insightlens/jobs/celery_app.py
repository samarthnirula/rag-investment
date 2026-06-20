"""Celery application instance for background task processing.

Broker and result backend both use Redis.  The broker URL is read from
``REDIS_URL`` (default: ``redis://localhost:6379/0``).  The result backend
uses Redis DB 1 on the same host (auto-derived from REDIS_URL).
"""
from __future__ import annotations

import os

from celery import Celery

_BROKER_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# Strip the trailing DB number and use DB 1 for the result backend
_base = _BROKER_URL.rsplit("/", 1)[0]
_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", f"{_base}/1")

celery_app = Celery(
    "insightlens",
    broker=_BROKER_URL,
    backend=_RESULT_BACKEND,
    include=["insightlens.jobs.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "insightlens.jobs.tasks.ingest_pdf_task": {"queue": "ingest"},
    },
)

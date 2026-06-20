"""Database-backed background job queue primitives."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg2

from insightlens.storage.snowflake_client import SnowflakeConnection


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    job_type: str
    status: str
    user_id: str | None
    case_id: str | None
    payload: dict
    error: str | None
    created_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None


class JobsRepository:
    def __init__(self, conn: SnowflakeConnection) -> None:
        self._conn = conn

    def enqueue(
        self,
        *,
        job_type: str,
        user_id: str | None = None,
        case_id: str | None = None,
        payload: dict | None = None,
    ) -> str:
        job_id = uuid.uuid4().hex[:16]
        cur = self._conn.cursor()
        try:
            cur.execute(
                """INSERT INTO background_jobs
                   (job_id, job_type, user_id, case_id, payload_json)
                   VALUES (%s, %s, %s, %s, %s)""",
                (job_id, job_type, user_id, case_id, json.dumps(payload or {})),
            )
        except psycopg2.Error:
            pass
        finally:
            cur.close()
        return job_id

    def claim_next(self, job_types: list[str] | None = None) -> JobRecord | None:
        """Atomically claim the oldest queued job. Returns None if queue is empty.

        Uses SELECT … FOR UPDATE SKIP LOCKED so multiple workers never double-claim.
        """
        cur = self._conn.cursor()
        try:
            type_filter = ""
            params: list[Any] = []
            if job_types:
                placeholders = ",".join(["%s"] * len(job_types))
                type_filter = f" AND job_type IN ({placeholders})"
                params.extend(job_types)

            cur.execute(
                f"""SELECT job_id, job_type, status, user_id, case_id,
                           payload_json, error, created_at, started_at, finished_at
                    FROM background_jobs
                    WHERE status = 'queued'{type_filter}
                    ORDER BY created_at ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED""",
                tuple(params),
            )
            row = cur.fetchone()
            if not row:
                return None
            job_id = row[0]
            cur.execute(
                "UPDATE background_jobs SET status = 'running', started_at = NOW() WHERE job_id = %s",
                (job_id,),
            )
            return _row_to_job(row)
        except psycopg2.Error:
            return None
        finally:
            cur.close()

    def mark_running(self, job_id: str) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "UPDATE background_jobs SET status = 'running', started_at = NOW() WHERE job_id = %s",
                (job_id,),
            )
        except psycopg2.Error:
            pass
        finally:
            cur.close()

    def mark_completed(self, job_id: str) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "UPDATE background_jobs SET status = 'completed', finished_at = NOW() WHERE job_id = %s",
                (job_id,),
            )
        except psycopg2.Error:
            pass
        finally:
            cur.close()

    def mark_failed(self, job_id: str, error: str) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "UPDATE background_jobs SET status = 'failed', error = %s, finished_at = NOW() WHERE job_id = %s",
                (error[:2000], job_id),
            )
        except psycopg2.Error:
            pass
        finally:
            cur.close()

    def get_job(self, job_id: str) -> JobRecord | None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT job_id, job_type, status, user_id, case_id,
                          payload_json, error, created_at, started_at, finished_at
                   FROM background_jobs WHERE job_id = %s""",
                (job_id,),
            )
            row = cur.fetchone()
            return _row_to_job(row) if row else None
        except psycopg2.Error:
            return None
        finally:
            cur.close()

    def get_jobs_for_case(self, case_id: str, job_type: str | None = None) -> list[JobRecord]:
        cur = self._conn.cursor()
        try:
            if job_type:
                cur.execute(
                    """SELECT job_id, job_type, status, user_id, case_id,
                              payload_json, error, created_at, started_at, finished_at
                       FROM background_jobs
                       WHERE case_id = %s AND job_type = %s
                       ORDER BY created_at ASC""",
                    (case_id, job_type),
                )
            else:
                cur.execute(
                    """SELECT job_id, job_type, status, user_id, case_id,
                              payload_json, error, created_at, started_at, finished_at
                       FROM background_jobs
                       WHERE case_id = %s
                       ORDER BY created_at ASC""",
                    (case_id,),
                )
            return [_row_to_job(row) for row in cur.fetchall()]
        except psycopg2.Error:
            return []
        finally:
            cur.close()

    def latest_job_for_case(self, case_id: str, job_type: str) -> JobRecord | None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT job_id, job_type, status, user_id, case_id,
                          payload_json, error, created_at, started_at, finished_at
                   FROM background_jobs
                   WHERE case_id = %s AND job_type = %s
                   ORDER BY created_at DESC LIMIT 1""",
                (case_id, job_type),
            )
            row = cur.fetchone()
            return _row_to_job(row) if row else None
        except psycopg2.Error:
            return None
        finally:
            cur.close()

    def stats(self) -> dict[str, int]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT status, COUNT(*) FROM background_jobs GROUP BY status"
            )
            return {row[0]: int(row[1]) for row in cur.fetchall()}
        except psycopg2.Error:
            return {}
        finally:
            cur.close()


def _row_to_job(row: tuple) -> JobRecord:
    return JobRecord(
        job_id=row[0],
        job_type=row[1],
        status=row[2],
        user_id=row[3],
        case_id=row[4],
        payload=json.loads(row[5] or "{}"),
        error=row[6],
        created_at=row[7],
        started_at=row[8],
        finished_at=row[9],
    )

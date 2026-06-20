"""Usage and cost telemetry repositories."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

import psycopg2

from insightlens.storage.snowflake_client import SnowflakeConnection


@dataclass(frozen=True)
class UploadEvent:
    upload_id: str
    user_id: str
    document_id: str | None
    file_name: str
    file_size_bytes: int
    page_count: int
    chunks_inserted: int
    estimated_cost_usd: float
    uploaded_at: datetime | None = None


class UsageRepository:
    def __init__(self, conn: SnowflakeConnection) -> None:
        self._conn = conn

    def log_upload(
        self,
        *,
        user_id: str,
        document_id: str,
        file_name: str,
        file_size_bytes: int,
        page_count: int,
        chunks_inserted: int,
        estimated_cost_usd: float,
    ) -> str:
        upload_id = uuid.uuid4().hex[:16]
        cur = self._conn.cursor()
        try:
            cur.execute(
                """INSERT INTO upload_events
                   (upload_id, user_id, document_id, file_name, file_size_bytes,
                    page_count, chunks_inserted, estimated_cost_usd)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    upload_id,
                    user_id,
                    document_id,
                    file_name[:500],
                    file_size_bytes,
                    page_count,
                    chunks_inserted,
                    estimated_cost_usd,
                ),
            )
        except psycopg2.Error:
            pass
        finally:
            cur.close()
        return upload_id

    def count_uploads_this_month(self, user_id: str) -> int:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT COUNT(*)
                   FROM upload_events
                   WHERE user_id = %s
                     AND DATE_TRUNC('month', uploaded_at) = DATE_TRUNC('month', CURRENT_DATE)""",
                (user_id,),
            )
            return int(cur.fetchone()[0] or 0)
        except psycopg2.Error:
            return 0
        finally:
            cur.close()

    def estimated_upload_cost_this_month(self, user_id: str) -> float:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT COALESCE(SUM(estimated_cost_usd), 0)
                   FROM upload_events
                   WHERE user_id = %s
                     AND DATE_TRUNC('month', uploaded_at) = DATE_TRUNC('month', CURRENT_DATE)""",
                (user_id,),
            )
            return float(cur.fetchone()[0] or 0)
        except psycopg2.Error:
            return 0.0
        finally:
            cur.close()

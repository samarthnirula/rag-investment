"""Immutable query audit log — every user query + response is recorded."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import psycopg2

from insightlens.storage.snowflake_client import SnowflakeConnection


@dataclass(frozen=True)
class QueryLogEntry:
    log_id: str
    user_id: str | None
    page: str
    query_text: str
    chunks_retrieved: int
    model_used: str
    response_length: int
    logged_at: datetime | None = None


class AuditRepository:
    def __init__(self, conn: SnowflakeConnection) -> None:
        self._conn = conn

    def log_query(
        self,
        user_id: str | None,
        page: str,
        query_text: str,
        chunks_retrieved: int,
        model_used: str,
        response_length: int,
        estimated_cost_usd: float | None = None,
    ) -> str:
        log_id = uuid.uuid4().hex[:16]
        cur = self._conn.cursor()
        try:
            cur.execute(
                """INSERT INTO query_log
                   (log_id, user_id, page, query_text, chunks_retrieved,
                    model_used, response_length, estimated_cost_usd)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (log_id, user_id, page, query_text[:4000], chunks_retrieved,
                 model_used, response_length, estimated_cost_usd),
            )
        except psycopg2.Error:
            # Older databases may not have the cost column yet. Keep logging.
            try:
                cur.execute(
                    """INSERT INTO query_log
                       (log_id, user_id, page, query_text, chunks_retrieved,
                        model_used, response_length)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (log_id, user_id, page, query_text[:4000], chunks_retrieved,
                     model_used, response_length),
                )
            except psycopg2.Error:
                pass  # audit log failures must never crash the app
        finally:
            cur.close()
        return log_id

    def get_user_stats(self, user_id: str) -> dict:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT
                     COUNT(*) AS total_queries,
                     COUNT(*) FILTER (WHERE DATE_TRUNC('month', logged_at) = DATE_TRUNC('month', CURRENT_DATE)) AS this_month,
                     COUNT(*) FILTER (WHERE logged_at::date = CURRENT_DATE) AS today,
                     SUM(chunks_retrieved) AS total_chunks_retrieved,
                     COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
                   FROM query_log WHERE user_id = %s""",
                (user_id,),
            )
            row = cur.fetchone()
            return {
                "total_queries":      int(row[0] or 0),
                "queries_this_month": int(row[1] or 0),
                "queries_today":      int(row[2] or 0),
                "chunks_retrieved":   int(row[3] or 0),
                "estimated_cost_usd": float(row[4] or 0),
            }
        except psycopg2.Error:
            try:
                cur.execute(
                    """SELECT
                         COUNT(*) AS total_queries,
                         COUNT(*) FILTER (WHERE DATE_TRUNC('month', logged_at) = DATE_TRUNC('month', CURRENT_DATE)) AS this_month,
                         COUNT(*) FILTER (WHERE logged_at::date = CURRENT_DATE) AS today,
                         SUM(chunks_retrieved) AS total_chunks_retrieved
                       FROM query_log WHERE user_id = %s""",
                    (user_id,),
                )
                row = cur.fetchone()
                return {
                    "total_queries":      int(row[0] or 0),
                    "queries_this_month": int(row[1] or 0),
                    "queries_today":      int(row[2] or 0),
                    "chunks_retrieved":   int(row[3] or 0),
                    "estimated_cost_usd": 0.0,
                }
            except psycopg2.Error:
                return {"total_queries": 0, "queries_this_month": 0,
                        "queries_today": 0, "chunks_retrieved": 0,
                        "estimated_cost_usd": 0.0}
        finally:
            cur.close()

    def get_daily_counts(self, user_id: str, days: int = 30) -> list[dict]:
        cutoff = datetime.now() - timedelta(days=days)
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT DATE_TRUNC('day', logged_at)::DATE AS query_date,
                          COUNT(*) AS cnt
                   FROM query_log
                   WHERE user_id = %s AND logged_at >= %s
                   GROUP BY 1 ORDER BY 1""",
                (user_id, cutoff),
            )
            return [{"date": str(row[0]), "Queries": int(row[1])} for row in cur.fetchall()]
        except psycopg2.Error:
            return []
        finally:
            cur.close()

    def get_page_breakdown(self, user_id: str, days: int = 30) -> list[dict]:
        cutoff = datetime.now() - timedelta(days=days)
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT COALESCE(page, 'unknown') AS page, COUNT(*) AS cnt
                   FROM query_log
                   WHERE user_id = %s AND logged_at >= %s
                   GROUP BY 1 ORDER BY 2 DESC""",
                (user_id, cutoff),
            )
            label = {"insightlens": "Investment Chat", "epstein": "Epstein Chat"}
            return [
                {"Page": label.get(row[0], row[0].title()), "Queries": int(row[1])}
                for row in cur.fetchall()
            ]
        except psycopg2.Error:
            return []
        finally:
            cur.close()

    def get_hourly_distribution(self, user_id: str, days: int = 30) -> list[dict]:
        cutoff = datetime.now() - timedelta(days=days)
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT EXTRACT(HOUR FROM logged_at)::INTEGER AS hr, COUNT(*) AS cnt
                   FROM query_log
                   WHERE user_id = %s AND logged_at >= %s
                   GROUP BY 1 ORDER BY 1""",
                (user_id, cutoff),
            )
            hour_map = {row[0]: int(row[1]) for row in cur.fetchall()}
            return [
                {"Hour": f"{h:02d}:00", "Queries": hour_map.get(h, 0)}
                for h in range(24)
            ]
        except psycopg2.Error:
            return []
        finally:
            cur.close()

    def get_recent_queries(
        self, user_id: str, limit: int = 25, page_filter: str | None = None
    ) -> list[dict]:
        cur = self._conn.cursor()
        try:
            if page_filter and page_filter != "All":
                cur.execute(
                    """SELECT logged_at, page, query_text, chunks_retrieved, response_length
                       FROM query_log
                       WHERE user_id = %s AND page = %s
                       ORDER BY logged_at DESC LIMIT %s""",
                    (user_id, page_filter, limit),
                )
            else:
                cur.execute(
                    """SELECT logged_at, page, query_text, chunks_retrieved, response_length
                       FROM query_log WHERE user_id = %s
                       ORDER BY logged_at DESC LIMIT %s""",
                    (user_id, limit),
                )
            label = {"insightlens": "Investment", "epstein": "Epstein"}
            return [
                {
                    "Time":     str(row[0])[:16] if row[0] else "",
                    "Chat":     label.get(row[1], str(row[1]).title()),
                    "Query":    (row[2][:80] + "…") if row[2] and len(row[2]) > 80 else (row[2] or ""),
                    "Sources":  int(row[3] or 0),
                    "Resp len": int(row[4] or 0),
                }
                for row in cur.fetchall()
            ]
        except psycopg2.Error:
            return []
        finally:
            cur.close()

    def get_chunks_over_time(self, user_id: str, days: int = 30) -> list[dict]:
        cutoff = datetime.now() - timedelta(days=days)
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT DATE_TRUNC('day', logged_at)::DATE AS query_date,
                          AVG(chunks_retrieved) AS avg_chunks
                   FROM query_log
                   WHERE user_id = %s AND logged_at >= %s
                   GROUP BY 1 ORDER BY 1""",
                (user_id, cutoff),
            )
            return [
                {"date": str(row[0]), "Avg Sources": round(float(row[1] or 0), 1)}
                for row in cur.fetchall()
            ]
        except psycopg2.Error:
            return []
        finally:
            cur.close()

    def delete_user_logs(self, user_id: str) -> int:
        cur = self._conn.cursor()
        try:
            cur.execute("DELETE FROM query_log WHERE user_id = %s", (user_id,))
            return cur.rowcount
        except psycopg2.Error:
            return 0
        finally:
            cur.close()

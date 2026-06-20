"""Persistence for extracted case insights and generated artifacts."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

import psycopg2

from insightlens.storage.snowflake_client import SnowflakeConnection


@dataclass(frozen=True)
class CaseInsight:
    insight_id: str
    case_id: str
    insight_type: str
    title: str
    body: str | None
    severity: str | None
    document_id: str | None
    page_number: int | None
    metadata_json: str | None
    created_at: datetime | None = None


class InsightsRepository:
    def __init__(self, conn: SnowflakeConnection) -> None:
        self._conn = conn

    def replace_case_insights(
        self,
        case_id: str,
        insight_type: str,
        insights: list[dict],
    ) -> int:
        cur = self._conn.cursor()
        inserted = 0
        try:
            cur.execute(
                "DELETE FROM case_insights WHERE case_id = %s AND insight_type = %s",
                (case_id, insight_type),
            )
            for item in insights:
                cur.execute(
                    """INSERT INTO case_insights
                       (insight_id, case_id, insight_type, title, body, severity,
                        document_id, page_number, metadata_json)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        uuid.uuid4().hex[:16],
                        case_id,
                        insight_type,
                        str(item.get("title", ""))[:500],
                        item.get("body"),
                        item.get("severity"),
                        item.get("document_id"),
                        item.get("page_number"),
                        item.get("metadata_json"),
                    ),
                )
                inserted += 1
        except psycopg2.Error:
            return inserted
        finally:
            cur.close()
        return inserted

    def list_case_insights(
        self,
        case_id: str,
        insight_type: str | None = None,
        limit: int = 200,
    ) -> list[CaseInsight]:
        cur = self._conn.cursor()
        try:
            if insight_type:
                cur.execute(
                    """SELECT insight_id, case_id, insight_type, title, body, severity,
                              document_id, page_number, metadata_json, created_at
                       FROM case_insights
                       WHERE case_id = %s AND insight_type = %s
                       ORDER BY created_at DESC LIMIT %s""",
                    (case_id, insight_type, limit),
                )
            else:
                cur.execute(
                    """SELECT insight_id, case_id, insight_type, title, body, severity,
                              document_id, page_number, metadata_json, created_at
                       FROM case_insights
                       WHERE case_id = %s
                       ORDER BY created_at DESC LIMIT %s""",
                    (case_id, limit),
                )
            return [
                CaseInsight(
                    insight_id=row[0],
                    case_id=row[1],
                    insight_type=row[2],
                    title=row[3],
                    body=row[4],
                    severity=row[5],
                    document_id=row[6],
                    page_number=row[7],
                    metadata_json=row[8],
                    created_at=row[9],
                )
                for row in cur.fetchall()
            ]
        except psycopg2.Error:
            return []
        finally:
            cur.close()

    def create_artifact(
        self,
        *,
        user_id: str,
        case_id: str | None,
        artifact_type: str,
        title: str,
        content: str,
    ) -> str:
        artifact_id = uuid.uuid4().hex[:16]
        cur = self._conn.cursor()
        try:
            cur.execute(
                """INSERT INTO generated_artifacts
                   (artifact_id, user_id, case_id, artifact_type, title, content)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (artifact_id, user_id, case_id, artifact_type, title[:500], content),
            )
        except psycopg2.Error:
            pass
        finally:
            cur.close()
        return artifact_id

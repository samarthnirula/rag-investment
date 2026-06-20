"""Named case collections — group documents for multi-matter workspaces."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

import psycopg2

from insightlens.storage.snowflake_client import SnowflakeConnection


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    user_id: str
    case_name: str
    description: str | None
    created_at: datetime | None
    updated_at: datetime | None
    document_count: int = 0


class CasesRepository:
    def __init__(self, conn: SnowflakeConnection) -> None:
        self._conn = conn

    def create_case(self, user_id: str, case_name: str, description: str | None = None) -> str:
        case_id = uuid.uuid4().hex[:16]
        cur = self._conn.cursor()
        try:
            cur.execute(
                "INSERT INTO cases (case_id, user_id, case_name, description) VALUES (%s,%s,%s,%s)",
                (case_id, user_id, case_name[:200], description),
            )
        finally:
            cur.close()
        return case_id

    def list_cases(self, user_id: str) -> list[CaseRecord]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT c.case_id, c.user_id, c.case_name, c.description,
                          c.created_at, c.updated_at, COUNT(cd.document_id) AS doc_count
                   FROM cases c
                   LEFT JOIN case_documents cd ON c.case_id = cd.case_id
                   WHERE c.user_id = %s
                   GROUP BY c.case_id, c.user_id, c.case_name, c.description,
                            c.created_at, c.updated_at
                   ORDER BY c.updated_at DESC""",
                (user_id,),
            )
            return [
                CaseRecord(case_id=r[0], user_id=r[1], case_name=r[2], description=r[3],
                           created_at=r[4], updated_at=r[5], document_count=int(r[6] or 0))
                for r in cur.fetchall()
            ]
        except psycopg2.Error:
            return []
        finally:
            cur.close()

    def get_case(self, case_id: str) -> CaseRecord | None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT case_id,user_id,case_name,description,created_at,updated_at FROM cases WHERE case_id=%s",
                (case_id,),
            )
            r = cur.fetchone()
            if r is None:
                return None
            return CaseRecord(case_id=r[0], user_id=r[1], case_name=r[2], description=r[3],
                              created_at=r[4], updated_at=r[5])
        except psycopg2.Error:
            return None
        finally:
            cur.close()

    def update_case(self, case_id: str, case_name: str, description: str | None) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "UPDATE cases SET case_name=%s, description=%s, updated_at=NOW() WHERE case_id=%s",
                (case_name[:200], description, case_id),
            )
        except psycopg2.Error:
            pass
        finally:
            cur.close()

    def delete_case(self, case_id: str) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute("DELETE FROM case_documents WHERE case_id=%s", (case_id,))
            cur.execute("DELETE FROM cases WHERE case_id=%s", (case_id,))
        except psycopg2.Error:
            pass
        finally:
            cur.close()

    def add_document_to_case(self, case_id: str, document_id: str) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "INSERT INTO case_documents (case_id,document_id) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                (case_id, document_id),
            )
            cur.execute("UPDATE cases SET updated_at=NOW() WHERE case_id=%s", (case_id,))
        except psycopg2.Error:
            pass
        finally:
            cur.close()

    def remove_document_from_case(self, case_id: str, document_id: str) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "DELETE FROM case_documents WHERE case_id=%s AND document_id=%s",
                (case_id, document_id),
            )
        except psycopg2.Error:
            pass
        finally:
            cur.close()

    def get_case_document_ids(self, case_id: str) -> list[str]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT document_id FROM case_documents WHERE case_id=%s ORDER BY added_at",
                (case_id,),
            )
            return [r[0] for r in cur.fetchall()]
        except psycopg2.Error:
            return []
        finally:
            cur.close()

    def get_case_documents_info(self, case_id: str) -> list[dict]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT d.document_id, d.file_name, d.company, d.document_type,
                          d.version_label, d.page_count, cd.added_at
                   FROM case_documents cd
                   JOIN documents d ON cd.document_id = d.document_id
                   WHERE cd.case_id = %s ORDER BY cd.added_at""",
                (case_id,),
            )
            return [
                {"document_id": r[0], "file_name": r[1], "company": r[2],
                 "document_type": r[3], "version_label": r[4], "page_count": r[5], "added_at": r[6]}
                for r in cur.fetchall()
            ]
        except psycopg2.Error:
            return []
        finally:
            cur.close()

    def delete_user_cases(self, user_id: str) -> int:
        cur = self._conn.cursor()
        try:
            cur.execute("SELECT case_id FROM cases WHERE user_id=%s", (user_id,))
            case_ids = [r[0] for r in cur.fetchall()]
            if case_ids:
                ph = ",".join(["%s"] * len(case_ids))
                cur.execute(f"DELETE FROM case_documents WHERE case_id IN ({ph})", tuple(case_ids))
            cur.execute("DELETE FROM cases WHERE user_id=%s", (user_id,))
            return len(case_ids)
        except psycopg2.Error:
            return 0
        finally:
            cur.close()

"""Persistence operations for documents and chunks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

from snowflake.connector import SnowflakeConnection
from snowflake.connector.errors import ProgrammingError


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    file_name: str
    company: str | None
    document_type: str | None
    version_label: str | None
    version_date: date | None
    page_count: int


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    chunk_text: str
    token_count: int
    embedding: Sequence[float]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    file_name: str
    company: str | None
    version_label: str | None
    page_number: int
    chunk_text: str
    similarity: float


class RepositoryError(Exception):
    """Raised when a database operation fails."""


class ChunkRepository:
    """Read and write operations for documents and chunks."""

    def __init__(self, conn: SnowflakeConnection) -> None:
        self._conn = conn

    def upsert_document(self, doc: DocumentRecord) -> None:
        sql = """
            MERGE INTO DOCUMENTS d
            USING (SELECT %s AS document_id) s
            ON d.document_id = s.document_id
            WHEN MATCHED THEN UPDATE SET
                file_name = %s,
                company = %s,
                document_type = %s,
                version_label = %s,
                version_date = %s,
                page_count = %s
            WHEN NOT MATCHED THEN INSERT
                (document_id, file_name, company, document_type, version_label, version_date, page_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            doc.document_id,
            doc.file_name, doc.company, doc.document_type, doc.version_label, doc.version_date, doc.page_count,
            doc.document_id, doc.file_name, doc.company, doc.document_type, doc.version_label, doc.version_date, doc.page_count,
        )
        cursor = self._conn.cursor()
        try:
            cursor.execute(sql, params)
        except ProgrammingError as exc:
            raise RepositoryError(
                f"Failed to upsert document '{doc.file_name}' (id={doc.document_id}): {exc.msg}"
            ) from exc
        finally:
            cursor.close()

    def insert_chunks(self, chunks: Iterable[ChunkRecord]) -> int:
        rows = list(chunks)
        if not rows:
            return 0

        cursor = self._conn.cursor()
        inserted = 0
        try:
            for chunk in rows:
                # Snowflake connector cannot bind Python lists/strings to VECTOR columns
                # via parameterized %s. Inline the float literal directly in the SQL;
                # values are machine-generated floats so there is no injection risk.
                # Snowflake does not allow VECTOR casts in VALUES clauses;
                # use SELECT with the literal inlined for the embedding column.
                vec = "[" + ",".join(str(float(v)) for v in chunk.embedding) + "]"
                sql = f"""
                    INSERT INTO CHUNKS
                        (chunk_id, document_id, page_number, chunk_index, chunk_text, token_count, embedding)
                    SELECT %s, %s, %s, %s, %s, %s, {vec}::VECTOR(FLOAT, 384)
                """
                cursor.execute(
                    sql,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.page_number,
                        chunk.chunk_index,
                        chunk.chunk_text,
                        chunk.token_count,
                    ),
                )
                inserted += 1
        except ProgrammingError as exc:
            raise RepositoryError(
                f"Failed inserting chunks (succeeded: {inserted}/{len(rows)}). "
                f"Last chunk: {chunk.chunk_id}. Error: {exc.msg}"
            ) from exc
        finally:
            cursor.close()
        return inserted

    def delete_document(self, document_id: str) -> None:
        cursor = self._conn.cursor()
        try:
            cursor.execute("DELETE FROM CHUNKS WHERE document_id = %s", (document_id,))
            cursor.execute("DELETE FROM DOCUMENTS WHERE document_id = %s", (document_id,))
        except ProgrammingError as exc:
            raise RepositoryError(
                f"Failed to delete document {document_id}: {exc.msg}"
            ) from exc
        finally:
            cursor.close()

    def search_similar(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        company_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        vec = "[" + ",".join(str(float(v)) for v in query_embedding) + "]"
        base_sql = f"""
            SELECT
                c.chunk_id,
                c.document_id,
                d.file_name,
                d.company,
                d.version_label,
                c.page_number,
                c.chunk_text,
                VECTOR_COSINE_SIMILARITY(c.embedding, {vec}::VECTOR(FLOAT, 384)) AS similarity
            FROM CHUNKS c
            JOIN DOCUMENTS d ON c.document_id = d.document_id
        """
        params: list = []
        if company_filter:
            base_sql += " WHERE d.company = %s"
            params.append(company_filter)
        base_sql += " ORDER BY similarity DESC LIMIT %s"
        params.append(top_k)

        cursor = self._conn.cursor()
        try:
            cursor.execute(base_sql, tuple(params))
            rows = cursor.fetchall()
        except ProgrammingError as exc:
            raise RepositoryError(
                f"Vector search failed (top_k={top_k}, company_filter={company_filter}): {exc.msg}"
            ) from exc
        finally:
            cursor.close()

        return [
            RetrievedChunk(
                chunk_id=row[0],
                document_id=row[1],
                file_name=row[2],
                company=row[3],
                version_label=row[4],
                page_number=row[5],
                chunk_text=row[6],
                similarity=float(row[7]),
            )
            for row in rows
        ]

    def list_companies(self) -> list[str]:
        cursor = self._conn.cursor()
        try:
            cursor.execute("SELECT DISTINCT company FROM DOCUMENTS WHERE company IS NOT NULL ORDER BY company")
            return [row[0] for row in cursor.fetchall()]
        except ProgrammingError as exc:
            raise RepositoryError(f"Failed to list companies: {exc.msg}") from exc
        finally:
            cursor.close()

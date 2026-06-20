"""Persistence operations for documents and chunks."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

import psycopg2

from insightlens.storage.snowflake_client import SnowflakeConnection


_ALLOWED_VECTOR_DIMS = frozenset({384, 1024})  # local SentenceTransformer | voyage-law-2


def _safe_vector_str(embedding: Sequence[float], dim: int | None = None) -> str:
    """Serialize an embedding to a pgvector-compatible string '[f1,f2,...]'.

    Validates every element is a finite float to prevent injection via
    crafted embedding values.

    `dim`, if given, pins the expected length (e.g. when a caller already
    knows it). When omitted, the embedding's own length is used, as long as
    it matches one of the dimensions this deployment actually supports
    (384 = local all-MiniLM-L6-v2, 1024 = voyage-law-2). Previously this
    defaulted to 384 unconditionally, which made every insert/search call
    raise ValueError once the embedder switched to Voyage's 1024-dim output
    (the production path whenever VOYAGE_API_KEY is set).
    """
    validated: list[str] = []
    for v in embedding:
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"Embedding contains non-finite value: {v}")
        validated.append(f"{f:.8g}")

    expected = dim if dim is not None else len(validated)
    if dim is None and expected not in _ALLOWED_VECTOR_DIMS:
        raise ValueError(
            f"Embedding has {expected} dims, which doesn't match a supported "
            f"model ({sorted(_ALLOWED_VECTOR_DIMS)}). Check VOYAGE_API_KEY / "
            f"embedding backend configuration."
        )
    if len(validated) != expected:
        raise ValueError(f"Embedding has {len(validated)} dims, expected {expected}")
    return "[" + ",".join(validated) + "]"


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    file_name: str
    company: str | None
    document_type: str | None
    version_label: str | None
    version_date: date | None
    page_count: int
    supersedes_document_id: str | None = None
    user_id: str | None = None


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    chunk_text: str
    token_count: int
    embedding: Sequence[float]
    section_header: str | None = None
    chunk_type: str = "body"
    structured_content: str | None = None


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
    section_header: str | None = None
    chunk_type: str = "body"
    structured_content: str | None = None
    supersedes_document_id: str | None = None
    document_type: str | None = None
    version_date: date | None = None
    source_type: str = "document"


class RepositoryError(Exception):
    """Raised when a database operation fails."""


class ChunkRepository:
    """Read and write operations for documents and chunks."""

    def __init__(self, conn: SnowflakeConnection) -> None:
        self._conn = conn

    def upsert_document(self, doc: DocumentRecord) -> None:
        sql = """
            INSERT INTO documents
                (document_id, file_name, company, document_type, version_label,
                 version_date, page_count, supersedes_document_id, user_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_id) DO UPDATE SET
                file_name              = EXCLUDED.file_name,
                company                = EXCLUDED.company,
                document_type          = EXCLUDED.document_type,
                version_label          = EXCLUDED.version_label,
                version_date           = EXCLUDED.version_date,
                page_count             = EXCLUDED.page_count,
                supersedes_document_id = EXCLUDED.supersedes_document_id,
                user_id                = EXCLUDED.user_id
        """
        cur = self._conn.cursor()
        try:
            cur.execute(sql, (
                doc.document_id, doc.file_name, doc.company, doc.document_type,
                doc.version_label, doc.version_date, doc.page_count,
                doc.supersedes_document_id, doc.user_id,
            ))
        except psycopg2.Error as exc:
            raise RepositoryError(
                f"Failed to upsert document '{doc.file_name}' (id={doc.document_id}): {exc}"
            ) from exc
        finally:
            cur.close()

    def insert_chunks(self, chunks: Iterable[ChunkRecord]) -> int:
        rows = list(chunks)
        if not rows:
            return 0

        cur = self._conn.cursor()
        inserted = 0
        chunk = None
        try:
            for chunk in rows:
                vec_str = _safe_vector_str(chunk.embedding)
                cur.execute(
                    """
                    INSERT INTO chunks
                        (chunk_id, document_id, page_number, chunk_index, chunk_text,
                         token_count, embedding, section_header, chunk_type, structured_content)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.page_number,
                        chunk.chunk_index,
                        chunk.chunk_text,
                        chunk.token_count,
                        vec_str,
                        chunk.section_header,
                        chunk.chunk_type,
                        chunk.structured_content,
                    ),
                )
                inserted += 1
        except psycopg2.Error as exc:
            last = chunk.chunk_id if chunk else "unknown"
            raise RepositoryError(
                f"Failed inserting chunks (succeeded: {inserted}/{len(rows)}). "
                f"Last chunk: {last}. Error: {exc}"
            ) from exc
        finally:
            cur.close()
        return inserted

    def delete_document(self, document_id: str, user_id: str | None = None) -> None:
        """Delete a document and its dependent rows.

        If user_id is provided, the document must belong to that user.  If it is
        omitted, only shared/system documents (user_id IS NULL) can be deleted.
        This keeps old ingestion scripts working without letting a caller delete
        another user's upload by ID.
        """
        cur = self._conn.cursor()
        try:
            if user_id is None:
                cur.execute(
                    "SELECT 1 FROM documents WHERE document_id = %s AND user_id IS NULL",
                    (document_id,),
                )
            else:
                cur.execute(
                    "SELECT 1 FROM documents WHERE document_id = %s AND user_id = %s",
                    (document_id, user_id),
                )
            if not cur.fetchone():
                raise RepositoryError(
                    f"Document {document_id} not found or not owned by the requesting user."
                )
            cur.execute("DELETE FROM images WHERE document_id = %s", (document_id,))
            cur.execute("DELETE FROM case_documents WHERE document_id = %s", (document_id,))
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
            if user_id is None:
                cur.execute(
                    "DELETE FROM documents WHERE document_id = %s AND user_id IS NULL",
                    (document_id,),
                )
            else:
                cur.execute(
                    "DELETE FROM documents WHERE document_id = %s AND user_id = %s",
                    (document_id, user_id),
                )
        except RepositoryError:
            raise
        except psycopg2.Error as exc:
            raise RepositoryError(
                f"Failed to delete document {document_id}: {exc}"
            ) from exc
        finally:
            cur.close()

    def delete_user_documents(self, user_id: str) -> int:
        """Delete all documents uploaded by user_id, including chunks and images."""
        cur = self._conn.cursor()
        try:
            cur.execute(
                """DELETE FROM images WHERE document_id IN (
                       SELECT document_id FROM documents WHERE user_id = %s
                   )""",
                (user_id,),
            )
            cur.execute(
                """DELETE FROM case_documents WHERE document_id IN (
                       SELECT document_id FROM documents WHERE user_id = %s
                   )""",
                (user_id,),
            )
            cur.execute(
                """DELETE FROM chunks WHERE document_id IN (
                       SELECT document_id FROM documents WHERE user_id = %s
                   )""",
                (user_id,),
            )
            cur.execute("DELETE FROM documents WHERE user_id = %s", (user_id,))
            return cur.rowcount
        except psycopg2.Error as exc:
            raise RepositoryError(f"Failed to delete documents for user {user_id}: {exc}") from exc
        finally:
            cur.close()

    def get_chunks_for_case(self, case_id: str, user_id: str, limit: int = 120) -> list[RetrievedChunk]:
        """Return all chunks that belong to a case (for overview/timeline generation)."""
        sql = """
            SELECT
                c.chunk_id, c.document_id, d.file_name, d.company, d.version_label,
                c.page_number, c.chunk_text, c.section_header, c.chunk_type,
                c.structured_content, d.supersedes_document_id, d.document_type, d.version_date
            FROM chunks c
            JOIN documents d ON c.document_id = d.document_id
            JOIN case_documents cd ON d.document_id = cd.document_id
            WHERE cd.case_id = %s AND d.user_id = %s
            ORDER BY d.file_name, c.page_number, c.chunk_index
            LIMIT %s
        """
        cur = self._conn.cursor()
        try:
            cur.execute(sql, (case_id, user_id, limit))
            rows = cur.fetchall()
        except psycopg2.Error as exc:
            raise RepositoryError(f"get_chunks_for_case failed: {exc}") from exc
        finally:
            cur.close()
        return [
            RetrievedChunk(
                chunk_id=row[0], document_id=row[1], file_name=row[2],
                company=row[3], version_label=row[4], page_number=row[5],
                chunk_text=row[6], similarity=0.0, section_header=row[7],
                chunk_type=row[8] or "body", structured_content=row[9],
                supersedes_document_id=row[10], document_type=row[11], version_date=row[12],
            )
            for row in rows
        ]

    def search_similar(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        company_filter: str | None = None,
        user_id: str | None = None,
        org_member_ids: list[str] | None = None,
        system_only: bool = False,
        user_only: bool = False,
        case_id: str | None = None,
    ) -> list[RetrievedChunk]:
        vec_str = _safe_vector_str(query_embedding)
        if case_id and user_id:
            # Case-scoped vector search — join through case_documents
            base_sql = """
                SELECT
                    c.chunk_id, c.document_id, d.file_name, d.company, d.version_label,
                    c.page_number, c.chunk_text,
                    1 - (c.embedding <=> %s::vector) AS similarity,
                    c.section_header, c.chunk_type, c.structured_content,
                    d.supersedes_document_id, d.document_type, d.version_date
                FROM chunks c
                JOIN documents d ON c.document_id = d.document_id
                JOIN case_documents cd ON d.document_id = cd.document_id
                WHERE cd.case_id = %s AND d.user_id = %s
                ORDER BY similarity DESC LIMIT %s
            """
            cur = self._conn.cursor()
            try:
                cur.execute(base_sql, (vec_str, case_id, user_id, top_k))
                rows = cur.fetchall()
            except psycopg2.Error as exc:
                raise RepositoryError(f"Case vector search failed: {exc}") from exc
            finally:
                cur.close()
            return [
                RetrievedChunk(
                    chunk_id=row[0], document_id=row[1], file_name=row[2],
                    company=row[3], version_label=row[4], page_number=row[5],
                    chunk_text=row[6], similarity=float(row[7]), section_header=row[8],
                    chunk_type=row[9] or "body", structured_content=row[10],
                    supersedes_document_id=row[11], document_type=row[12], version_date=row[13],
                )
                for row in rows
            ]

        base_sql = """
            SELECT
                c.chunk_id,
                c.document_id,
                d.file_name,
                d.company,
                d.version_label,
                c.page_number,
                c.chunk_text,
                1 - (c.embedding <=> %s::vector) AS similarity,
                c.section_header,
                c.chunk_type,
                c.structured_content,
                d.supersedes_document_id,
                d.document_type,
                d.version_date
            FROM chunks c
            JOIN documents d ON c.document_id = d.document_id
        """
        params: list = [vec_str]
        conditions: list[str] = []
        if system_only:
            conditions.append("d.user_id IS NULL")
        elif user_id:
            if user_only:
                conditions.append("d.user_id = %s")
                params.append(user_id)
            else:
                # Own docs + system docs + org teammates' docs
                visible_ids = [user_id] + (org_member_ids or [])
                placeholders = ",".join(["%s"] * len(visible_ids))
                conditions.append(f"(d.user_id IN ({placeholders}) OR d.user_id IS NULL)")
                params.extend(visible_ids)
        if company_filter:
            conditions.append("UPPER(d.company) = UPPER(%s)")
            params.append(company_filter)
        if conditions:
            base_sql += " WHERE " + " AND ".join(conditions)
        base_sql += " ORDER BY similarity DESC LIMIT %s"
        params.append(top_k)

        cur = self._conn.cursor()
        try:
            cur.execute(base_sql, tuple(params))
            rows = cur.fetchall()
        except psycopg2.Error as exc:
            raise RepositoryError(
                f"Vector search failed (top_k={top_k}, company_filter={company_filter}): {exc}"
            ) from exc
        finally:
            cur.close()

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
                section_header=row[8],
                chunk_type=row[9] or "body",
                structured_content=row[10],
                supersedes_document_id=row[11],
                document_type=row[12],
                version_date=row[13],
            )
            for row in rows
        ]

    def get_all_chunks(
        self,
        company_filter: str | None = None,
        user_id: str | None = None,
        system_only: bool = False,
        user_only: bool = False,
        case_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Load chunks (no embeddings) for BM25 corpus construction.

        When *case_id* is provided the corpus is restricted to documents that
        belong to that case via the case_documents join table.
        """
        if case_id and user_id:
            # Case-scoped query — join through case_documents
            sql = """
                SELECT
                    c.chunk_id,
                    c.document_id,
                    d.file_name,
                    d.company,
                    d.version_label,
                    c.page_number,
                    c.chunk_text,
                    c.section_header,
                    c.chunk_type,
                    c.structured_content,
                    d.supersedes_document_id,
                    d.document_type,
                    d.version_date
                FROM chunks c
                JOIN documents d ON c.document_id = d.document_id
                JOIN case_documents cd ON d.document_id = cd.document_id
                WHERE cd.case_id = %s AND d.user_id = %s
                ORDER BY c.chunk_id
            """
            cur = self._conn.cursor()
            try:
                cur.execute(sql, (case_id, user_id))
                rows = cur.fetchall()
            except psycopg2.Error as exc:
                raise RepositoryError(f"Failed to load case BM25 corpus: {exc}") from exc
            finally:
                cur.close()
            return [
                RetrievedChunk(
                    chunk_id=row[0], document_id=row[1], file_name=row[2],
                    company=row[3], version_label=row[4], page_number=row[5],
                    chunk_text=row[6], similarity=0.0, section_header=row[7],
                    chunk_type=row[8] or "body", structured_content=row[9],
                    supersedes_document_id=row[10], document_type=row[11], version_date=row[12],
                )
                for row in rows
            ]

        sql = """
            SELECT
                c.chunk_id,
                c.document_id,
                d.file_name,
                d.company,
                d.version_label,
                c.page_number,
                c.chunk_text,
                c.section_header,
                c.chunk_type,
                c.structured_content,
                d.supersedes_document_id,
                d.document_type,
                d.version_date
            FROM chunks c
            JOIN documents d ON c.document_id = d.document_id
        """
        params: list = []
        conditions: list[str] = []
        if system_only:
            conditions.append("d.user_id IS NULL")
        elif user_id:
            if user_only:
                conditions.append("d.user_id = %s")
            else:
                conditions.append("(d.user_id = %s OR d.user_id IS NULL)")
            params.append(user_id)
        if company_filter:
            conditions.append("UPPER(d.company) = UPPER(%s)")
            params.append(company_filter)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY c.chunk_id"

        cur = self._conn.cursor()
        try:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        except psycopg2.Error as exc:
            raise RepositoryError(f"Failed to load BM25 corpus: {exc}") from exc
        finally:
            cur.close()

        return [
            RetrievedChunk(
                chunk_id=row[0],
                document_id=row[1],
                file_name=row[2],
                company=row[3],
                version_label=row[4],
                page_number=row[5],
                chunk_text=row[6],
                similarity=0.0,
                section_header=row[7],
                chunk_type=row[8] or "body",
                structured_content=row[9],
                supersedes_document_id=row[10],
                document_type=row[11],
                version_date=row[12],
            )
            for row in rows
        ]

    def get_chunks_for_documents(self, document_ids: list[str]) -> list[RetrievedChunk]:
        """Load chunks for a known set of document IDs."""
        if not document_ids:
            return []
        placeholders = ",".join(["%s"] * len(document_ids))
        sql = f"""
            SELECT
                c.chunk_id,
                c.document_id,
                d.file_name,
                d.company,
                d.version_label,
                c.page_number,
                c.chunk_text,
                c.section_header,
                c.chunk_type,
                c.structured_content,
                d.supersedes_document_id,
                d.document_type,
                d.version_date
            FROM chunks c
            JOIN documents d ON c.document_id = d.document_id
            WHERE c.document_id IN ({placeholders})
            ORDER BY d.file_name, c.page_number, c.chunk_index
        """
        cur = self._conn.cursor()
        try:
            cur.execute(sql, tuple(document_ids))
            rows = cur.fetchall()
        except psycopg2.Error as exc:
            raise RepositoryError(f"Failed to load case chunks: {exc}") from exc
        finally:
            cur.close()

        return [
            RetrievedChunk(
                chunk_id=row[0],
                document_id=row[1],
                file_name=row[2],
                company=row[3],
                version_label=row[4],
                page_number=row[5],
                chunk_text=row[6],
                similarity=0.0,
                section_header=row[7],
                chunk_type=row[8] or "body",
                structured_content=row[9],
                supersedes_document_id=row[10],
                document_type=row[11],
                version_date=row[12],
            )
            for row in rows
        ]

    def get_all_documents(self) -> list[DocumentRecord]:
        sql = """
            SELECT document_id, file_name, company, document_type,
                   version_label, version_date, page_count, supersedes_document_id, user_id
            FROM documents
            ORDER BY company, version_date
        """
        cur = self._conn.cursor()
        try:
            cur.execute(sql)
            rows = cur.fetchall()
        except psycopg2.Error as exc:
            raise RepositoryError(f"Failed to list documents: {exc}") from exc
        finally:
            cur.close()

        return [
            DocumentRecord(
                document_id=row[0],
                file_name=row[1],
                company=row[2],
                document_type=row[3],
                version_label=row[4],
                version_date=row[5],
                page_count=row[6],
                supersedes_document_id=row[7],
                user_id=row[8],
            )
            for row in rows
        ]

    def list_documents(self, user_id: str | None = None) -> list[DocumentRecord]:
        sql = """
            SELECT document_id, file_name, company, document_type,
                   version_label, version_date, page_count, supersedes_document_id,
                   user_id, ingested_at
            FROM documents
        """
        params: list = []
        if user_id is not None:
            sql += " WHERE user_id = %s"
            params.append(user_id)
        else:
            sql += " WHERE user_id IS NULL"
        sql += " ORDER BY ingested_at DESC NULLS LAST"

        cur = self._conn.cursor()
        try:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        except psycopg2.Error as exc:
            raise RepositoryError(f"Failed to list documents: {exc}") from exc
        finally:
            cur.close()

        return [
            DocumentRecord(
                document_id=row[0],
                file_name=row[1],
                company=row[2],
                document_type=row[3],
                version_label=row[4],
                version_date=row[5],
                page_count=row[6],
                supersedes_document_id=row[7],
                user_id=row[8],
            )
            for row in rows
        ]

    def set_supersedes(self, newer_document_id: str, older_document_id: str) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "UPDATE documents SET supersedes_document_id = %s WHERE document_id = %s",
                (older_document_id, newer_document_id),
            )
        except psycopg2.Error as exc:
            raise RepositoryError(
                f"Failed to set supersedes relationship: {exc}"
            ) from exc
        finally:
            cur.close()

    def list_companies(self, user_id: str | None = None) -> list[str]:
        cur = self._conn.cursor()
        try:
            if user_id:
                cur.execute(
                    "SELECT DISTINCT company FROM documents WHERE company IS NOT NULL AND user_id = %s ORDER BY company",
                    (user_id,),
                )
            else:
                cur.execute(
                    "SELECT DISTINCT company FROM documents WHERE company IS NOT NULL AND user_id IS NULL ORDER BY company"
                )
            return [row[0] for row in cur.fetchall()]
        except psycopg2.Error as exc:
            raise RepositoryError(f"Failed to list companies: {exc}") from exc
        finally:
            cur.close()

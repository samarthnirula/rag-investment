"""Persistence operations for document images."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import psycopg2

from insightlens.storage.snowflake_client import SnowflakeConnection


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    document_id: str
    page_number: int
    image_index: int
    file_path: str
    media_type: str | None = None
    width: int | None = None
    height: int | None = None
    ai_description: str | None = None


@dataclass(frozen=True)
class ScoredImageRecord:
    image_id: str
    document_id: str
    page_number: int
    image_index: int
    file_path: str
    media_type: str | None
    width: int | None
    height: int | None
    ai_description: str | None
    similarity: float


class ImageRepositoryError(Exception):
    pass


def _row_to_record(row: tuple) -> ImageRecord:
    return ImageRecord(
        image_id=row[0], document_id=row[1], page_number=row[2],
        image_index=row[3], file_path=row[4], media_type=row[5],
        width=row[6], height=row[7], ai_description=row[8],
    )


def _safe_vec(embedding: Sequence[float]) -> str:
    vals = []
    for v in embedding:
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"Non-finite embedding value: {v}")
        vals.append(f"{f:.8g}")
    return "[" + ",".join(vals) + "]"


class ImageRepository:
    def __init__(self, conn: SnowflakeConnection) -> None:
        self._conn = conn

    def insert_image(self, record: ImageRecord) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """INSERT INTO images
                   (image_id, document_id, page_number, image_index,
                    file_path, media_type, width, height, ai_description)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (image_id) DO NOTHING""",
                (record.image_id, record.document_id, record.page_number,
                 record.image_index, record.file_path, record.media_type,
                 record.width, record.height, record.ai_description),
            )
        except psycopg2.Error as exc:
            raise ImageRepositoryError(f"Failed to insert image {record.image_id}: {exc}") from exc
        finally:
            cur.close()

    def insert_images_batch(self, records: list[ImageRecord]) -> int:
        inserted = 0
        for record in records:
            self.insert_image(record)
            inserted += 1
        return inserted

    def get_images_for_page(self, document_id: str, page_number: int) -> list[ImageRecord]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT image_id,document_id,page_number,image_index,
                          file_path,media_type,width,height,ai_description
                   FROM images WHERE document_id=%s AND page_number=%s ORDER BY image_index""",
                (document_id, page_number),
            )
            return [_row_to_record(r) for r in cur.fetchall()]
        except psycopg2.Error as exc:
            raise ImageRepositoryError(f"Failed to fetch images: {exc}") from exc
        finally:
            cur.close()

    def get_image(self, image_id: str) -> ImageRecord | None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT image_id,document_id,page_number,image_index,
                          file_path,media_type,width,height,ai_description
                   FROM images WHERE image_id=%s""",
                (image_id,),
            )
            row = cur.fetchone()
            return _row_to_record(row) if row else None
        except psycopg2.Error as exc:
            raise ImageRepositoryError(f"Failed to fetch image {image_id}: {exc}") from exc
        finally:
            cur.close()

    def get_all_image_metadata(self, document_id: str | None = None) -> list[ImageRecord]:
        sql = """SELECT image_id,document_id,page_number,image_index,
                        file_path,media_type,width,height,ai_description FROM images"""
        params: list = []
        if document_id:
            sql += " WHERE document_id=%s"
            params.append(document_id)
        sql += " ORDER BY document_id, page_number, image_index"
        cur = self._conn.cursor()
        try:
            cur.execute(sql, tuple(params))
            return [_row_to_record(r) for r in cur.fetchall()]
        except psycopg2.Error as exc:
            raise ImageRepositoryError(f"Failed to load image metadata: {exc}") from exc
        finally:
            cur.close()

    def update_description_embedding(self, image_id: str, embedding: Sequence[float]) -> None:
        vec_str = _safe_vec(embedding)
        cur = self._conn.cursor()
        try:
            cur.execute(
                "UPDATE images SET description_embedding = %s::vector WHERE image_id = %s",
                (vec_str, image_id),
            )
        except psycopg2.Error as exc:
            raise ImageRepositoryError(f"Failed to update embedding for {image_id}: {exc}") from exc
        finally:
            cur.close()

    def search_by_description(
        self,
        query_embedding: Sequence[float],
        top_k: int = 3,
        min_similarity: float = 0.35,
        company_filter: str | None = None,
    ) -> list[ScoredImageRecord]:
        """Return images whose AI descriptions are semantically similar to the query."""
        vec_str = _safe_vec(query_embedding)
        sql = """
            SELECT i.image_id, i.document_id, i.page_number, i.image_index,
                   i.file_path, i.media_type, i.width, i.height, i.ai_description,
                   1 - (i.description_embedding <=> %s::vector) AS similarity
            FROM images i
            JOIN documents d ON i.document_id = d.document_id
            WHERE i.description_embedding IS NOT NULL
              AND 1 - (i.description_embedding <=> %s::vector) >= %s
        """
        params: list = [vec_str, vec_str, min_similarity]
        if company_filter:
            sql += " AND UPPER(d.company) = UPPER(%s)"
            params.append(company_filter)
        sql += " ORDER BY similarity DESC LIMIT %s"
        params.append(top_k)

        cur = self._conn.cursor()
        try:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        except psycopg2.Error as exc:
            raise ImageRepositoryError(f"Image description search failed: {exc}") from exc
        finally:
            cur.close()

        return [
            ScoredImageRecord(
                image_id=row[0], document_id=row[1], page_number=row[2],
                image_index=row[3], file_path=row[4], media_type=row[5],
                width=row[6], height=row[7], ai_description=row[8],
                similarity=float(row[9]),
            )
            for row in rows
        ]

    def search_by_text_terms(
        self,
        terms: list[str],
        top_k: int = 5,
        system_only: bool = False,
    ) -> list[ImageRecord]:
        """Return images whose filename/path/AI description contains query terms."""
        cleaned = [term.strip().lower() for term in terms if len(term.strip()) >= 3]
        if not cleaned:
            return []

        conditions = []
        params: list = []
        for term in cleaned:
            like = f"%{term}%"
            conditions.append(
                "(LOWER(i.ai_description) LIKE %s OR LOWER(i.file_path) LIKE %s OR LOWER(d.file_name) LIKE %s)"
            )
            params.extend([like, like, like])

        sql = """
            SELECT i.image_id, i.document_id, i.page_number, i.image_index,
                   i.file_path, i.media_type, i.width, i.height, i.ai_description
            FROM images i
            JOIN documents d ON i.document_id = d.document_id
            WHERE ({conditions})
        """.format(conditions=" OR ".join(conditions))
        if system_only:
            sql += " AND d.user_id IS NULL"
        sql += " ORDER BY i.image_id LIMIT %s"
        params.append(top_k)

        cur = self._conn.cursor()
        try:
            cur.execute(sql, tuple(params))
            return [_row_to_record(row) for row in cur.fetchall()]
        except psycopg2.Error as exc:
            raise ImageRepositoryError(f"Image text search failed: {exc}") from exc
        finally:
            cur.close()

    def get_images_missing_embedding(self) -> list[ImageRecord]:
        """Return images that have an ai_description but no embedding yet."""
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT image_id,document_id,page_number,image_index,
                          file_path,media_type,width,height,ai_description
                   FROM images
                   WHERE ai_description IS NOT NULL
                     AND description_embedding IS NULL
                   ORDER BY image_id"""
            )
            return [_row_to_record(r) for r in cur.fetchall()]
        except psycopg2.Error as exc:
            raise ImageRepositoryError(f"Failed to fetch un-embedded images: {exc}") from exc
        finally:
            cur.close()

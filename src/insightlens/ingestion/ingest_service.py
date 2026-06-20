"""In-process ingestion pipeline — usable from the UI without subprocess."""
from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from insightlens.billing import default_plan, estimate_ingestion_cost_usd
from insightlens.config import AppConfig, RAW_PDF_DIR
from insightlens.embeddings.embedder import Embedder
from insightlens.ingestion.chunker import SlideAwareChunker
from insightlens.ingestion.document_metadata import extract_metadata
from insightlens.ingestion.image_extractor import extract_images_from_pdf
from insightlens.ingestion.pdf_parser import PDFParsingError, parse_pdf
from insightlens.ingestion.pptx_parser import PPTXParsingError, parse_pptx
from insightlens.storage.chunk_repository import (
    ChunkRecord,
    ChunkRepository,
    DocumentRecord,
)
from insightlens.storage.image_repository import ImageRecord, ImageRepository
from insightlens.storage.snowflake_client import open_connection

_IMAGES_DIR = RAW_PDF_DIR.parent / "images"
_SUPPORTED_EXTENSIONS = {".pdf", ".pptx"}


def _document_id(path: Path) -> str:
    return hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:16]


def _document_id_from_name(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]


def _chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}-{chunk_index:05d}"


@dataclass
class IngestResult:
    document_id: str
    file_name: str
    chunks_inserted: int
    images_found: int
    page_count: int = 0
    estimated_cost_usd: float = 0.0
    skipped: bool = False
    error: str | None = None


class IngestService:
    """Run the full ingestion pipeline from a file path — used by the data page."""

    def __init__(
        self,
        cfg: AppConfig,
        embedder: Embedder,
        images_dir: Path | None = None,
    ) -> None:
        self._cfg = cfg
        self._embedder = embedder
        self._chunker = SlideAwareChunker()
        self._images_dir = images_dir or _IMAGES_DIR
        self._images_dir.mkdir(parents=True, exist_ok=True)

    def ingest(
        self,
        path: Path,
        user_id: str | None = None,
        progress: Callable[[str], None] | None = None,
        original_file_name: str | None = None,
    ) -> IngestResult:
        """Ingest a single supported document into PostgreSQL/pgvector.

        progress: optional callable that receives status strings for display.
        """

        def _log(msg: str) -> None:
            if progress:
                progress(msg)

        display_name = original_file_name or path.name
        document_id = _document_id_from_name(f"{user_id or 'system'}:{display_name}")
        suffix = Path(display_name).suffix.lower() or path.suffix.lower()

        if suffix not in _SUPPORTED_EXTENSIONS:
            return IngestResult(
                document_id=document_id,
                file_name=display_name,
                chunks_inserted=0,
                images_found=0,
                page_count=0,
                skipped=True,
                error="Unsupported file type. Upload a PDF or PPTX file.",
            )

        try:
            _log(f"Parsing {display_name}…")
            if suffix == ".pptx":
                parsed = parse_pptx(path)
            else:
                parsed = parse_pdf(path)
        except (PDFParsingError, PPTXParsingError) as exc:
            return IngestResult(
                document_id=document_id,
                file_name=display_name,
                chunks_inserted=0,
                images_found=0,
                page_count=0,
                error=f"Parse failed: {exc}",
            )

        first_page_text = parsed.pages[0].text if parsed.pages else ""
        metadata = extract_metadata(Path(display_name), first_page_text)

        plan = default_plan()
        if user_id and parsed.total_pages > plan.max_pages_per_pdf:
            return IngestResult(
                document_id=document_id,
                file_name=display_name,
                chunks_inserted=0,
                images_found=0,
                page_count=parsed.total_pages,
                skipped=True,
                error=(
                    f"Document has {parsed.total_pages} pages/slides; the {plan.name} limit is "
                    f"{plan.max_pages_per_pdf} pages/slides per file."
                ),
            )

        _log(f"Chunking {len(parsed.pages)} pages…")
        slide_chunks = self._chunker.chunk_document(parsed.pages)

        if not slide_chunks:
            return IngestResult(
                document_id=document_id,
                file_name=display_name,
                chunks_inserted=0,
                images_found=0,
                page_count=parsed.total_pages,
                skipped=True,
                error="No text chunks produced (visual-only document?)",
            )

        _log(f"Embedding {len(slide_chunks)} chunks…")
        try:
            embeddings = self._embedder.embed_texts([c.text for c in slide_chunks])
        except Exception as exc:
            return IngestResult(
                document_id=document_id,
                file_name=display_name,
                chunks_inserted=0,
                images_found=0,
                page_count=parsed.total_pages,
                error=f"Embedding failed: {exc}",
            )

        records = [
            ChunkRecord(
                chunk_id=_chunk_id(document_id, chunk.chunk_index),
                document_id=document_id,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                chunk_text=chunk.text,
                token_count=chunk.token_count,
                embedding=result.vector,
                section_header=chunk.section_header,
                chunk_type=chunk.chunk_type,
                structured_content=chunk.structured_content,
            )
            for chunk, result in zip(slide_chunks, embeddings, strict=True)
        ]

        _log("Storing in PostgreSQL…")
        try:
            with open_connection(self._cfg.db) as conn:
                repo = ChunkRepository(conn)
                try:
                    repo.delete_document(document_id, user_id)
                except Exception:
                    pass
                repo.upsert_document(
                    DocumentRecord(
                        document_id=document_id,
                        file_name=display_name,
                        company=metadata.company,
                        document_type=metadata.document_type,
                        version_label=metadata.version_label,
                        version_date=metadata.version_date,
                        page_count=parsed.total_pages,
                        user_id=user_id,
                    ),
                )
                inserted = repo.insert_chunks(records)

                image_repo = ImageRepository(conn)
                # Extract embedded raster images for PDFs. PPTX text ingestion is
                # currently slide-text only; image/OCR support can be added later.
                image_dicts = (
                    extract_images_from_pdf(path, document_id, self._images_dir, api_key="")
                    if suffix == ".pdf"
                    else []
                )
                for img in image_dicts:
                    image_repo.insert_image(
                        ImageRecord(
                            image_id=img["image_id"],
                            document_id=img["document_id"],
                            page_number=img["page_number"],
                            image_index=img["image_index"],
                            file_path=img["file_path"],
                            media_type=img["media_type"],
                            width=img["width"],
                            height=img["height"],
                            ai_description=img.get("ai_description"),
                        )
                    )
        except Exception as exc:
            return IngestResult(
                document_id=document_id,
                file_name=display_name,
                chunks_inserted=0,
                images_found=0,
                page_count=parsed.total_pages,
                error=f"Storage failed: {exc}",
            )

        _log(f"Done — {inserted} chunks, {len(image_dicts)} images.")
        return IngestResult(
            document_id=document_id,
            file_name=display_name,
            chunks_inserted=inserted,
            images_found=len(image_dicts),
            page_count=parsed.total_pages,
            estimated_cost_usd=estimate_ingestion_cost_usd(
                pages=parsed.total_pages,
                file_size_bytes=path.stat().st_size,
            ),
        )

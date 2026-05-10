"""Walks data/raw_pdfs, parses + chunks + embeds + persists every PDF."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from insightlens.config import RAW_PDF_DIR, ConfigError, load_config
from insightlens.embeddings.embedder import EmbeddingError, Embedder
from insightlens.ingestion.chunker import ChunkingError, RecursiveTokenChunker
from insightlens.ingestion.document_metadata import extract_metadata
from insightlens.ingestion.pdf_parser import PDFParsingError, parse_pdf
from insightlens.storage.chunk_repository import (
    ChunkRecord,
    ChunkRepository,
    DocumentRecord,
    RepositoryError,
)
from insightlens.storage.snowflake_client import (
    SnowflakeConnectionError,
    open_connection,
)


def _document_id(path: Path) -> str:
    return hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:16]


def _chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}-{chunk_index:05d}"


def _process_pdf(
    path: Path,
    chunker: RecursiveTokenChunker,
    embedder: Embedder,
    repository: ChunkRepository,
) -> int:
    print(f"[ingest] Parsing {path.name}")
    parsed = parse_pdf(path)

    first_page_text = parsed.pages[0].text if parsed.pages else ""
    metadata = extract_metadata(path, first_page_text)

    document_id = _document_id(path)
    repository.delete_document(document_id)
    repository.upsert_document(
        DocumentRecord(
            document_id=document_id,
            file_name=path.name,
            company=metadata.company,
            document_type=metadata.document_type,
            version_label=metadata.version_label,
            version_date=metadata.version_date,
            page_count=parsed.total_pages,
        )
    )

    chunks_with_meta: list[tuple[str, int, int, int]] = []
    running_index = 0
    for page in parsed.pages:
        page_chunks = chunker.chunk_page(page.text, page.page_number, running_index)
        for chunk in page_chunks:
            chunks_with_meta.append((chunk.text, chunk.page_number, chunk.chunk_index, chunk.token_count))
            running_index += 1

    if not chunks_with_meta:
        print(f"[ingest] {path.name} produced no chunks (empty or visual-only document)")
        return 0

    print(f"[ingest] Embedding {len(chunks_with_meta)} chunks from {path.name}")
    embeddings = embedder.embed_texts([text for text, _, _, _ in chunks_with_meta])

    records = [
        ChunkRecord(
            chunk_id=_chunk_id(document_id, chunk_index),
            document_id=document_id,
            page_number=page_number,
            chunk_index=chunk_index,
            chunk_text=text,
            token_count=token_count,
            embedding=result.vector,
        )
        for (text, page_number, chunk_index, token_count), result in zip(chunks_with_meta, embeddings, strict=True)
    ]

    inserted = repository.insert_chunks(records)
    print(f"[ingest] Inserted {inserted} chunks for {path.name}")
    return inserted


def main() -> int:
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"[ingest] Configuration error: {exc}", file=sys.stderr)
        return 1

    pdf_paths = sorted(RAW_PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"[ingest] No PDFs found in {RAW_PDF_DIR}", file=sys.stderr)
        return 1

    chunker = RecursiveTokenChunker(
        chunk_size_tokens=cfg.chunk_size_tokens,
        overlap_tokens=cfg.chunk_overlap_tokens,
    )

    print("[ingest] Loading embedding model (downloads ~90 MB on first run)...")
    embedder = Embedder(model=cfg.embedding_model)

    total_inserted = 0
    try:
        with open_connection(cfg.snowflake) as conn:
            repository = ChunkRepository(conn)
            for path in pdf_paths:
                try:
                    total_inserted += _process_pdf(path, chunker, embedder, repository)
                except PDFParsingError as exc:
                    print(f"[ingest] Skipping {path.name}: {exc}", file=sys.stderr)
                except ChunkingError as exc:
                    print(f"[ingest] Chunking failed for {path.name}: {exc}", file=sys.stderr)
                    return 1
                except EmbeddingError as exc:
                    print(f"[ingest] Embedding failed for {path.name}: {exc}", file=sys.stderr)
                    return 1
                except RepositoryError as exc:
                    print(f"[ingest] Database write failed for {path.name}: {exc}", file=sys.stderr)
                    return 1
    except SnowflakeConnectionError as exc:
        print(f"[ingest] {exc}", file=sys.stderr)
        return 1

    print(f"[ingest] Done. Total chunks inserted: {total_inserted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

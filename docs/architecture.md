# InsightLens — Complete Build Guide

A version-aware RAG system over investment documents. This guide walks through every file, every command, and every design decision needed to ship the project end-to-end.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites & Accounts](#2-prerequisites--accounts)
3. [Environment & Dependencies](#3-environment--dependencies)
4. [Snowflake Setup](#4-snowflake-setup)
5. [Configuration Layer](#5-configuration-layer)
6. [Storage Layer](#6-storage-layer)
7. [Ingestion Layer](#7-ingestion-layer)
8. [Embeddings Layer](#8-embeddings-layer)
9. [Retrieval Layer](#9-retrieval-layer)
10. [Generation Layer](#10-generation-layer)
11. [Streamlit UI](#11-streamlit-ui)
12. [Operational Scripts](#12-operational-scripts)
13. [Tests](#13-tests)
14. [Run Sequence](#14-run-sequence)
15. [README for Submission](#15-readme-for-submission)

---

## 1. Architecture Overview

```
PDFs (data/raw_pdfs/)
    │
    ▼
┌─────────────────┐
│  pdf_parser.py  │  Extracts text + page metadata using PyMuPDF
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ document_metadata.py │  Detects company, version, doc type
└────────┬─────────────┘
         │
         ▼
┌─────────────────┐
│   chunker.py    │  Recursive splitting, ~800 tokens, 150 overlap
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   embedder.py   │  OpenAI text-embedding-3-small (1536-dim)
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ Snowflake (VECTOR)   │  Chunks + embeddings + metadata
└────────┬─────────────┘
         │
         ▼  (query time)
┌──────────────────────┐
│ vector_search.py     │  Cosine similarity + version filtering
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ llm_client.py        │  Claude (Anthropic) for answer synthesis
│ + answer_builder.py  │  Format with citations
└────────┬─────────────┘
         │
         ▼
┌─────────────────┐
│ streamlit_app.py│
└─────────────────┘
```

**Stack rationale:**

- **OpenAI for embeddings** — `text-embedding-3-small` gives strong semantic recall at $0.02/M tokens. Cheap, fast, well-documented.
- **Claude for generation** — Anthropic models follow structured citation instructions reliably and handle multi-source conflict reasoning, which is core to this spec.
- **Snowflake for storage** — preferred by the spec, native `VECTOR` type with `VECTOR_COSINE_SIMILARITY`, scales to millions of chunks without extra infrastructure.
- **Streamlit for UI** — minimal surface area, lets the system design speak for itself.

---

## 2. Prerequisites & Accounts

You need:

- macOS with Python 3.10+
- An OpenAI API key (platform.openai.com → API keys)
- An Anthropic API key (console.anthropic.com → API Keys)
- A Snowflake account (signup.snowflake.com → 30-day free trial, $400 credits)

Add ~$5 credit to each LLM provider account — total project cost typically lands under $5.

---

## 3. Environment & Dependencies

### `requirements.txt`

```
openai==1.54.4
anthropic==0.39.0
snowflake-connector-python==3.12.3
pymupdf==1.24.13
pdfplumber==0.11.4
tiktoken==0.8.0
python-dotenv==1.0.1
streamlit==1.40.1
pydantic==2.9.2
pytest==8.3.3
```

### Install

```bash
cd ~/Desktop/rag-investment
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "insightlens"
version = "0.1.0"
description = "Version-aware RAG over investment documents"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

Then make the package importable:

```bash
pip install -e .
```

### `.env.example` (commit this)

```
OPENAI_API_KEY=sk-replace-me
ANTHROPIC_API_KEY=sk-ant-replace-me

SNOWFLAKE_ACCOUNT=xxxxx-yyyyyy
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=INSIGHTLENS
SNOWFLAKE_SCHEMA=PUBLIC
SNOWFLAKE_ROLE=ACCOUNTADMIN

EMBEDDING_MODEL=text-embedding-3-small
GENERATION_MODEL=claude-sonnet-4-6
CHUNK_SIZE_TOKENS=800
CHUNK_OVERLAP_TOKENS=150
RETRIEVAL_TOP_K=8
```

### `.env` (do NOT commit — fill in real values)

Copy `.env.example` to `.env` and replace placeholders.

### `.gitignore`

```
venv/
.env
__pycache__/
*.pyc
*.egg-info/
.DS_Store
data/raw_pdfs/*
!data/raw_pdfs/.gitkeep
data/processed/*
!data/processed/.gitkeep
.streamlit/secrets.toml
.pytest_cache/
build/
dist/
```

---

## 4. Snowflake Setup

After signup, find your account identifier in the URL: `https://app.snowflake.com/<region>/<account>/...` → the format you want is `<account>-<region>` or check Account → Account Details.

### `src/insightlens/storage/schema.sql`

```sql
CREATE DATABASE IF NOT EXISTS INSIGHTLENS;
USE DATABASE INSIGHTLENS;
USE SCHEMA PUBLIC;

CREATE TABLE IF NOT EXISTS DOCUMENTS (
    document_id     VARCHAR PRIMARY KEY,
    file_name       VARCHAR NOT NULL,
    company         VARCHAR,
    document_type   VARCHAR,
    version_label   VARCHAR,
    version_date    DATE,
    page_count      INTEGER,
    ingested_at     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS CHUNKS (
    chunk_id        VARCHAR PRIMARY KEY,
    document_id     VARCHAR NOT NULL,
    page_number     INTEGER NOT NULL,
    chunk_index     INTEGER NOT NULL,
    chunk_text      VARCHAR NOT NULL,
    token_count     INTEGER,
    embedding       VECTOR(FLOAT, 1536),
    FOREIGN KEY (document_id) REFERENCES DOCUMENTS(document_id)
);

-- Snowflake does not support secondary indexes; micro-partition pruning handles document_id lookups.
-- For large corpora, add: ALTER TABLE CHUNKS CLUSTER BY (document_id);
```

You'll run this through the script in section 12, not by hand.

---

## 5. Configuration Layer

### `src/insightlens/config.py`

```python
"""Centralized configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class ConfigError(Exception):
    """Raised when a required environment variable is missing or invalid."""


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigError(f"Environment variable {name} is not set. Check your .env file.")
    return value


def _require_int(name: str) -> int:
    raw = _require(name)
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be an integer, got '{raw}'.") from exc


@dataclass(frozen=True)
class SnowflakeConfig:
    account: str
    user: str
    password: str
    warehouse: str
    database: str
    schema: str
    role: str


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str
    anthropic_api_key: str
    embedding_model: str
    generation_model: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    retrieval_top_k: int
    snowflake: SnowflakeConfig


def load_config() -> AppConfig:
    """Load and validate all configuration. Raises ConfigError on missing values."""
    snowflake = SnowflakeConfig(
        account=_require("SNOWFLAKE_ACCOUNT"),
        user=_require("SNOWFLAKE_USER"),
        password=_require("SNOWFLAKE_PASSWORD"),
        warehouse=_require("SNOWFLAKE_WAREHOUSE"),
        database=_require("SNOWFLAKE_DATABASE"),
        schema=_require("SNOWFLAKE_SCHEMA"),
        role=_require("SNOWFLAKE_ROLE"),
    )

    return AppConfig(
        openai_api_key=_require("OPENAI_API_KEY"),
        anthropic_api_key=_require("ANTHROPIC_API_KEY"),
        embedding_model=_require("EMBEDDING_MODEL"),
        generation_model=_require("GENERATION_MODEL"),
        chunk_size_tokens=_require_int("CHUNK_SIZE_TOKENS"),
        chunk_overlap_tokens=_require_int("CHUNK_OVERLAP_TOKENS"),
        retrieval_top_k=_require_int("RETRIEVAL_TOP_K"),
        snowflake=snowflake,
    )
```

---

## 6. Storage Layer

### `src/insightlens/storage/snowflake_client.py`

```python
"""Snowflake connection management."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import snowflake.connector
from snowflake.connector import SnowflakeConnection
from snowflake.connector.errors import DatabaseError, ProgrammingError

from insightlens.config import SnowflakeConfig


class SnowflakeConnectionError(Exception):
    """Raised when Snowflake cannot be reached or authenticated."""


@contextmanager
def open_connection(cfg: SnowflakeConfig) -> Iterator[SnowflakeConnection]:
    """Open a Snowflake connection with the given config. Yields a connection."""
    try:
        conn = snowflake.connector.connect(
            account=cfg.account,
            user=cfg.user,
            password=cfg.password,
            warehouse=cfg.warehouse,
            database=cfg.database,
            schema=cfg.schema,
            role=cfg.role,
        )
    except DatabaseError as exc:
        raise SnowflakeConnectionError(
            f"Failed to authenticate to Snowflake account '{cfg.account}' as user '{cfg.user}'. "
            f"Underlying error: {exc}"
        ) from exc

    try:
        yield conn
    finally:
        conn.close()


def execute_script(conn: SnowflakeConnection, sql_text: str) -> None:
    """Execute a multi-statement SQL script."""
    cursor = conn.cursor()
    try:
        for statement in _split_statements(sql_text):
            if statement.strip():
                cursor.execute(statement)
    except ProgrammingError as exc:
        raise SnowflakeConnectionError(
            f"Snowflake rejected SQL statement. Underlying error: {exc.msg} "
            f"(error code: {exc.errno})"
        ) from exc
    finally:
        cursor.close()


def _split_statements(sql_text: str) -> list[str]:
    return [stmt for stmt in sql_text.split(";") if stmt.strip()]
```

### `src/insightlens/storage/chunk_repository.py`

```python
"""Persistence operations for documents and chunks."""
from __future__ import annotations

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

        sql = """
            INSERT INTO CHUNKS
                (chunk_id, document_id, page_number, chunk_index, chunk_text, token_count, embedding)
            SELECT
                column1, column2, column3, column4, column5, column6, column7::VECTOR(FLOAT, 1536)
            FROM VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor = self._conn.cursor()
        inserted = 0
        try:
            for chunk in rows:
                cursor.execute(
                    sql,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.page_number,
                        chunk.chunk_index,
                        chunk.chunk_text,
                        chunk.token_count,
                        list(chunk.embedding),
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
        base_sql = """
            SELECT
                c.chunk_id,
                c.document_id,
                d.file_name,
                d.company,
                d.version_label,
                c.page_number,
                c.chunk_text,
                VECTOR_COSINE_SIMILARITY(c.embedding, %s::VECTOR(FLOAT, 1536)) AS similarity
            FROM CHUNKS c
            JOIN DOCUMENTS d ON c.document_id = d.document_id
        """
        params: list = [list(query_embedding)]
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
```

---

## 7. Ingestion Layer

### `src/insightlens/ingestion/pdf_parser.py`

```python
"""PDF text extraction with page-level metadata."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz


class PDFParsingError(Exception):
    """Raised when a PDF cannot be opened or parsed."""


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str
    char_count: int
    is_likely_visual: bool


@dataclass(frozen=True)
class ParsedDocument:
    file_path: Path
    pages: list[ParsedPage]

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


def parse_pdf(path: Path) -> ParsedDocument:
    """Extract text from a PDF, one entry per page."""
    if not path.exists():
        raise PDFParsingError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise PDFParsingError(f"Expected .pdf extension, got: {path.suffix}")

    try:
        document = fitz.open(path)
    except fitz.FileDataError as exc:
        raise PDFParsingError(f"Corrupt or unreadable PDF: {path.name}. Error: {exc}") from exc

    pages: list[ParsedPage] = []
    try:
        for index, page in enumerate(document):
            text = page.get_text("text") or ""
            stripped = text.strip()
            pages.append(
                ParsedPage(
                    page_number=index + 1,
                    text=stripped,
                    char_count=len(stripped),
                    is_likely_visual=len(stripped) < 50,
                )
            )
    finally:
        document.close()

    if not any(page.text for page in pages):
        raise PDFParsingError(
            f"No extractable text found in {path.name}. "
            f"This may be a scanned PDF requiring OCR."
        )

    return ParsedDocument(file_path=path, pages=pages)
```

### `src/insightlens/ingestion/document_metadata.py`

```python
"""Heuristic detection of company, document type, and version from filenames and content."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class DocumentMetadata:
    company: str | None
    document_type: str | None
    version_label: str | None
    version_date: date | None


_VERSION_PATTERNS = [
    re.compile(r"\bv(\d+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bversion[\s_-]?(\d+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\b(20\d{2})[\s_-]?(Q[1-4])\b", re.IGNORECASE),
    re.compile(r"\b(Q[1-4])[\s_-]?(20\d{2})\b", re.IGNORECASE),
    re.compile(r"\b(20\d{2})\b"),
]

_DATE_PATTERN = re.compile(r"(20\d{2})[-_/](\d{1,2})[-_/](\d{1,2})")

_DOC_TYPE_KEYWORDS = {
    "investor_presentation": ["investor", "presentation", "deck", "pitch"],
    "strategy": ["strategy", "strategic", "plan"],
    "third_party_report": ["report", "analysis", "research"],
    "annual_report": ["annual", "10-k", "10k"],
    "quarterly": ["10-q", "10q", "quarterly"],
}


def extract_metadata(file_path: Path, first_page_text: str) -> DocumentMetadata:
    """Combine filename signals and first-page content to infer metadata."""
    stem = file_path.stem
    haystack = f"{stem} {first_page_text[:2000]}"

    return DocumentMetadata(
        company=_detect_company(stem, first_page_text),
        document_type=_detect_doc_type(haystack),
        version_label=_detect_version_label(stem),
        version_date=_detect_date(haystack),
    )


def _detect_company(stem: str, first_page_text: str) -> str | None:
    candidate = stem.split("_")[0].split("-")[0].strip()
    if candidate and len(candidate) >= 2 and not candidate.isdigit():
        return candidate.title()

    lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
    for line in lines[:5]:
        if 2 <= len(line.split()) <= 6 and line[0].isupper():
            return line
    return None


def _detect_doc_type(haystack: str) -> str | None:
    lowered = haystack.lower()
    for label, keywords in _DOC_TYPE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return label
    return None


def _detect_version_label(stem: str) -> str | None:
    for pattern in _VERSION_PATTERNS:
        match = pattern.search(stem)
        if match:
            return "_".join(group for group in match.groups() if group)
    return None


def _detect_date(haystack: str) -> date | None:
    match = _DATE_PATTERN.search(haystack)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None
```

### `src/insightlens/ingestion/chunker.py`

```python
"""Token-aware recursive text chunking."""
from __future__ import annotations

from dataclasses import dataclass

import tiktoken


class ChunkingError(Exception):
    """Raised when chunking parameters are invalid or text cannot be tokenized."""


@dataclass(frozen=True)
class TextChunk:
    text: str
    token_count: int
    page_number: int
    chunk_index: int


_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveTokenChunker:
    """Splits text into chunks bounded by token count, preferring natural boundaries."""

    def __init__(self, chunk_size_tokens: int, overlap_tokens: int, encoding_name: str = "cl100k_base") -> None:
        if chunk_size_tokens <= 0:
            raise ChunkingError(f"chunk_size_tokens must be positive, got {chunk_size_tokens}")
        if overlap_tokens < 0 or overlap_tokens >= chunk_size_tokens:
            raise ChunkingError(
                f"overlap_tokens ({overlap_tokens}) must be in [0, chunk_size_tokens={chunk_size_tokens})"
            )

        self._chunk_size = chunk_size_tokens
        self._overlap = overlap_tokens
        self._encoder = tiktoken.get_encoding(encoding_name)

    def chunk_page(self, text: str, page_number: int, starting_chunk_index: int) -> list[TextChunk]:
        if not text.strip():
            return []

        pieces = self._split_recursively(text, _SEPARATORS)
        chunks = self._merge_pieces_to_chunks(pieces)

        return [
            TextChunk(
                text=chunk_text,
                token_count=len(self._encoder.encode(chunk_text)),
                page_number=page_number,
                chunk_index=starting_chunk_index + offset,
            )
            for offset, chunk_text in enumerate(chunks)
        ]

    def _split_recursively(self, text: str, separators: list[str]) -> list[str]:
        if len(self._encoder.encode(text)) <= self._chunk_size:
            return [text]

        if not separators:
            return self._hard_split_by_tokens(text)

        separator = separators[0]
        rest = separators[1:]

        if separator == "":
            return self._hard_split_by_tokens(text)

        parts = text.split(separator)
        result: list[str] = []
        for part in parts:
            if len(self._encoder.encode(part)) <= self._chunk_size:
                result.append(part)
            else:
                result.extend(self._split_recursively(part, rest))
        return result

    def _hard_split_by_tokens(self, text: str) -> list[str]:
        tokens = self._encoder.encode(text)
        return [
            self._encoder.decode(tokens[i : i + self._chunk_size])
            for i in range(0, len(tokens), self._chunk_size)
        ]

    def _merge_pieces_to_chunks(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current_tokens: list[int] = []

        for piece in pieces:
            piece_tokens = self._encoder.encode(piece)
            if not piece_tokens:
                continue

            if len(current_tokens) + len(piece_tokens) <= self._chunk_size:
                if current_tokens:
                    current_tokens.extend(self._encoder.encode(" "))
                current_tokens.extend(piece_tokens)
            else:
                if current_tokens:
                    chunks.append(self._encoder.decode(current_tokens))
                    overlap_start = max(0, len(current_tokens) - self._overlap)
                    current_tokens = current_tokens[overlap_start:] + piece_tokens
                else:
                    current_tokens = piece_tokens

        if current_tokens:
            chunks.append(self._encoder.decode(current_tokens))

        return chunks
```

---

## 8. Embeddings Layer

### `src/insightlens/embeddings/embedder.py`

```python
"""OpenAI embedding client with batching."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from openai import APIError, APIStatusError, OpenAI, RateLimitError


class EmbeddingError(Exception):
    """Raised when embeddings cannot be generated."""


@dataclass(frozen=True)
class EmbeddingResult:
    text: str
    vector: list[float]


class OpenAIEmbedder:
    """Wraps OpenAI's embedding endpoint with batched calls."""

    def __init__(self, api_key: str, model: str, batch_size: int = 64) -> None:
        if not api_key:
            raise EmbeddingError("OpenAI API key is empty.")
        if batch_size <= 0:
            raise EmbeddingError(f"batch_size must be positive, got {batch_size}")

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._batch_size = batch_size

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        results: list[EmbeddingResult] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            results.extend(self._embed_batch(batch))
        return results

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise EmbeddingError("Cannot embed an empty query.")
        return self._embed_batch([query])[0].vector

    def _embed_batch(self, batch: list[str]) -> list[EmbeddingResult]:
        try:
            response = self._client.embeddings.create(input=batch, model=self._model)
        except RateLimitError as exc:
            raise EmbeddingError(
                f"OpenAI rate limit hit while embedding batch of {len(batch)}. "
                f"Wait and retry, or reduce batch size. Detail: {exc}"
            ) from exc
        except APIStatusError as exc:
            raise EmbeddingError(
                f"OpenAI returned status {exc.status_code} for embedding request. "
                f"Detail: {exc.message}"
            ) from exc
        except APIError as exc:
            raise EmbeddingError(f"OpenAI API error during embedding: {exc}") from exc

        if len(response.data) != len(batch):
            raise EmbeddingError(
                f"OpenAI returned {len(response.data)} embeddings for batch of {len(batch)} inputs."
            )

        return [
            EmbeddingResult(text=text, vector=item.embedding)
            for text, item in zip(batch, response.data, strict=True)
        ]
```

---

## 9. Retrieval Layer

### `src/insightlens/retrieval/vector_search.py`

```python
"""High-level retrieval orchestration: embed query → search → return ranked chunks."""
from __future__ import annotations

from dataclasses import dataclass

from insightlens.embeddings.embedder import OpenAIEmbedder
from insightlens.storage.chunk_repository import ChunkRepository, RetrievedChunk


class RetrievalError(Exception):
    """Raised when the retrieval pipeline fails."""


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    top_k: int
    company_filter: str | None = None


class VectorSearchService:
    """Coordinates query embedding and vector lookup."""

    def __init__(self, embedder: OpenAIEmbedder, repository: ChunkRepository) -> None:
        self._embedder = embedder
        self._repository = repository

    def retrieve(self, request: RetrievalRequest) -> list[RetrievedChunk]:
        if not request.query.strip():
            raise RetrievalError("Query is empty.")
        if request.top_k <= 0:
            raise RetrievalError(f"top_k must be positive, got {request.top_k}")

        query_vector = self._embedder.embed_query(request.query)
        return self._repository.search_similar(
            query_embedding=query_vector,
            top_k=request.top_k,
            company_filter=request.company_filter,
        )
```

---

## 10. Generation Layer

### `src/insightlens/generation/prompts.py`

```python
"""Prompt templates for answer generation."""
from __future__ import annotations

from insightlens.storage.chunk_repository import RetrievedChunk

SYSTEM_PROMPT = """You are an analyst assistant that answers questions about investment documents.

Rules you must follow:
1. Ground every factual claim in the provided sources. If the sources do not contain the answer, say so plainly.
2. When sources from different document versions disagree, present each view separately with attribution. Never silently merge conflicting numbers.
3. Cite sources inline using the format [Source N]. Each source corresponds to one entry in the source list.
4. If a question asks about charts, images, or visual content, acknowledge that visual extraction is limited and answer from any available text.
5. Keep answers concise. Lead with the direct answer; supporting detail follows."""


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return (
            f"Question: {question}\n\n"
            "No source material was retrieved for this question. "
            "Reply that the corpus does not appear to contain information relevant to the question."
        )

    source_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        version = chunk.version_label or "unversioned"
        company = chunk.company or "unknown company"
        header = f"[Source {index}] {chunk.file_name} (company: {company}, version: {version}, page: {chunk.page_number})"
        source_blocks.append(f"{header}\n{chunk.chunk_text}")

    sources_text = "\n\n".join(source_blocks)
    return (
        f"Question: {question}\n\n"
        f"Sources:\n{sources_text}\n\n"
        "Provide an answer that follows the rules. Use [Source N] inline citations."
    )
```

### `src/insightlens/generation/llm_client.py`

```python
"""Anthropic Claude client for answer generation."""
from __future__ import annotations

from anthropic import Anthropic, APIError, APIStatusError, RateLimitError


class GenerationError(Exception):
    """Raised when the LLM fails to produce an answer."""


class ClaudeClient:
    """Thin wrapper around Anthropic's messages API."""

    def __init__(self, api_key: str, model: str, max_tokens: int = 1024) -> None:
        if not api_key:
            raise GenerationError("Anthropic API key is empty.")
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except RateLimitError as exc:
            raise GenerationError(f"Anthropic rate limit reached: {exc}") from exc
        except APIStatusError as exc:
            raise GenerationError(
                f"Anthropic returned status {exc.status_code}: {exc.message}"
            ) from exc
        except APIError as exc:
            raise GenerationError(f"Anthropic API error: {exc}") from exc

        if not response.content:
            raise GenerationError("Anthropic returned an empty response.")

        text_blocks = [block.text for block in response.content if block.type == "text"]
        if not text_blocks:
            raise GenerationError("Anthropic response contained no text blocks.")

        return "\n".join(text_blocks).strip()
```

### `src/insightlens/generation/answer_builder.py`

```python
"""Combines retrieval + generation and packages the response with citations."""
from __future__ import annotations

from dataclasses import dataclass

from insightlens.generation.llm_client import ClaudeClient
from insightlens.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from insightlens.retrieval.vector_search import RetrievalRequest, VectorSearchService
from insightlens.storage.chunk_repository import RetrievedChunk


@dataclass(frozen=True)
class Citation:
    label: str
    file_name: str
    company: str | None
    version_label: str | None
    page_number: int
    similarity: float


@dataclass(frozen=True)
class AnswerWithSources:
    answer_text: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]


class AnswerService:
    """End-to-end question answering."""

    def __init__(
        self,
        retrieval: VectorSearchService,
        llm: ClaudeClient,
        default_top_k: int,
    ) -> None:
        self._retrieval = retrieval
        self._llm = llm
        self._default_top_k = default_top_k

    def answer(self, question: str, company_filter: str | None = None) -> AnswerWithSources:
        chunks = self._retrieval.retrieve(
            RetrievalRequest(
                query=question,
                top_k=self._default_top_k,
                company_filter=company_filter,
            )
        )

        user_prompt = build_user_prompt(question, chunks)
        answer_text = self._llm.generate(SYSTEM_PROMPT, user_prompt)

        citations = [
            Citation(
                label=f"Source {index}",
                file_name=chunk.file_name,
                company=chunk.company,
                version_label=chunk.version_label,
                page_number=chunk.page_number,
                similarity=chunk.similarity,
            )
            for index, chunk in enumerate(chunks, start=1)
        ]

        return AnswerWithSources(
            answer_text=answer_text,
            citations=citations,
            retrieved_chunks=chunks,
        )
```

---

## 11. Streamlit UI

### `src/insightlens/ui/streamlit_app.py`

```python
"""Streamlit application: question input, answer display, source citations."""
from __future__ import annotations

import streamlit as st

from insightlens.config import ConfigError, load_config
from insightlens.embeddings.embedder import OpenAIEmbedder
from insightlens.generation.answer_builder import AnswerService
from insightlens.generation.llm_client import ClaudeClient
from insightlens.retrieval.vector_search import VectorSearchService
from insightlens.storage.chunk_repository import ChunkRepository
from insightlens.storage.snowflake_client import open_connection


st.set_page_config(page_title="InsightLens", layout="wide")
st.title("InsightLens")
st.caption("Ask questions about your investment document corpus.")


@st.cache_resource
def _bootstrap():
    cfg = load_config()
    embedder = OpenAIEmbedder(api_key=cfg.openai_api_key, model=cfg.embedding_model)
    llm = ClaudeClient(api_key=cfg.anthropic_api_key, model=cfg.generation_model)
    return cfg, embedder, llm


@st.cache_data(ttl=300)
def _load_companies(_cfg):
    with open_connection(_cfg.snowflake) as conn:
        return ChunkRepository(conn).list_companies()


try:
    cfg, embedder, llm = _bootstrap()
except ConfigError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()


companies = _load_companies(cfg)

with st.sidebar:
    st.header("Filters")
    company_choice = st.selectbox(
        "Restrict to company",
        options=["All companies"] + companies,
        index=0,
    )
    top_k = st.slider("Sources to retrieve", min_value=3, max_value=15, value=cfg.retrieval_top_k)

question = st.text_area("Question", height=100, placeholder="e.g. What is Company X's key strategy?")
ask_button = st.button("Ask", type="primary", disabled=not question.strip())

if ask_button:
    company_filter = None if company_choice == "All companies" else company_choice

    with st.spinner("Retrieving and generating..."):
        with open_connection(cfg.snowflake) as conn:
            repository = ChunkRepository(conn)
            retrieval = VectorSearchService(embedder=embedder, repository=repository)
            service = AnswerService(retrieval=retrieval, llm=llm, default_top_k=top_k)
            result = service.answer(question, company_filter=company_filter)

    st.subheader("Answer")
    st.write(result.answer_text)

    st.subheader("Sources")
    for citation in result.citations:
        with st.expander(
            f"{citation.label} — {citation.file_name} (page {citation.page_number}, similarity {citation.similarity:.3f})"
        ):
            st.markdown(f"**Company:** {citation.company or 'unknown'}")
            st.markdown(f"**Version:** {citation.version_label or 'unversioned'}")
            matching = next(
                (c for c in result.retrieved_chunks if c.chunk_id and c.page_number == citation.page_number and c.file_name == citation.file_name),
                None,
            )
            if matching:
                st.text_area("Excerpt", value=matching.chunk_text, height=200, disabled=True, key=citation.label)
```

---

## 12. Operational Scripts

### `scripts/setup_database.py`

```python
"""Creates Snowflake database, schema, and tables defined in schema.sql."""
from __future__ import annotations

import sys
from pathlib import Path

from insightlens.config import ConfigError, load_config
from insightlens.storage.snowflake_client import (
    SnowflakeConnectionError,
    execute_script,
    open_connection,
)


def main() -> int:
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"[setup_database] Configuration error: {exc}", file=sys.stderr)
        return 1

    schema_path = Path(__file__).resolve().parents[1] / "src" / "insightlens" / "storage" / "schema.sql"
    if not schema_path.exists():
        print(f"[setup_database] Schema file missing: {schema_path}", file=sys.stderr)
        return 1

    sql_text = schema_path.read_text()

    try:
        with open_connection(cfg.snowflake) as conn:
            execute_script(conn, sql_text)
    except SnowflakeConnectionError as exc:
        print(f"[setup_database] {exc}", file=sys.stderr)
        return 1

    print("[setup_database] Schema applied successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### `scripts/ingest_documents.py`

```python
"""Walks data/raw_pdfs, parses + chunks + embeds + persists every PDF."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from insightlens.config import RAW_PDF_DIR, ConfigError, load_config
from insightlens.embeddings.embedder import EmbeddingError, OpenAIEmbedder
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
    embedder: OpenAIEmbedder,
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
    embedder = OpenAIEmbedder(api_key=cfg.openai_api_key, model=cfg.embedding_model)

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
```

### `run.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

source venv/bin/activate
streamlit run src/insightlens/ui/streamlit_app.py
```

Make it executable: `chmod +x run.sh`

---

## 13. Tests

### `tests/test_chunker.py`

```python
from insightlens.ingestion.chunker import ChunkingError, RecursiveTokenChunker

import pytest


def test_short_text_yields_single_chunk():
    chunker = RecursiveTokenChunker(chunk_size_tokens=100, overlap_tokens=10)
    chunks = chunker.chunk_page("This is a short paragraph.", page_number=1, starting_chunk_index=0)
    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0


def test_empty_text_yields_no_chunks():
    chunker = RecursiveTokenChunker(chunk_size_tokens=100, overlap_tokens=10)
    assert chunker.chunk_page("   ", page_number=2, starting_chunk_index=5) == []


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ChunkingError):
        RecursiveTokenChunker(chunk_size_tokens=50, overlap_tokens=50)


def test_long_text_produces_multiple_chunks():
    chunker = RecursiveTokenChunker(chunk_size_tokens=20, overlap_tokens=5)
    long_text = " ".join(["sentence number {}".format(i) for i in range(60)])
    chunks = chunker.chunk_page(long_text, page_number=3, starting_chunk_index=10)
    assert len(chunks) > 1
    assert chunks[0].chunk_index == 10
    assert all(chunk.page_number == 3 for chunk in chunks)
```

### `tests/test_metadata_extraction.py`

```python
from pathlib import Path

from insightlens.ingestion.document_metadata import extract_metadata


def test_extracts_version_from_filename():
    metadata = extract_metadata(Path("Acme_InvestorDeck_v2.pdf"), "Acme Corp\nInvestor Update")
    assert metadata.company == "Acme"
    assert metadata.version_label == "2"
    assert metadata.document_type == "investor_presentation"


def test_extracts_quarter_year():
    metadata = extract_metadata(Path("Acme_2024_Q3.pdf"), "Acme quarterly report")
    assert metadata.version_label is not None
    assert "2024" in metadata.version_label or "Q3" in metadata.version_label.upper()


def test_unknown_filename_falls_back_to_first_page():
    metadata = extract_metadata(Path("file.pdf"), "Initech Holdings\nStrategy Review 2023")
    assert metadata.company is not None
```

Run with: `pytest`

---

## 14. Run Sequence

Execute in this order:

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Install dependencies and the package
pip install -r requirements.txt
pip install -e .

# 3. Create Snowflake schema (one-time)
python scripts/setup_database.py

# 4. Ingest documents (re-run after adding/changing PDFs)
python scripts/ingest_documents.py

# 5. Run tests
pytest

# 6. Launch the app
./run.sh
```

The Streamlit app opens at `http://localhost:8501`.

---

## 15. README for Submission

Save this as `README.md` at the project root:

```markdown
# InsightLens

A version-aware retrieval-augmented generation system over investment documents. Ask natural-language questions across investor decks, strategy materials, and third-party reports; receive answers grounded in the source corpus with inline citations.

## Architecture

PDFs are parsed page-by-page, tagged with company and version metadata, chunked with a recursive token-aware splitter, embedded with OpenAI `text-embedding-3-small`, and stored in Snowflake's native `VECTOR` column type. At query time, the question is embedded with the same model, ranked by `VECTOR_COSINE_SIMILARITY`, and the top chunks are passed to Claude with a prompt that enforces citation discipline and conflict preservation.

## Database

Snowflake stores two tables. `DOCUMENTS` holds per-file metadata: company, document type, version label, version date, page count. `CHUNKS` holds the chunked text, token counts, and a `VECTOR(FLOAT, 1536)` embedding, joined to `DOCUMENTS` by `document_id`. Cosine similarity is computed in-database, which avoids shipping all vectors to the application.

## Chunking Strategy

Recursive splitting at 800 tokens with 150-token overlap. Split candidates are tried in order: paragraph, line, sentence, word, character. This keeps semantically coherent units together when possible and falls back gracefully on dense layouts. Chunks carry their source page number so citations remain page-accurate.

## Retrieval Approach

Top-k cosine similarity (default k=8) over the full corpus. The UI exposes an optional company filter, applied at SQL level. The retrieved chunks are passed to the LLM with their full metadata block (file name, company, version, page) so the model can attribute claims correctly.

## Version Awareness

Each document is tagged with a `version_label` parsed from the filename (e.g. `v2`, `2024_Q3`) and a `version_date` when extractable. When multiple versions of the same company's materials are retrieved for a single query, the LLM sees them as distinct sources and is instructed to keep claims attributed to the specific version. The UI exposes companies as filter options for queries that need to be locked to a specific version family.

## Conflict Handling

The system prompt explicitly forbids merging conflicting numbers across sources. When the retrieved chunks contain contradicting facts, the model is required to present each view with its own citation. The UI renders every retrieved source so the user can verify the conflict directly.

## Charts and Tables

Text inside tables is captured by PyMuPDF when the PDF is text-based. Chart content rendered as images is not extracted; pages with very low text density are flagged in `is_likely_visual` during parsing and contribute reduced material to retrieval. This is documented as a known limitation rather than masked by hallucination — the prompt explicitly instructs the model to acknowledge visual-content limits.

## Known Limitations

- Scanned PDFs without OCR will fail ingestion; an OCR fallback (Tesseract or a vision model) is the natural next step.
- Chart and image content is not embedded; multimodal embeddings (CLIP or a vision-language model) would close this gap.
- Conflict detection is currently the LLM's responsibility; an explicit numerical extraction layer would catch contradictions deterministically.
- Single-query retrieval — no query rewriting, no multi-hop reasoning, no agentic retrieval.

## What I Would Improve With More Time

- Hybrid retrieval combining BM25 with dense vectors for proper-noun heavy queries.
- A cross-encoder reranker to lift precision in the top 3.
- Structured table extraction with `pdfplumber` and a separate "fact" index for numerical claims.
- Eval harness measuring retrieval recall@k and answer faithfulness on a small labeled set.
- Caching of embeddings keyed by chunk hash to make re-ingestion idempotent and fast.

## Setup

Requires Python 3.10+, an OpenAI API key, an Anthropic API key, and a Snowflake account.

```bash
git clone <repo>
cd rag-investment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

cp .env.example .env
# Fill in API keys and Snowflake credentials in .env

python scripts/setup_database.py
# Drop PDFs into data/raw_pdfs/
python scripts/ingest_documents.py

./run.sh
```

## Running Tests

```bash
pytest
```
```

---

## Final Sanity Checklist Before Demo

- All PDFs ingested without errors → `python scripts/ingest_documents.py` exits 0
- Snowflake `CHUNKS` table populated → `SELECT COUNT(*) FROM CHUNKS;` returns expected count
- Streamlit launches without ConfigError → all `.env` values present
- A query returns citations and the expanders show real chunk text
- A version-comparison query (e.g. "How did revenue change between v1 and v2 of CompanyX's deck?") triggers the model to cite both versions separately
- The README architecture section maps onto the actual code structure

---

End of build guide.
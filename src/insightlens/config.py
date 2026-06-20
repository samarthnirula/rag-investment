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

# ── Upload limits (shared by billing, ingest, and nginx) ─────────────────────
MAX_FILE_SIZE_MB: int = 50
MAX_PAGES: int = 500


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


# Keep SnowflakeConfig as an alias so any code that imports it by name still works.
@dataclass(frozen=True)
class PostgresConfig:
    database_url: str


SnowflakeConfig = PostgresConfig  # backward-compat alias


@dataclass(frozen=True)
class AppConfig:
    anthropic_api_key: str
    zep_api_key: str
    zep_enabled: bool
    embedding_model: str
    generation_model: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    retrieval_top_k: int
    db: PostgresConfig
    voyage_api_key: str = ""
    redis_url: str = "redis://localhost:6379/0"


def load_config() -> AppConfig:
    """Load and validate all configuration. Raises ConfigError on missing values."""
    db = PostgresConfig(
        database_url=_require("DATABASE_URL"),
    )

    return AppConfig(
        anthropic_api_key=_require("ANTHROPIC_API_KEY"),
        zep_api_key=os.getenv("ZEP_API_KEY", ""),
        zep_enabled=os.getenv("ZEP_ENABLED", "true").lower() not in {"0", "false", "no"},
        embedding_model=_require("EMBEDDING_MODEL"),
        generation_model=_require("GENERATION_MODEL"),
        chunk_size_tokens=int(os.getenv("CHUNK_SIZE_TOKENS", "400")),
        chunk_overlap_tokens=int(os.getenv("CHUNK_OVERLAP_TOKENS", "50")),
        retrieval_top_k=_require_int("RETRIEVAL_TOP_K"),
        db=db,
        voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    )

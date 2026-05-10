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
        anthropic_api_key=_require("ANTHROPIC_API_KEY"),
        embedding_model=_require("EMBEDDING_MODEL"),
        generation_model=_require("GENERATION_MODEL"),
        chunk_size_tokens=_require_int("CHUNK_SIZE_TOKENS"),
        chunk_overlap_tokens=_require_int("CHUNK_OVERLAP_TOKENS"),
        retrieval_top_k=_require_int("RETRIEVAL_TOP_K"),
        snowflake=snowflake,
    )

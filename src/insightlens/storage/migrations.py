"""Small PostgreSQL migration runner for Atticus."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import psycopg2

from insightlens.storage.snowflake_client import SnowflakeConnection, execute_script


MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"


@dataclass(frozen=True)
class MigrationResult:
    version: str
    name: str
    applied: bool


def ensure_migration_table(conn: SnowflakeConnection) -> None:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version     TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                applied_at  TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
    finally:
        cur.close()


def applied_versions(conn: SnowflakeConnection) -> set[str]:
    ensure_migration_table(conn)
    cur = conn.cursor()
    try:
        cur.execute("SELECT version FROM schema_migrations")
        return {row[0] for row in cur.fetchall()}
    finally:
        cur.close()


def list_migration_files(migrations_dir: Path = MIGRATIONS_DIR) -> list[Path]:
    if not migrations_dir.exists():
        return []
    return sorted(path for path in migrations_dir.glob("*.sql") if path.is_file())


def _parse_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    version, _, name = stem.partition("_")
    return version, name.replace("_", " ") or stem


def apply_migrations(
    conn: SnowflakeConnection,
    migrations_dir: Path = MIGRATIONS_DIR,
) -> list[MigrationResult]:
    ensure_migration_table(conn)
    done = applied_versions(conn)
    results: list[MigrationResult] = []

    for path in list_migration_files(migrations_dir):
        version, name = _parse_name(path)
        if version in done:
            results.append(MigrationResult(version, name, applied=False))
            continue

        execute_script(conn, path.read_text())
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO schema_migrations (version, name) VALUES (%s, %s)",
                (version, name),
            )
        except psycopg2.Error as exc:
            raise RuntimeError(f"Failed recording migration {path.name}: {exc}") from exc
        finally:
            cur.close()
        results.append(MigrationResult(version, name, applied=True))

    return results

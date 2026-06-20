"""Creates/updates the PostgreSQL database schema and applies migrations."""
from __future__ import annotations

import sys
from pathlib import Path

from insightlens.config import ConfigError, load_config
from insightlens.storage.snowflake_client import (
    SnowflakeConnectionError,
    execute_script,
    open_connection,
)
from insightlens.storage.migrations import apply_migrations


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
        with open_connection(cfg.db) as conn:
            execute_script(conn, sql_text)
            migrations = apply_migrations(conn)
    except SnowflakeConnectionError as exc:
        print(f"[setup_database] {exc}", file=sys.stderr)
        return 1

    print("[setup_database] Schema applied successfully.")
    applied = [m for m in migrations if m.applied]
    skipped = [m for m in migrations if not m.applied]
    if applied:
        print("[setup_database] Applied migrations: " + ", ".join(m.version for m in applied))
    if skipped:
        print("[setup_database] Already applied: " + ", ".join(m.version for m in skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Run all pending migrations against the database.

Usage:
    python scripts/run_migrations.py

Requires DATABASE_URL in .env or environment.  Reads all .sql files from the
``migrations/`` directory in alphabetical order and applies any that have not
yet been recorded in the ``schema_migrations`` tracking table.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Bootstrap path so dotenv and psycopg2 are importable even when run from
# the project root without installing the package.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

import psycopg2


_MIGRATIONS_DIR = _ROOT / "migrations"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_name TEXT PRIMARY KEY,
    applied_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def _connect() -> psycopg2.extensions.connection:
    url = os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL is not set.  Add it to .env or the environment.", file=sys.stderr)
        sys.exit(1)
    return psycopg2.connect(url)


def _applied(conn, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM schema_migrations WHERE migration_name = %s", (name,))
        return cur.fetchone() is not None


def _record(conn, name: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO schema_migrations (migration_name) VALUES (%s) ON CONFLICT DO NOTHING",
            (name,),
        )


def main() -> None:
    sql_files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
    if not sql_files:
        print("No migration files found in", _MIGRATIONS_DIR)
        sys.exit(0)

    conn = _connect()
    conn.autocommit = False

    try:
        # Ensure the tracking table exists.
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLE)
        conn.commit()

        failed = False
        for sql_path in sql_files:
            name = sql_path.name
            if _applied(conn, name):
                print(f"⏭  Skipped {name} (already applied)")
                continue

            print(f"⏳ Applying {name}…", end=" ", flush=True)
            try:
                sql = sql_path.read_text(encoding="utf-8")
                with conn.cursor() as cur:
                    cur.execute(sql)
                _record(conn, name)
                conn.commit()
                print(f"✅ Applied {name}")
            except Exception as exc:
                conn.rollback()
                print(f"❌ FAILED {name}: {exc}", file=sys.stderr)
                failed = True
                break  # stop on first failure to avoid cascading issues

    finally:
        conn.close()

    if failed:
        sys.exit(1)
    print("\nAll migrations complete.")


if __name__ == "__main__":
    main()

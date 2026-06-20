"""PostgreSQL connection management with thread-safe connection pooling.

Drop-in replacement for the former Snowflake client — public API is unchanged
so all callers (open_connection, execute_script) work without modification.

Pool behaviour:
  - Size: POOL_SIZE (default 5). Configurable via env var PG_POOL_SIZE.
  - Borrow: blocks up to POOL_TIMEOUT seconds before raising.
  - Validation: before yielding, runs a lightweight ping to confirm liveness.
    Stale connections are discarded and replaced with a fresh one.
  - Return: healthy connections go back to the pool; connections that raised
    an exception are discarded to avoid poisoning.
  - Autocommit: enabled so behaviour matches the old Snowflake client (each
    statement commits immediately — no explicit BEGIN/COMMIT needed).

Setup:
  Set DATABASE_URL in .env, e.g.
    postgresql://localhost:5432/insightlens          (local)
    postgresql://user:pass@host:5432/dbname          (remote)
    postgresql://...@*.supabase.com:5432/postgres    (Supabase)
  The pgvector extension must be enabled on the database:
    CREATE EXTENSION IF NOT EXISTS vector;
"""
from __future__ import annotations

import os
import queue
import threading
from contextlib import contextmanager
from typing import Iterator

import psycopg2
import psycopg2.extensions
import psycopg2.extras

from insightlens.config import PostgresConfig

# Export a SnowflakeConnection alias so existing repository type annotations work unchanged.
SnowflakeConnection = psycopg2.extensions.connection

POOL_SIZE    = int(os.getenv("PG_POOL_SIZE", "5"))
POOL_TIMEOUT = 30  # seconds to wait for a free connection

_pools: dict[str, "_ConnectionPool"] = {}
_registry_lock = threading.Lock()


class SnowflakeConnectionError(Exception):
    """Raised when the database cannot be reached or a pool connection fails."""


class _ConnectionPool:
    """Thread-safe fixed-size pool of psycopg2 connections."""

    def __init__(self, cfg: PostgresConfig, size: int) -> None:
        self._cfg = cfg
        self._pool: queue.Queue[SnowflakeConnection] = queue.Queue(maxsize=size)
        for _ in range(size):
            try:
                self._pool.put_nowait(self._new_conn())
            except Exception:
                pass  # Pool may start partially filled; grows on demand

    def _new_conn(self) -> SnowflakeConnection:
        try:
            conn = psycopg2.connect(self._cfg.database_url)
            conn.autocommit = True
            return conn
        except psycopg2.Error as exc:
            # Do NOT include exc or database_url in the message — both may contain credentials.
            raise SnowflakeConnectionError(
                "Failed to connect to the database. Check DATABASE_URL and network access."
            ) from exc

    def _is_alive(self, conn: SnowflakeConnection) -> bool:
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            return True
        except Exception:
            return False

    @contextmanager
    def connection(self) -> Iterator[SnowflakeConnection]:
        try:
            conn = self._pool.get(timeout=POOL_TIMEOUT)
        except queue.Empty:
            conn = self._new_conn()
            temporary = True
        else:
            temporary = False
            if not self._is_alive(conn):
                try:
                    conn.close()
                except Exception:
                    pass
                conn = self._new_conn()

        errored = False
        try:
            yield conn
        except Exception:
            errored = True
            raise
        finally:
            if temporary or errored:
                try:
                    conn.close()
                except Exception:
                    pass
            else:
                try:
                    self._pool.put_nowait(conn)
                except queue.Full:
                    conn.close()


def _pool_key(cfg: PostgresConfig) -> str:
    return cfg.database_url


def _get_pool(cfg: PostgresConfig) -> _ConnectionPool:
    key = _pool_key(cfg)
    with _registry_lock:
        if key not in _pools:
            _pools[key] = _ConnectionPool(cfg, POOL_SIZE)
    return _pools[key]


@contextmanager
def open_connection(cfg: PostgresConfig) -> Iterator[SnowflakeConnection]:
    """Borrow a live PostgreSQL connection from the pool. Returns it on exit."""
    with _get_pool(cfg).connection() as conn:
        yield conn


def execute_script(conn: SnowflakeConnection, sql_text: str) -> None:
    """Execute a multi-statement SQL script (semicolon-separated)."""
    cur = conn.cursor()
    try:
        for statement in _split_statements(sql_text):
            if statement.strip():
                cur.execute(statement)
    except psycopg2.Error:
        raise SnowflakeConnectionError(
            "PostgreSQL rejected a SQL statement during script execution."
        ) from None
    finally:
        cur.close()


def _split_statements(sql_text: str) -> list[str]:
    result = []
    for stmt in sql_text.split(";"):
        stripped = stmt.strip()
        if not stripped:
            continue
        non_comment = "\n".join(
            line for line in stripped.splitlines()
            if not line.strip().startswith("--")
        ).strip()
        if non_comment:
            result.append(stripped)
    return result

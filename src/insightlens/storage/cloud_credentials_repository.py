"""Repository for per-user cloud storage OAuth credentials."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from insightlens.storage.token_crypto import decrypt_token, encrypt_token

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PGConnection


@dataclass
class CloudCredentialRecord:
    id: int
    user_id: str
    provider: str
    refresh_token: str
    access_token: str | None
    token_expires_at: datetime | None
    created_at: datetime
    updated_at: datetime


class CloudCredentialsRepository:
    def __init__(self, conn: "PGConnection") -> None:
        self._conn = conn

    def upsert(
        self,
        user_id: str,
        provider: str,
        refresh_token: str,
        access_token: str | None = None,
        token_expires_at: datetime | None = None,
    ) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO cloud_credentials
                    (user_id, provider, refresh_token, access_token, token_expires_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id, provider) DO UPDATE SET
                    refresh_token   = EXCLUDED.refresh_token,
                    access_token    = EXCLUDED.access_token,
                    token_expires_at = EXCLUDED.token_expires_at,
                    updated_at      = NOW()
                """,
                (
                    user_id,
                    provider,
                    encrypt_token(refresh_token),
                    encrypt_token(access_token),
                    token_expires_at,
                ),
            )
        finally:
            cur.close()

    def get(self, user_id: str, provider: str) -> CloudCredentialRecord | None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                SELECT id, user_id, provider, refresh_token,
                       access_token, token_expires_at, created_at, updated_at
                FROM   cloud_credentials
                WHERE  user_id = %s AND provider = %s
                """,
                (user_id, provider),
            )
            row = cur.fetchone()
        finally:
            cur.close()
        if not row:
            return None
        return CloudCredentialRecord(
            id=row[0],
            user_id=row[1],
            provider=row[2],
            refresh_token=decrypt_token(row[3]),
            access_token=decrypt_token(row[4]),
            token_expires_at=row[5],
            created_at=row[6],
            updated_at=row[7],
        )

    def delete(self, user_id: str, provider: str) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "DELETE FROM cloud_credentials WHERE user_id = %s AND provider = %s",
                (user_id, provider),
            )
        finally:
            cur.close()

    def list_providers(self, user_id: str) -> list[str]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT provider FROM cloud_credentials WHERE user_id = %s",
                (user_id,),
            )
            return [r[0] for r in cur.fetchall()]
        finally:
            cur.close()
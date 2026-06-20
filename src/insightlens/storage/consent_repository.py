"""GDPR consent log — record when a user accepts Terms and Privacy Policy."""
from __future__ import annotations

import uuid
from datetime import datetime

import psycopg2

from insightlens.storage.snowflake_client import SnowflakeConnection

_CONSENT_TYPES = ("terms_v1", "privacy_v1")


class ConsentRepository:
    def __init__(self, conn: SnowflakeConnection) -> None:
        self._conn = conn

    def log_acceptance(self, user_id: str) -> None:
        """Record acceptance of both Terms and Privacy Policy for a user."""
        cur = self._conn.cursor()
        try:
            for ctype in _CONSENT_TYPES:
                cur.execute(
                    """INSERT INTO consent_log (consent_id, user_id, consent_type)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (user_id, consent_type) DO NOTHING""",
                    (uuid.uuid4().hex[:16], user_id, ctype),
                )
        except psycopg2.Error:
            pass
        finally:
            cur.close()

    def get_consent_dates(self, user_id: str) -> dict[str, datetime | None]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT consent_type, accepted_at FROM consent_log WHERE user_id=%s",
                (user_id,),
            )
            return {r[0]: r[1] for r in cur.fetchall()}
        except psycopg2.Error:
            return {}
        finally:
            cur.close()

    def delete_user_consents(self, user_id: str) -> None:
        """GDPR erasure."""
        cur = self._conn.cursor()
        try:
            cur.execute("DELETE FROM consent_log WHERE user_id=%s", (user_id,))
        except psycopg2.Error:
            pass
        finally:
            cur.close()

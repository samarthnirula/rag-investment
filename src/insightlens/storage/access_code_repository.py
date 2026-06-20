"""Access code gating — issue, validate, and revoke invite codes."""
from __future__ import annotations

import secrets
import string
from datetime import datetime

import psycopg2

from insightlens.storage.snowflake_client import SnowflakeConnection


def _generate_code(length: int = 12) -> str:
    chars = string.ascii_uppercase + string.digits
    raw = "".join(secrets.choice(chars) for _ in range(length))
    return f"{raw[:4]}-{raw[4:8]}-{raw[8:12]}"


class AccessCodeRepository:
    def __init__(self, conn: SnowflakeConnection) -> None:
        self._conn = conn

    def create_code(
        self,
        created_by: str,
        max_uses: int = 1,
        note: str | None = None,
        code: str | None = None,
    ) -> str:
        c = code or _generate_code()
        cur = self._conn.cursor()
        try:
            cur.execute(
                "INSERT INTO access_codes (code,created_by,max_uses,is_active,note) VALUES (%s,%s,%s,TRUE,%s)",
                (c, created_by, max_uses, note),
            )
        finally:
            cur.close()
        return c

    def list_codes(self) -> list[dict]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT code,created_by,created_at,max_uses,uses_count,
                          used_by,used_at,is_active,note
                   FROM access_codes ORDER BY created_at DESC"""
            )
            cols = ["code","created_by","created_at","max_uses",
                    "uses_count","used_by","used_at","is_active","note"]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        except psycopg2.Error:
            return []
        finally:
            cur.close()

    def revoke_code(self, code: str) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "UPDATE access_codes SET is_active=FALSE WHERE code=%s",
                (code.strip().upper(),),
            )
        except psycopg2.Error:
            pass
        finally:
            cur.close()

    def codes_exist(self) -> bool:
        cur = self._conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM access_codes WHERE is_active=TRUE")
            row = cur.fetchone()
            return bool(row and row[0] > 0)
        except psycopg2.Error:
            return False
        finally:
            cur.close()

    def validate_and_claim(self, code: str, user_id: str) -> tuple[bool, str]:
        """Atomically validate + claim a code in one statement.

        Previously this did a SELECT (check is_active/uses_count) followed by
        a separate UPDATE -- a classic TOCTOU race. Two concurrent requests
        for a max_uses=1 code could both pass the SELECT check before either
        UPDATE ran, letting the code be claimed twice. Fixed by folding the
        validity check into the UPDATE's WHERE clause so the increment only
        happens if the code is still claimable, and checking rowcount instead
        of pre-checking with a separate read.
        """
        code = code.strip().upper()
        cur = self._conn.cursor()
        try:
            cur.execute(
                """UPDATE access_codes
                   SET uses_count = uses_count + 1,
                       used_by    = COALESCE(used_by, %s),
                       used_at    = COALESCE(used_at, NOW()),
                       is_active  = CASE WHEN max_uses > 0 AND uses_count + 1 >= max_uses
                                         THEN FALSE ELSE is_active END
                   WHERE code = %s
                     AND is_active = TRUE
                     AND (max_uses <= 0 OR uses_count < max_uses)""",
                (user_id, code),
            )
            if cur.rowcount > 0:
                return True, "OK"

            # No row updated -- figure out why, for an accurate message.
            cur.execute(
                "SELECT max_uses,uses_count,is_active FROM access_codes WHERE code=%s",
                (code,),
            )
            row = cur.fetchone()
            if not row:
                return False, "Invalid access code."
            max_uses, uses_count, is_active = row
            if not is_active:
                return False, "This access code has been revoked."
            if max_uses > 0 and uses_count >= max_uses:
                return False, "This access code has already been used."
            return False, "Could not validate access code. Please try again."
        except psycopg2.Error:
            return False, "Could not validate access code. Please try again."
        finally:
            cur.close()

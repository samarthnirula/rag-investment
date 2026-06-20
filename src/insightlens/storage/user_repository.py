"""PostgreSQL-backed user registry.

Keeps Firebase user metadata in sync and stores the active plan so the
backend never needs to decode custom claims to know a user's plan tier.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class UserRecord:
    uid: str
    email: Optional[str]
    display_name: Optional[str]
    plan: str
    plan_updated_at: Optional[datetime]
    created_at: Optional[datetime]


class UserRepository:
    def __init__(self, conn) -> None:
        self._conn = conn

    def upsert_user(
        self,
        uid: str,
        email: str = "",
        display_name: str = "",
        plan: str = "trial",
    ) -> bool:
        """Insert or update a user row. Returns True if this was a brand-new user."""
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO users (uid, email, display_name, plan, plan_updated_at, created_at)
                VALUES (%s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (uid) DO UPDATE
                    SET email        = EXCLUDED.email,
                        display_name = EXCLUDED.display_name
                RETURNING (xmax = 0) AS inserted
                """,
                (uid, email or None, display_name or None, plan),
            )
            row = cur.fetchone()
            self._conn.commit()
            return bool(row and row[0])
        except Exception:
            self._conn.rollback()
            logger.exception("UserRepository.upsert_user uid=%s", uid)
            return False
        finally:
            cur.close()

    def get_user(self, uid: str) -> Optional[UserRecord]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT uid, email, display_name, plan, plan_updated_at, created_at "
                "FROM users WHERE uid = %s",
                (uid,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return UserRecord(
                uid=row[0],
                email=row[1],
                display_name=row[2],
                plan=row[3] or "trial",
                plan_updated_at=row[4],
                created_at=row[5],
            )
        finally:
            cur.close()

    def get_user_subscription(self, uid: str) -> dict:
        """Return subscription status for uid as a plain dict."""
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT plan, trial_expires_at, subscription_active, created_at "
                "FROM users WHERE uid = %s",
                (uid,),
            )
            row = cur.fetchone()
            if row is None:
                return {
                    "plan": "trial",
                    "trial_expires_at": None,
                    "subscription_active": False,
                    "is_trial_expired": False,
                }
            plan, trial_expires_at, subscription_active, created_at = row
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            is_trial_expired = (
                trial_expires_at is not None
                and trial_expires_at.replace(tzinfo=timezone.utc) < now
                and not bool(subscription_active)
            )
            return {
                "plan": plan or "trial",
                "trial_expires_at": trial_expires_at,
                "subscription_active": bool(subscription_active),
                "is_trial_expired": is_trial_expired,
            }
        finally:
            cur.close()

    def is_trial_expired(self, uid: str) -> bool:
        """Return True if the user's trial window has closed and they are not subscribed."""
        sub = self.get_user_subscription(uid)
        return sub["is_trial_expired"]

    def count_user_cases(self, uid: str) -> int:
        """Return the number of cases owned by uid (used to enforce trial limit)."""
        cur = self._conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM cases WHERE user_id = %s", (uid,))
            row = cur.fetchone()
            return int(row[0]) if row else 0
        finally:
            cur.close()

    def set_trial_expires_at(self, uid: str, trial_expires_at) -> None:
        """Set trial_expires_at for a newly-registered user."""
        cur = self._conn.cursor()
        try:
            cur.execute(
                "UPDATE users SET trial_expires_at = %s WHERE uid = %s AND trial_expires_at IS NULL",
                (trial_expires_at, uid),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            logger.exception("UserRepository.set_trial_expires_at uid=%s", uid)
        finally:
            cur.close()

    def update_plan(self, uid: str, new_plan: str) -> tuple[str, str]:
        """Update a user's plan. Returns (old_plan, new_plan)."""
        cur = self._conn.cursor()
        try:
            cur.execute("SELECT plan FROM users WHERE uid = %s FOR UPDATE", (uid,))
            row = cur.fetchone()
            old_plan = row[0] if row else "trial"
            cur.execute(
                "UPDATE users SET plan = %s, plan_updated_at = NOW() WHERE uid = %s",
                (new_plan, uid),
            )
            if cur.rowcount == 0:
                # User row doesn't exist yet — create it
                cur.execute(
                    "INSERT INTO users (uid, plan, plan_updated_at) VALUES (%s, %s, NOW()) "
                    "ON CONFLICT (uid) DO UPDATE SET plan = EXCLUDED.plan, plan_updated_at = NOW()",
                    (uid, new_plan),
                )
            self._conn.commit()
            return old_plan, new_plan
        except Exception:
            self._conn.rollback()
            logger.exception("UserRepository.update_plan uid=%s", uid)
            return "trial", new_plan
        finally:
            cur.close()

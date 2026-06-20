"""Organization and team membership primitives."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import psycopg2

from insightlens.storage.snowflake_client import SnowflakeConnection


@dataclass(frozen=True)
class Organization:
    org_id: str
    org_name: str
    owner_id: str
    role: str


class OrgRepository:
    def __init__(self, conn: SnowflakeConnection) -> None:
        self._conn = conn

    def create_org(self, owner_id: str, org_name: str) -> str:
        org_id = uuid.uuid4().hex[:16]
        cur = self._conn.cursor()
        try:
            cur.execute(
                "INSERT INTO organizations (org_id, org_name, owner_id) VALUES (%s, %s, %s)",
                (org_id, org_name[:200], owner_id),
            )
            cur.execute(
                "INSERT INTO organization_members (org_id, user_id, role) VALUES (%s, %s, %s)",
                (org_id, owner_id, "owner"),
            )
        except psycopg2.Error:
            pass
        finally:
            cur.close()
        return org_id

    def add_member(self, org_id: str, user_id: str, role: str = "member") -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """INSERT INTO organization_members (org_id, user_id, role)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (org_id, user_id) DO UPDATE SET role = EXCLUDED.role""",
                (org_id, user_id, role),
            )
        except psycopg2.Error:
            pass
        finally:
            cur.close()

    def list_user_orgs(self, user_id: str) -> list[Organization]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT o.org_id, o.org_name, o.owner_id, m.role
                   FROM organization_members m
                   JOIN organizations o ON o.org_id = m.org_id
                   WHERE m.user_id = %s
                   ORDER BY o.created_at DESC""",
                (user_id,),
            )
            return [
                Organization(org_id=row[0], org_name=row[1], owner_id=row[2], role=row[3])
                for row in cur.fetchall()
            ]
        except psycopg2.Error:
            return []
        finally:
            cur.close()

    def get_org(self, org_id: str) -> Organization | None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT org_id, org_name, owner_id FROM organizations WHERE org_id = %s",
                (org_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return Organization(org_id=row[0], org_name=row[1], owner_id=row[2], role="owner")
        except psycopg2.Error:
            return None
        finally:
            cur.close()

    def list_members(self, org_id: str) -> list[dict]:
        """Return [{user_id, role, joined_at}] for all members of an org."""
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT user_id, role, created_at
                   FROM organization_members
                   WHERE org_id = %s
                   ORDER BY created_at ASC""",
                (org_id,),
            )
            return [
                {"user_id": row[0], "role": row[1], "joined_at": row[2]}
                for row in cur.fetchall()
            ]
        except psycopg2.Error:
            return []
        finally:
            cur.close()

    def remove_member(self, org_id: str, user_id: str) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "DELETE FROM organization_members WHERE org_id = %s AND user_id = %s",
                (org_id, user_id),
            )
        except psycopg2.Error:
            pass
        finally:
            cur.close()

    def rename_org(self, org_id: str, new_name: str) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "UPDATE organizations SET org_name = %s WHERE org_id = %s",
                (new_name[:200], org_id),
            )
        except psycopg2.Error:
            pass
        finally:
            cur.close()

    def get_member_role(self, org_id: str, user_id: str) -> str | None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT role FROM organization_members WHERE org_id = %s AND user_id = %s",
                (org_id, user_id),
            )
            row = cur.fetchone()
            return row[0] if row else None
        except psycopg2.Error:
            return None
        finally:
            cur.close()

    def get_org_member_ids(self, org_id: str) -> list[str]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT user_id FROM organization_members WHERE org_id = %s",
                (org_id,),
            )
            return [row[0] for row in cur.fetchall()]
        except psycopg2.Error:
            return []
        finally:
            cur.close()

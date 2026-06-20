"""Global discussion board — shared across all users."""
from __future__ import annotations

import uuid
from datetime import datetime

import psycopg2

from insightlens.storage.snowflake_client import SnowflakeConnection


class DiscussionRepository:
    def __init__(self, conn: SnowflakeConnection) -> None:
        self._conn = conn

    def list_posts(self, limit: int = 200) -> list[dict]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT post_id, author, post_type, content, posted_at
                   FROM discussion_posts ORDER BY posted_at ASC LIMIT %s""",
                (limit,),
            )
            return [
                {
                    "post_id": r[0],
                    "author":  r[1],
                    "type":    r[2],
                    "content": r[3],
                    "time":    r[4].strftime("%b %d, %H:%M") if r[4] else "",
                }
                for r in cur.fetchall()
            ]
        except psycopg2.Error:
            return []
        finally:
            cur.close()

    def add_post(self, author: str, post_type: str, content: str, user_id: str = "") -> str:
        post_id = uuid.uuid4().hex[:16]
        cur = self._conn.cursor()
        try:
            cur.execute(
                "INSERT INTO discussion_posts (post_id,author,post_type,content,user_id) VALUES (%s,%s,%s,%s,%s)",
                (post_id, author[:120], post_type, content[:8000], user_id or None),
            )
        finally:
            cur.close()
        return post_id

    def delete_post(self, post_id: str, requesting_uid: str) -> None:
        """Delete a post. Raises ValueError if post doesn't exist or isn't owned by requesting_uid."""
        cur = self._conn.cursor()
        try:
            cur.execute("SELECT user_id FROM discussion_posts WHERE post_id = %s", (post_id,))
            row = cur.fetchone()
            if row is None:
                raise ValueError("Post not found.")
            owner_uid = row[0]
            if owner_uid and owner_uid != requesting_uid:
                raise PermissionError("You can only delete your own posts.")
            cur.execute("DELETE FROM discussion_posts WHERE post_id = %s", (post_id,))
        finally:
            cur.close()

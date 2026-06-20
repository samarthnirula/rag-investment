"""Persistent chat history stored in PostgreSQL (per-user, per-page)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

import psycopg2

from insightlens.storage.snowflake_client import SnowflakeConnection


@dataclass(frozen=True)
class ChatRecord:
    chat_id: str
    user_id: str
    page: str
    chat_name: str | None
    created_at: datetime | None
    updated_at: datetime | None
    case_id: str | None = None
    chat_type: str | None = None


@dataclass(frozen=True)
class ChatMessage:
    message_id: str
    chat_id: str
    role: str
    content: str
    chunks_json: str | None
    created_at: datetime | None


class PersistentChatRepository:
    def __init__(self, conn: SnowflakeConnection) -> None:
        self._conn = conn

    def create_chat(
        self,
        user_id: str,
        page: str,
        chat_name: str | None = None,
        chat_id: str | None = None,
        case_id: str | None = None,
        chat_type: str = "chat",
    ) -> str:
        cid = chat_id or uuid.uuid4().hex[:16]
        cur = self._conn.cursor()
        try:
            cur.execute(
                """INSERT INTO chats (chat_id, user_id, page, chat_name, case_id, chat_type)
                   VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (chat_id) DO NOTHING""",
                (cid, user_id, page, chat_name or "New Chat", case_id, chat_type),
            )
        finally:
            cur.close()
        return cid

    def create_case_chats(self, user_id: str, case_id: str, case_name: str) -> list[dict]:
        """Create the 3 auto-generated tabs for a new case."""
        tabs = [
            ("chat",     f"{case_name} — Chat"),
            ("timeline", f"{case_name} — Timeline"),
            ("overview", f"{case_name} — Overview"),
        ]
        result = []
        for chat_type, name in tabs:
            cid = self.create_chat(
                user_id=user_id,
                page="insightlens",
                chat_name=name,
                case_id=case_id,
                chat_type=chat_type,
            )
            result.append({"chat_id": cid, "chat_type": chat_type, "name": name})
        return result

    def list_case_chats(self, case_id: str, user_id: str) -> list[dict]:
        """Return the auto-created chats for a case."""
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT chat_id, chat_name, chat_type
                   FROM chats WHERE case_id = %s AND user_id = %s
                   ORDER BY created_at""",
                (case_id, user_id),
            )
            return [
                {"chat_id": r[0], "name": r[1], "chat_type": r[2] or "chat"}
                for r in cur.fetchall()
            ]
        except psycopg2.Error:
            return []
        finally:
            cur.close()

    def list_chats(self, user_id: str, page: str | None = None) -> list[ChatRecord]:
        cur = self._conn.cursor()
        try:
            if page:
                cur.execute(
                    """SELECT chat_id,user_id,page,chat_name,created_at,updated_at,case_id,chat_type
                       FROM chats WHERE user_id=%s AND page=%s
                       ORDER BY updated_at DESC LIMIT 50""",
                    (user_id, page),
                )
            else:
                cur.execute(
                    """SELECT chat_id,user_id,page,chat_name,created_at,updated_at,case_id,chat_type
                       FROM chats WHERE user_id=%s ORDER BY updated_at DESC LIMIT 50""",
                    (user_id,),
                )
            return [
                ChatRecord(chat_id=r[0], user_id=r[1], page=r[2], chat_name=r[3],
                           created_at=r[4], updated_at=r[5], case_id=r[6], chat_type=r[7])
                for r in cur.fetchall()
            ]
        except psycopg2.Error:
            return []
        finally:
            cur.close()

    def update_chat_name(self, chat_id: str, name: str, user_id: str) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "UPDATE chats SET chat_name=%s, updated_at=NOW() WHERE chat_id=%s AND user_id=%s",
                (name[:100], chat_id, user_id),
            )
        except psycopg2.Error:
            pass
        finally:
            cur.close()

    def delete_chat(self, chat_id: str, user_id: str) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "DELETE FROM chat_messages WHERE chat_id=%s AND EXISTS "
                "(SELECT 1 FROM chats WHERE chat_id=%s AND user_id=%s)",
                (chat_id, chat_id, user_id),
            )
            cur.execute(
                "DELETE FROM chats WHERE chat_id=%s AND user_id=%s",
                (chat_id, user_id),
            )
        except psycopg2.Error:
            pass
        finally:
            cur.close()

    def save_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        chunks_json: str | None = None,
    ) -> str:
        message_id = uuid.uuid4().hex[:16]
        cur = self._conn.cursor()
        try:
            cur.execute(
                """INSERT INTO chat_messages (message_id,chat_id,role,content,chunks_json)
                   VALUES (%s,%s,%s,%s,%s)""",
                (message_id, chat_id, role, content[:50000], chunks_json),
            )
            cur.execute(
                "UPDATE chats SET updated_at=NOW() WHERE chat_id=%s", (chat_id,)
            )
        except psycopg2.Error:
            pass
        finally:
            cur.close()
        return message_id

    def load_messages(self, chat_id: str, user_id: str) -> list[ChatMessage]:
        """Load messages only if chat_id is owned by user_id."""
        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT cm.message_id, cm.chat_id, cm.role, cm.content,
                          cm.chunks_json, cm.created_at
                   FROM chat_messages cm
                   JOIN chats ch ON cm.chat_id = ch.chat_id
                   WHERE cm.chat_id = %s AND ch.user_id = %s
                   ORDER BY cm.created_at""",
                (chat_id, user_id),
            )
            return [
                ChatMessage(message_id=r[0], chat_id=r[1], role=r[2], content=r[3],
                            chunks_json=r[4], created_at=r[5])
                for r in cur.fetchall()
            ]
        except psycopg2.Error:
            return []
        finally:
            cur.close()

    def delete_user_chats(self, user_id: str) -> int:
        cur = self._conn.cursor()
        try:
            cur.execute("SELECT chat_id FROM chats WHERE user_id=%s", (user_id,))
            chat_ids = [r[0] for r in cur.fetchall()]
            if chat_ids:
                ph = ",".join(["%s"] * len(chat_ids))
                cur.execute(f"DELETE FROM chat_messages WHERE chat_id IN ({ph})", tuple(chat_ids))
            cur.execute("DELETE FROM chats WHERE user_id=%s", (user_id,))
            return len(chat_ids)
        except psycopg2.Error:
            return 0
        finally:
            cur.close()

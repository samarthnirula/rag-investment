"""Optional Zep memory layer for chat context."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZepActor:
    user_id: str
    email: str = ""
    display_name: str = ""

    @property
    def first_name(self) -> str:
        parts = self.display_name.strip().split()
        return parts[0] if parts else "User"

    @property
    def last_name(self) -> str:
        parts = self.display_name.strip().split()
        return " ".join(parts[1:]) if len(parts) > 1 else ""

    @property
    def message_name(self) -> str:
        return self.display_name.strip() or self.email.strip() or self.user_id


class ZepMemory:
    """Small defensive wrapper around zep-cloud.

    The app should keep working if Zep is not configured, not installed, or
    temporarily unavailable. In those cases these methods become no-ops and
    the existing RAG pipeline remains authoritative for document citations.
    """

    def __init__(self, api_key: str = "", enabled: bool = True) -> None:
        self.enabled = bool(enabled and api_key)
        self._client = None
        self._ensured_users: set[str] = set()
        self._ensured_threads: set[str] = set()
        if not self.enabled:
            return

        try:
            try:
                from zep_cloud.client import Zep
            except ImportError:
                from zep_cloud import Zep

            self._client = Zep(api_key=api_key)
        except Exception:
            logger.warning("Zep initialization failed; memory layer disabled.", exc_info=True)
            self.enabled = False

    def ensure_user(self, actor: ZepActor) -> None:
        if not self._client:
            return
        if actor.user_id in self._ensured_users:
            return
        try:
            self._client.user.add(
                user_id=actor.user_id,
                email=actor.email or None,
                first_name=actor.first_name,
                last_name=actor.last_name or None,
            )
            self._ensured_users.add(actor.user_id)
        except Exception:
            self._ensured_users.add(actor.user_id)
            logger.debug("Zep user already exists or could not be created.", exc_info=True)

    def ensure_thread(self, thread_id: str, actor: ZepActor) -> None:
        if not self._client or not thread_id:
            return
        if thread_id in self._ensured_threads:
            return
        self.ensure_user(actor)
        try:
            self._client.thread.create(thread_id=thread_id, user_id=actor.user_id)
            self._ensured_threads.add(thread_id)
        except Exception:
            self._ensured_threads.add(thread_id)
            logger.debug("Zep thread already exists or could not be created.", exc_info=True)

    def add_message(self, thread_id: str, role: str, content: str, actor: ZepActor) -> None:
        """Add a message without fetching context. Use add_message_get_context for queries."""
        if not self._client or not thread_id or not content:
            return
        self.ensure_thread(thread_id, actor)
        try:
            from zep_cloud.types import Message

            name = "Atticus" if role == "assistant" else actor.message_name
            self._client.thread.add_messages(
                thread_id,
                messages=[
                    Message(
                        created_at=datetime.now(timezone.utc).isoformat(),
                        name=name,
                        role=role,
                        content=content[:50000],
                    )
                ],
            )
        except Exception:
            logger.warning("Failed to add %s message to Zep.", role, exc_info=True)

    def add_message_get_context(
        self, thread_id: str, role: str, content: str, actor: ZepActor
    ) -> str:
        """Add a message and return context in a single Zep API call.

        Uses return_context=True so we avoid the separate get_user_context round-trip.
        Falls back to empty string on any error so the query still completes.
        """
        if not self._client or not thread_id or not content:
            return ""
        self.ensure_thread(thread_id, actor)
        try:
            from zep_cloud.types import Message

            name = "Atticus" if role == "assistant" else actor.message_name
            resp = self._client.thread.add_messages(
                thread_id,
                messages=[
                    Message(
                        created_at=datetime.now(timezone.utc).isoformat(),
                        name=name,
                        role=role,
                        content=content[:50000],
                    )
                ],
                return_context=True,
            )
            return (getattr(resp, "context", "") or "").strip()
        except Exception:
            logger.warning("Failed to add message + get context from Zep.", exc_info=True)
            return ""

    def get_context(self, thread_id: str, actor: ZepActor) -> str:
        if not self._client or not thread_id:
            return ""
        self.ensure_thread(thread_id, actor)
        try:
            user_context = self._client.thread.get_user_context(thread_id=thread_id)
            return (getattr(user_context, "context", "") or "").strip()
        except Exception:
            logger.warning("Failed to retrieve Zep context.", exc_info=True)
            return ""

    def add_system_event(self, thread_id: str, content: str, actor: ZepActor) -> None:
        """Record a system-level event (account creation, plan change, upload, etc.).

        Uses role='system' so the event is stored in Zep's memory graph but is
        not surfaced as a visible chat message in the UI.
        """
        if not self._client or not thread_id or not content:
            return
        self.ensure_thread(thread_id, actor)
        try:
            from zep_cloud.types import Message

            self._client.thread.add_messages(
                thread_id,
                messages=[
                    Message(
                        created_at=datetime.now(timezone.utc).isoformat(),
                        name="system",
                        role="system",
                        content=content[:50000],
                    )
                ],
            )
        except Exception:
            logger.warning("Failed to add system event to Zep.", exc_info=True)

    def add_business_data(self, actor: ZepActor, data: str) -> None:
        if not self._client or not data:
            return
        self.ensure_user(actor)
        try:
            self._client.graph.add(user_id=actor.user_id, type="text", data=data[:50000])
        except Exception:
            logger.warning("Failed to add business data to Zep.", exc_info=True)

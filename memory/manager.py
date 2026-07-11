"""
MemoryManager — short-term conversation memory lifecycle.

Lifecycle:
  begin_session(session_id)  → create + bind memory
  record_turn / update_selected_docs during execute
  end_session(session_id)    → cleanup
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from memory.base import ConversationMemory
from memory.in_memory_store import InMemoryStorage

if TYPE_CHECKING:
    from models.schemas import AnswerResult

    from memory.storage_interface import MemoryStorage


class MemoryManager:
    """
    Central memory access point — only Runtime should use this class.

    Tools and Skills must not import this module.
    """

    def __init__(self, storage: MemoryStorage | None = None) -> None:
        self._storage = storage or InMemoryStorage()

    @property
    def storage(self) -> MemoryStorage:
        return self._storage

    def begin_session(self, session_id: str) -> ConversationMemory:
        """
        Create and bind memory for a session.

        If the session already exists, return the existing memory.
        """
        existing = self._storage.load(session_id)
        if existing is not None:
            return existing

        memory = ConversationMemory(session_id=session_id)
        self._storage.save(memory)
        return memory

    def get_session(self, session_id: str) -> ConversationMemory | None:
        """Return bound memory for a session, or None if not started."""
        return self._storage.load(session_id)

    def update_selected_docs(self, session_id: str, doc_ids: list[str]) -> ConversationMemory:
        memory = self.begin_session(session_id)
        memory.selected_doc_ids = list(doc_ids)
        self._storage.save(memory)
        return memory

    def record_turn(self, session_id: str, query: str, result: AnswerResult) -> ConversationMemory:
        memory = self.begin_session(session_id)
        memory.record_turn(query, result)
        self._storage.save(memory)
        return memory

    def end_session(self, session_id: str) -> None:
        """End a session and cleanup its memory."""
        self._storage.delete(session_id)

    def has_session(self, session_id: str) -> bool:
        return self._storage.exists(session_id)

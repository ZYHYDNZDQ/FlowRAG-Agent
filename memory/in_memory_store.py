"""
In-memory storage backend for short-term conversation memory.
"""

from __future__ import annotations

from memory.base import ConversationMemory


class InMemoryStorage:
    """Process-local dict-backed memory storage."""

    def __init__(self) -> None:
        self._data: dict[str, ConversationMemory] = {}

    def load(self, session_id: str) -> ConversationMemory | None:
        return self._data.get(session_id)

    def save(self, memory: ConversationMemory) -> None:
        self._data[memory.session_id] = memory

    def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)

    def exists(self, session_id: str) -> bool:
        return session_id in self._data

    def clear_all(self) -> None:
        """Remove every session — useful in tests."""
        self._data.clear()

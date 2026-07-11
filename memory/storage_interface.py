"""
Memory storage backends — pluggable persistence interface.

Implementations:
  - InMemoryStorage (default)
  - RedisStorage (reserved, not implemented)
  - JsonFileStorage (reserved, not implemented)
"""

from __future__ import annotations

from typing import Protocol

from memory.base import ConversationMemory


class MemoryStorage(Protocol):
    """Contract for short-term conversation memory backends."""

    def load(self, session_id: str) -> ConversationMemory | None:
        """Load memory for a session, or None if missing."""

    def save(self, memory: ConversationMemory) -> None:
        """Persist memory for a session."""

    def delete(self, session_id: str) -> None:
        """Remove memory for a session."""

    def exists(self, session_id: str) -> bool:
        """Return True when the session has stored memory."""


class RedisStorage:
    """
    Reserved Redis backend — not implemented.

    Intended for multi-process / distributed session memory.
    """

    def __init__(self, *, url: str = "redis://localhost:6379/0") -> None:
        self.url = url
        raise NotImplementedError("RedisStorage is reserved for future use")

    def load(self, session_id: str) -> ConversationMemory | None:
        raise NotImplementedError

    def save(self, memory: ConversationMemory) -> None:
        raise NotImplementedError

    def delete(self, session_id: str) -> None:
        raise NotImplementedError

    def exists(self, session_id: str) -> bool:
        raise NotImplementedError


class JsonFileStorage:
    """
    Reserved JSON file backend — not implemented.

    Intended for local durable short-term memory across restarts.
    """

    def __init__(self, *, directory: str = "data/memory") -> None:
        self.directory = directory
        raise NotImplementedError("JsonFileStorage is reserved for future use")

    def load(self, session_id: str) -> ConversationMemory | None:
        raise NotImplementedError

    def save(self, memory: ConversationMemory) -> None:
        raise NotImplementedError

    def delete(self, session_id: str) -> None:
        raise NotImplementedError

    def exists(self, session_id: str) -> bool:
        raise NotImplementedError

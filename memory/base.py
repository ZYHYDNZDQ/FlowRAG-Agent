"""
Conversation memory data models — short-term session state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from models.schemas import AnswerResult, IntentType


@dataclass
class ConversationTurn:
    """One user query and agent response."""

    query: str
    answer: str
    intent: IntentType
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ConversationMemory:
    """
    Short-term memory bound to a single session_id.

    Managed exclusively by MemoryManager via Runtime.
    """

    session_id: str
    selected_doc_ids: list[str] = field(default_factory=list)
    turns: list[ConversationTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def record_turn(self, query: str, result: AnswerResult) -> None:
        self.turns.append(
            ConversationTurn(
                query=query,
                answer=result.answer,
                intent=result.intent,
            )
        )
        self.updated_at = datetime.now(timezone.utc)

    def recent_turns(self, limit: int = 5) -> list[ConversationTurn]:
        """Return the most recent conversation turns."""
        if limit <= 0:
            return []
        return self.turns[-limit:]

    def clear_turns(self) -> None:
        self.turns.clear()
        self.updated_at = datetime.now(timezone.utc)

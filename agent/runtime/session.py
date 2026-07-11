"""
Backward-compatible session aliases — delegate to memory module.

New code should use ``memory.MemoryManager`` via Runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memory.base import ConversationMemory, ConversationTurn
from memory.manager import MemoryManager
from models.schemas import AnswerResult, IntentType


@dataclass
class SessionTurn:
    """Legacy alias for ConversationTurn."""

    query: str
    answer: str
    intent: IntentType

    @classmethod
    def from_conversation(cls, turn: ConversationTurn) -> SessionTurn:
        return cls(query=turn.query, answer=turn.answer, intent=turn.intent)


@dataclass
class SessionState:
    """Legacy view over ConversationMemory."""

    session_id: str
    selected_doc_ids: list[str] = field(default_factory=list)
    turns: list[SessionTurn] = field(default_factory=list)

    @classmethod
    def from_memory(cls, memory: ConversationMemory) -> SessionState:
        return cls(
            session_id=memory.session_id,
            selected_doc_ids=list(memory.selected_doc_ids),
            turns=[
                SessionTurn.from_conversation(turn)
                for turn in memory.turns
            ],
        )

    def record_turn(self, query: str, result: AnswerResult) -> None:
        self.turns.append(
            SessionTurn(
                query=query,
                answer=result.answer,
                intent=result.intent,
            )
        )


class SessionStore:
    """
    Deprecated — use ``MemoryManager`` through ``AgentRuntime.memory``.

    Thin adapter kept for backward-compatible tests.
    """

    def __init__(self, manager: MemoryManager | None = None) -> None:
        self._manager = manager or MemoryManager()

    @property
    def manager(self) -> MemoryManager:
        return self._manager

    def get_or_create(self, session_id: str) -> SessionState:
        memory = self._manager.begin_session(session_id)
        return SessionState.from_memory(memory)

    def update_selected_docs(self, session_id: str, doc_ids: list[str]) -> SessionState:
        memory = self._manager.update_selected_docs(session_id, doc_ids)
        return SessionState.from_memory(memory)

    def clear(self, session_id: str) -> None:
        self._manager.end_session(session_id)

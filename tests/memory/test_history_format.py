"""Conversation history formatting and truncation tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memory.base import ConversationTurn
from memory.history_format import (
    build_conversation_history_section,
    estimate_tokens,
    format_turn,
    select_turns_for_history,
)
from models.schemas import IntentType

pytestmark = pytest.mark.memory


def _turn(index: int, *, query: str | None = None, answer: str | None = None) -> ConversationTurn:
    return ConversationTurn(
        query=query or f"question-{index}",
        answer=answer or f"answer-{index}" * 20,
        intent=IntentType.QA,
        recorded_at=datetime.now(timezone.utc),
    )


def test_format_turn_serializes_user_and_assistant():
    line = format_turn(ConversationTurn(query="hi", answer="hello", intent=IntentType.QA))
    assert "User: hi" in line
    assert "Assistant: hello" in line


def test_select_turns_prioritizes_recent_turns():
    turns = [_turn(i) for i in range(10)]
    selected = select_turns_for_history(turns, max_turns=3, max_tokens=10_000)
    assert len(selected) == 3
    assert [t.query for t in selected] == ["question-7", "question-8", "question-9"]


def test_select_turns_truncates_by_max_tokens():
    turns = [
        _turn(0, query="old", answer="x" * 400),
        _turn(1, query="mid", answer="y" * 400),
        _turn(2, query="new", answer="z" * 50),
    ]
    selected = select_turns_for_history(turns, max_turns=10, max_tokens=120)
    assert len(selected) == 1
    assert selected[0].query == "new"


def test_build_history_section_empty():
    assert build_conversation_history_section([], max_turns=5, max_tokens=500) == "（无）"


def test_build_history_section_contains_recent_dialogue():
    turns = [_turn(0, query="first", answer="a1"), _turn(1, query="second", answer="a2")]
    section = build_conversation_history_section(turns, max_turns=5, max_tokens=2000)
    assert "User: first" in section
    assert "User: second" in section
    assert estimate_tokens(section) > 0

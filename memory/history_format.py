"""
Format session conversation history for LLM prompts.

Used by Runtime only — Skills/Tools receive pre-formatted text via injection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from config.prompts import EMPTY_CONVERSATION_HISTORY

if TYPE_CHECKING:
    from memory.base import ConversationTurn


def estimate_tokens(text: str) -> int:
    """Rough token estimate for mixed Chinese/English text."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def format_turn(turn: ConversationTurn) -> str:
    """Serialize one turn for the history section."""
    return f"User: {turn.query}\nAssistant: {turn.answer}"


def select_turns_for_history(
    turns: Sequence[ConversationTurn],
    *,
    max_turns: int,
    max_tokens: int,
) -> list[ConversationTurn]:
    """
    Select turns for prompt injection, prioritizing the most recent dialogue.

    Walks from newest to oldest until ``max_turns`` or ``max_tokens`` is reached.
    """
    if max_turns <= 0 or max_tokens <= 0 or not turns:
        return []

    selected_newest_first: list[ConversationTurn] = []
    token_count = 0

    for turn in reversed(turns):
        if len(selected_newest_first) >= max_turns:
            break
        line = format_turn(turn)
        line_tokens = estimate_tokens(line)
        if selected_newest_first and token_count + line_tokens > max_tokens:
            break
        selected_newest_first.append(turn)
        token_count += line_tokens

    return list(reversed(selected_newest_first))


def build_conversation_history_section(
    turns: Sequence[ConversationTurn],
    *,
    max_turns: int,
    max_tokens: int,
) -> str:
    """Build the Conversation History block injected into generation prompts."""
    selected = select_turns_for_history(
        turns,
        max_turns=max_turns,
        max_tokens=max_tokens,
    )
    if not selected:
        return EMPTY_CONVERSATION_HISTORY
    return "\n\n".join(format_turn(turn) for turn in selected)

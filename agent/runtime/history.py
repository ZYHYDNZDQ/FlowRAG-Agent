"""Runtime bridge — read session memory and build injectable history text."""

from __future__ import annotations

from typing import TYPE_CHECKING

from memory.history_format import build_conversation_history_section

if TYPE_CHECKING:
    from config.settings import Settings
    from memory.base import ConversationMemory


def build_history_for_prompt(
    memory: ConversationMemory | None,
    settings: Settings,
) -> str:
    """Format prior session turns for Skill/Tool prompt injection."""
    if memory is None or not memory.turns:
        from config.prompts import EMPTY_CONVERSATION_HISTORY

        return EMPTY_CONVERSATION_HISTORY
    return build_conversation_history_section(
        memory.turns,
        max_turns=settings.memory_max_turns,
        max_tokens=settings.memory_max_tokens,
    )

"""
Skill execution context — Tool registry + Tool context.

Independent of Agent Runtime and Memory modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from config.settings import Settings, get_settings
from tools.context import ToolExecutionContext
from tools.registry import ToolRegistry, get_tool_registry

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from retrieval.chroma_store import ChromaStore


@dataclass
class SkillContext:
    """Dependencies available to Skills during run()."""

    settings: Settings
    tool_context: ToolExecutionContext
    tools: ToolRegistry
    conversation_history: str = ""

    @classmethod
    def create(
        cls,
        *,
        settings: Settings | None = None,
        vector_store: ChromaStore | None = None,
        llm: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
        tools: ToolRegistry | None = None,
        conversation_history: str = "",
    ) -> SkillContext:
        cfg = settings or get_settings()
        return cls(
            settings=cfg,
            tool_context=ToolExecutionContext.create(
                settings=cfg,
                vector_store=vector_store,
                llm=llm,
                embeddings=embeddings,
            ),
            tools=tools or get_tool_registry(),
            conversation_history=conversation_history,
        )

"""
Tool execution context — dependencies for Tool.execute().

Independent of Agent Runtime; no Memory or Skill references.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from config.settings import Settings, get_settings
from llm.factory import get_embeddings, get_llm
from retrieval.chroma_store import ChromaStore
from retrieval.rag_service import RAGService, get_rag_service

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel


@dataclass
class ToolExecutionContext:
    """
    Injected dependencies available to every Tool during execute().

    Assembled by callers (skills, tests); Tools must not construct this
    by importing Runtime.
    """

    settings: Settings
    llm: BaseChatModel
    embeddings: Embeddings
    rag_service: RAGService
    vector_store: ChromaStore | None = None

    @classmethod
    def create(
        cls,
        *,
        settings: Settings | None = None,
        vector_store: ChromaStore | None = None,
        llm: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
        rag_service: RAGService | None = None,
        connect_store: bool = True,
    ) -> ToolExecutionContext:
        cfg = settings or get_settings()
        store = vector_store
        if store is not None and connect_store:
            store.connect()
        return cls(
            settings=cfg,
            llm=llm or get_llm(cfg),
            embeddings=embeddings or get_embeddings(cfg),
            rag_service=rag_service or get_rag_service(),
            vector_store=store,
        )

    @classmethod
    def stateless(
        cls,
        *,
        settings: Settings | None = None,
        llm: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
    ) -> ToolExecutionContext:
        """Context for tools that do not require vector store access."""
        return cls.create(
            settings=settings,
            llm=llm,
            embeddings=embeddings,
            vector_store=None,
            connect_store=False,
        )

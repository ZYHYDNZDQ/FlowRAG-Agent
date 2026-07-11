"""
AgentContext — dependency container for Skills and Tools.

Runtime initializes this; Skills/Tools consume services through it.
Runtime must not call vector_index.store.query() directly.
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


class VectorIndexGateway:
    """
    Lifecycle holder for the vector index connection.

    Only Tools access ``store``; Runtime never queries it.
    """

    def __init__(
        self,
        settings: Settings,
        store: ChromaStore | None = None,
    ) -> None:
        self._store = store or ChromaStore(settings)
        self._store.connect()

    @property
    def store(self) -> ChromaStore:
        return self._store


@dataclass
class AgentContext:
    """Shared dependencies for a single Runtime execution."""

    settings: Settings
    llm: BaseChatModel
    embeddings: Embeddings
    rag_service: RAGService
    vector_index: VectorIndexGateway | None = None

    @classmethod
    def create(
        cls,
        *,
        settings: Settings | None = None,
        store: ChromaStore | None = None,
        llm: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
        rag_service: RAGService | None = None,
    ) -> AgentContext:
        """Build a fully wired context (used by Runtime on each execute)."""
        cfg = settings or get_settings()
        cfg.ensure_dirs()
        return cls(
            settings=cfg,
            llm=llm or get_llm(cfg),
            embeddings=embeddings or get_embeddings(cfg),
            rag_service=rag_service or get_rag_service(),
            vector_index=VectorIndexGateway(cfg, store=store),
        )

    @classmethod
    def for_stateless(
        cls,
        *,
        settings: Settings | None = None,
        llm: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
    ) -> AgentContext:
        """Context for Tools that do not access the vector index."""
        cfg = settings or get_settings()
        return cls(
            settings=cfg,
            llm=llm or get_llm(cfg),
            embeddings=embeddings or get_embeddings(cfg),
            rag_service=get_rag_service(),
            vector_index=None,
        )

    @classmethod
    def for_tool_call(
        cls,
        *,
        store: ChromaStore,
        embeddings: Embeddings | None = None,
        settings: Settings | None = None,
        llm: BaseChatModel | None = None,
    ) -> AgentContext:
        """Minimal context when callers invoke Tools without Runtime."""
        cfg = settings or get_settings()
        return cls(
            settings=cfg,
            llm=llm or get_llm(cfg),
            embeddings=embeddings or get_embeddings(cfg),
            rag_service=get_rag_service(),
            vector_index=VectorIndexGateway(cfg, store=store),
        )

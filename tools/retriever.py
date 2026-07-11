"""
Retriever helper — backward-compatible function API.

Delegates to ``tools.rag_tool.RagTool`` via ToolRegistry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from models.schemas import RetrievedChunk, RetrievalScope
from retrieval.chroma_store import ChromaStore
from tools.context import ToolExecutionContext
from tools.registry import get_tool_registry

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from config.settings import Settings


def retrieve(
    store: ChromaStore,
    query: str,
    scope: RetrievalScope,
    *,
    top_k: int | None = None,
    score_threshold: float | None = None,
    embeddings: Embeddings | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Run similarity search via rag.retrieve Tool (no Agent trace)."""
    ctx = ToolExecutionContext.create(
        settings=settings,
        vector_store=store,
        embeddings=embeddings,
    )
    return get_tool_registry().run(
        "rag.retrieve",
        ctx,
        query=query,
        scope=scope,
        top_k=top_k,
        score_threshold=score_threshold,
    )

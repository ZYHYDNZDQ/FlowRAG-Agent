"""
Retriever tool — vector search for Agent workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.workflows.common import retrieve_chunks
from config.settings import Settings, get_settings
from models.schemas import RetrievedChunk, RetrievalScope
from retrieval.chroma_store import ChromaStore

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


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
    """
    Run similarity search with scope filter.

    Thin wrapper around ``retrieve_chunks`` without trace emission.
    Workflows should call ``retrieve_chunks`` directly when trace is needed.
    """
    cfg = settings or get_settings()
    where = ChromaStore.build_where_filter(scope)
    chunks = store.query(
        query,
        top_k=top_k or cfg.default_top_k,
        where=where,
        embeddings=embeddings,
    )
    if score_threshold is not None:
        chunks = [
            c for c in chunks if c.score is None or c.score >= score_threshold
        ]
    return chunks

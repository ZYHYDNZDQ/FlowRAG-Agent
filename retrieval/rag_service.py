"""
RAG retrieval service — single business entry for vector search and context formatting.

Pipeline position: above ChromaStore, below Agent workflows.
Does not emit Agent trace or invoke LLMs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.settings import Settings, get_settings
from models.schemas import RetrievedChunk, RetrievalScope
from retrieval.chunk_utils import dedupe_chunks_by_id, filter_by_score
from retrieval.chroma_store import ChromaStore

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


class RAGService:
    """Encapsulates retrieval query, filtering, deduplication, and context formatting."""

    def query_chunks(
        self,
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
        Run similarity search with scope filter, score filter, and deduplication.

        Returns chunks sorted by score (highest first).
        """
        cfg = settings or get_settings()
        k = top_k or cfg.default_top_k
        threshold = score_threshold if score_threshold is not None else cfg.score_threshold

        where = ChromaStore.build_where_filter(scope)
        chunks = store.query(
            query,
            top_k=k,
            where=where,
            embeddings=embeddings,
        )
        chunks = filter_by_score(chunks, threshold)
        return dedupe_chunks_by_id(chunks)

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a single context block for the LLM prompt."""
        if not chunks:
            return "（无检索结果）"

        parts: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata
            header = f"[片段{index}] 来源: {meta.source_file} 第{meta.page}页"
            parts.append(f"{header}\n{chunk.text}")
        return "\n\n".join(parts)


_default_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """Return module-level RAGService singleton."""
    global _default_service
    if _default_service is None:
        _default_service = RAGService()
    return _default_service

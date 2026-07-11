"""
Retriever factory — build LangChain retrievers backed by ChromaStore.

Returned Documents always include ``source_file`` and ``page`` in metadata
so downstream RAG / citation logic can attribute answers.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from config.settings import Settings, get_settings
from models.schemas import RetrievalScope
from retrieval.chroma_store import ChromaStore
from retrieval.rag_service import get_rag_service


class ChromaRetriever(BaseRetriever):
    """
    LangChain retriever that queries ``ChromaStore`` with optional scope filters.

    Each retrieved Document exposes:
      - ``metadata['source_file']`` — filename
      - ``metadata['page']`` — 1-based page number
      - ``metadata['doc_id']``, ``chunk_id``, ``score``
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: ChromaStore
    scope: RetrievalScope = Field(default_factory=RetrievalScope)
    top_k: int = 6
    score_threshold: float | None = None
    embeddings: Any | None = Field(default=None, exclude=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve chunks and map them to LangChain Documents with source metadata."""
        chunks = get_rag_service().query_chunks(
            self.store,
            query,
            self.scope,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            embeddings=self.embeddings,
        )
        return [_chunk_to_document(chunk) for chunk in chunks]


def create_retriever(
    store: ChromaStore,
    scope: RetrievalScope,
    *,
    top_k: int | None = None,
    score_threshold: float | None = None,
    embeddings: Any | None = None,
    settings: Settings | None = None,
) -> ChromaRetriever:
    """
    Build a configured ``ChromaRetriever`` for a given document scope.

    Args:
        store: Connected ``ChromaStore`` instance.
        scope: Document / page filter (ALL, SINGLE, SELECTED, page range).
        top_k: Number of chunks to return; defaults to settings.default_top_k.
        score_threshold: Drop chunks below this similarity score.
        embeddings: Optional Embeddings for query encoding (tests).
        settings: Global settings fallback for top_k / threshold.

    Returns:
        LangChain-compatible retriever; call ``.invoke(query)`` or ``.get_relevant_documents``.
    """
    cfg = settings or get_settings()
    return ChromaRetriever(
        store=store,
        scope=scope,
        top_k=top_k or cfg.default_top_k,
        score_threshold=score_threshold if score_threshold is not None else cfg.score_threshold,
        embeddings=embeddings,
    )


def retrieve_with_sources(
    store: ChromaStore,
    query: str,
    scope: RetrievalScope,
    *,
    top_k: int | None = None,
    score_threshold: float | None = None,
    embeddings: Any | None = None,
) -> list[Document]:
    """
    One-shot retrieval helper returning Documents with source metadata.

    Example::

        docs = retrieve_with_sources(store, "付款周期", scope)
        print(docs[0].metadata["source_file"], docs[0].metadata["page"])
    """
    retriever = create_retriever(
        store,
        scope,
        top_k=top_k,
        score_threshold=score_threshold,
        embeddings=embeddings,
    )
    return retriever.invoke(query)


def _chunk_to_document(chunk: Any) -> Document:
    """Map ``RetrievedChunk`` → LangChain ``Document`` with source fields."""
    meta = chunk.metadata
    return Document(
        page_content=chunk.text,
        metadata={
            "source_file": meta.source_file,
            "page": meta.page,
            "doc_id": meta.doc_id,
            "chunk_id": meta.chunk_id,
            "source_path": meta.source_path,
            "score": chunk.score,
        },
    )

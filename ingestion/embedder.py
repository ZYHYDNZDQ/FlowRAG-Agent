"""
Embedding service — batch-encode chunk texts into dense vectors.

Pipeline position: after chunking, before Chroma storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from llm.factory import get_embeddings

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings


def embed_documents(
    documents: list[Document],
    *,
    batch_size: int = 32,
    on_progress: Callable[[int, int], None] | None = None,
    embeddings: Embeddings | None = None,
) -> list[list[float]]:
    """
    Compute embedding vectors for a list of chunk Documents.

    Args:
        documents: Chunk Documents produced by ``chunk_documents``.
        batch_size: Texts encoded per batch (memory / throughput trade-off).
        on_progress: Optional ``callback(done_count, total_count)``.
        embeddings: Inject a custom Embeddings instance (useful in tests).

    Returns:
        List of float vectors aligned 1:1 with ``documents``.
    """
    if not documents:
        return []

    model = embeddings or get_embeddings()
    texts = [doc.page_content for doc in documents]
    vectors: list[list[float]] = []
    total = len(texts)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_vectors = model.embed_documents(texts[start:end])
        vectors.extend(batch_vectors)
        if on_progress:
            on_progress(end, total)

    return vectors


def embed_query(
    query: str,
    *,
    embeddings: Embeddings | None = None,
) -> list[float]:
    """
    Encode a single user query into the same vector space as stored chunks.

    Used by ``ChromaStore.query`` when no pre-computed query embedding is supplied.
    """
    model = embeddings or get_embeddings()
    return model.embed_query(query)

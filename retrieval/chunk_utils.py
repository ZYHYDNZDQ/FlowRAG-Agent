"""
Pure helpers for retrieved chunk post-processing.

No I/O, no Agent trace — shared by RAGService and workflows.
"""

from __future__ import annotations

from models.schemas import RetrievedChunk


def chunk_score(chunk: RetrievedChunk) -> float:
    """Return similarity score, defaulting missing scores to 0."""
    return chunk.score if chunk.score is not None else 0.0


def filter_by_score(
    chunks: list[RetrievedChunk],
    threshold: float | None,
) -> list[RetrievedChunk]:
    """Drop chunks below threshold; no-op when threshold is None."""
    if threshold is None:
        return chunks
    return [c for c in chunks if c.score is None or c.score >= threshold]


def dedupe_chunks_by_id(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Deduplicate by chunk_id, keep highest score, sort descending."""
    best: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        existing = best.get(chunk.chunk_id)
        if existing is None or chunk_score(chunk) > chunk_score(existing):
            best[chunk.chunk_id] = chunk
    return sorted(best.values(), key=lambda c: -chunk_score(c))

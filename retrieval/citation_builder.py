"""
Citation builder — convert retrieval results to Citation objects.

Rules:
  - Citations built directly from chunk metadata (not LLM output)
  - Dedupe by (source_file, page, chunk_id), keep best score
  - excerpt = first N chars of chunk text
"""

from __future__ import annotations

from models.schemas import Citation, RetrievedChunk


def build_citations(
    chunks: list[RetrievedChunk],
    *,
    excerpt_length: int = 200,
    max_citations: int = 10,
) -> list[Citation]:
    """Build deduplicated, score-sorted citation list."""
    best: dict[tuple[str, int, str], Citation] = {}

    for chunk in chunks:
        meta = chunk.metadata
        key = (meta.source_file, meta.page, meta.chunk_id)
        citation = Citation(
            source_file=meta.source_file,
            page=meta.page,
            chunk_id=meta.chunk_id,
            doc_id=meta.doc_id,
            excerpt=chunk.text[:excerpt_length].strip(),
            score=chunk.score,
        )
        existing = best.get(key)
        if existing is None or _score(citation) > _score(existing):
            best[key] = citation

    ordered = sorted(
        best.values(),
        key=lambda item: (-_score(item), item.source_file, item.page),
    )
    return ordered[:max_citations]


def format_citation_label(citation: Citation) -> str:
    """Human-readable label, e.g. '合同_2024.pdf 第12页'."""
    return f"{citation.source_file} 第{citation.page}页"


def _score(citation: Citation) -> float:
    return citation.score if citation.score is not None else 0.0

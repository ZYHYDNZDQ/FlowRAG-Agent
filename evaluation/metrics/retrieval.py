"""Retrieval evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from config.settings import Settings
from evaluation.datasets.schema import RetrievalCase
from models.schemas import RetrievalScope
from retrieval.chroma_store import ChromaStore
from retrieval.rag_service import RAGService


@dataclass(frozen=True)
class RetrievalCaseResult:
    case_id: str
    query: str
    expected_page: int
    hit: bool
    top_k_pages: list[int]
    top_k_chunk_ids: list[str]


@dataclass(frozen=True)
class RetrievalEvalSummary:
    top_k: int
    total: int
    hits: int
    hit_rate: float
    case_results: list[RetrievalCaseResult]


def top_k_hit(
    chunks: list,
    *,
    expected_page: int,
    keyword: str | None,
    top_k: int,
) -> bool:
    """Return True if expected page or keyword appears in top-k chunks."""
    selected = chunks[:top_k]
    pages = [c.metadata.page for c in selected]
    if expected_page in pages:
        return True
    if keyword:
        return any(keyword.lower() in c.text.lower() for c in selected)
    return False


def evaluate_retrieval(
    cases: list[RetrievalCase],
    *,
    store: ChromaStore,
    scope: RetrievalScope,
    embeddings: Embeddings,
    settings: Settings,
    top_k: int = 5,
    rag_service: RAGService | None = None,
) -> RetrievalEvalSummary:
    """Compute Top-k hit rate over retrieval benchmark cases."""
    service = rag_service or RAGService()
    results: list[RetrievalCaseResult] = []

    for case in cases:
        chunks = service.query_chunks(
            store,
            case.query,
            scope,
            top_k=top_k,
            score_threshold=0.0,
            embeddings=embeddings,
            settings=settings,
        )
        hit = top_k_hit(
            chunks,
            expected_page=case.expected_page,
            keyword=case.keyword_in_chunk,
            top_k=top_k,
        )
        results.append(
            RetrievalCaseResult(
                case_id=case.id,
                query=case.query,
                expected_page=case.expected_page,
                hit=hit,
                top_k_pages=[c.metadata.page for c in chunks[:top_k]],
                top_k_chunk_ids=[c.chunk_id for c in chunks[:top_k]],
            )
        )

    hits = sum(1 for r in results if r.hit)
    total = len(results)
    rate = hits / total if total else 0.0
    return RetrievalEvalSummary(
        top_k=top_k,
        total=total,
        hits=hits,
        hit_rate=rate,
        case_results=results,
    )

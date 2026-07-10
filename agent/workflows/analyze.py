"""
Analyze Workflow — multi-angle retrieval then structured analysis.

Pipeline: sub-queries → multi-retrieval → merge context → LLM analysis → citations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from agent.workflows.base import BaseWorkflow
from agent.workflows.common import (
    NO_CONTEXT_ANSWER,
    build_answer_citations,
    format_context,
    generate_text,
    retrieve_chunks,
)
from config.prompts import ANALYZE_SYSTEM_PROMPT, ANALYZE_USER_TEMPLATE
from config.settings import Settings, get_settings
from models.schemas import AgentStep, AnswerResult, IntentType, RetrievedChunk, RetrievalScope
from retrieval.chroma_store import ChromaStore

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel

_ANALYZE_SUBQUERY_TOP_K = 4


def build_analysis_subqueries(query: str) -> list[str]:
    """
    Build 2~3 retrieval sub-queries covering different analysis angles.

    Rule-based (no extra LLM call) for simplicity and reliability.
    """
    base = query.strip() or "文档分析"
    candidates = [
        base,
        f"{base} 核心内容与要点",
        f"{base} 风险影响与结论",
    ]
    # Preserve order, drop duplicates
    seen: set[str] = set()
    unique: list[str] = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


class AnalyzeWorkflow(BaseWorkflow):
    """Perform multi-angle retrieval and produce structured analysis."""

    name = "analyze"

    def __init__(
        self,
        store: ChromaStore,
        llm: BaseChatModel,
        *,
        settings: Settings | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        self._store = store
        self._llm = llm
        self._settings = settings or get_settings()
        self._embeddings = embeddings

    def run(
        self,
        query: str,
        scope: RetrievalScope,
        *,
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> AnswerResult:
        trace: list[AgentStep] = []

        def _on_step(step: AgentStep) -> None:
            trace.append(step)
            if on_step:
                on_step(step)

        subqueries = build_analysis_subqueries(query)
        all_chunks: list[RetrievedChunk] = []

        for subquery in subqueries:
            hits = retrieve_chunks(
                self._store,
                subquery,
                scope,
                top_k=_ANALYZE_SUBQUERY_TOP_K,
                score_threshold=0.0,
                embeddings=self._embeddings,
                settings=self._settings,
                on_step=_on_step,
            )
            all_chunks.extend(hits)

        merged = _merge_chunks(all_chunks)
        # Cap context size
        merged = merged[: self._settings.max_context_chunks]

        if not merged:
            return AnswerResult(
                answer=NO_CONTEXT_ANSWER,
                intent=IntentType.ANALYZE,
                citations=[],
                trace=trace,
                retrieved_chunks=[],
            )

        context = format_context(merged)
        user_prompt = ANALYZE_USER_TEMPLATE.format(query=query, context=context)
        answer = generate_text(
            self._llm,
            ANALYZE_SYSTEM_PROMPT,
            user_prompt,
            on_step=_on_step,
        )
        citations = build_answer_citations(merged, on_step=_on_step)

        return AnswerResult(
            answer=answer,
            intent=IntentType.ANALYZE,
            citations=citations,
            trace=trace,
            retrieved_chunks=merged,
        )


def _merge_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Deduplicate by chunk_id, keep best score, sort descending."""
    best: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        existing = best.get(chunk.chunk_id)
        if existing is None or _chunk_score(chunk) > _chunk_score(existing):
            best[chunk.chunk_id] = chunk
    return sorted(best.values(), key=lambda c: -_chunk_score(c))


def _chunk_score(chunk: RetrievedChunk) -> float:
    return chunk.score if chunk.score is not None else 0.0

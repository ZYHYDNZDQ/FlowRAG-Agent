"""
Shared helpers for QA / Summarize / Analyze workflows.

Keeps each workflow thin: retrieve → format context → LLM → citations → trace.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import Settings, get_settings
from models.schemas import (
    AgentStep,
    AgentStepType,
    IntentType,
    RetrievedChunk,
    RetrievalScope,
)
from retrieval.citation_builder import build_citations
from retrieval.chroma_store import ChromaStore

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel

NO_CONTEXT_ANSWER = "知识库中未找到相关内容，请先上传 PDF 或调整检索范围。"


def emit_step(
    on_step: Callable[[AgentStep], None] | None,
    *,
    step_type: AgentStepType,
    name: str,
    detail: str = "",
    duration_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentStep:
    """Build an AgentStep, optionally notify UI callback, and return it."""
    step = AgentStep(
        step_type=step_type,
        name=name,
        detail=detail,
        duration_ms=duration_ms,
        metadata=metadata or {},
    )
    if on_step:
        on_step(step)
    return step


def retrieve_chunks(
    store: ChromaStore,
    query: str,
    scope: RetrievalScope,
    *,
    top_k: int | None = None,
    score_threshold: float | None = None,
    embeddings: Embeddings | None = None,
    settings: Settings | None = None,
    on_step: Callable[[AgentStep], None] | None = None,
) -> list[RetrievedChunk]:
    """
    Query Chroma and emit a Retriever trace step.

    Returns deduplicated chunks sorted by score (highest first).
    """
    cfg = settings or get_settings()
    k = top_k or cfg.default_top_k
    threshold = score_threshold if score_threshold is not None else cfg.score_threshold

    started = time.perf_counter()
    where = ChromaStore.build_where_filter(scope)
    chunks = store.query(
        query,
        top_k=k,
        where=where,
        embeddings=embeddings,
    )

    if threshold is not None:
        chunks = [
            c for c in chunks if c.score is None or c.score >= threshold
        ]

    chunks = _dedupe_chunks(chunks)
    elapsed = (time.perf_counter() - started) * 1000

    emit_step(
        on_step,
        step_type=AgentStepType.RETRIEVAL,
        name="Retriever",
        detail=f"找到 {len(chunks)} 个相关文档片段",
        duration_ms=elapsed,
        metadata={
            "query": query,
            "top_k": k,
            "hits": [
                {
                    "source_file": c.metadata.source_file,
                    "page": c.metadata.page,
                    "score": c.score,
                }
                for c in chunks[:5]
            ],
        },
    )
    return chunks


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


def generate_text(
    llm: BaseChatModel,
    system_prompt: str,
    user_prompt: str,
    *,
    on_step: Callable[[AgentStep], None] | None = None,
    step_name: str = "Generator",
) -> str:
    """Invoke chat model and emit a Generator trace step."""
    started = time.perf_counter()
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    content = str(response.content).strip()
    elapsed = (time.perf_counter() - started) * 1000

    emit_step(
        on_step,
        step_type=AgentStepType.GENERATION,
        name=step_name,
        detail="生成最终答案",
        duration_ms=elapsed,
        metadata={"answer_length": len(content)},
    )
    return content


def build_answer_citations(
    chunks: list[RetrievedChunk],
    *,
    on_step: Callable[[AgentStep], None] | None = None,
) -> list:
    """Build Citation list from retrieval hits and emit Postprocess step."""
    citations = build_citations(chunks)
    emit_step(
        on_step,
        step_type=AgentStepType.POSTPROCESS,
        name="Citations",
        detail=f"构建 {len(citations)} 条引用来源",
        metadata={"citation_count": len(citations)},
    )
    return citations


def intent_label(intent: IntentType) -> str:
    """Human-readable intent label for trace display."""
    return {
        IntentType.QA: "qa",
        IntentType.SUMMARIZE: "summary",
        IntentType.ANALYZE: "analysis",
    }[intent]


def _dedupe_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Deduplicate by chunk_id, keep highest score."""
    best: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        key = chunk.chunk_id
        existing = best.get(key)
        if existing is None or _score(chunk) > _score(existing):
            best[key] = chunk
    return sorted(best.values(), key=lambda c: -_score(c))


def _score(chunk: RetrievedChunk) -> float:
    return chunk.score if chunk.score is not None else 0.0

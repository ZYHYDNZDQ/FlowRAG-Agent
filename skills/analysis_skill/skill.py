"""Analysis Skill — multi-angle document analysis."""

from __future__ import annotations

from typing import Callable

from models.schemas import (
    AgentStep,
    AgentStepType,
    AnswerResult,
    IntentType,
    RetrievedChunk,
    RetrievalScope,
)
from retrieval.chunk_utils import dedupe_chunks_by_id
from skills.base import BaseSkill
from skills.context import SkillContext
from skills.helpers import NO_CONTEXT_ANSWER, invoke_tool

_ANALYZE_SUBQUERY_TOP_K = 4


def build_analysis_subqueries(query: str) -> list[str]:
    """Build retrieval sub-queries for multi-angle analysis."""
    base = query.strip() or "文档分析"
    candidates = [
        base,
        f"{base} 核心内容与要点",
        f"{base} 风险影响与结论",
    ]
    seen: set[str] = set()
    unique: list[str] = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


class AnalysisSkill(BaseSkill):
    """
    Analysis business capability.

    Flow: multi search_document → merge → format_context → analyze → build_citations
    """

    name = "analysis"
    intent = IntentType.ANALYZE
    description = "Perform structured multi-angle document analysis."

    def run(
        self,
        ctx: SkillContext,
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

        all_chunks: list[RetrievedChunk] = []
        for subquery in build_analysis_subqueries(query):
            hits = invoke_tool(
                ctx,
                "search_document",
                {
                    "query": subquery,
                    "scope": scope,
                    "top_k": _ANALYZE_SUBQUERY_TOP_K,
                    "score_threshold": 0.0,
                },
                on_step=_on_step,
                step_type=AgentStepType.RETRIEVAL,
                step_name="Retriever",
                detail=f"分析检索: {subquery}",
                extra_metadata={"query": subquery},
            )
            all_chunks.extend(hits)

        merged = dedupe_chunks_by_id(all_chunks)[: ctx.settings.max_context_chunks]

        if not merged:
            return AnswerResult(
                answer=NO_CONTEXT_ANSWER,
                intent=self.intent,
                citations=[],
                trace=trace,
                retrieved_chunks=[],
            )

        context = invoke_tool(
            ctx,
            "rag.format_context",
            {"chunks": merged},
            on_step=_on_step,
            step_type=AgentStepType.RETRIEVAL,
            step_name="ContextBuilder",
            detail="构建分析上下文",
        )

        answer = invoke_tool(
            ctx,
            "analyze",
            {"query": query, "context": context, "history": ctx.conversation_history},
            on_step=_on_step,
            step_type=AgentStepType.GENERATION,
            step_name="Analyzer",
            detail="生成结构化分析",
        )

        citations = invoke_tool(
            ctx,
            "rag.build_citations",
            {"chunks": merged},
            on_step=_on_step,
            step_type=AgentStepType.POSTPROCESS,
            step_name="Citations",
            detail="构建引用来源",
        )

        return AnswerResult(
            answer=answer,
            intent=self.intent,
            citations=citations,
            trace=trace,
            retrieved_chunks=merged,
        )

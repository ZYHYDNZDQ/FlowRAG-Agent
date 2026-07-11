"""Summary Skill — document summarization."""

from __future__ import annotations

from typing import Callable

from models.schemas import (
    AgentStep,
    AgentStepType,
    AnswerResult,
    IntentType,
    RetrievalScope,
)
from skills.base import BaseSkill
from skills.context import SkillContext
from skills.helpers import NO_CONTEXT_ANSWER, invoke_tool

_SUMMARIZE_TOP_K = 12


class SummarySkill(BaseSkill):
    """
    Summary business capability.

    Flow: search_document → format_context → summarize → build_citations
    """

    name = "summary"
    intent = IntentType.SUMMARIZE
    description = "Summarize knowledge-base documents for the user."

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

        retrieval_query = query.strip() or "文档主要内容概述"
        chunks = invoke_tool(
            ctx,
            "search_document",
            {
                "query": retrieval_query,
                "scope": scope,
                "top_k": _SUMMARIZE_TOP_K,
                "score_threshold": 0.0,
            },
            on_step=_on_step,
            step_type=AgentStepType.RETRIEVAL,
            step_name="Retriever",
            detail="检索总结所需文档片段",
            extra_metadata={"query": retrieval_query},
        )

        if not chunks:
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
            {"chunks": chunks},
            on_step=_on_step,
            step_type=AgentStepType.RETRIEVAL,
            step_name="ContextBuilder",
            detail="构建总结上下文",
        )

        answer = invoke_tool(
            ctx,
            "summarize",
            {"query": query, "context": context, "history": ctx.conversation_history},
            on_step=_on_step,
            step_type=AgentStepType.GENERATION,
            step_name="Summarizer",
            detail="生成文档总结",
        )

        citations = invoke_tool(
            ctx,
            "rag.build_citations",
            {"chunks": chunks},
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
            retrieved_chunks=chunks,
        )

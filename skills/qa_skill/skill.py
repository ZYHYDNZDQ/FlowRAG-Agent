"""QA Skill — retrieval-augmented question answering."""

from __future__ import annotations

from typing import Callable

from config.prompts import QA_SYSTEM_PROMPT, build_agent_user_prompt
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


class QASkill(BaseSkill):
    """
    QA business capability.

    Flow: search_document → format_context → llm.generate → build_citations
    """

    name = "qa"
    intent = IntentType.QA
    description = "Answer factual questions grounded in retrieved documents."

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

        chunks = invoke_tool(
            ctx,
            "search_document",
            {
                "query": query,
                "scope": scope,
                "top_k": ctx.settings.default_top_k,
                "score_threshold": 0.0,
            },
            on_step=_on_step,
            step_type=AgentStepType.RETRIEVAL,
            step_name="Retriever",
            detail="检索知识库文档",
            extra_metadata={"query": query},
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
            detail="构建 LLM 上下文",
        )

        user_prompt = build_agent_user_prompt(
            history=ctx.conversation_history,
            context=context,
            question=query,
        )
        answer = invoke_tool(
            ctx,
            "llm.generate",
            {"system_prompt": QA_SYSTEM_PROMPT, "user_prompt": user_prompt},
            on_step=_on_step,
            step_type=AgentStepType.GENERATION,
            step_name="Generator",
            detail="生成最终答案",
            extra_metadata={"answer_length": 0},
        )

        citations = invoke_tool(
            ctx,
            "rag.build_citations",
            {"chunks": chunks},
            on_step=_on_step,
            step_type=AgentStepType.POSTPROCESS,
            step_name="Citations",
            detail=f"构建引用来源",
        )

        return AnswerResult(
            answer=answer,
            intent=self.intent,
            citations=citations,
            trace=trace,
            retrieved_chunks=chunks,
        )

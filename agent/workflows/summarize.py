"""
Summarize Workflow — retrieve relevant content then generate a summary.

Pipeline: retrieve (broader top_k) → single LLM summary → citations
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
from config.prompts import SUMMARIZE_REDUCE_TEMPLATE, SUMMARIZE_SYSTEM_PROMPT
from config.settings import Settings, get_settings
from models.schemas import AgentStep, AnswerResult, IntentType, RetrievalScope
from retrieval.chroma_store import ChromaStore

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel

_SUMMARIZE_TOP_K = 12


class SummarizeWorkflow(BaseWorkflow):
    """Summarize documents based on retrieved chunks."""

    name = "summarize"

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

        # Use the user query or a default retrieval phrase
        retrieval_query = query.strip() or "文档主要内容概述"
        chunks = retrieve_chunks(
            self._store,
            retrieval_query,
            scope,
            top_k=_SUMMARIZE_TOP_K,
            score_threshold=0.0,
            embeddings=self._embeddings,
            settings=self._settings,
            on_step=_on_step,
        )

        if not chunks:
            return AnswerResult(
                answer=NO_CONTEXT_ANSWER,
                intent=IntentType.SUMMARIZE,
                citations=[],
                trace=trace,
                retrieved_chunks=[],
            )

        context = format_context(chunks)
        user_prompt = SUMMARIZE_REDUCE_TEMPLATE.format(partial_summaries=context)
        if query.strip():
            user_prompt = f"用户总结要求：{query}\n\n{user_prompt}"

        answer = generate_text(
            self._llm,
            SUMMARIZE_SYSTEM_PROMPT,
            user_prompt,
            on_step=_on_step,
        )
        citations = build_answer_citations(chunks, on_step=_on_step)

        return AnswerResult(
            answer=answer,
            intent=IntentType.SUMMARIZE,
            citations=citations,
            trace=trace,
            retrieved_chunks=chunks,
        )

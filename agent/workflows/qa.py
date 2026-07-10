"""
QA Workflow — retrieval-augmented question answering.

Pipeline: retrieve → format context → LLM generate → build citations
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
from config.prompts import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE
from config.settings import Settings, get_settings
from models.schemas import AgentStep, AnswerResult, IntentType, RetrievalScope
from retrieval.chroma_store import ChromaStore

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel


class QAWorkflow(BaseWorkflow):
    """Answer factual questions using retrieved knowledge base context."""

    name = "qa"

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

        chunks = retrieve_chunks(
            self._store,
            query,
            scope,
            top_k=self._settings.default_top_k,
            score_threshold=0.0,
            embeddings=self._embeddings,
            settings=self._settings,
            on_step=_on_step,
        )

        if not chunks:
            return AnswerResult(
                answer=NO_CONTEXT_ANSWER,
                intent=IntentType.QA,
                citations=[],
                trace=trace,
                retrieved_chunks=[],
            )

        context = format_context(chunks)
        user_prompt = QA_USER_TEMPLATE.format(context=context, query=query)
        answer = generate_text(
            self._llm,
            QA_SYSTEM_PROMPT,
            user_prompt,
            on_step=_on_step,
        )
        citations = build_answer_citations(chunks, on_step=_on_step)

        return AnswerResult(
            answer=answer,
            intent=IntentType.QA,
            citations=citations,
            trace=trace,
            retrieved_chunks=chunks,
        )

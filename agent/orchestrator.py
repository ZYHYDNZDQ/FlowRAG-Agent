"""
Agent orchestrator — single entry point for all user queries.

Flow: route → select workflow → retrieve → generate → return AnswerResult + trace
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from agent.router import route
from agent.workflows.analyze import AnalyzeWorkflow
from agent.workflows.common import emit_step, intent_label
from agent.workflows.qa import QAWorkflow
from agent.workflows.summarize import SummarizeWorkflow
from config.settings import Settings, get_settings
from llm.factory import get_embeddings, get_llm
from models.schemas import (
    AgentStep,
    AgentStepType,
    AnswerResult,
    IntentType,
    RetrievalScope,
)
from retrieval.chroma_store import ChromaStore

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel

_WORKFLOW_MAP: dict[IntentType, type] = {
    IntentType.QA: QAWorkflow,
    IntentType.SUMMARIZE: SummarizeWorkflow,
    IntentType.ANALYZE: AnalyzeWorkflow,
}


def run(
    query: str,
    *,
    scope_override: RetrievalScope | None = None,
    selected_doc_ids: list[str] | None = None,
    on_step: Callable[[AgentStep], None] | None = None,
    store: ChromaStore | None = None,
    llm: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    settings: Settings | None = None,
) -> AnswerResult:
    """
    Execute the full Agent pipeline for a user query.

    1. Router — classify intent (qa / summarize / analyze)
    2. Workflow — dispatch to the matching pipeline
    3. Trace — collect AgentStep list for UI display

    Args:
        query: User question or task.
        scope_override: Force retrieval scope (bypass router scope).
        selected_doc_ids: Documents selected in UI sidebar.
        on_step: Callback invoked for each AgentStep (UI trace).
        store: Inject ChromaStore (tests).
        llm: Inject chat model (tests).
        embeddings: Inject Embeddings (tests).
        settings: Override global settings.

    Returns:
        AnswerResult with answer, citations, and execution trace.
    """
    cfg = settings or get_settings()
    cfg.ensure_dirs()

    chroma = store or ChromaStore(cfg)
    chroma.connect()
    chat_model = llm or get_llm(cfg)
    embed_model = embeddings or get_embeddings(cfg)

    router_result = route(query, selected_doc_ids=selected_doc_ids)
    scope = scope_override or router_result.scope

    trace: list[AgentStep] = []

    def _on_step(step: AgentStep) -> None:
        trace.append(step)
        if on_step:
            on_step(step)

    router_step = emit_step(
        _on_step,
        step_type=AgentStepType.ROUTING,
        name="Router",
        detail=f"任务类型: {intent_label(router_result.intent)}",
        metadata={
            "intent": router_result.intent.value,
            "scope": scope.mode.value,
            "confidence": router_result.confidence,
            "reasoning": router_result.reasoning,
        },
    )

    workflow_cls = _WORKFLOW_MAP[router_result.intent]
    workflow = workflow_cls(
        chroma,
        chat_model,
        settings=cfg,
        embeddings=embed_model,
    )

    result = workflow.run(query, scope, on_step=_on_step)

    # Prepend router step to full trace
    full_trace = [router_step, *result.trace]
    return result.model_copy(update={"trace": full_trace})

"""
Agent Runtime — execution entry point.

Responsibilities: init context, memory, trace, skill dispatch.
Prohibited: RAG logic, business rules, direct vector DB queries.

Memory access is exclusive to Runtime via MemoryManager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.runtime.history import build_history_for_prompt
from agent.router import intent_label, route
from agent.runtime.context import AgentContext
from agent.runtime.requests import ExecuteRequest, ExecuteResult
from agent.runtime.trace import TraceCollector
from skills.helpers import emit_step
from memory.manager import MemoryManager
from models.schemas import AgentStepType, RetrievalScope
from skills.context import SkillContext
from skills.registry import SkillRegistry, get_skill_registry

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from retrieval.chroma_store import ChromaStore

    from config.settings import Settings


class AgentRuntime:
    """
    Lightweight Agent execution harness.

    Flow: begin session → route intent → dispatch skill → record memory → optional cleanup
    """

    def __init__(
        self,
        *,
        skill_registry: SkillRegistry | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self._skills = skill_registry or get_skill_registry()
        self._memory = memory_manager or MemoryManager()

    @property
    def memory(self) -> MemoryManager:
        """Expose memory manager for session lifecycle control."""
        return self._memory

    def execute(
        self,
        request: ExecuteRequest,
        *,
        settings: Settings | None = None,
        store: ChromaStore | None = None,
        llm: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
    ) -> ExecuteResult:
        """Run one agent request end-to-end."""
        ctx = AgentContext.create(
            settings=settings,
            store=store,
            llm=llm,
            embeddings=embeddings,
        )

        memory = self._memory.begin_session(request.session_id)
        if request.selected_doc_ids is not None:
            self._memory.update_selected_docs(
                request.session_id,
                request.selected_doc_ids,
            )
            memory = self._memory.get_session(request.session_id) or memory

        trace = TraceCollector(
            run_id=request.run_id,
            on_step=request.on_step,
        )

        doc_ids = memory.selected_doc_ids or request.selected_doc_ids
        router_result = route(request.query, selected_doc_ids=doc_ids, llm=ctx.llm)
        scope = request.scope_override or router_result.scope

        router_step = trace.emit(
            emit_step(
                None,
                step_type=AgentStepType.ROUTING,
                name="Router",
                detail=f"任务类型: {intent_label(router_result.intent)}",
                metadata={
                    "intent": router_result.intent.value,
                    "scope": scope.mode.value,
                    "confidence": router_result.confidence,
                    "reasoning": router_result.reasoning,
                    "run_id": request.run_id,
                    "session_id": request.session_id,
                },
            )
        )

        skill = self._skills.get(router_result.intent)
        conversation_history = build_history_for_prompt(memory, ctx.settings)
        skill_ctx = SkillContext.create(
            settings=ctx.settings,
            vector_store=ctx.vector_index.store if ctx.vector_index else None,
            llm=ctx.llm,
            embeddings=ctx.embeddings,
            conversation_history=conversation_history,
        )
        result = skill.run(skill_ctx, request.query, scope, on_step=trace.emit)
        self._memory.record_turn(request.session_id, request.query, result)

        if request.end_session:
            self._memory.end_session(request.session_id)

        full_trace = [router_step, *result.trace]
        answer = result.model_copy(update={"trace": full_trace})

        return ExecuteResult(
            answer=answer,
            run_id=request.run_id,
            session_id=request.session_id,
        )

    def end_session(self, session_id: str) -> None:
        """Explicitly end a session and cleanup its memory."""
        self._memory.end_session(session_id)


_default_runtime: AgentRuntime | None = None


def get_runtime() -> AgentRuntime:
    """Return module-level AgentRuntime singleton."""
    global _default_runtime
    if _default_runtime is None:
        _default_runtime = AgentRuntime()
    return _default_runtime

"""

Agent orchestrator — compatibility entry point delegating to AgentRuntime.



Prefer ``agent.runtime.get_runtime().execute()`` for new integrations.

"""



from __future__ import annotations



from typing import TYPE_CHECKING, Callable



from agent.runtime import AgentRuntime, ExecuteRequest, get_runtime

from models.schemas import AgentStep, AnswerResult, RetrievalScope



if TYPE_CHECKING:

    from langchain_core.embeddings import Embeddings

    from langchain_core.language_models import BaseChatModel

    from retrieval.chroma_store import ChromaStore



    from config.settings import Settings





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

    session_id: str = "default",

    runtime: AgentRuntime | None = None,

) -> AnswerResult:

    """

    Execute the full Agent pipeline for a user query via AgentRuntime.



    Backward-compatible wrapper around ``AgentRuntime.execute()``.

    """

    engine = runtime or get_runtime()

    result = engine.execute(

        ExecuteRequest(

            query=query,

            session_id=session_id,

            selected_doc_ids=selected_doc_ids,

            scope_override=scope_override,

            on_step=on_step,

        ),

        settings=settings,

        store=store,

        llm=llm,

        embeddings=embeddings,

    )

    return result.answer



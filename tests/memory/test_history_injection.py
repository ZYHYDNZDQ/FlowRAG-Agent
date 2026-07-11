"""Memory prompt injection via Runtime tests."""

from __future__ import annotations

import pytest
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.embeddings import FakeEmbeddings

from agent.runtime import AgentRuntime, ExecuteRequest
from agent.runtime.history import build_history_for_prompt
from ingestion.indexer import ingest_pdf
from memory import MemoryManager
from memory.base import ConversationMemory, ConversationTurn
from models.schemas import AnswerResult, IntentType
from skills.context import SkillContext
from skills.qa_skill import QASkill

pytestmark = pytest.mark.memory


class CapturingFakeLLM(FakeListChatModel):
    """Fake LLM that records user prompt content from each invoke."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses=responses)
        object.__setattr__(self, "user_prompts", [])

    def invoke(self, input, config=None, **kwargs):  # noqa: ANN001
        messages = input if isinstance(input, list) else [input]
        if messages:
            self.user_prompts.append(str(messages[-1].content))
        return super().invoke(input, config, **kwargs)


def test_build_history_for_prompt_from_runtime_bridge(rag_settings):
    memory = ConversationMemory(session_id="s1")
    memory.turns.append(
        ConversationTurn(query="上一轮问题", answer="上一轮回答", intent=IntentType.QA)
    )
    history = build_history_for_prompt(memory, rag_settings)
    assert "User: 上一轮问题" in history
    assert "Assistant: 上一轮回答" in history


def test_qa_skill_prompt_contains_history_sections(
    ingested_contract,
    rag_settings,
    fake_embeddings: FakeEmbeddings,
):
    result, store, _registry, _ = ingested_contract
    llm = CapturingFakeLLM(responses=["带历史的回答"])
    ctx = SkillContext.create(
        settings=rag_settings,
        vector_store=store,
        embeddings=fake_embeddings,
        llm=llm,
        conversation_history="User: 之前问过付款\nAssistant: 30天内付款",
    )
    from models.schemas import RetrievalMode, RetrievalScope

    QASkill().run(
        ctx,
        "还有违约金吗？",
        RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id]),
    )
    assert llm.user_prompts
    prompt = llm.user_prompts[0]
    assert "Conversation History:" in prompt
    assert "User: 之前问过付款" in prompt
    assert "Retrieved Context:" in prompt
    assert "Question:" in prompt
    assert "还有违约金吗？" in prompt


def test_runtime_multi_turn_injects_previous_dialogue(
    sample_contract_pdf,
    rag_settings,
    chroma_store,
    doc_registry,
    fake_embeddings: FakeEmbeddings,
):
    ingest_pdf(
        sample_contract_pdf,
        settings=rag_settings,
        store=chroma_store,
        registry=doc_registry,
        embeddings=fake_embeddings,
    )
    llm = CapturingFakeLLM(
        responses=[
            "第一轮答案",
            "第二轮答案",
        ]
    )
    runtime = AgentRuntime(memory_manager=MemoryManager())
    runtime.execute(
        ExecuteRequest(query="payment within 30 days", session_id="multi-turn"),
        settings=rag_settings,
        store=chroma_store,
        llm=llm,
        embeddings=fake_embeddings,
    )
    runtime.execute(
        ExecuteRequest(query="late payment penalty", session_id="multi-turn"),
        settings=rag_settings,
        store=chroma_store,
        llm=llm,
        embeddings=fake_embeddings,
    )
    assert len(llm.user_prompts) == 2
    assert "（无）" in llm.user_prompts[0] or "Conversation History:" in llm.user_prompts[0]
    second_prompt = llm.user_prompts[1]
    assert "payment within 30 days" in second_prompt
    assert "第一轮答案" in second_prompt


def test_runtime_truncates_long_history(
    rag_settings,
    chroma_store,
    fake_embeddings: FakeEmbeddings,
):
    memory = MemoryManager()
    session = memory.begin_session("long-history")
    for index in range(20):
        session.record_turn(
            f"q-{index}",
            AnswerResult(answer="x" * 200, intent=IntentType.QA),
        )

    settings = rag_settings.model_copy(
        update={"memory_max_turns": 2, "memory_max_tokens": 200},
    )
    history = build_history_for_prompt(memory.get_session("long-history"), settings)
    assert "q-18" in history or "q-19" in history
    assert "q-0" not in history

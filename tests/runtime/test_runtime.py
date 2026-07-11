"""Agent Runtime tests."""

from __future__ import annotations

import pytest
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.embeddings import FakeEmbeddings

from agent.orchestrator import run
from agent.runtime import AgentRuntime, ExecuteRequest
from ingestion.indexer import ingest_pdf
from memory import MemoryManager
from models.schemas import AgentStepType, IntentType
from skills.registry import build_default_skill_registry
from tools.registry import get_tool_registry

pytestmark = pytest.mark.runtime


def test_tool_registry_default_tools():
    names = get_tool_registry().names()
    assert "search_document" in names
    assert "summarize" in names
    assert "analyze" in names


def test_runtime_execute_qa(
    sample_contract_pdf,
    rag_settings,
    chroma_store,
    doc_registry,
    fake_embeddings: FakeEmbeddings,
    fake_llm: FakeListChatModel,
):
    ingest_pdf(
        sample_contract_pdf,
        settings=rag_settings,
        store=chroma_store,
        registry=doc_registry,
        embeddings=fake_embeddings,
    )
    runtime = AgentRuntime(
        skill_registry=build_default_skill_registry(),
        memory_manager=MemoryManager(),
    )
    steps: list = []
    result = runtime.execute(
        ExecuteRequest(
            query="payment within 30 days",
            session_id="runtime-qa",
            on_step=steps.append,
        ),
        settings=rag_settings,
        store=chroma_store,
        llm=fake_llm,
        embeddings=fake_embeddings,
    )
    assert result.answer.intent == IntentType.QA
    assert steps[0].name == "Router"
    assert any(s.step_type == AgentStepType.RETRIEVAL for s in result.answer.trace)


def test_runtime_records_memory_turn(
    rag_settings,
    chroma_store,
    fake_llm: FakeListChatModel,
    fake_embeddings: FakeEmbeddings,
):
    memory = MemoryManager()
    runtime = AgentRuntime(memory_manager=memory)
    runtime.execute(
        ExecuteRequest(query="付款周期是多少", session_id="s1"),
        settings=rag_settings,
        store=chroma_store,
        llm=fake_llm,
        embeddings=fake_embeddings,
    )
    session = memory.get_session("s1")
    assert session is not None
    assert len(session.turns) == 1


def test_orchestrator_compat_wrapper(
    sample_contract_pdf,
    rag_settings,
    chroma_store,
    doc_registry,
    fake_embeddings,
    fake_llm,
):
    ingest_pdf(
        sample_contract_pdf,
        settings=rag_settings,
        store=chroma_store,
        registry=doc_registry,
        embeddings=fake_embeddings,
    )
    result = run(
        "请总结合同要点",
        store=chroma_store,
        llm=fake_llm,
        embeddings=fake_embeddings,
        settings=rag_settings,
    )
    assert result.intent == IntentType.SUMMARIZE
    assert result.answer

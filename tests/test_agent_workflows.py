"""Tests for Agent workflows and orchestrator."""

from __future__ import annotations

import pytest
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.embeddings import FakeEmbeddings

from agent.orchestrator import run
from agent.router import route
from agent.workflows.analyze import AnalyzeWorkflow, build_analysis_subqueries
from agent.workflows.qa import QAWorkflow
from agent.workflows.summarize import SummarizeWorkflow
from ingestion.indexer import ingest_pdf
from models.schemas import AgentStepType, IntentType, RetrievalMode, RetrievalScope


@pytest.fixture
def fake_llm() -> FakeListChatModel:
    return FakeListChatModel(
        responses=[
            "根据文档，甲方应在30个自然日内完成付款。",
            "文档主题：合同付款与违约责任。核心要点：30日内付款；逾期违约金。",
            "分析结论：付款周期明确，违约成本较高。",
        ]
    )


def test_router_selects_workflow_type():
    assert route("付款周期是多少").intent == IntentType.QA
    assert route("请总结这份文档").intent == IntentType.SUMMARIZE
    assert route("分析合同风险").intent == IntentType.ANALYZE


def test_build_analysis_subqueries():
    queries = build_analysis_subqueries("合同风险")
    assert len(queries) == 3
    assert queries[0] == "合同风险"


def test_qa_workflow_with_trace(
    ingested_contract,
    rag_settings,
    fake_llm: FakeListChatModel,
    fake_embeddings: FakeEmbeddings,
):
    result, store, _registry, _ = ingested_contract
    scope = RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id])
    steps: list = []

    workflow = QAWorkflow(
        store,
        fake_llm,
        settings=rag_settings,
        embeddings=fake_embeddings,
    )
    answer = workflow.run(
        "payment within 30 days",
        scope,
        on_step=steps.append,
    )

    assert answer.answer
    assert answer.intent == IntentType.QA
    assert any(s.step_type == AgentStepType.RETRIEVAL for s in steps)
    assert any(s.step_type == AgentStepType.GENERATION for s in steps)
    assert steps[0].name == "Retriever"
    assert "找到" in steps[0].detail


def test_summarize_workflow(
    ingested_contract,
    rag_settings,
    fake_llm: FakeListChatModel,
    fake_embeddings: FakeEmbeddings,
):
    result, store, _registry, _ = ingested_contract
    scope = RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id])

    workflow = SummarizeWorkflow(
        store,
        fake_llm,
        settings=rag_settings,
        embeddings=fake_embeddings,
    )
    answer = workflow.run("总结文档", scope)

    assert answer.intent == IntentType.SUMMARIZE
    assert answer.answer
    assert answer.citations


def test_analyze_workflow(
    ingested_contract,
    rag_settings,
    fake_llm: FakeListChatModel,
    fake_embeddings: FakeEmbeddings,
):
    result, store, _registry, _ = ingested_contract
    scope = RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id])

    workflow = AnalyzeWorkflow(
        store,
        fake_llm,
        settings=rag_settings,
        embeddings=fake_embeddings,
    )
    answer = workflow.run("分析付款与违约条款", scope)

    assert answer.intent == IntentType.ANALYZE
    assert answer.answer
    assert len(answer.retrieved_chunks) > 0


def test_orchestrator_full_pipeline(
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

    steps: list = []
    result = run(
        "请总结合同要点",
        selected_doc_ids=[],
        on_step=steps.append,
        store=chroma_store,
        llm=fake_llm,
        embeddings=fake_embeddings,
        settings=rag_settings,
    )

    assert result.intent == IntentType.SUMMARIZE
    assert result.answer
    assert result.trace[0].name == "Router"
    assert "任务类型: summary" in result.trace[0].detail
    assert any(s.name == "Retriever" for s in result.trace)
    assert any(s.name == "Generator" for s in result.trace)


def test_orchestrator_qa_no_context(
    rag_settings,
    chroma_store,
    fake_llm: FakeListChatModel,
    fake_embeddings: FakeEmbeddings,
):
    """Empty knowledge base returns graceful message without LLM call for generation."""
    result = run(
        "付款周期是多少",
        store=chroma_store,
        llm=fake_llm,
        embeddings=fake_embeddings,
        settings=rag_settings,
    )

    assert "未找到" in result.answer

"""Skill tests — Skill orchestrates Tools directly."""

from __future__ import annotations

import pytest
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.embeddings import FakeEmbeddings

from models.schemas import IntentType, RetrievalMode, RetrievalScope
from skills.analysis_skill import AnalysisSkill, build_analysis_subqueries
from skills.context import SkillContext
from skills.qa_skill import QASkill
from skills.registry import build_default_skill_registry
from skills.summary_skill import SummarySkill

pytestmark = pytest.mark.skill


def test_skill_registry_builtin():
    registry = build_default_skill_registry()
    names = {item["name"] for item in registry.list_skills()}
    assert names == {"qa", "summary", "analysis"}


def test_analysis_subqueries():
    queries = build_analysis_subqueries("合同风险")
    assert len(queries) == 3
    assert queries[0] == "合同风险"


class TestQASkill:
    def test_tool_chain(self, ingested_contract, rag_settings, fake_embeddings, fake_llm):
        result, store, _registry, _ = ingested_contract
        ctx = SkillContext.create(
            settings=rag_settings,
            vector_store=store,
            embeddings=fake_embeddings,
            llm=fake_llm,
        )
        steps: list = []
        answer = QASkill().run(
            ctx,
            "payment within 30 days",
            RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id]),
            on_step=steps.append,
        )
        assert answer.intent == IntentType.QA
        assert any(s.metadata.get("tool") == "search_document" for s in steps)


class TestSummarySkill:
    def test_search_then_summarize(
        self, ingested_contract, rag_settings, fake_embeddings, fake_llm
    ):
        result, store, _registry, _ = ingested_contract
        ctx = SkillContext.create(
            settings=rag_settings,
            vector_store=store,
            embeddings=fake_embeddings,
            llm=fake_llm,
        )
        steps: list = []
        answer = SummarySkill().run(
            ctx,
            "请总结文档",
            RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id]),
            on_step=steps.append,
        )
        tool_names = {s.metadata.get("tool") for s in steps}
        assert "search_document" in tool_names
        assert "summarize" in tool_names
        assert answer.intent == IntentType.SUMMARIZE


class TestAnalysisSkill:
    def test_multi_retrieval(self, ingested_contract, rag_settings, fake_embeddings, fake_llm):
        result, store, _registry, _ = ingested_contract
        ctx = SkillContext.create(
            settings=rag_settings,
            vector_store=store,
            embeddings=fake_embeddings,
            llm=fake_llm,
        )
        answer = AnalysisSkill().run(
            ctx,
            "分析合同风险",
            RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id]),
        )
        assert answer.intent == IntentType.ANALYZE
        assert answer.retrieved_chunks

"""RAG answer generation and citation tests."""

from __future__ import annotations

import pytest
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.embeddings import FakeEmbeddings

from models.schemas import AgentStepType, IntentType, RetrievalMode, RetrievalScope
from retrieval.citation_builder import build_citations, format_citation_label
from skills.context import SkillContext
from skills.helpers import NO_CONTEXT_ANSWER
from skills.qa_skill import QASkill
from tests.fixtures.data.documents import SAMPLE_CONTRACT_FILENAME

pytestmark = pytest.mark.rag_answer


def test_citation_builder_dedupes_and_formats():
    from models.schemas import ChunkMetadata, RetrievedChunk

    meta = ChunkMetadata(
        doc_id="d1",
        source_file="a.pdf",
        source_path="data/a.pdf",
        page=2,
        chunk_index=0,
        chunk_id="c1",
    )
    chunk = RetrievedChunk(chunk_id="c1", text="payment terms", metadata=meta, score=0.9)
    citations = build_citations([chunk, chunk])
    assert len(citations) == 1
    assert format_citation_label(citations[0]) == "a.pdf 第2页"


def test_qa_skill_returns_answer_with_citations(
    ingested_contract,
    rag_settings,
    fake_embeddings: FakeEmbeddings,
    fake_llm: FakeListChatModel,
):
    result, store, _registry, _ = ingested_contract
    ctx = SkillContext.create(
        settings=rag_settings,
        vector_store=store,
        embeddings=fake_embeddings,
        llm=fake_llm,
    )
    answer = QASkill().run(
        ctx,
        "payment within 30 days",
        RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id]),
    )
    assert answer.intent == IntentType.QA
    assert answer.answer
    assert answer.citations
    assert answer.citations[0].source_file == SAMPLE_CONTRACT_FILENAME


def test_qa_skill_empty_kb_returns_fallback(
    rag_settings,
    chroma_store,
    fake_llm: FakeListChatModel,
    fake_embeddings: FakeEmbeddings,
):
    ctx = SkillContext.create(
        settings=rag_settings,
        vector_store=chroma_store,
        embeddings=fake_embeddings,
        llm=fake_llm,
    )
    answer = QASkill().run(
        ctx,
        "payment within 30 days",
        RetrievalScope(mode=RetrievalMode.ALL),
    )
    assert NO_CONTEXT_ANSWER in answer.answer
    assert answer.citations == []


def test_qa_skill_trace_contains_generation_step(
    ingested_contract,
    rag_settings,
    fake_embeddings: FakeEmbeddings,
    fake_llm: FakeListChatModel,
):
    result, store, _registry, _ = ingested_contract
    ctx = SkillContext.create(
        settings=rag_settings,
        vector_store=store,
        embeddings=fake_embeddings,
        llm=fake_llm,
    )
    steps: list = []
    QASkill().run(
        ctx,
        "payment within 30 days",
        RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id]),
        on_step=steps.append,
    )
    assert any(s.step_type == AgentStepType.GENERATION for s in steps)

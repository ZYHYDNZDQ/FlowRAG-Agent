"""End-to-end Agent pipeline tests."""

from __future__ import annotations

import pytest
from langchain_community.embeddings import FakeEmbeddings

from agent.orchestrator import run
from agent.runtime import AgentRuntime, ExecuteRequest
from config.settings import Settings
from ingestion.indexer import ingest_pdf
from models.schemas import DocStatus, IntentType
from retrieval.chroma_store import ChromaStore
from tests.fixtures.data.documents import SAMPLE_CONTRACT_FILENAME
from tests.fixtures.data.qa_scenarios import QA_SCENARIOS

pytestmark = pytest.mark.e2e


def test_e2e_ingest_pdf_pipeline(
    sample_contract_pdf,
    rag_settings: Settings,
    chroma_store: ChromaStore,
    doc_registry,
    fake_embeddings: FakeEmbeddings,
):
    result = ingest_pdf(
        sample_contract_pdf,
        settings=rag_settings,
        store=chroma_store,
        registry=doc_registry,
        embeddings=fake_embeddings,
    )
    assert result.status == DocStatus.INDEXED
    assert result.page_count == 2
    assert result.chunk_count > 0
    assert chroma_store.count() == result.chunk_count
    record = doc_registry.get(result.doc_id)
    assert record is not None
    assert record.source_file == SAMPLE_CONTRACT_FILENAME


def test_e2e_ingest_skips_duplicate(
    sample_contract_pdf,
    rag_settings,
    chroma_store,
    doc_registry,
    fake_embeddings,
):
    first = ingest_pdf(
        sample_contract_pdf,
        settings=rag_settings,
        store=chroma_store,
        registry=doc_registry,
        embeddings=fake_embeddings,
    )
    second = ingest_pdf(
        sample_contract_pdf,
        settings=rag_settings,
        store=chroma_store,
        registry=doc_registry,
        embeddings=fake_embeddings,
    )
    assert first.status == DocStatus.INDEXED
    assert second.doc_id == first.doc_id
    assert "already indexed" in second.error_message


@pytest.mark.parametrize("scenario", QA_SCENARIOS, ids=lambda s: s["intent"].value)
def test_e2e_runtime_scenarios(
    ingested_contract,
    rag_settings,
    fake_llm,
    fake_embeddings,
    scenario,
):
    result, store, _registry, _ = ingested_contract
    runtime = AgentRuntime()
    execute_result = runtime.execute(
        ExecuteRequest(
            query=str(scenario["query"]),
            session_id=f"e2e-{scenario['intent'].value}",
        ),
        settings=rag_settings,
        store=store,
        llm=fake_llm,
        embeddings=fake_embeddings,
    )
    answer = execute_result.answer
    assert answer.intent == scenario["intent"]
    assert answer.answer
    if scenario["expect_citations"]:
        assert answer.citations
    if scenario["expect_chunks"]:
        assert answer.retrieved_chunks


def test_e2e_full_stack_via_orchestrator(ingested_contract, rag_settings, fake_llm, fake_embeddings):
    result, store, _registry, _ = ingested_contract
    answer = run(
        "payment within 30 days",
        selected_doc_ids=[result.doc_id],
        store=store,
        llm=fake_llm,
        embeddings=fake_embeddings,
        settings=rag_settings,
    )
    assert answer.intent == IntentType.QA
    assert answer.trace[0].name == "Router"
    assert answer.citations

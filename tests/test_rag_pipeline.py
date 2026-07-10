"""End-to-end tests for the RAG ingestion + retrieval pipeline."""

import pytest
from langchain_community.embeddings import FakeEmbeddings

from config.settings import Settings
from ingestion.indexer import ingest_pdf
from models.schemas import DocStatus, RetrievalMode, RetrievalScope
from retrieval.chroma_store import ChromaStore
from retrieval.retriever_factory import create_retriever, retrieve_with_sources
from tests.fixtures.sample_content import SAMPLE_CONTRACT_FILENAME


def test_ingest_pdf_pipeline(
    sample_contract_pdf,
    rag_settings: Settings,
    chroma_store: ChromaStore,
    doc_registry,
    fake_embeddings: FakeEmbeddings,
):
    """PDF → parse → chunk → embed → Chroma; registry marked indexed."""
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
    assert record.status == DocStatus.INDEXED
    assert record.source_file == SAMPLE_CONTRACT_FILENAME


def test_retriever_returns_source_metadata(
    ingested_contract,
):
    """Retriever results must expose filename and page number."""
    result, chroma_store, _registry, fake_embeddings = ingested_contract

    scope = RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id])
    docs = retrieve_with_sources(
        chroma_store,
        "payment within 30 days",
        scope,
        top_k=3,
        score_threshold=0.0,
        embeddings=fake_embeddings,
    )

    assert len(docs) > 0
    for doc in docs:
        assert doc.metadata["source_file"] == SAMPLE_CONTRACT_FILENAME
        assert isinstance(doc.metadata["page"], int)
        assert doc.metadata["page"] >= 1
        assert doc.metadata["doc_id"] == result.doc_id


def test_retriever_invoke_via_langchain(
    ingested_contract,
    rag_settings: Settings,
):
    """LangChain retriever.invoke() returns Documents with source fields."""
    result, chroma_store, _registry, fake_embeddings = ingested_contract

    retriever = create_retriever(
        chroma_store,
        RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id]),
        top_k=2,
        score_threshold=0.0,
        embeddings=fake_embeddings,
        settings=rag_settings,
    )
    docs = retriever.invoke("penalty")

    assert docs
    assert docs[0].metadata["source_file"] == SAMPLE_CONTRACT_FILENAME
    assert "page" in docs[0].metadata


def test_ingest_skips_duplicate_hash(
    sample_contract_pdf,
    rag_settings: Settings,
    chroma_store: ChromaStore,
    doc_registry,
    fake_embeddings: FakeEmbeddings,
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

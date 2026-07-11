"""Shared pytest fixtures for FlowRAG-Agent test suite."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.embeddings import FakeEmbeddings

from config.settings import Settings
from ingestion.indexer import ingest_pdf
from models.doc_registry import DocRegistry
from models.schemas import DocStatus, IngestResult
from retrieval.chroma_store import ChromaStore
from tests.fixtures.data.documents import SAMPLE_CONTRACT_FILENAME, SAMPLE_CONTRACT_PAGES
from tests.fixtures.data.llm_responses import FAKE_LLM_RESPONSES
from tests.fixtures.factories import make_sample_pdf

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


def pytest_configure(config: pytest.Config) -> None:
    markers = [
        ("rag_retrieval", "RAG retrieval layer tests"),
        ("rag_answer", "RAG answer generation and citation tests"),
        ("router", "Intent router tests"),
        ("tool", "Tool system tests"),
        ("skill", "Skill workflow tests"),
        ("runtime", "Agent Runtime tests"),
        ("memory", "Memory manager and isolation tests"),
        ("e2e", "End-to-end pipeline tests"),
        ("unit", "Supporting unit tests"),
        ("example", "RAG usage examples (API documentation)"),
    ]
    for name, description in markers:
        config.addinivalue_line("markers", f"{name}: {description}")


@pytest.fixture
def rag_settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    return Settings(
        data_dir=data_dir,
        uploads_dir=data_dir / "uploads",
        chroma_persist_dir=data_dir / "chroma",
        registry_path=data_dir / "registry.db",
        chroma_collection_name="test_flowrag",
        chunk_size=120,
        chunk_overlap=20,
    )


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings(size=128)


@pytest.fixture
def fake_llm() -> FakeListChatModel:
    return FakeListChatModel(responses=list(FAKE_LLM_RESPONSES))


@pytest.fixture
def chroma_store(rag_settings: Settings) -> ChromaStore:
    store = ChromaStore(rag_settings)
    store.connect()
    return store


@pytest.fixture
def doc_registry(rag_settings: Settings) -> DocRegistry:
    return DocRegistry(rag_settings.registry_path)


@pytest.fixture
def sample_contract_pdf(tmp_path: Path) -> Path:
    return make_sample_pdf(tmp_path / SAMPLE_CONTRACT_FILENAME, SAMPLE_CONTRACT_PAGES)


@pytest.fixture
def ingested_contract(
    sample_contract_pdf: Path,
    rag_settings: Settings,
    chroma_store: ChromaStore,
    doc_registry: DocRegistry,
    fake_embeddings: FakeEmbeddings,
) -> tuple[IngestResult, ChromaStore, DocRegistry, Embeddings]:
    result = ingest_pdf(
        sample_contract_pdf,
        settings=rag_settings,
        store=chroma_store,
        registry=doc_registry,
        embeddings=fake_embeddings,
    )
    assert result.status == DocStatus.INDEXED
    return result, chroma_store, doc_registry, fake_embeddings

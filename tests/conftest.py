"""Shared pytest fixtures for RAG pipeline tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import fitz
import pytest
from langchain_community.embeddings import FakeEmbeddings

from config.settings import Settings
from ingestion.indexer import ingest_pdf
from models.doc_registry import DocRegistry
from models.schemas import DocStatus, IngestResult
from retrieval.chroma_store import ChromaStore
from tests.fixtures.sample_content import SAMPLE_CONTRACT_FILENAME, SAMPLE_CONTRACT_PAGES

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "example: RAG usage examples (also serve as API documentation)",
    )


@pytest.fixture
def rag_settings(tmp_path: Path) -> Settings:
    """Isolated data directories and small chunks for fast tests."""
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
    """Deterministic 128-d vectors — no model download required."""
    return FakeEmbeddings(size=128)


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
    """Two-page sample PDF used across pipeline and example tests."""
    return make_sample_pdf(
        tmp_path / SAMPLE_CONTRACT_FILENAME,
        SAMPLE_CONTRACT_PAGES,
    )


@pytest.fixture
def ingested_contract(
    sample_contract_pdf: Path,
    rag_settings: Settings,
    chroma_store: ChromaStore,
    doc_registry: DocRegistry,
    fake_embeddings: FakeEmbeddings,
) -> tuple[IngestResult, ChromaStore, DocRegistry, Embeddings]:
    """
    Pre-ingested contract PDF for retrieval / citation examples.

    Returns:
        (ingest_result, chroma_store, doc_registry, embeddings)
    """
    result = ingest_pdf(
        sample_contract_pdf,
        settings=rag_settings,
        store=chroma_store,
        registry=doc_registry,
        embeddings=fake_embeddings,
    )
    assert result.status == DocStatus.INDEXED
    return result, chroma_store, doc_registry, fake_embeddings


def make_sample_pdf(path: Path, pages: list[str]) -> Path:
    """
    Create a minimal PDF with known text per page (for parser / pipeline tests).

    Args:
        path: Output ``.pdf`` path.
        pages: List of page texts; one page per item.
    """
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path

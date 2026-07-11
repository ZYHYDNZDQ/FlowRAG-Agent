"""MCP server adapter tests — no MCP protocol wiring required."""

from __future__ import annotations

import json

import pytest
from langchain_community.embeddings import FakeEmbeddings

from config.settings import Settings
from ingestion.indexer import ingest_pdf
from mcp_server.resources import _record_to_dict
from mcp_server.tools import build_retrieval_scope, serialize_chunks
from models.doc_registry import DocRegistry
from models.schemas import DocStatus, RetrievalMode
from retrieval.chroma_store import ChromaStore
from tests.fixtures.data.documents import SAMPLE_CONTRACT_FILENAME, SAMPLE_CONTRACT_PAGES
from tests.fixtures.factories import make_sample_pdf
from tools.context import ToolExecutionContext
from tools.registry import get_tool_registry

pytestmark = pytest.mark.unit


def test_build_retrieval_scope():
    assert build_retrieval_scope(None).mode == RetrievalMode.ALL
    assert build_retrieval_scope(["d1"]).mode == RetrievalMode.SINGLE
    assert build_retrieval_scope(["d1", "d2"]).mode == RetrievalMode.SELECTED


def test_search_document_delegates_to_rag_service(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        uploads_dir=tmp_path / "data" / "uploads",
        chroma_persist_dir=tmp_path / "data" / "chroma",
        registry_path=tmp_path / "data" / "registry.db",
        chroma_collection_name="mcp_test",
        chunk_size=120,
        chunk_overlap=20,
        score_threshold=0.0,
    )
    settings.ensure_dirs()
    pdf = make_sample_pdf(settings.uploads_dir / SAMPLE_CONTRACT_FILENAME, SAMPLE_CONTRACT_PAGES)
    store = ChromaStore(settings)
    store.connect()
    registry = DocRegistry(settings.registry_path)
    embeddings = FakeEmbeddings(size=128)
    result = ingest_pdf(pdf, settings=settings, store=store, registry=registry, embeddings=embeddings)
    assert result.status == DocStatus.INDEXED

    ctx = ToolExecutionContext.create(settings=settings, vector_store=store, embeddings=embeddings)
    chunks = get_tool_registry().run(
        "search_document",
        ctx,
        query="payment within 30 days",
        scope=build_retrieval_scope([result.doc_id]),
        top_k=3,
        score_threshold=0.0,
    )
    payload = json.loads(serialize_chunks(chunks))
    assert payload
    assert payload[0]["metadata"]["page"] >= 1


def test_record_to_dict_shape():
    from datetime import datetime, timezone

    from models.schemas import DocRecord

    record = DocRecord(
        doc_id="d1",
        source_file="a.pdf",
        source_path="data/a.pdf",
        file_hash="hash",
        page_count=2,
        chunk_count=5,
        status=DocStatus.INDEXED,
        ingested_at=datetime.now(timezone.utc),
    )
    data = _record_to_dict(record)
    assert data["doc_id"] == "d1"
    assert data["status"] == "indexed"

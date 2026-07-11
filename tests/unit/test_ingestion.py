"""Ingestion unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from config.settings import Settings
from ingestion.chunker import build_chunk_metadata, chunk_documents
from ingestion.pdf_parser import parse_pdf
from models.doc_registry import DocRegistry
from models.schemas import ChunkMetadata, DocRecord, DocStatus, RetrievedChunk
from retrieval.chunk_utils import dedupe_chunks_by_id, filter_by_score
from retrieval.rag_service import RAGService
from tests.fixtures.factories import make_sample_pdf

pytestmark = pytest.mark.unit


def test_parse_pdf_returns_page_metadata(tmp_path: Path):
    pdf_path = make_sample_pdf(
        tmp_path / "contract.pdf",
        ["Page 1: payment terms", "Page 2: breach liability"],
    )
    docs = parse_pdf(pdf_path, doc_id="doc-1", source_file="contract.pdf")
    assert len(docs) == 2
    assert docs[0].metadata["page"] == 1
    assert "payment" in docs[0].page_content


def test_parse_pdf_raises_on_empty_text(tmp_path: Path):
    import fitz

    blank = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(blank)
    doc.close()
    with pytest.raises(ValueError, match="No extractable text"):
        parse_pdf(blank, doc_id="doc-2", source_file="blank.pdf")


def test_chunk_id_format():
    meta = build_chunk_metadata(
        "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "test.pdf",
        "uploads/test.pdf",
        page=12,
        chunk_index=3,
    )
    assert meta.chunk_id == "a1b2c3d4_p0012_c0003"


def test_chunk_documents_preserves_metadata():
    settings = Settings(chunk_size=50, chunk_overlap=10)
    long_text = "test chunk content. " * 10
    page_docs = [
        Document(
            page_content=long_text,
            metadata={
                "doc_id": "doc-1",
                "source_file": "report.pdf",
                "source_path": "data/uploads/report.pdf",
                "page": 3,
                "page_count": 5,
            },
        )
    ]
    chunks = chunk_documents(page_docs, doc_id="doc-1", settings=settings)
    assert len(chunks) >= 2
    assert chunks[0].metadata["source_file"] == "report.pdf"


def test_doc_registry_crud(tmp_path):
    registry = DocRegistry(tmp_path / "registry.db")
    record = DocRecord(
        doc_id="doc-test-1",
        source_file="sample.pdf",
        source_path="data/uploads/sample.pdf",
        file_hash="sha256:deadbeef",
        page_count=5,
        chunk_count=12,
        status=DocStatus.INDEXED,
    )
    registry.create(record)
    assert registry.get("doc-test-1") is not None
    assert registry.clear_all() == 1


def _make_chunk(chunk_id: str, *, score: float | None = 1.0, text: str = "sample"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            doc_id="doc-1",
            source_file="test.pdf",
            source_path="data/uploads/test.pdf",
            page=1,
            chunk_index=0,
            chunk_id=chunk_id,
        ),
        score=score,
    )


def test_chunk_utils_filter_and_dedupe():
    chunks = [_make_chunk("a", score=0.9), _make_chunk("b", score=0.1)]
    assert [c.chunk_id for c in filter_by_score(chunks, 0.5)] == ["a"]
    duped = [_make_chunk("dup", score=0.4), _make_chunk("dup", score=0.9)]
    assert dedupe_chunks_by_id(duped)[0].text == "sample"


def test_rag_service_format_context():
    assert RAGService.format_context([]) == "（无检索结果）"

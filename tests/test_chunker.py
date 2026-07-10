"""Tests for ingestion/chunker.py."""

from langchain_core.documents import Document

from ingestion.chunker import build_chunk_metadata, chunk_documents
from config.settings import Settings


def test_chunk_id_format():
    meta = build_chunk_metadata(
        "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "test.pdf",
        "uploads/test.pdf",
        page=12,
        chunk_index=3,
    )
    assert meta.chunk_id == "a1b2c3d4_p0012_c0003"
    assert meta.page == 12
    assert meta.chunk_index == 3


def test_chunk_metadata_required_fields():
    meta = build_chunk_metadata(
        "doc-uuid",
        "合同.pdf",
        "data/uploads/合同.pdf",
        page=1,
        chunk_index=0,
        file_hash="sha256:abc",
        page_count=10,
    )
    chroma_dict = meta.to_chroma_dict()
    assert chroma_dict["doc_id"] == "doc-uuid"
    assert chroma_dict["source_file"] == "合同.pdf"
    assert chroma_dict["page"] == 1
    assert "file_hash" in chroma_dict


def test_chunk_documents_preserves_filename_and_page():
    settings = Settings(chunk_size=50, chunk_overlap=10)
    long_text = "这是一段用于测试切分的长文本。" * 5
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
    for chunk in chunks:
        assert chunk.metadata["source_file"] == "report.pdf"
        assert chunk.metadata["page"] == 3
        assert "chunk_id" in chunk.metadata

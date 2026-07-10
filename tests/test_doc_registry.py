"""Tests for models/doc_registry.py."""

from models.doc_registry import DocRegistry
from models.schemas import DocRecord, DocStatus


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

    fetched = registry.get("doc-test-1")
    assert fetched is not None
    assert fetched.source_file == "sample.pdf"
    assert fetched.chunk_count == 12

    by_hash = registry.find_by_hash("sha256:deadbeef")
    assert by_hash is not None
    assert by_hash.doc_id == "doc-test-1"

    registry.update_status("doc-test-1", DocStatus.FAILED, error_message="parse error")
    updated = registry.get("doc-test-1")
    assert updated is not None
    assert updated.status == DocStatus.FAILED

    assert registry.clear_all() == 1
    assert registry.get("doc-test-1") is None

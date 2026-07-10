"""Tests for retrieval/chroma_store.py and retriever_factory.py."""

import pytest

from models.schemas import RetrievalMode, RetrievalScope
from retrieval.chroma_store import ChromaStore


def test_build_where_filter_single_doc():
    scope = RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=["doc-1"])
    assert ChromaStore.build_where_filter(scope) == {"doc_id": {"$eq": "doc-1"}}


def test_build_where_filter_selected_docs():
    scope = RetrievalScope(mode=RetrievalMode.SELECTED, doc_ids=["a", "b"])
    assert ChromaStore.build_where_filter(scope) == {"doc_id": {"$in": ["a", "b"]}}


def test_build_where_filter_all_returns_none():
    scope = RetrievalScope(mode=RetrievalMode.ALL)
    assert ChromaStore.build_where_filter(scope) is None


def test_build_where_filter_page_range():
    scope = RetrievalScope(
        mode=RetrievalMode.SINGLE,
        doc_ids=["doc-1"],
        page_start=10,
        page_end=25,
    )
    result = ChromaStore.build_where_filter(scope)
    assert result == {
        "$and": [
            {"doc_id": {"$eq": "doc-1"}},
            {"page": {"$gte": 10}},
            {"page": {"$lte": 25}},
        ]
    }


def test_build_where_filter_single_without_doc_id_raises():
    scope = RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[])
    with pytest.raises(ValueError, match="requires exactly one doc_id"):
        ChromaStore.build_where_filter(scope)

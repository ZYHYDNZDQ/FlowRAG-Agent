"""RAG retrieval layer tests."""

from __future__ import annotations

import pytest

from models.schemas import RetrievalMode, RetrievalScope
from retrieval.chroma_store import ChromaStore
from retrieval.rag_service import RAGService
from retrieval.retriever_factory import create_retriever, retrieve_with_sources
from tests.fixtures.data.documents import SAMPLE_CONTRACT_FILENAME, SAMPLE_QUERIES

pytestmark = pytest.mark.rag_retrieval


class TestChromaWhereFilter:
    def test_single_doc(self):
        scope = RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=["doc-1"])
        assert ChromaStore.build_where_filter(scope) == {"doc_id": {"$eq": "doc-1"}}

    def test_selected_docs(self):
        scope = RetrievalScope(mode=RetrievalMode.SELECTED, doc_ids=["a", "b"])
        assert ChromaStore.build_where_filter(scope) == {"doc_id": {"$in": ["a", "b"]}}

    def test_all_returns_none(self):
        scope = RetrievalScope(mode=RetrievalMode.ALL)
        assert ChromaStore.build_where_filter(scope) is None

    def test_page_range(self):
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

    def test_single_without_doc_id_raises(self):
        scope = RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[])
        with pytest.raises(ValueError, match="requires exactly one doc_id"):
            ChromaStore.build_where_filter(scope)


class TestRAGServiceRetrieval:
    def test_query_returns_chunks_with_metadata(self, ingested_contract, rag_settings):
        result, store, _registry, fake_embeddings = ingested_contract
        scope = RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id])
        chunks = RAGService().query_chunks(
            store,
            "payment within 30 days",
            scope,
            top_k=3,
            score_threshold=0.0,
            embeddings=fake_embeddings,
            settings=rag_settings,
        )
        assert chunks
        assert chunks[0].metadata.source_file == SAMPLE_CONTRACT_FILENAME
        assert chunks[0].metadata.page >= 1

    @pytest.mark.parametrize("case", SAMPLE_QUERIES, ids=lambda c: str(c["query"]))
    def test_parametrized_queries_hit_expected_page(self, ingested_contract, rag_settings, case):
        result, store, _registry, fake_embeddings = ingested_contract
        scope = RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id])
        chunks = RAGService().query_chunks(
            store,
            str(case["query"]),
            scope,
            top_k=3,
            score_threshold=0.0,
            embeddings=fake_embeddings,
            settings=rag_settings,
        )
        assert chunks
        pages = {chunk.metadata.page for chunk in chunks}
        assert case["expected_page"] in pages


class TestLangChainRetriever:
    def test_retrieve_with_sources_metadata(self, ingested_contract):
        result, store, _registry, fake_embeddings = ingested_contract
        scope = RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id])
        docs = retrieve_with_sources(
            store,
            "payment within 30 days",
            scope,
            top_k=3,
            score_threshold=0.0,
            embeddings=fake_embeddings,
        )
        assert docs
        for doc in docs:
            assert doc.metadata["source_file"] == SAMPLE_CONTRACT_FILENAME
            assert doc.metadata["doc_id"] == result.doc_id

    def test_retriever_invoke(self, ingested_contract, rag_settings):
        result, store, _registry, fake_embeddings = ingested_contract
        retriever = create_retriever(
            store,
            RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id]),
            top_k=2,
            score_threshold=0.0,
            embeddings=fake_embeddings,
            settings=rag_settings,
        )
        docs = retriever.invoke("penalty")
        assert docs
        assert docs[0].metadata["page"] >= 1

"""
RAG pipeline usage examples.

These tests double as copy-paste references for the ingestion + retrieval API.
Run only examples:

    pytest tests/examples/ -v

Run full suite:

    pytest tests/ -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_documents
from ingestion.indexer import ingest_pdf
from ingestion.pdf_parser import parse_pdf
from models.schemas import DocStatus, RetrievalMode, RetrievalScope
from retrieval.chroma_store import ChromaStore
from retrieval.citation_builder import build_citations, format_citation_label
from retrieval.retriever_factory import create_retriever, retrieve_with_sources
from tests.fixtures.data.documents import (
    SAMPLE_CONTRACT_FILENAME,
    SAMPLE_CONTRACT_PAGES,
    SAMPLE_QUERIES,
)

pytestmark = pytest.mark.example


# ---------------------------------------------------------------------------
# Example 1 — Step-by-step pipeline (parse → chunk → embed → store → retrieve)
# ---------------------------------------------------------------------------


def test_example_step_by_step_pipeline(
    sample_contract_pdf: Path,
    rag_settings,
    chroma_store: ChromaStore,
    doc_registry,
    fake_embeddings,
):
    """
    Example: manually walk each RAG stage (useful when debugging one layer).

    Stages:
      1. parse_pdf       — PyMuPDF per-page Documents
      2. chunk_documents — RecursiveCharacterTextSplitter
      3. embed_documents — batch vectors
      4. add_chunks      — Chroma persistence
      5. query           — similarity search with source metadata
    """
    doc_id = "example-doc-001"

    # 1) Parse
    page_docs = parse_pdf(
        sample_contract_pdf,
        doc_id=doc_id,
        source_file=SAMPLE_CONTRACT_FILENAME,
    )
    assert len(page_docs) == 2
    assert page_docs[0].metadata["source_file"] == SAMPLE_CONTRACT_FILENAME
    assert page_docs[0].metadata["page"] == 1

    # 2) Chunk
    for doc in page_docs:
        doc.metadata["file_hash"] = "sha256:example"
        doc.metadata["ingest_version"] = 1
    chunks = chunk_documents(page_docs, doc_id=doc_id, settings=rag_settings)
    assert all(c.metadata["source_file"] == SAMPLE_CONTRACT_FILENAME for c in chunks)
    assert all(isinstance(c.metadata["page"], int) for c in chunks)

    # 3) Embed
    vectors = embed_documents(chunks, embeddings=fake_embeddings)
    assert len(vectors) == len(chunks)

    # 4) Store
    chroma_store.add_chunks(
        ids=[c.metadata["chunk_id"] for c in chunks],
        documents=[c.page_content for c in chunks],
        metadatas=[dict(c.metadata) for c in chunks],
        embeddings=vectors,
    )
    assert chroma_store.count() == len(chunks)

    # 5) Retrieve
    hits = chroma_store.query(
        "payment within days",
        top_k=2,
        where={"doc_id": {"$eq": doc_id}},
        embeddings=fake_embeddings,
    )
    assert hits
    assert hits[0].metadata.source_file == SAMPLE_CONTRACT_FILENAME
    assert hits[0].metadata.page >= 1


# ---------------------------------------------------------------------------
# Example 2 — One-call ingestion via indexer
# ---------------------------------------------------------------------------


def test_example_ingest_pdf_one_liner(
    sample_contract_pdf: Path,
    rag_settings,
    chroma_store: ChromaStore,
    doc_registry,
    fake_embeddings,
):
    """
    Example: production path — single ``ingest_pdf`` handles the full pipeline.

    ```python
    from ingestion.indexer import ingest_pdf

    result = ingest_pdf("data/uploads/contract.pdf")
    assert result.status == DocStatus.INDEXED
    print(result.doc_id, result.chunk_count)
    ```
    """
    result = ingest_pdf(
        sample_contract_pdf,
        settings=rag_settings,
        store=chroma_store,
        registry=doc_registry,
        embeddings=fake_embeddings,
    )

    assert result.status == DocStatus.INDEXED
    assert result.source_file == SAMPLE_CONTRACT_FILENAME
    assert result.page_count == len(SAMPLE_CONTRACT_PAGES)
    assert result.chunk_count > 0

    record = doc_registry.get(result.doc_id)
    assert record is not None
    assert record.chunk_count == result.chunk_count


# ---------------------------------------------------------------------------
# Example 3 — Retriever with source metadata
# ---------------------------------------------------------------------------


def test_example_retrieve_with_sources(ingested_contract):
    """
    Example: retrieve chunks as LangChain Documents with filename + page.

    ```python
    from models.schemas import RetrievalScope, RetrievalMode
    from retrieval.retriever_factory import retrieve_with_sources

    scope = RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[doc_id])
    docs = retrieve_with_sources(store, "payment terms", scope)
    print(docs[0].metadata["source_file"], docs[0].metadata["page"])
    ```
    """
    result, store, _registry, embeddings = ingested_contract
    scope = RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id])

    docs = retrieve_with_sources(
        store,
        "payment within 30 days",
        scope,
        top_k=3,
        score_threshold=0.0,
        embeddings=embeddings,
    )

    assert docs
    first = docs[0]
    assert first.metadata["source_file"] == SAMPLE_CONTRACT_FILENAME
    assert first.metadata["page"] in (1, 2)
    assert first.metadata["doc_id"] == result.doc_id
    assert "chunk_id" in first.metadata


# ---------------------------------------------------------------------------
# Example 4 — LangChain retriever.invoke()
# ---------------------------------------------------------------------------


def test_example_langchain_retriever_invoke(ingested_contract, rag_settings):
    """
    Example: use ``create_retriever`` when you need a LangChain Retriever.

    ```python
    retriever = create_retriever(store, scope, top_k=5)
    docs = retriever.invoke("penalty rate")
    ```
    """
    result, store, _registry, embeddings = ingested_contract
    retriever = create_retriever(
        store,
        RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id]),
        top_k=2,
        score_threshold=0.0,
        embeddings=embeddings,
        settings=rag_settings,
    )

    docs = retriever.invoke("penalty")
    assert isinstance(docs[0], Document)
    assert docs[0].metadata["source_file"] == SAMPLE_CONTRACT_FILENAME


# ---------------------------------------------------------------------------
# Example 5 — Page-range filter (analyze a chapter)
# ---------------------------------------------------------------------------


def test_example_page_range_filter(ingested_contract, fake_embeddings):
    """
    Example: restrict retrieval to pages 2–2 (liability chapter only).

    ```python
    scope = RetrievalScope(
        mode=RetrievalMode.SINGLE,
        doc_ids=[doc_id],
        page_start=2,
        page_end=2,
    )
    ```
    """
    result, store, _registry, embeddings = ingested_contract
    scope = RetrievalScope(
        mode=RetrievalMode.SINGLE,
        doc_ids=[result.doc_id],
        page_start=2,
        page_end=2,
    )

    docs = retrieve_with_sources(
        store,
        "payment",  # keyword appears on page 1, should not dominate page-2 filter
        scope,
        top_k=5,
        score_threshold=0.0,
        embeddings=embeddings,
    )

    assert docs
    assert all(doc.metadata["page"] == 2 for doc in docs)


# ---------------------------------------------------------------------------
# Example 6 — Build citations from retrieval hits
# ---------------------------------------------------------------------------


def test_example_build_citations_from_hits(ingested_contract, fake_embeddings):
    """
    Example: turn raw retrieval hits into deduplicated Citation objects.

    ```python
    from retrieval.citation_builder import build_citations, format_citation_label

    hits = store.query("payment", where=where, embeddings=embeddings)
    citations = build_citations(hits)
    for c in citations:
        print(format_citation_label(c))  # contract.pdf 第1页
    ```
    """
    result, store, _registry, embeddings = ingested_contract
    hits = store.query(
        "payment terms",
        top_k=3,
        where={"doc_id": {"$eq": result.doc_id}},
        embeddings=embeddings,
    )

    citations = build_citations(hits, max_citations=5)
    assert citations
    assert citations[0].source_file == SAMPLE_CONTRACT_FILENAME
    assert citations[0].page >= 1
    assert format_citation_label(citations[0]).startswith(SAMPLE_CONTRACT_FILENAME)


# ---------------------------------------------------------------------------
# Example 7 — Parametrized queries (table-driven smoke tests)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", SAMPLE_QUERIES, ids=lambda c: c["query"])
def test_example_parametrized_queries(ingested_contract, case):
    """
    Example: table-driven retrieval checks for multiple user questions.

    Extend ``tests/fixtures/data/documents.py:SAMPLE_QUERIES`` with new rows.
    """
    result, store, _registry, embeddings = ingested_contract
    scope = RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id])

    docs = retrieve_with_sources(
        store,
        str(case["query"]),
        scope,
        top_k=3,
        score_threshold=0.0,
        embeddings=embeddings,
    )

    assert docs, f"no hits for query: {case['query']}"
    pages = {doc.metadata["page"] for doc in docs}
    assert int(case["expected_page"]) in pages
    assert any(
        str(case["keyword_in_chunk"]).lower() in doc.page_content.lower()
        for doc in docs
    )

"""Tests for retrieval/citation_builder.py."""

from models.schemas import ChunkMetadata, RetrievedChunk
from retrieval.citation_builder import build_citations, format_citation_label


def _chunk(chunk_id: str, page: int, text: str, score: float) -> RetrievedChunk:
    meta = ChunkMetadata(
        doc_id="doc-1",
        source_file="test.pdf",
        source_path="uploads/test.pdf",
        page=page,
        chunk_index=0,
        chunk_id=chunk_id,
    )
    return RetrievedChunk(chunk_id=chunk_id, text=text, metadata=meta, score=score)


def test_build_citations_dedupes_by_chunk():
    chunks = [
        _chunk("c1", 1, "alpha content", 0.9),
        _chunk("c1", 1, "alpha content duplicate", 0.5),
        _chunk("c2", 2, "beta content", 0.8),
    ]
    citations = build_citations(chunks, max_citations=10)
    assert len(citations) == 2
    assert citations[0].chunk_id == "c1"
    assert citations[0].score == 0.9


def test_format_citation_label():
    from models.schemas import Citation

    label = format_citation_label(
        Citation(
            source_file="合同.pdf",
            page=12,
            chunk_id="c1",
            doc_id="doc-1",
        )
    )
    assert label == "合同.pdf 第12页"

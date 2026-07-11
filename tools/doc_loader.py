"""
Document loader tool — fetch full text for a document by doc_id.

Planned for full-document context when needed.
Not yet implemented; retrieval goes through RAGService today.
"""

from __future__ import annotations

from models.schemas import RetrievedChunk


def load_document_chunks(
    doc_id: str,
    *,
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[RetrievedChunk]:
    """Load all chunks for a document, optionally filtered by page range."""
    raise NotImplementedError

"""PDF ingestion pipeline: parse → chunk → embed → index."""

from ingestion.chunker import build_chunk_metadata, chunk_documents
from ingestion.embedder import embed_documents, embed_query
from ingestion.pdf_parser import parse_pdf
from ingestion.utils import compute_file_hash, ensure_unique_upload_path, generate_doc_id

__all__ = [
    "build_chunk_metadata",
    "chunk_documents",
    "compute_file_hash",
    "embed_documents",
    "embed_query",
    "ensure_unique_upload_path",
    "generate_doc_id",
    "parse_pdf",
]


def __getattr__(name: str):
    """Lazy-load indexer to avoid circular import with retrieval.chroma_store."""
    if name == "ingest_pdf":
        from ingestion.indexer import ingest_pdf

        return ingest_pdf
    if name == "ingest_uploaded_bytes":
        from ingestion.indexer import ingest_uploaded_bytes

        return ingest_uploaded_bytes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""
Indexer — orchestrate the full PDF → Chroma ingestion pipeline.

Flow:
  upload/save → parse_pdf → chunk_documents → embed_documents → ChromaStore.add_chunks
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from config.settings import Settings, get_settings
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_documents
from ingestion.pdf_parser import parse_pdf
from ingestion.utils import compute_file_hash, ensure_unique_upload_path, generate_doc_id
from models.doc_registry import DocRegistry
from models.schemas import DocRecord, DocStatus, IngestResult
from retrieval.chroma_store import ChromaStore

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


def ingest_pdf(
    file_path: Path,
    *,
    force: bool = False,
    on_progress: Callable[[str, float], None] | None = None,
    settings: Settings | None = None,
    store: ChromaStore | None = None,
    registry: DocRegistry | None = None,
    embeddings: Embeddings | None = None,
) -> IngestResult:
    """
    Ingest one PDF end-to-end into the local knowledge base.

    Stages:
      1. **save** — copy file into ``data/uploads/`` (collision-safe name)
      2. **register** — create ``doc_registry`` row (status=pending)
      3. **parse** — PyMuPDF per-page text extraction
      4. **chunk** — RecursiveCharacterTextSplitter per page
      5. **embed** — batch embedding vectors
      6. **index** — upsert into Chroma with metadata (filename + page)
      7. **finalize** — update registry (status=indexed)

    Args:
        file_path: Path to the PDF (may be a temp upload path).
        force: Re-index even when an identical ``file_hash`` already exists.
        on_progress: ``callback(stage_name, progress_0_to_1)`` for UI/CLI.
        settings: Override global settings (tests).
        store: Inject ChromaStore (tests).
        registry: Inject DocRegistry (tests).
        embeddings: Inject Embeddings model (tests).

    Returns:
        ``IngestResult`` with doc_id, counts, and final status.
    """
    cfg = settings or get_settings()
    cfg.ensure_dirs()

    source_path = Path(file_path)
    if not source_path.is_file():
        return IngestResult(
            doc_id="",
            source_file=source_path.name,
            page_count=0,
            chunk_count=0,
            status=DocStatus.FAILED,
            error_message=f"File not found: {source_path}",
        )

    source_file = source_path.name
    file_hash = compute_file_hash(source_path)
    doc_registry = registry or DocRegistry(cfg.registry_path)
    chroma = store or ChromaStore(cfg)

    existing = doc_registry.find_by_hash(file_hash)
    if existing and existing.status == DocStatus.INDEXED and not force:
        return IngestResult(
            doc_id=existing.doc_id,
            source_file=existing.source_file,
            page_count=existing.page_count,
            chunk_count=existing.chunk_count,
            status=DocStatus.INDEXED,
            error_message="already indexed (use force=True to re-ingest)",
        )

    doc_id = existing.doc_id if existing and force else generate_doc_id()
    ingest_version = (existing.ingest_version + 1) if existing and force else 1

    dest_path = ensure_unique_upload_path(cfg.uploads_dir, source_file)
    if source_path.resolve() != dest_path.resolve():
        shutil.copy2(source_path, dest_path)

    try:
        relative_path = str(dest_path.relative_to(cfg.data_dir.parent))
    except ValueError:
        relative_path = str(dest_path)

    if existing and force:
        chroma.connect()
        chroma.delete_by_doc_id(existing.doc_id)
        doc_registry.update_status(
            doc_id,
            DocStatus.PENDING,
            ingest_version=ingest_version,
            error_message="",
        )
    else:
        doc_registry.create(
            DocRecord(
                doc_id=doc_id,
                source_file=source_file,
                source_path=relative_path,
                file_hash=file_hash,
                status=DocStatus.PENDING,
                ingest_version=ingest_version,
            )
        )

    try:
        _report(on_progress, "parse", 0.1)
        page_docs = parse_pdf(dest_path, doc_id=doc_id, source_file=source_file)
        for doc in page_docs:
            doc.metadata["file_hash"] = file_hash
            doc.metadata["ingest_version"] = ingest_version

        page_count = int(page_docs[0].metadata.get("page_count", len(page_docs)))

        _report(on_progress, "chunk", 0.35)
        chunks = chunk_documents(page_docs, doc_id=doc_id, settings=cfg)
        if not chunks:
            raise ValueError("Chunking produced zero segments")

        _report(on_progress, "embed", 0.55)

        def embed_progress(done: int, total: int) -> None:
            if on_progress:
                on_progress("embed", 0.55 + 0.25 * (done / total))

        vectors = embed_documents(
            chunks,
            on_progress=embed_progress,
            embeddings=embeddings,
        )

        _report(on_progress, "index", 0.85)
        chroma.connect()
        ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [dict(chunk.metadata) for chunk in chunks]
        chroma.add_chunks(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=vectors,
        )

        _report(on_progress, "done", 1.0)
        doc_registry.update_status(
            doc_id,
            DocStatus.INDEXED,
            chunk_count=len(chunks),
            page_count=page_count,
            ingest_version=ingest_version,
        )

        return IngestResult(
            doc_id=doc_id,
            source_file=source_file,
            page_count=page_count,
            chunk_count=len(chunks),
            status=DocStatus.INDEXED,
        )

    except Exception as exc:
        doc_registry.update_status(
            doc_id,
            DocStatus.FAILED,
            error_message=str(exc),
        )
        return IngestResult(
            doc_id=doc_id,
            source_file=source_file,
            page_count=0,
            chunk_count=0,
            status=DocStatus.FAILED,
            error_message=str(exc),
        )


def ingest_uploaded_bytes(
    filename: str,
    data: bytes,
    *,
    force: bool = False,
    on_progress: Callable[[str, float], None] | None = None,
    settings: Settings | None = None,
    store: ChromaStore | None = None,
    registry: DocRegistry | None = None,
    embeddings: Embeddings | None = None,
) -> IngestResult:
    """
    Convenience entry for Streamlit uploads — write bytes to a temp file then ingest.

    Args:
        filename: Original upload filename (e.g. ``report.pdf``).
        data: Raw PDF bytes from ``UploadedFile.getvalue()``.
    """
    cfg = settings or get_settings()
    cfg.ensure_dirs()

    temp_path = cfg.uploads_dir / f"_tmp_{filename}"
    temp_path.write_bytes(data)
    try:
        return ingest_pdf(
            temp_path,
            force=force,
            on_progress=on_progress,
            settings=cfg,
            store=store,
            registry=registry,
            embeddings=embeddings,
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _report(
    callback: Callable[[str, float], None] | None,
    stage: str,
    progress: float,
) -> None:
    if callback:
        callback(stage, progress)

"""
Text chunker — split page Documents with RecursiveCharacterTextSplitter.

Pipeline position: after PDF parsing, before embedding.
Default strategy splits **within each page** so every chunk keeps a stable page number.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import PageBreakStrategy, Settings, get_settings
from ingestion.utils import doc_id_short
from models.schemas import ChunkMetadata


def chunk_documents(
    documents: list[Document],
    *,
    doc_id: str,
    settings: Settings | None = None,
) -> list[Document]:
    """
    Split page-level Documents into smaller chunks for vector indexing.

    Each returned Document contains:
      - ``page_content``: chunk text
      - ``metadata.source_file``: original filename (required by product spec)
      - ``metadata.page``: 1-based page number (required by product spec)
      - plus full Chroma fields (chunk_id, doc_id, char offsets, …)

    Args:
        documents: Output of ``parse_pdf`` (one item per page).
        doc_id: Document UUID shared by all chunks.
        settings: Chunk size / overlap; defaults to global settings.

    Returns:
        Flat list of chunk Documents ready for embedding.
    """
    settings = settings or get_settings()
    splitter = _build_splitter(settings)
    chunked: list[Document] = []
    global_index = 0

    for page_doc in documents:
        meta = page_doc.metadata
        source_file = str(meta["source_file"])
        source_path = str(meta.get("source_path", ""))
        page = int(meta["page"])
        page_count = meta.get("page_count")
        file_hash = str(meta.get("file_hash", ""))
        ingest_version = int(meta.get("ingest_version", 1))

        page_text = page_doc.page_content
        segments = (
            _split_intra_page(splitter, page_text)
            if settings.page_break_strategy == PageBreakStrategy.INTRA_PAGE
            else splitter.split_text(page_text)
        )

        char_cursor = 0
        for segment in segments:
            if not segment.strip():
                continue

            char_start = page_text.find(segment, char_cursor)
            if char_start < 0:
                char_start = char_cursor
            char_end = char_start + len(segment)
            char_cursor = max(char_cursor, char_end)

            chunk_meta = build_chunk_metadata(
                doc_id,
                source_file,
                source_path,
                page,
                global_index,
                page_count=int(page_count) if page_count is not None else None,
                char_start=char_start,
                char_end=char_end,
                file_hash=file_hash,
                ingest_version=ingest_version,
                settings=settings,
            )
            chunked.append(_document_from_metadata(chunk_meta, segment))
            global_index += 1

    return chunked


def build_chunk_metadata(
    doc_id: str,
    source_file: str,
    source_path: str,
    page: int,
    chunk_index: int,
    *,
    page_count: int | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
    file_hash: str = "",
    ingest_version: int = 1,
    section_hint: str = "",
    settings: Settings | None = None,
) -> ChunkMetadata:
    """
    Build a validated ``ChunkMetadata`` record with a deterministic ``chunk_id``.

    ``chunk_id`` format: ``{doc_short}_p{page:04d}_c{chunk_index:04d}``
    """
    settings = settings or get_settings()
    short = doc_id_short(doc_id)
    chunk_id = settings.chunk_id_template.format(
        doc_short=short,
        page=page,
        chunk_index=chunk_index,
    )
    token_estimate = None
    if char_start is not None and char_end is not None:
        token_estimate = max(1, (char_end - char_start) // 2)

    return ChunkMetadata(
        doc_id=doc_id,
        source_file=source_file,
        source_path=source_path,
        page=page,
        chunk_index=chunk_index,
        chunk_id=chunk_id,
        ingest_version=ingest_version,
        page_count=page_count,
        char_start=char_start,
        char_end=char_end,
        token_estimate=token_estimate,
        section_hint=section_hint,
        file_hash=file_hash,
    )


def _build_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    """Create a RecursiveCharacterTextSplitter from global chunk settings."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )


def _split_intra_page(
    splitter: RecursiveCharacterTextSplitter,
    page_text: str,
) -> list[str]:
    """Split text within a single page without merging adjacent pages."""
    return splitter.split_text(page_text)


def _document_from_metadata(meta: ChunkMetadata, text: str) -> Document:
    """
    Wrap chunk text as a LangChain Document.

    Ensures ``source_file`` and ``page`` are always present in metadata.
    """
    chroma_meta = meta.to_chroma_dict()
    chroma_meta["source_file"] = meta.source_file
    chroma_meta["page"] = meta.page
    return Document(page_content=text, metadata=chroma_meta)

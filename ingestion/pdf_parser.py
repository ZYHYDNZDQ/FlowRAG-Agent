"""
PDF parser — extract text per page using PyMuPDF.

Pipeline position: first step after file upload.
Each output Document carries filename (source_file) and 1-based page number.
"""

from __future__ import annotations

from pathlib import Path

import fitz
from langchain_core.documents import Document


def parse_pdf(
    file_path: Path,
    *,
    doc_id: str,
    source_file: str,
) -> list[Document]:
    """
    Parse a PDF into one LangChain Document per non-empty page.

    Uses PyMuPDF (fitz) to extract plain text. Page metadata is attached so
    downstream chunking and retrieval can cite ``source_file`` and ``page``.

    Args:
        file_path: Absolute or relative path to the PDF on disk.
        doc_id: UUID assigned to this document in the registry / Chroma.
        source_file: Original filename shown to users (e.g. ``合同.pdf``).

    Returns:
        List of page-level Documents. ``metadata`` always includes:
        ``source_file``, ``page`` (1-based), ``doc_id``, ``source_path``,
        ``page_count``.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError: If the PDF has no extractable text (e.g. scanned image PDF).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    source_path = str(path.resolve())
    page_docs: list[Document] = []

    with fitz.open(path) as pdf:
        page_count = pdf.page_count
        for page_index in range(page_count):
            page = pdf.load_page(page_index)
            text = page.get_text().strip()
            if not text:
                continue

            page_number = page_index + 1
            page_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "doc_id": doc_id,
                        "source_file": source_file,
                        "source_path": source_path,
                        "page": page_number,
                        "page_count": page_count,
                    },
                )
            )

    if not page_docs:
        raise ValueError(
            f"No extractable text in '{source_file}'. "
            "Scanned PDFs without OCR are not supported."
        )

    return page_docs

"""Tests for ingestion/pdf_parser.py."""

from pathlib import Path

import pytest

from ingestion.pdf_parser import parse_pdf
from tests.conftest import make_sample_pdf


def test_parse_pdf_returns_page_metadata(tmp_path: Path):
    pdf_path = make_sample_pdf(
        tmp_path / "contract.pdf",
        ["Page 1: payment terms", "Page 2: breach liability"],
    )

    docs = parse_pdf(pdf_path, doc_id="doc-1", source_file="contract.pdf")

    assert len(docs) == 2
    assert docs[0].metadata["source_file"] == "contract.pdf"
    assert docs[0].metadata["page"] == 1
    assert docs[1].metadata["page"] == 2
    assert "payment" in docs[0].page_content


def test_parse_pdf_raises_on_empty_text(tmp_path: Path):
    # Blank page — no insert_text
    import fitz

    blank = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(blank)
    doc.close()

    with pytest.raises(ValueError, match="No extractable text"):
        parse_pdf(blank, doc_id="doc-2", source_file="blank.pdf")

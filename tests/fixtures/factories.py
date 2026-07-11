"""Test factories — build artifacts without touching business logic."""

from __future__ import annotations

from pathlib import Path

import fitz


def make_sample_pdf(path: Path, pages: list[str]) -> Path:
    """Create a minimal PDF with known text per page."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path

"""
Sample document content for RAG tests.

ASCII page text for reliable PyMuPDF extraction on all platforms.
"""

from __future__ import annotations

SAMPLE_CONTRACT_FILENAME = "contract.pdf"

SAMPLE_CONTRACT_PAGES: list[str] = [
    "Chapter 1 Payment: Party A shall complete payment within 30 calendar days.",
    "Chapter 2 Liability: Late payment incurs a daily penalty of 0.05 percent.",
]

SAMPLE_QUERIES: list[dict[str, str | int]] = [
    {
        "query": "payment within 30 days",
        "expected_page": 1,
        "keyword_in_chunk": "payment",
    },
    {
        "query": "late payment penalty",
        "expected_page": 2,
        "keyword_in_chunk": "penalty",
    },
]

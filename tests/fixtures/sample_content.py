"""
Sample document content for RAG pipeline tests and examples.

Use ASCII text in page bodies for reliable PyMuPDF extraction across platforms.
Chinese queries are still valid for retrieval smoke tests.
"""

from __future__ import annotations

# A minimal two-page "contract" used across example tests.
SAMPLE_CONTRACT_FILENAME = "contract.pdf"

SAMPLE_CONTRACT_PAGES: list[str] = [
    "Chapter 1 Payment: Party A shall complete payment within 30 calendar days.",
    "Chapter 2 Liability: Late payment incurs a daily penalty of 0.05 percent.",
]

# Example user queries mapped to expected source pages (by content keyword).
SAMPLE_QUERIES: list[dict[str, str | int]] = [
    {
        "query": "payment deadline days",
        "expected_page": 1,
        "keyword_in_chunk": "payment",
    },
    {
        "query": "late payment penalty",
        "expected_page": 2,
        "keyword_in_chunk": "penalty",
    },
]

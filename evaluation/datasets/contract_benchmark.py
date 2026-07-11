"""
Contract benchmark — evaluation data only.

ASCII page text for reliable cross-platform PDF extraction.
"""

from __future__ import annotations

from evaluation.datasets.schema import (
    EvalBenchmark,
    EvalDocument,
    GenerationCase,
    RetrievalCase,
)
from models.schemas import IntentType

_CONTRACT_DOCUMENT = EvalDocument(
    filename="eval_contract.pdf",
    pages=[
        "Chapter 1 Payment: Party A shall complete payment within 30 calendar days.",
        "Chapter 2 Liability: Late payment incurs a daily penalty of 0.05 percent.",
    ],
)

_RETRIEVAL_CASES: list[RetrievalCase] = [
    RetrievalCase(
        id="ret-payment-deadline",
        query="payment within 30 days",
        expected_page=1,
        keyword_in_chunk="payment",
    ),
    RetrievalCase(
        id="ret-late-penalty",
        query="late payment penalty",
        expected_page=2,
        keyword_in_chunk="penalty",
    ),
]

_GENERATION_CASES: list[GenerationCase] = [
    GenerationCase(
        id="gen-payment-deadline",
        query="What is the payment deadline?",
        reference_answer="Party A shall complete payment within 30 calendar days.",
        criteria="Must mention 30 days (or 30 calendar days) payment period.",
        intent=IntentType.QA,
    ),
    GenerationCase(
        id="gen-late-penalty",
        query="What happens if payment is late?",
        reference_answer="Late payment incurs a daily penalty of 0.05 percent.",
        criteria="Must mention late payment penalty or 0.05 percent daily rate.",
        intent=IntentType.QA,
    ),
    GenerationCase(
        id="gen-summarize",
        query="请总结文档要点",
        reference_answer="Contract covers payment within 30 days and late payment penalties.",
        criteria="Summary should cover both payment terms and penalty clauses.",
        intent=IntentType.SUMMARIZE,
    ),
]

CONTRACT_BENCHMARK = EvalBenchmark(
    name="contract",
    description="Two-page contract PDF for retrieval and generation evaluation.",
    document=_CONTRACT_DOCUMENT,
    retrieval_cases=_RETRIEVAL_CASES,
    generation_cases=_GENERATION_CASES,
)

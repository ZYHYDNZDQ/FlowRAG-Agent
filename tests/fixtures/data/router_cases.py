"""Router intent and scope test cases."""

from __future__ import annotations

from models.schemas import IntentType, RetrievalMode

ROUTER_TEST_CASES: list[dict[str, object]] = [
    {"query": "请总结这份合同的核心条款", "intent": IntentType.SUMMARIZE},
    {"query": "对比甲乙双方的风险差异", "intent": IntentType.ANALYZE},
    {"query": "付款周期是多少天？", "intent": IntentType.QA},
    {"query": "payment deadline", "intent": IntentType.QA},
    {"query": "请概括文档要点", "intent": IntentType.SUMMARIZE},
    {"query": "评估合同风险", "intent": IntentType.ANALYZE},
]

ROUTER_SCOPE_CASES: list[dict[str, object]] = [
    {
        "doc_ids": ["doc-1"],
        "expected_mode": RetrievalMode.SINGLE,
        "expected_ids": ["doc-1"],
    },
    {
        "doc_ids": ["doc-1", "doc-2"],
        "expected_mode": RetrievalMode.SELECTED,
        "expected_ids": ["doc-1", "doc-2"],
    },
    {
        "doc_ids": [],
        "expected_mode": RetrievalMode.ALL,
        "expected_ids": [],
    },
]

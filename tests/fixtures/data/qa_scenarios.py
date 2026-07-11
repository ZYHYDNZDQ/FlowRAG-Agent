"""QA / Skill / E2E scenario definitions."""

from __future__ import annotations

from models.schemas import IntentType

QA_SCENARIOS: list[dict[str, object]] = [
    {
        "query": "payment within 30 days",
        "intent": IntentType.QA,
        "expect_citations": True,
        "expect_chunks": True,
    },
    {
        "query": "请总结文档",
        "intent": IntentType.SUMMARIZE,
        "expect_citations": True,
        "expect_chunks": True,
    },
    {
        "query": "分析合同风险",
        "intent": IntentType.ANALYZE,
        "expect_citations": True,
        "expect_chunks": True,
    },
]

EMPTY_KB_QUERY = "完全不相关的随机问题xyz"

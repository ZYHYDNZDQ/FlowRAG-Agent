"""Deterministic fake LLM outputs for offline evaluation (--fake)."""

from __future__ import annotations

FAKE_AGENT_RESPONSES: list[str] = [
    "Party A shall complete payment within 30 calendar days according to the contract.",
    "Late payment incurs a daily penalty of 0.05 percent.",
    "The document covers a 30-day payment term and late payment penalty clauses.",
]

FAKE_JUDGE_RESPONSES: list[str] = [
    '{"score": 5, "pass": true, "reasoning": "Correctly states the 30-day payment deadline."}',
    '{"score": 5, "pass": true, "reasoning": "Correctly describes the late payment penalty."}',
    '{"score": 4, "pass": true, "reasoning": "Summary covers payment and penalty themes."}',
]

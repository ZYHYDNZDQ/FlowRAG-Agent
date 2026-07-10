"""Tests for agent/router.py."""

from agent.router import build_scope_from_selection, route
from models.schemas import IntentType, RetrievalMode


def test_rule_route_summarize_keywords():
    result = route("请总结这份合同的核心条款")
    assert result.intent == IntentType.SUMMARIZE
    assert result.scope.mode == RetrievalMode.ALL


def test_rule_route_analyze_keywords():
    result = route("对比甲乙双方的风险差异")
    assert result.intent == IntentType.ANALYZE


def test_rule_route_defaults_to_qa():
    result = route("付款周期是多少天？")
    assert result.intent == IntentType.QA


def test_build_scope_from_single_selection():
    scope = build_scope_from_selection(["doc-1"])
    assert scope.mode == RetrievalMode.SINGLE
    assert scope.doc_ids == ["doc-1"]


def test_build_scope_from_multi_selection():
    scope = build_scope_from_selection(["doc-1", "doc-2"])
    assert scope.mode == RetrievalMode.SELECTED
    assert scope.doc_ids == ["doc-1", "doc-2"]

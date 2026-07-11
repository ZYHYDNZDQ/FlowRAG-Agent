"""Router intent and scope tests."""

from __future__ import annotations

import pytest

from agent.router import (
    RouterSkillDecision,
    _try_llm_route,
    build_scope_from_selection,
    route,
)
from models.schemas import IntentType
from tests.fixtures.data.router_cases import ROUTER_SCOPE_CASES, ROUTER_TEST_CASES

pytestmark = pytest.mark.router


@pytest.mark.parametrize("case", ROUTER_TEST_CASES, ids=lambda c: str(c["query"])[:20])
def test_route_intent_rule_fallback_without_llm(case):
    """Without LLM, route uses rule-based fallback (backward compatible)."""
    result = route(str(case["query"]))
    assert result.intent == case["intent"]
    assert result.reasoning.startswith("rule fallback:")


@pytest.mark.parametrize("case", ROUTER_SCOPE_CASES, ids=lambda c: str(c["doc_ids"]))
def test_build_scope_from_selection(case):
    scope = build_scope_from_selection(list(case["doc_ids"]))  # type: ignore[arg-type]
    assert scope.mode == case["expected_mode"]
    assert scope.doc_ids == case["expected_ids"]


def test_llm_router_normal_classification(monkeypatch):
    """LLM returns valid skill + confidence → use LLM result."""

    def _mock_llm_route(query: str, llm: object, scope: object) -> object:
        from models.schemas import RouterResult

        return RouterResult(
            intent=IntentType.SUMMARIZE,
            scope=scope,  # type: ignore[arg-type]
            confidence=0.92,
            reasoning="llm router: SUMMARY (confidence=0.92)",
        )

    monkeypatch.setattr("agent.router._try_llm_route", _mock_llm_route)
    result = route("随便什么问题", llm=object())
    assert result.intent == IntentType.SUMMARIZE
    assert result.confidence == 0.92
    assert result.reasoning.startswith("llm router:")


def test_llm_router_failure_falls_back_to_rules(monkeypatch):
    """LLM invocation failure → rule-based router."""

    def _fail_llm_route(query: str, llm: object, scope: object) -> None:
        return None

    monkeypatch.setattr("agent.router._try_llm_route", _fail_llm_route)
    result = route("请总结文档要点", llm=object())
    assert result.intent == IntentType.SUMMARIZE
    assert "rule fallback" in result.reasoning


def test_llm_router_invalid_output_falls_back_to_rules(monkeypatch):
    """Invalid skill or low confidence → rule-based router."""

    class BadDecision:
        skill = "INVALID"
        confidence = 0.95

    def _bad_llm_route(query: str, llm: object, scope: object) -> object:
        from agent.router import _llm_decision_to_result

        return _llm_decision_to_result(BadDecision(), scope)  # type: ignore[arg-type]

    monkeypatch.setattr("agent.router._try_llm_route", _bad_llm_route)
    result = route("评估合同风险", llm=object())
    assert result.intent == IntentType.ANALYZE
    assert "rule fallback" in result.reasoning


def test_llm_router_low_confidence_falls_back(monkeypatch):
    """Low confidence from LLM → rule-based router."""

    def _low_conf_route(query: str, llm: object, scope: object) -> object:
        from agent.router import _llm_decision_to_result

        decision = RouterSkillDecision(skill="QA", confidence=0.2)
        return _llm_decision_to_result(decision, scope)  # type: ignore[arg-type]

    monkeypatch.setattr("agent.router._try_llm_route", _low_conf_route)
    result = route("请总结文档", llm=object())
    assert result.intent == IntentType.SUMMARIZE
    assert "rule fallback" in result.reasoning


def test_try_llm_route_parses_structured_decision():
    """_llm_decision_to_result maps QA/SUMMARY/ANALYSIS to IntentType."""

    from agent.router import _llm_decision_to_result
    from models.schemas import RetrievalScope

    scope = RetrievalScope()
    for skill, intent in [
        ("QA", IntentType.QA),
        ("SUMMARY", IntentType.SUMMARIZE),
        ("ANALYSIS", IntentType.ANALYZE),
    ]:
        decision = RouterSkillDecision(skill=skill, confidence=0.88)  # type: ignore[arg-type]
        result = _llm_decision_to_result(decision, scope)
        assert result is not None
        assert result.intent == intent
        assert skill in result.reasoning

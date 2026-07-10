"""
Intent router — classify user query into workflow.

Two-layer strategy:
  1. Rule-based keyword matching (fast path)
  2. LLM structured output → RouterResult (fallback, Day 3+)
"""

from __future__ import annotations

from models.schemas import IntentType, RetrievalMode, RetrievalScope, RouterResult

# Rule-based keyword hints (extend as needed)
_SUMMARIZE_KEYWORDS = ("总结", "摘要", "概述", "归纳", "概括", "summarize", "summary")
_ANALYZE_KEYWORDS = ("分析", "对比", "评估", "风险", "比较", "analyze", "区别", "差异")


def route(
    query: str,
    *,
    selected_doc_ids: list[str] | None = None,
) -> RouterResult:
    """
    Determine intent and retrieval scope for a user query.

    Args:
        query: User question or task description.
        selected_doc_ids: Documents selected in UI sidebar.

    Returns:
        RouterResult with intent, scope, and optional reasoning.
    """
    scope = build_scope_from_selection(selected_doc_ids)
    return _rule_based_route(query, scope)


def build_scope_from_selection(selected_doc_ids: list[str] | None) -> RetrievalScope:
    """Map UI document selection to RetrievalScope."""
    ids = selected_doc_ids or []
    if not ids:
        return RetrievalScope(mode=RetrievalMode.ALL)
    if len(ids) == 1:
        return RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=ids)
    return RetrievalScope(mode=RetrievalMode.SELECTED, doc_ids=ids)


def _rule_based_route(query: str, scope: RetrievalScope) -> RouterResult:
    """Fast keyword-based routing."""
    normalized = query.strip().lower()
    if not normalized:
        return RouterResult(
            intent=IntentType.QA,
            scope=scope,
            confidence=1.0,
            reasoning="empty query defaults to qa",
        )

    for keyword in _SUMMARIZE_KEYWORDS:
        if keyword in normalized:
            return RouterResult(
                intent=IntentType.SUMMARIZE,
                scope=scope,
                confidence=0.9,
                reasoning=f"keyword match: {keyword}",
            )

    for keyword in _ANALYZE_KEYWORDS:
        if keyword in normalized:
            return RouterResult(
                intent=IntentType.ANALYZE,
                scope=scope,
                confidence=0.9,
                reasoning=f"keyword match: {keyword}",
            )

    return RouterResult(
        intent=IntentType.QA,
        scope=scope,
        confidence=0.8,
        reasoning="default qa",
    )

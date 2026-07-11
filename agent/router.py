"""

Intent router — LLM structured output with rule-based fallback.



Flow:

  1. LLM Router (skill + confidence only — no Tool selection)

  2. On failure / low confidence / invalid skill → Rule Router

"""



from __future__ import annotations



import logging

from typing import TYPE_CHECKING, Literal



from pydantic import BaseModel, Field



from config.prompts import ROUTER_SYSTEM_PROMPT

from langchain_core.messages import HumanMessage, SystemMessage

from models.schemas import IntentType, RetrievalMode, RetrievalScope, RouterResult



if TYPE_CHECKING:

    from langchain_core.language_models import BaseChatModel



logger = logging.getLogger(__name__)



# Rule-based keyword hints (extend as needed)

_SUMMARIZE_KEYWORDS = ("总结", "摘要", "概述", "归纳", "概括", "summarize", "summary")

_ANALYZE_KEYWORDS = ("分析", "对比", "评估", "风险", "比较", "analyze", "区别", "差异")



LLM_CONFIDENCE_THRESHOLD = 0.5



_SKILL_TO_INTENT: dict[str, IntentType] = {

    "QA": IntentType.QA,

    "SUMMARY": IntentType.SUMMARIZE,

    "ANALYSIS": IntentType.ANALYZE,

}





class RouterSkillDecision(BaseModel):

    """LLM structured router output — skill selection only."""



    skill: Literal["QA", "SUMMARY", "ANALYSIS"]

    confidence: float = Field(ge=0.0, le=1.0)





def route(

    query: str,

    *,

    selected_doc_ids: list[str] | None = None,

    llm: BaseChatModel | None = None,

) -> RouterResult:

    """

    Determine intent (Skill) and retrieval scope for a user query.



    Scope comes from UI document selection only; Router does not choose Tools.

    """

    scope = build_scope_from_selection(selected_doc_ids)



    if llm is not None:

        llm_result = _try_llm_route(query, llm, scope)

        if llm_result is not None:

            return llm_result



    return _rule_based_route(query, scope)





def build_scope_from_selection(selected_doc_ids: list[str] | None) -> RetrievalScope:

    """Map UI document selection to RetrievalScope."""

    ids = selected_doc_ids or []

    if not ids:

        return RetrievalScope(mode=RetrievalMode.ALL)

    if len(ids) == 1:

        return RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=ids)

    return RetrievalScope(mode=RetrievalMode.SELECTED, doc_ids=ids)





def _try_llm_route(

    query: str,

    llm: BaseChatModel,

    scope: RetrievalScope,

) -> RouterResult | None:

    """Invoke LLM structured router; return None to trigger rule fallback."""

    normalized = query.strip()

    if not normalized:

        return None



    try:

        structured_llm = llm.with_structured_output(RouterSkillDecision)

        decision = structured_llm.invoke(

            [

                SystemMessage(content=ROUTER_SYSTEM_PROMPT),

                HumanMessage(content=f"用户问题：{normalized}"),

            ]

        )

    except Exception as exc:

        logger.warning("LLM router failed, using rule fallback: %s", exc)

        return None



    return _llm_decision_to_result(decision, scope)





def _llm_decision_to_result(

    decision: RouterSkillDecision | object,

    scope: RetrievalScope,

) -> RouterResult | None:

    """Map LLM decision to RouterResult; None triggers rule fallback."""

    skill = getattr(decision, "skill", None)

    confidence = getattr(decision, "confidence", None)



    if not isinstance(skill, str) or skill not in _SKILL_TO_INTENT:

        logger.warning("LLM router returned invalid skill: %s", skill)

        return None



    try:

        confidence_value = float(confidence)  # type: ignore[arg-type]

    except (TypeError, ValueError):

        logger.warning("LLM router returned invalid confidence: %s", confidence)

        return None



    if confidence_value < LLM_CONFIDENCE_THRESHOLD:

        logger.info(

            "LLM router low confidence %.2f < %.2f, using rule fallback",

            confidence_value,

            LLM_CONFIDENCE_THRESHOLD,

        )

        return None



    intent = _SKILL_TO_INTENT[skill]

    return RouterResult(

        intent=intent,

        scope=scope,

        confidence=confidence_value,

        reasoning=f"llm router: {skill} (confidence={confidence_value:.2f})",

    )





def _rule_based_route(query: str, scope: RetrievalScope) -> RouterResult:

    """Fast keyword-based routing fallback."""

    normalized = query.strip().lower()

    if not normalized:

        return RouterResult(

            intent=IntentType.QA,

            scope=scope,

            confidence=1.0,

            reasoning="rule fallback: empty query defaults to qa",

        )



    for keyword in _SUMMARIZE_KEYWORDS:

        if keyword in normalized:

            return RouterResult(

                intent=IntentType.SUMMARIZE,

                scope=scope,

                confidence=0.9,

                reasoning=f"rule fallback: keyword match: {keyword}",

            )



    for keyword in _ANALYZE_KEYWORDS:

        if keyword in normalized:

            return RouterResult(

                intent=IntentType.ANALYZE,

                scope=scope,

                confidence=0.9,

                reasoning=f"rule fallback: keyword match: {keyword}",

            )



    return RouterResult(

        intent=IntentType.QA,

        scope=scope,

        confidence=0.8,

        reasoning="rule fallback: default qa",

    )





def intent_label(intent: IntentType) -> str:

    """Human-readable intent label for trace / UI."""

    return {

        IntentType.QA: "qa",

        IntentType.SUMMARIZE: "summary",

        IntentType.ANALYZE: "analysis",

    }[intent]



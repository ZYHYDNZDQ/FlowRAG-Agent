"""Generation evaluation via LLM-as-judge."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agent.runtime import AgentRuntime, ExecuteRequest
from config.settings import Settings
from evaluation.datasets.schema import GenerationCase
from langchain_core.embeddings import Embeddings
from models.schemas import RetrievalScope
from retrieval.chroma_store import ChromaStore

JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator for RAG-based QA systems.

Given a user question, reference answer, evaluation criteria, and the model answer,
score the model answer from 1 to 5:
  5 = fully correct, grounded, and complete
  4 = mostly correct with minor omissions
  3 = partially correct
  2 = mostly wrong or unsupported
  1 = completely wrong or irrelevant

Respond with JSON only:
{"score": <int 1-5>, "pass": <bool>, "reasoning": "<short explanation>"}

Set pass=true when score >= 4.
"""

JUDGE_USER_TEMPLATE = """## Question
{query}

## Reference answer
{reference_answer}

## Evaluation criteria
{criteria}

## Model answer
{model_answer}

Return JSON only."""


@dataclass(frozen=True)
class JudgeVerdict:
    score: int
    passed: bool
    reasoning: str
    raw_response: str


@dataclass(frozen=True)
class GenerationCaseResult:
    case_id: str
    query: str
    model_answer: str
    verdict: JudgeVerdict
    citation_count: int


@dataclass(frozen=True)
class GenerationEvalSummary:
    total: int
    passed: int
    pass_rate: float
    average_score: float
    case_results: list[GenerationCaseResult]


def _parse_judge_response(text: str) -> JudgeVerdict:
    """Parse judge JSON; fall back to heuristic extraction."""
    raw = text.strip()
    payload: dict | None = None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group())
            except json.JSONDecodeError:
                payload = None

    if payload is not None:
        score = int(payload.get("score", 1))
        score = max(1, min(5, score))
        passed = bool(payload.get("pass", score >= 4))
        reasoning = str(payload.get("reasoning", "")).strip() or "No reasoning provided."
        return JudgeVerdict(score=score, passed=passed, reasoning=reasoning, raw_response=raw)

    lowered = raw.lower()
    if "pass" in lowered and "true" in lowered:
        return JudgeVerdict(score=4, passed=True, reasoning=raw, raw_response=raw)
    return JudgeVerdict(score=2, passed=False, reasoning=raw, raw_response=raw)


def judge_answer(
    *,
    judge_llm: BaseChatModel,
    query: str,
    reference_answer: str,
    criteria: str,
    model_answer: str,
) -> JudgeVerdict:
    """Score one model answer with LLM-as-judge."""
    user_prompt = JUDGE_USER_TEMPLATE.format(
        query=query,
        reference_answer=reference_answer,
        criteria=criteria,
        model_answer=model_answer,
    )
    response = judge_llm.invoke(
        [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )
    return _parse_judge_response(str(response.content))


def evaluate_generation(
    cases: list[GenerationCase],
    *,
    runtime: AgentRuntime,
    judge_llm: BaseChatModel,
    settings: Settings,
    store: ChromaStore,
    llm: BaseChatModel,
    embeddings: Embeddings,
    scope: RetrievalScope,
    doc_id: str,
) -> GenerationEvalSummary:
    """Run AgentRuntime on generation cases and score with LLM-as-judge."""
    results: list[GenerationCaseResult] = []

    for case in cases:
        exec_result = runtime.execute(
            ExecuteRequest(
                query=case.query,
                session_id=f"eval-{case.id}",
                selected_doc_ids=[doc_id],
                end_session=True,
            ),
            settings=settings,
            store=store,
            llm=llm,
            embeddings=embeddings,
        )
        answer = exec_result.answer
        verdict = judge_answer(
            judge_llm=judge_llm,
            query=case.query,
            reference_answer=case.reference_answer,
            criteria=case.criteria,
            model_answer=answer.answer,
        )
        results.append(
            GenerationCaseResult(
                case_id=case.id,
                query=case.query,
                model_answer=answer.answer,
                verdict=verdict,
                citation_count=len(answer.citations),
            )
        )

    passed = sum(1 for r in results if r.verdict.passed)
    total = len(results)
    avg_score = sum(r.verdict.score for r in results) / total if total else 0.0
    return GenerationEvalSummary(
        total=total,
        passed=passed,
        pass_rate=passed / total if total else 0.0,
        average_score=avg_score,
        case_results=results,
    )

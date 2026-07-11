"""Evaluation dataset schemas — separate from business models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from models.schemas import IntentType


class RetrievalCase(BaseModel):
    """One retrieval benchmark row."""

    id: str
    query: str
    expected_page: int = Field(ge=1)
    keyword_in_chunk: str | None = None


class GenerationCase(BaseModel):
    """One generation benchmark row for LLM-as-judge."""

    id: str
    query: str
    reference_answer: str
    criteria: str
    intent: IntentType = IntentType.QA


class EvalDocument(BaseModel):
    """Synthetic document used to build an isolated eval knowledge base."""

    filename: str
    pages: list[str]


class EvalBenchmark(BaseModel):
    """Full benchmark bundle."""

    name: str
    description: str
    document: EvalDocument
    retrieval_cases: list[RetrievalCase]
    generation_cases: list[GenerationCase]

"""
RAG retrieval and context formatting tools.

Delegates to RAGService — no duplicated retrieval logic.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from models.schemas import RetrievedChunk, RetrievalScope
from tools.base import BaseTool
from tools.context import ToolExecutionContext


class RagToolInput(BaseModel):
    """Input for vector retrieval."""

    query: str
    scope: RetrievalScope = Field(default_factory=RetrievalScope)
    top_k: int | None = None
    score_threshold: float | None = None


class RagFormatContextInput(BaseModel):
    """Input for formatting retrieved chunks into LLM context."""

    chunks: list[RetrievedChunk]


class RagTool(BaseTool):
    """Atomic tool: semantic retrieval over the knowledge base."""

    name = "rag.retrieve"
    description = "Retrieve relevant document chunks from the vector knowledge base."
    input_schema = RagToolInput

    def execute(
        self,
        ctx: ToolExecutionContext,
        input_data: BaseModel,
        **_: Any,
    ) -> list[RetrievedChunk]:
        payload = self.validate_input(input_data)
        assert isinstance(payload, RagToolInput)
        if ctx.vector_store is None:
            raise RuntimeError("rag.retrieve requires vector_store in ToolExecutionContext")
        return ctx.rag_service.query_chunks(
            ctx.vector_store,
            payload.query,
            payload.scope,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
            embeddings=ctx.embeddings,
            settings=ctx.settings,
        )


class RagFormatContextTool(BaseTool):
    """Atomic tool: format retrieval hits into a prompt context block."""

    name = "rag.format_context"
    description = "Format retrieved chunks into a single context string for the LLM."
    input_schema = RagFormatContextInput

    def execute(
        self,
        ctx: ToolExecutionContext,
        input_data: BaseModel,
        **_: Any,
    ) -> str:
        payload = self.validate_input(input_data)
        assert isinstance(payload, RagFormatContextInput)
        return ctx.rag_service.format_context(payload.chunks)

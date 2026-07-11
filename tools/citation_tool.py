"""Citation building tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from models.schemas import Citation, RetrievedChunk
from retrieval.citation_builder import build_citations
from tools.base import BaseTool
from tools.context import ToolExecutionContext


class CitationToolInput(BaseModel):
    chunks: list[RetrievedChunk]


class CitationTool(BaseTool):
    """Atomic tool: build citation list from retrieval hits."""

    name = "rag.build_citations"
    description = "Build deduplicated citation references from retrieved chunks."
    input_schema = CitationToolInput

    def execute(
        self,
        ctx: ToolExecutionContext,
        input_data: BaseModel,
        **_: Any,
    ) -> list[Citation]:
        payload = self.validate_input(input_data)
        assert isinstance(payload, CitationToolInput)
        return build_citations(payload.chunks)

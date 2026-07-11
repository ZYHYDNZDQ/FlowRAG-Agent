"""
Analysis tool — atomic LLM structured analysis over prepared context.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from config.prompts import ANALYZE_SYSTEM_PROMPT, build_agent_user_prompt
from tools.base import BaseTool
from tools.context import ToolExecutionContext


class AnalyzeToolInput(BaseModel):
    query: str
    context: str
    history: str = ""


class AnalyzeTool(BaseTool):
    """Generate structured analysis from pre-retrieved context."""

    name = "analyze"
    description = "Analyze document context and produce structured conclusions."
    input_schema = AnalyzeToolInput

    def execute(
        self,
        ctx: ToolExecutionContext,
        input_data: BaseModel,
        **_: Any,
    ) -> str:
        payload = self.validate_input(input_data)
        assert isinstance(payload, AnalyzeToolInput)
        user_prompt = build_agent_user_prompt(
            history=payload.history,
            context=payload.context,
            question=payload.query,
        )
        response = ctx.llm.invoke(
            [
                SystemMessage(content=ANALYZE_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        return str(response.content).strip()

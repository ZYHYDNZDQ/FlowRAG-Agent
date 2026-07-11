"""
Summarize tool — atomic LLM summarization over a prepared context.

Does not retrieve documents; Skills call search_document first.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from config.prompts import SUMMARIZE_REDUCE_TEMPLATE, SUMMARIZE_SYSTEM_PROMPT, build_agent_user_prompt
from langchain_core.messages import HumanMessage, SystemMessage
from tools.base import BaseTool
from tools.context import ToolExecutionContext


class SummarizeToolInput(BaseModel):
    query: str
    context: str
    history: str = ""


class SummarizeTool(BaseTool):
    """Generate a structured summary from pre-retrieved context."""

    name = "summarize"
    description = "Summarize document context into a structured answer."
    input_schema = SummarizeToolInput

    def execute(
        self,
        ctx: ToolExecutionContext,
        input_data: BaseModel,
        **_: Any,
    ) -> str:
        payload = self.validate_input(input_data)
        assert isinstance(payload, SummarizeToolInput)
        processed_context = SUMMARIZE_REDUCE_TEMPLATE.format(partial_summaries=payload.context)
        user_prompt = build_agent_user_prompt(
            history=payload.history,
            context=processed_context,
            question=payload.query.strip() or "请总结文档",
        )
        response = ctx.llm.invoke(
            [
                SystemMessage(content=SUMMARIZE_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        return str(response.content).strip()

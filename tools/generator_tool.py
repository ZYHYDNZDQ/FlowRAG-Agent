"""LLM text generation tool."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from tools.base import BaseTool
from tools.context import ToolExecutionContext


class GenerateToolInput(BaseModel):
    system_prompt: str
    user_prompt: str


class GenerateTool(BaseTool):
    """Atomic tool: invoke chat model with system + user prompts."""

    name = "llm.generate"
    description = "Generate text using the configured chat model."
    input_schema = GenerateToolInput

    def execute(
        self,
        ctx: ToolExecutionContext,
        input_data: BaseModel,
        **_: Any,
    ) -> str:
        payload = self.validate_input(input_data)
        assert isinstance(payload, GenerateToolInput)
        response = ctx.llm.invoke(
            [
                SystemMessage(content=payload.system_prompt),
                HumanMessage(content=payload.user_prompt),
            ]
        )
        return str(response.content).strip()

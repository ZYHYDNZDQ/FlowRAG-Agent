"""
BaseTool — atomic capability contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from tools.context import ToolExecutionContext


class BaseTool(ABC):
    """
    Atomic Agent capability.

    Rules:
      - Must not call other Tools
      - Must not call Skills
      - Must not operate Memory
      - Must not depend on Agent Runtime
    """

    name: str
    description: str
    input_schema: type[BaseModel]

    @abstractmethod
    def execute(
        self,
        ctx: ToolExecutionContext,
        input_data: BaseModel,
        **kwargs: Any,
    ) -> Any:
        """Run the tool against validated input and injected context."""

    def validate_input(self, data: BaseModel | dict[str, Any]) -> BaseModel:
        if isinstance(data, self.input_schema):
            return data
        return self.input_schema.model_validate(data)

    def run(
        self,
        ctx: ToolExecutionContext,
        input_data: BaseModel | dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Validate input then execute."""
        validated = self.validate_input(input_data)
        return self.execute(ctx, validated, **kwargs)

    def metadata(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.__name__,
        }

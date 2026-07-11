"""
BaseSkill — complete business capability contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from models.schemas import AgentStep, AnswerResult, IntentType, RetrievalScope
from skills.context import SkillContext


class BaseSkill(ABC):
    """
    Business-level capability orchestrating multiple Tools.

    Skills may call Tools and encode task logic.
    Skills must not access Chroma, invoke LLM directly, import Memory, or depend on Runtime.
    """

    name: str
    intent: IntentType
    description: str

    @abstractmethod
    def run(
        self,
        ctx: SkillContext,
        query: str,
        scope: RetrievalScope,
        *,
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> AnswerResult:
        """Execute the skill pipeline and return a structured answer."""

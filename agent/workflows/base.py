"""
Workflow base protocol.

Each workflow implements run() returning AnswerResult.
Workflows are LangChain LCEL Runnable chains (Day 2+).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from models.schemas import AgentStep, AnswerResult, RetrievalScope


class BaseWorkflow(ABC):
    """Abstract base for QA, Summarize, and Analyze workflows."""

    name: str = "base"

    @abstractmethod
    def run(
        self,
        query: str,
        scope: RetrievalScope,
        *,
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> AnswerResult:
        """Execute workflow and return structured result."""
        ...

"""
Streamlit callback handler — bridge LangChain events to AgentStep list.

Captures: routing decisions, retrieval queries/hits, LLM generation phases.
Consumed by ui/components/agent_trace.py for live display.

Implementation planned for Day 4.
"""

from __future__ import annotations

from typing import Callable

from models.schemas import AgentStep


class StreamlitTraceHandler:
    """Accumulates AgentStep events and notifies UI callback."""

    def __init__(
        self,
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> None:
        self.steps: list[AgentStep] = []
        self._on_step = on_step

    def emit(self, step: AgentStep) -> None:
        """Record a step and optionally notify UI."""
        self.steps.append(step)
        if self._on_step:
            self._on_step(step)

    def reset(self) -> None:
        """Clear accumulated steps for a new query."""
        self.steps.clear()

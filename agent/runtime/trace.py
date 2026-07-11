"""
Agent execution trace collection.
"""

from __future__ import annotations

from typing import Callable

from models.schemas import AgentStep


class TraceCollector:
    """Accumulates AgentStep events for a single Runtime run."""

    def __init__(
        self,
        *,
        run_id: str,
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> None:
        self.run_id = run_id
        self.steps: list[AgentStep] = []
        self._on_step = on_step

    def emit(self, step: AgentStep) -> AgentStep:
        """Record a step and optionally notify an external callback."""
        enriched = step.model_copy(
            update={
                "metadata": {
                    **step.metadata,
                    "run_id": self.run_id,
                }
            }
        )
        self.steps.append(enriched)
        if self._on_step:
            self._on_step(enriched)
        return enriched

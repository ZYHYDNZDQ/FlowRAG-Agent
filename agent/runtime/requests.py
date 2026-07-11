"""
Runtime request/response contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
from uuid import uuid4

from models.schemas import AgentStep, AnswerResult, RetrievalScope


@dataclass
class ExecuteRequest:
    """Input to AgentRuntime.execute()."""

    query: str
    session_id: str = "default"
    selected_doc_ids: list[str] | None = None
    scope_override: RetrievalScope | None = None
    on_step: Callable[[AgentStep], None] | None = None
    end_session: bool = False
    run_id: str = field(default_factory=lambda: uuid4().hex[:12])


@dataclass
class ExecuteResult:
    """Output from AgentRuntime.execute()."""

    answer: AnswerResult
    run_id: str
    session_id: str

"""
Shared helpers for Skill tool orchestration and trace emission.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from models.schemas import AgentStep, AgentStepType
from skills.context import SkillContext
from tools.registry import ToolRegistry

NO_CONTEXT_ANSWER = "知识库中未找到相关内容，请先上传 PDF 或调整检索范围。"


def emit_step(
    on_step: Callable[[AgentStep], None] | None,
    *,
    step_type: AgentStepType,
    name: str,
    detail: str = "",
    duration_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentStep:
    step = AgentStep(
        step_type=step_type,
        name=name,
        detail=detail,
        duration_ms=duration_ms,
        metadata=metadata or {},
    )
    if on_step:
        on_step(step)
    return step


def invoke_tool(
    ctx: SkillContext,
    tool_name: str,
    payload: dict[str, Any],
    *,
    on_step: Callable[[AgentStep], None] | None = None,
    step_type: AgentStepType,
    step_name: str,
    detail: str,
    extra_metadata: dict[str, Any] | None = None,
) -> Any:
    """Call a Tool through the registry and emit a trace step."""
    started = time.perf_counter()
    result = ctx.tools.run(tool_name, ctx.tool_context, **payload)
    elapsed = (time.perf_counter() - started) * 1000
    metadata = {"tool": tool_name, **(extra_metadata or {})}
    emit_step(
        on_step,
        step_type=step_type,
        name=step_name,
        detail=detail,
        duration_ms=elapsed,
        metadata=metadata,
    )
    return result

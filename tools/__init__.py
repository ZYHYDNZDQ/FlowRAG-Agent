"""Modular Agent Tool System."""

from tools.base import BaseTool
from tools.context import ToolExecutionContext
from tools.registry import ToolRegistry, get_tool_registry
from tools.retriever import retrieve

# Backward-compatible alias used by legacy imports.
Tool = BaseTool

__all__ = [
    "BaseTool",
    "Tool",
    "ToolExecutionContext",
    "ToolRegistry",
    "get_tool_registry",
    "retrieve",
]

"""
Tool registry — registration, lookup, and lifecycle management.
"""

from __future__ import annotations

from typing import Any

from tools.base import BaseTool
from tools.context import ToolExecutionContext


class ToolRegistry:
    """Register, query, and manage Tool instances."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        self._ensure_initialized()
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def list_tools(self) -> list[dict[str, str]]:
        self._ensure_initialized()
        return [tool.metadata() for tool in self._tools.values()]

    def names(self) -> list[str]:
        self._ensure_initialized()
        return list(self._tools.keys())

    def initialize(self) -> None:
        """Register built-in tools once (lifecycle entry)."""
        if self._initialized:
            return
        self._register_builtin_tools()
        self._initialized = True

    def shutdown(self) -> None:
        """Clear registry (lifecycle exit; useful in tests)."""
        self._tools.clear()
        self._initialized = False

    def run(
        self,
        name: str,
        ctx: ToolExecutionContext,
        input_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        tool = self.get(name)
        payload: dict[str, Any] = dict(input_data or {})
        payload.update(kwargs)
        return tool.run(ctx, payload, **kwargs)

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    def _register_builtin_tools(self) -> None:
        from tools.analyze_tool import AnalyzeTool
        from tools.citation_tool import CitationTool
        from tools.generator_tool import GenerateTool
        from tools.rag_tool import RagFormatContextTool, RagTool
        from tools.search_document_tool import SearchDocumentTool
        from tools.summarize_tool import SummarizeTool

        for tool in (
            RagTool(),
            SearchDocumentTool(),
            RagFormatContextTool(),
            GenerateTool(),
            SummarizeTool(),
            AnalyzeTool(),
            CitationTool(),
        ):
            self.register(tool)


_default_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Return module-level ToolRegistry singleton."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
        _default_registry.initialize()
    return _default_registry

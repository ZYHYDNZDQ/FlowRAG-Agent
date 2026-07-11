"""
FlowRAG MCP Server entry point.

Exposes knowledge-base retrieval to external MCP clients (Cursor, Claude Desktop, etc.)
via stdio transport. Independent from Streamlit / AgentRuntime.

Run:
    python -m mcp_server.server
    flowrag-mcp
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import config.env_bootstrap  # noqa: F401

from config.settings import get_settings
from mcp.server.fastmcp import FastMCP
from mcp_server.resources import register_resources
from mcp_server.tools import register_tools
from models.doc_registry import DocRegistry
from retrieval.chroma_store import ChromaStore
from tools.context import ToolExecutionContext


@dataclass
class McpServerContext:
    """Shared dependencies initialized at server startup."""

    tool_ctx: ToolExecutionContext
    registry: DocRegistry


@asynccontextmanager
async def app_lifespan(_server: FastMCP) -> AsyncIterator[McpServerContext]:
    """Wire production retrieval services once at startup."""
    settings = get_settings()
    settings.ensure_dirs()
    store = ChromaStore(settings)
    store.connect()
    tool_ctx = ToolExecutionContext.create(
        settings=settings,
        vector_store=store,
        connect_store=False,
    )
    registry = DocRegistry(settings.registry_path)
    try:
        yield McpServerContext(tool_ctx=tool_ctx, registry=registry)
    finally:
        pass


def create_server() -> FastMCP:
    """Build and configure the MCP server."""
    mcp = FastMCP(
        "FlowRAG Knowledge Base",
        instructions=(
            "Provides semantic search over a local PDF knowledge base. "
            "Use search_document to retrieve chunks; read flowrag://documents/index "
            "to list available documents."
        ),
        lifespan=app_lifespan,
    )
    register_tools(mcp)
    register_resources(mcp)
    return mcp


mcp = create_server()


def main() -> int:
    """Run MCP server over stdio (default transport)."""
    try:
        mcp.run()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

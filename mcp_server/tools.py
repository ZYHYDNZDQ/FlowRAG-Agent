"""
MCP tools — thin adapters over existing Agent tools.

Does NOT re-implement RAG; delegates to SearchDocumentTool → RAGService.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mcp.server.fastmcp import Context, FastMCP

from models.schemas import RetrievalMode, RetrievalScope, RetrievedChunk
from tools.registry import get_tool_registry

if TYPE_CHECKING:
    from mcp.server.session import ServerSession


def build_retrieval_scope(doc_ids: list[str] | None) -> RetrievalScope:
    """Map optional doc_ids to RetrievalScope (same rules as agent router)."""
    ids = doc_ids or []
    if not ids:
        return RetrievalScope(mode=RetrievalMode.ALL)
    if len(ids) == 1:
        return RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=ids)
    return RetrievalScope(mode=RetrievalMode.SELECTED, doc_ids=ids)


def serialize_chunks(chunks: list[RetrievedChunk]) -> str:
    """JSON payload for MCP clients."""
    return json.dumps([chunk.model_dump() for chunk in chunks], ensure_ascii=False, indent=2)


def register_tools(mcp: FastMCP) -> None:
    """Register MCP tools on the given server instance."""

    @mcp.tool(
        name="search_document",
        description=(
            "Semantic search over the FlowRAG knowledge base. "
            "Returns relevant document chunks with metadata and scores."
        ),
        annotations={
            "title": "Search Document",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    def search_document(
        query: str,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        *,
        ctx: Context,
    ) -> str:
        """
        Search indexed PDF chunks by natural-language query.

        Args:
            query: User question or search phrase.
            doc_ids: Optional document IDs to restrict scope.
            top_k: Maximum chunks to return (defaults to app settings).
        """
        app_ctx = ctx.request_context.lifespan_context
        tool_ctx = app_ctx.tool_ctx
        scope = build_retrieval_scope(doc_ids)

        chunks = get_tool_registry().run(
            "search_document",
            tool_ctx,
            query=query,
            scope=scope,
            top_k=top_k,
            score_threshold=0.0,
        )
        return serialize_chunks(chunks)

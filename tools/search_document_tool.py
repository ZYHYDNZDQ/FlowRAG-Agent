"""Search document tool — semantic retrieval over the knowledge base."""

from __future__ import annotations

from tools.rag_tool import RagTool


class SearchDocumentTool(RagTool):
    """Alias tool: search_document → RAGService.query_chunks."""

    name = "search_document"
    description = "Search the knowledge base and return relevant document chunks."

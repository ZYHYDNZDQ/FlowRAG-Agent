"""
MCP resources — read-only document catalog from DocRegistry.

Does NOT duplicate ingestion or Chroma logic.
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from config.settings import get_settings
from models.doc_registry import DocRegistry
from models.schemas import DocStatus


def _get_registry() -> DocRegistry:
    settings = get_settings()
    return DocRegistry(settings.registry_path)


def _record_to_dict(record) -> dict:
    return {
        "doc_id": record.doc_id,
        "source_file": record.source_file,
        "source_path": record.source_path,
        "page_count": record.page_count,
        "chunk_count": record.chunk_count,
        "status": record.status.value,
        "ingested_at": record.ingested_at.isoformat() if record.ingested_at else None,
    }


def register_resources(mcp: FastMCP) -> None:
    """Register MCP resources on the given server instance."""

    @mcp.resource(
        "flowrag://documents/index",
        name="documents",
        description="List all indexed documents in the knowledge base.",
        mime_type="application/json",
    )
    def list_documents() -> str:
        """Return indexed document metadata as JSON."""
        records = _get_registry().list_all(status=DocStatus.INDEXED)
        payload = [_record_to_dict(record) for record in records]
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @mcp.resource(
        "flowrag://document/{doc_id}",
        name="document",
        description="Metadata for a single indexed document.",
        mime_type="application/json",
    )
    def get_document(doc_id: str) -> str:
        """Return one document record by doc_id."""
        record = _get_registry().get(doc_id)
        if record is None:
            return json.dumps({"error": f"Document not found: {doc_id}"}, ensure_ascii=False)
        return json.dumps(_record_to_dict(record), ensure_ascii=False, indent=2)

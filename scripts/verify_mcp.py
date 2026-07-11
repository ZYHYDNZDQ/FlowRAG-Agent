"""
Verify FlowRAG MCP Server end-to-end over stdio.

Usage:
    python scripts/verify_mcp.py
    python scripts/verify_mcp.py --query "payment within 30 days"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def run_verify(query: str) -> int:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=str(PROJECT_ROOT),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"[OK] list_tools: {tool_names}")
            if "search_document" not in tool_names:
                print("[FAIL] search_document tool not registered")
                return 1

            resources = await session.list_resources()
            templates = await session.list_resource_templates()
            resource_uris = [r.uri for r in resources.resources]
            template_uris = [t.uriTemplate for t in templates.resourceTemplates]
            print(f"[OK] list_resources: {resource_uris}")
            print(f"[OK] list_resource_templates: {template_uris}")

            doc_list = await session.read_resource("flowrag://documents/index")
            doc_text = doc_list.contents[0].text if doc_list.contents else "[]"
            documents = json.loads(doc_text)
            print(f"[OK] documents resource: {len(documents)} indexed document(s)")
            if documents:
                print(f"     first: {documents[0].get('source_file')} ({documents[0].get('doc_id')})")

            result = await session.call_tool(
                "search_document",
                {"query": query, "top_k": 3},
            )
            if result.isError:
                err = result.content[0].text if result.content else "unknown error"
                print(f"[FAIL] search_document: {err}")
                return 1
            chunk_text = result.content[0].text if result.content else "[]"
            chunks = json.loads(chunk_text)
            print(f"[OK] search_document: {len(chunks)} chunk(s) for query={query!r}")
            if chunks:
                first = chunks[0]
                print(
                    f"     top hit: page {first['metadata']['page']} "
                    f"from {first['metadata']['source_file']}"
                )
            else:
                print("[WARN] no chunks returned — upload PDFs via Streamlit or flowrag-ingest first")

            print("\nMCP verification passed.")
            return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify FlowRAG MCP server")
    parser.add_argument("--query", default="payment within 30 days")
    args = parser.parse_args()
    return asyncio.run(run_verify(args.query))


if __name__ == "__main__":
    sys.exit(main())

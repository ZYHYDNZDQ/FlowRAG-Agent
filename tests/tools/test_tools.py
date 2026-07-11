"""Tool system tests."""

from __future__ import annotations

import pytest
from langchain_community.chat_models.fake import FakeListChatModel

from models.schemas import RetrievalMode, RetrievalScope
from tools.context import ToolExecutionContext
from tools.rag_tool import RagTool, RagToolInput
from tools.registry import ToolRegistry
from tools.search_document_tool import SearchDocumentTool
from tools.summarize_tool import SummarizeTool, SummarizeToolInput

pytestmark = pytest.mark.tool


def test_tool_registry_lifecycle():
    registry = ToolRegistry()
    assert not registry.initialized
    registry.initialize()
    names = registry.names()
    assert "search_document" in names
    assert "summarize" in names
    assert "analyze" in names
    assert "rag.retrieve" in names
    registry.shutdown()
    assert not registry.initialized


def test_tool_metadata_schema():
    registry = ToolRegistry()
    registry.initialize()
    meta = {item["name"]: item for item in registry.list_tools()}
    assert meta["search_document"]["input_schema"] == "RagToolInput"
    assert meta["summarize"]["input_schema"] == "SummarizeToolInput"


def test_search_document_tool(ingested_contract, rag_settings, fake_embeddings):
    result, store, _registry, _ = ingested_contract
    ctx = ToolExecutionContext.create(
        settings=rag_settings,
        vector_store=store,
        embeddings=fake_embeddings,
    )
    chunks = SearchDocumentTool().execute(
        ctx,
        RagToolInput(
            query="payment within 30 days",
            scope=RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id]),
            top_k=3,
            score_threshold=0.0,
        ),
    )
    assert chunks


def test_rag_tool_requires_vector_store(rag_settings, fake_llm: FakeListChatModel):
    ctx = ToolExecutionContext.stateless(settings=rag_settings, llm=fake_llm)
    with pytest.raises(RuntimeError, match="requires vector_store"):
        RagTool().execute(
            ctx,
            RagToolInput(query="test", scope=RetrievalScope()),
        )


def test_summarize_tool_atomic(fake_llm: FakeListChatModel, rag_settings):
    ctx = ToolExecutionContext.stateless(settings=rag_settings, llm=fake_llm)
    answer = SummarizeTool().execute(
        ctx,
        SummarizeToolInput(query="总结", context="付款周期30天"),
    )
    assert answer

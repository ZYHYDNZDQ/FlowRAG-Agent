"""
Core data contracts for FlowRAG-Agent.

These schemas define the interface between ingestion, retrieval,
agent workflows, and the Streamlit UI. See docs/METADATA_SCHEMA.md.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class IntentType(str, Enum):
    QA = "qa"
    SUMMARIZE = "summarize"
    ANALYZE = "analyze"


class RetrievalMode(str, Enum):
    ALL = "all"
    SINGLE = "single"
    SELECTED = "selected"


class DocStatus(str, Enum):
    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"


class AgentStepType(str, Enum):
    ROUTING = "routing"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    POSTPROCESS = "postprocess"


# ---------------------------------------------------------------------------
# Chroma metadata contract
# ---------------------------------------------------------------------------


class ChunkMetadata(BaseModel):
    """Flat metadata stored alongside each Chroma vector."""

    doc_id: str
    source_file: str
    source_path: str
    page: int = Field(ge=1, description="1-based page number")
    chunk_index: int = Field(ge=0)
    chunk_id: str
    ingest_version: int = Field(default=1, ge=1)

    # Recommended fields
    page_count: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    token_estimate: int | None = None
    section_hint: str = ""
    language: str = "zh"
    file_hash: str = ""

    def to_chroma_dict(self) -> dict[str, Any]:
        """Serialize to Chroma-compatible flat metadata."""
        data = self.model_dump(exclude_none=True)
        # Chroma does not accept None; drop empty optional strings if needed
        return {k: v for k, v in data.items() if v is not None and v != ""}


# ---------------------------------------------------------------------------
# Document registry
# ---------------------------------------------------------------------------


class DocRecord(BaseModel):
    """Document-level metadata stored in doc_registry."""

    doc_id: str
    source_file: str
    source_path: str
    file_hash: str
    page_count: int = 0
    chunk_count: int = 0
    status: DocStatus = DocStatus.PENDING
    error_message: str = ""
    ingested_at: datetime | None = None
    ingest_version: int = 1


class IngestResult(BaseModel):
    """Result returned after PDF ingestion."""

    doc_id: str
    source_file: str
    page_count: int
    chunk_count: int
    status: DocStatus
    error_message: str = ""


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class RetrievalScope(BaseModel):
    """Describes which documents/pages to search."""

    mode: RetrievalMode = RetrievalMode.ALL
    doc_ids: list[str] = Field(default_factory=list)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    # Filter construction: retrieval/chroma_store.py → ChromaStore.build_where_filter()


class Citation(BaseModel):
    """A single source reference attached to an answer."""

    source_file: str
    page: int
    chunk_id: str
    doc_id: str
    excerpt: str = ""
    score: float | None = None


class RetrievedChunk(BaseModel):
    """One chunk returned from vector search."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata
    score: float | None = None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class RouterResult(BaseModel):
    """Output of intent routing."""

    intent: IntentType
    scope: RetrievalScope
    confidence: float = 1.0
    reasoning: str = ""


class AgentStep(BaseModel):
    """One observable step in the Agent execution trace."""

    step_type: AgentStepType
    name: str
    detail: str = ""
    duration_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnswerResult(BaseModel):
    """Final output returned by the orchestrator."""

    answer: str
    intent: IntentType
    citations: list[Citation] = Field(default_factory=list)
    trace: list[AgentStep] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)

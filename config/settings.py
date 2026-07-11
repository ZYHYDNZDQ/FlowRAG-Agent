"""
Global application settings.

Loads from environment variables and .env file via pydantic-settings.
All paths are resolved relative to project root at runtime.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PageBreakStrategy(str, Enum):
    """How PDF pages are handled during chunking."""

    INTRA_PAGE = "intra_page"
    ALLOW_CROSS_PAGE = "allow_cross_page"


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"


class EmbeddingProvider(str, Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "FlowRAG-Agent"
    debug: bool = False

    # Paths
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    uploads_dir: Path = Field(default=PROJECT_ROOT / "data" / "uploads")
    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma")
    registry_path: Path = Field(default=PROJECT_ROOT / "data" / "registry.db")

    # ChromaDB
    chroma_collection_name: str = "flowrag_docs"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120
    page_break_strategy: PageBreakStrategy = PageBreakStrategy.INTRA_PAGE

    # Retrieval
    default_top_k: int = 6
    score_threshold: float = 0.42
    max_context_chunks: int = 8
    max_context_tokens: int = 4000

    # Session memory (short-term conversation history for prompt injection)
    memory_max_turns: int = 5
    memory_max_tokens: int = 1500

    # Chunk ID template (doc_short, page, chunk_index)
    chunk_id_template: str = "{doc_short}_p{page:04d}_c{chunk_index:04d}"

    # LLM
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    # Prefer LLM_API_KEY to avoid being overridden by a global OPENAI_API_KEY env var.
    llm_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-4o-mini"

    @property
    def effective_llm_api_key(self) -> str:
        """Return the API key used for chat/completion providers."""
        return self.llm_api_key or self.openai_api_key

    # Embeddings
    embedding_provider: EmbeddingProvider = EmbeddingProvider.HUGGINGFACE
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"
    hf_endpoint: str = ""
    openai_embedding_model: str = "text-embedding-3-small"

    def ensure_dirs(self) -> None:
        """Create runtime data directories if missing."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()

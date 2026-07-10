"""
LLM and Embedding model factory.

``get_embeddings`` is required by the RAG ingestion and retrieval pipeline.
``get_llm`` will be used by Agent workflows (Day 2+).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import config.env_bootstrap  # noqa: F401
from config.env_bootstrap import apply_hf_endpoint
from config.settings import EmbeddingProvider, LLMProvider, Settings, get_settings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel

_embeddings_cache: dict[tuple[str, str, str], Embeddings] = {}


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """
    Return a configured embedding model for vector indexing and query encoding.

    Providers:
      - ``huggingface``: local sentence-transformers (default)
      - ``openai``: OpenAI-compatible embedding API

    The instance is cached by provider + model name to avoid reloading weights.
    """
    cfg = settings or get_settings()
    cache_key = (
        cfg.embedding_provider.value,
        cfg.embedding_model,
        cfg.embedding_device,
    )
    if cache_key in _embeddings_cache:
        return _embeddings_cache[cache_key]

    if cfg.embedding_provider == EmbeddingProvider.HUGGINGFACE:
        apply_hf_endpoint(cfg.hf_endpoint or None)

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        model = HuggingFaceEmbeddings(
            model_name=cfg.embedding_model,
            model_kwargs={"device": cfg.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
        _embeddings_cache[cache_key] = model
        return model

    if cfg.embedding_provider == EmbeddingProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings

        model = OpenAIEmbeddings(
            model=cfg.openai_embedding_model,
            api_key=cfg.effective_llm_api_key or None,
            base_url=cfg.openai_base_url or None,
        )
        _embeddings_cache[cache_key] = model
        return model

    raise ValueError(f"Unsupported embedding provider: {cfg.embedding_provider}")


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    """
    Return a configured chat model for Agent generation (Day 2+).

    Not required for the RAG indexing / retrieval pipeline.
    """
    cfg = settings or get_settings()

    if cfg.llm_provider == LLMProvider.OLLAMA:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
        )

    if cfg.llm_provider in (LLMProvider.OPENAI, LLMProvider.OPENAI_COMPATIBLE):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.openai_model,
            api_key=cfg.effective_llm_api_key or None,
            base_url=cfg.openai_base_url or None,
        )

    raise ValueError(f"Unsupported LLM provider: {cfg.llm_provider}")

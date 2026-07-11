"""
Isolated evaluation harness.

Builds a temporary knowledge base without touching production ``data/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz
from langchain_core.embeddings import Embeddings

from config.settings import Settings
from evaluation.datasets.schema import EvalBenchmark
from ingestion.indexer import ingest_pdf
from models.doc_registry import DocRegistry
from models.schemas import DocStatus, RetrievalMode, RetrievalScope
from retrieval.chroma_store import ChromaStore


@dataclass(frozen=True)
class EvalEnvironment:
    """Isolated KB + dependencies for one evaluation run."""

    settings: Settings
    store: ChromaStore
    registry: DocRegistry
    doc_id: str
    scope: RetrievalScope
    work_dir: Path


def _make_pdf(path: Path, pages: list[str]) -> Path:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


def build_eval_environment(
    benchmark: EvalBenchmark,
    *,
    work_dir: Path,
    embeddings: Embeddings,
) -> EvalEnvironment:
    """
    Ingest benchmark document into an isolated Chroma + registry under ``work_dir``.

    Does not read or write the application's default ``data/`` paths.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir = work_dir / "data"
    settings = Settings(
        data_dir=data_dir,
        uploads_dir=data_dir / "uploads",
        chroma_persist_dir=data_dir / "chroma",
        registry_path=data_dir / "registry.db",
        chroma_collection_name=f"eval_{benchmark.name}",
        chunk_size=120,
        chunk_overlap=20,
        score_threshold=0.0,
    )
    settings.ensure_dirs()

    pdf_path = _make_pdf(
        settings.uploads_dir / benchmark.document.filename,
        benchmark.document.pages,
    )
    store = ChromaStore(settings)
    store.connect()
    registry = DocRegistry(settings.registry_path)

    result = ingest_pdf(
        pdf_path,
        settings=settings,
        store=store,
        registry=registry,
        embeddings=embeddings,
    )
    if result.status != DocStatus.INDEXED:
        raise RuntimeError(f"Eval ingest failed: {result.status}")

    scope = RetrievalScope(mode=RetrievalMode.SINGLE, doc_ids=[result.doc_id])
    return EvalEnvironment(
        settings=settings,
        store=store,
        registry=registry,
        doc_id=result.doc_id,
        scope=scope,
        work_dir=work_dir,
    )

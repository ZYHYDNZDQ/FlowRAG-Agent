"""
ChromaDB store wrapper.

Pipeline position: final persistence layer for embeddings; also powers retrieval.
"""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import Settings, get_settings
from models.schemas import ChunkMetadata, RetrievedChunk, RetrievalScope


class ChromaStore:
    """
    Thin wrapper around Chroma ``PersistentClient``.

    Handles collection lifecycle, chunk upsert, similarity search, and
    metadata-filtered deletion.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    @property
    def collection_name(self) -> str:
        """Active Chroma collection name from settings."""
        return self._settings.chroma_collection_name

    def connect(self) -> None:
        """
        Open (or create) a persistent Chroma client and collection.

        Uses cosine distance to align with normalized HuggingFace embeddings.
        Safe to call multiple times — reuses existing handles.
        """
        if self._client is not None and self._collection is not None:
            return

        self._settings.ensure_dirs()
        self._client = chromadb.PersistentClient(
            path=str(self._settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
    ) -> int:
        """
        Insert or update chunk vectors in the collection.

        Args:
            ids: Unique chunk IDs (same as ``metadata['chunk_id']``).
            documents: Raw chunk texts.
            metadatas: Flat dicts — must include ``source_file`` and ``page``.
            embeddings: Pre-computed vectors; if omitted Chroma embeds internally.

        Returns:
            Number of chunks written.
        """
        self.connect()
        assert self._collection is not None

        sanitized = [_sanitize_metadata(meta) for meta in metadatas]
        kwargs: dict[str, Any] = {
            "ids": ids,
            "documents": documents,
            "metadatas": sanitized,
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings

        self._collection.add(**kwargs)
        return len(ids)

    def query(
        self,
        query_text: str,
        *,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        query_embedding: list[float] | None = None,
        embeddings: Any | None = None,
    ) -> list[RetrievedChunk]:
        """
        Run similarity search and return chunks with full source metadata.

        Each ``RetrievedChunk`` includes:
          - ``metadata.source_file`` — filename
          - ``metadata.page`` — 1-based page number
          - ``score`` — similarity in [0, 1] (higher is better)

        Args:
            query_text: Natural language query.
            top_k: Max results; defaults to ``settings.default_top_k``.
            where: Chroma metadata filter (from ``build_where_filter``).
            query_embedding: Optional pre-computed query vector.
            embeddings: Optional Embeddings instance for query encoding (tests).
        """
        self.connect()
        assert self._collection is not None

        k = top_k or self._settings.default_top_k
        vector = query_embedding
        if vector is None:
            if embeddings is not None:
                vector = embeddings.embed_query(query_text)
            else:
                # Lazy import avoids circular dependency with ingestion.indexer
                from ingestion.embedder import embed_query

                vector = embed_query(query_text)

        result = self._collection.query(
            query_embeddings=[vector],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return _results_to_chunks(result)

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Remove all chunks belonging to a document.

        Returns:
            Chroma-reported delete count (may be 0 if nothing matched).
        """
        self.connect()
        assert self._collection is not None

        existing = self._collection.get(where={"doc_id": {"$eq": doc_id}})
        ids = existing.get("ids") or []
        if not ids:
            return 0

        self._collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        """Return total number of vectors in the collection."""
        self.connect()
        assert self._collection is not None
        return self._collection.count()

    @staticmethod
    def build_where_filter(scope: RetrievalScope) -> dict[str, Any] | None:
        """
        Convert ``RetrievalScope`` to a Chroma ``where`` clause.

        Presets: F0 (all), F1 (single doc), F2 (multi doc), F4 (page range).
        """
        from models.schemas import RetrievalMode

        clauses: list[dict[str, Any]] = []

        if scope.mode == RetrievalMode.SINGLE:
            if not scope.doc_ids:
                raise ValueError("SINGLE scope requires exactly one doc_id")
            clauses.append({"doc_id": {"$eq": scope.doc_ids[0]}})
        elif scope.mode == RetrievalMode.SELECTED:
            if not scope.doc_ids:
                raise ValueError("SELECTED scope requires at least one doc_id")
            clauses.append({"doc_id": {"$in": scope.doc_ids}})

        if scope.page_start is not None:
            clauses.append({"page": {"$gte": scope.page_start}})
        if scope.page_end is not None:
            clauses.append({"page": {"$lte": scope.page_end}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def metadata_from_chunk(meta: ChunkMetadata) -> dict[str, Any]:
        """Serialize ``ChunkMetadata`` for Chroma storage."""
        return _sanitize_metadata(meta.to_chroma_dict())


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """
    Coerce metadata to Chroma-accepted scalar types.

    Chroma rejects ``None`` and nested structures.
    """
    clean: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def _results_to_chunks(result: dict[str, Any]) -> list[RetrievedChunk]:
    """Map raw Chroma query result to ``RetrievedChunk`` list."""
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    chunks: list[RetrievedChunk] = []
    for index, chunk_id in enumerate(ids):
        raw_meta = metadatas[index] or {}
        text = documents[index] or ""
        distance = distances[index] if index < len(distances) else None
        score = _distance_to_score(distance)

        metadata = ChunkMetadata(
            doc_id=str(raw_meta["doc_id"]),
            source_file=str(raw_meta["source_file"]),
            source_path=str(raw_meta.get("source_path", "")),
            page=int(raw_meta["page"]),
            chunk_index=int(raw_meta.get("chunk_index", 0)),
            chunk_id=str(raw_meta.get("chunk_id", chunk_id)),
            ingest_version=int(raw_meta.get("ingest_version", 1)),
            page_count=int(raw_meta["page_count"]) if raw_meta.get("page_count") else None,
            char_start=int(raw_meta["char_start"]) if raw_meta.get("char_start") is not None else None,
            char_end=int(raw_meta["char_end"]) if raw_meta.get("char_end") is not None else None,
            file_hash=str(raw_meta.get("file_hash", "")),
        )

        chunks.append(
            RetrievedChunk(
                chunk_id=metadata.chunk_id,
                text=text,
                metadata=metadata,
                score=score,
            )
        )

    return chunks


def _distance_to_score(distance: float | None) -> float | None:
    """Convert Chroma cosine distance to a higher-is-better similarity score."""
    if distance is None:
        return None
    return max(0.0, min(1.0, 1.0 - float(distance)))

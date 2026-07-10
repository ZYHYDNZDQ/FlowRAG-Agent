"""
Document registry — CRUD for indexed PDF metadata.

Persistence: SQLite at settings.registry_path (default data/registry.db).
Chroma holds chunk vectors; registry holds document-level records.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from config.settings import get_settings
from models.schemas import DocRecord, DocStatus


class DocRegistry:
    """Manage document-level metadata outside Chroma."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        source_file TEXT NOT NULL,
        source_path TEXT NOT NULL,
        file_hash TEXT NOT NULL,
        page_count INTEGER NOT NULL DEFAULT 0,
        chunk_count INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'pending',
        error_message TEXT NOT NULL DEFAULT '',
        ingested_at TEXT,
        ingest_version INTEGER NOT NULL DEFAULT 1
    );
    CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
    CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
    """

    def __init__(self, registry_path: str | Path | None = None) -> None:
        settings = get_settings()
        self._registry_path = Path(registry_path or settings.registry_path)
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._registry_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(self._SCHEMA)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DocRecord:
        ingested_at = None
        if row["ingested_at"]:
            ingested_at = datetime.fromisoformat(row["ingested_at"])
        return DocRecord(
            doc_id=row["doc_id"],
            source_file=row["source_file"],
            source_path=row["source_path"],
            file_hash=row["file_hash"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            status=DocStatus(row["status"]),
            error_message=row["error_message"] or "",
            ingested_at=ingested_at,
            ingest_version=row["ingest_version"],
        )

    def create(self, record: DocRecord) -> DocRecord:
        """Insert a new document record."""
        ingested_at = record.ingested_at
        if ingested_at is None:
            ingested_at = datetime.now(timezone.utc)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, source_file, source_path, file_hash,
                    page_count, chunk_count, status, error_message,
                    ingested_at, ingest_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.doc_id,
                    record.source_file,
                    record.source_path,
                    record.file_hash,
                    record.page_count,
                    record.chunk_count,
                    record.status.value,
                    record.error_message,
                    ingested_at.isoformat(),
                    record.ingest_version,
                ),
            )
        return record.model_copy(update={"ingested_at": ingested_at})

    def get(self, doc_id: str) -> DocRecord | None:
        """Fetch a document by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_all(self, status: DocStatus | None = None) -> list[DocRecord]:
        """List documents, optionally filtered by status."""
        with self._connect() as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM documents ORDER BY ingested_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE status = ? ORDER BY ingested_at DESC",
                    (status.value,),
                ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def update_status(
        self,
        doc_id: str,
        status: DocStatus,
        *,
        chunk_count: int | None = None,
        page_count: int | None = None,
        error_message: str = "",
        ingest_version: int | None = None,
    ) -> None:
        """Update indexing status after ingestion completes or fails."""
        fields: list[str] = ["status = ?", "error_message = ?"]
        values: list[object] = [status.value, error_message]

        if chunk_count is not None:
            fields.append("chunk_count = ?")
            values.append(chunk_count)
        if page_count is not None:
            fields.append("page_count = ?")
            values.append(page_count)
        if ingest_version is not None:
            fields.append("ingest_version = ?")
            values.append(ingest_version)
        if status == DocStatus.INDEXED:
            fields.append("ingested_at = ?")
            values.append(datetime.now(timezone.utc).isoformat())

        values.append(doc_id)
        sql = f"UPDATE documents SET {', '.join(fields)} WHERE doc_id = ?"
        with self._connect() as conn:
            conn.execute(sql, values)

    def delete(self, doc_id: str) -> None:
        """Remove document record (caller must also delete Chroma vectors)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def find_by_hash(self, file_hash: str) -> DocRecord | None:
        """Check for duplicate upload by file hash."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE file_hash = ? AND status != 'failed' "
                "ORDER BY ingested_at DESC LIMIT 1",
                (file_hash,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def clear_all(self) -> int:
        """Delete all document records. Returns count removed."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM documents")
            return cursor.rowcount

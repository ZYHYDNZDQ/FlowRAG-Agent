"""
Ingestion utilities — file hashing, ID generation, safe paths.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path


def compute_file_hash(file_path: Path, *, chunk_size: int = 65536) -> str:
    """Return SHA256 hex digest prefixed with 'sha256:'."""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def generate_doc_id() -> str:
    """Generate a new document UUID."""
    return str(uuid.uuid4())


def doc_id_short(doc_id: str, *, length: int = 8) -> str:
    """Short prefix for chunk_id generation (alphanumeric, no hyphens)."""
    return doc_id.replace("-", "")[:length]


def ensure_unique_upload_path(uploads_dir: Path, filename: str) -> Path:
    """
    Resolve a collision-safe path under uploads_dir.

    If `filename` already exists, append _1, _2, ...
    """
    uploads_dir.mkdir(parents=True, exist_ok=True)
    candidate = uploads_dir / filename
    if not candidate.exists():
        return candidate

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        candidate = uploads_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1

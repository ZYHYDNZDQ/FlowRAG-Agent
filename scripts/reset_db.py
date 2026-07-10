#!/usr/bin/env python3
"""
Reset knowledge base — delete Chroma collection and doc registry.

Usage:
  python -m scripts.reset_db --confirm
  flowrag-reset --confirm

WARNING: Destructive operation. Requires --confirm flag.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from config.settings import get_settings
from models.doc_registry import DocRegistry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reset FlowRAG-Agent knowledge base")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm deletion of all indexed documents",
    )
    args = parser.parse_args(argv)

    if not args.confirm:
        print("Pass --confirm to delete all data.")
        return 1

    settings = get_settings()
    settings.ensure_dirs()

    removed_docs = DocRegistry().clear_all()
    print(f"Removed {removed_docs} document record(s) from registry.")

    chroma_dir = settings.chroma_persist_dir
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cleared Chroma persist directory: {chroma_dir}")

    uploads_dir = settings.uploads_dir
    if uploads_dir.exists():
        for path in uploads_dir.iterdir():
            if path.name == ".gitkeep":
                continue
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        print(f"Cleared uploads directory: {uploads_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

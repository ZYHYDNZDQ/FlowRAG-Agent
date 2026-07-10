#!/usr/bin/env python3
"""
CLI tool for batch PDF ingestion (development / debugging).

Usage:
  python -m scripts.ingest_cli path/to/file.pdf
  flowrag-ingest path/to/file.pdf   (after pip install -e .)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config.env_bootstrap  # noqa: F401  # must run before huggingface_hub import

from config.settings import get_settings
from ingestion.indexer import ingest_pdf
from models.schemas import DocStatus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest PDF files into FlowRAG-Agent knowledge base")
    parser.add_argument("files", nargs="+", type=Path, help="PDF files to ingest")
    parser.add_argument("--force", action="store_true", help="Re-index even if file hash exists")
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_dirs()

    exit_code = 0
    for pdf_path in args.files:
        print(f"\n=== Ingesting: {pdf_path} ===")
        result = ingest_pdf(pdf_path, force=args.force)

        if result.status == DocStatus.INDEXED:
            print(
                f"OK  doc_id={result.doc_id}  pages={result.page_count}  "
                f"chunks={result.chunk_count}"
            )
            if result.error_message:
                print(f"    note: {result.error_message}")
        else:
            print(f"FAIL  {result.error_message}")
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

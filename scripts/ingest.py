#!/usr/bin/env python
"""CLI entry point for ingesting data into AryaGPT's ChromaDB knowledge base.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --data-dir data/ --chroma-dir chroma_db/
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure the project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.ingest import ingest


def main():
    parser = argparse.ArgumentParser(description="Ingest data files into AryaGPT ChromaDB.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing source files (PDF, CSV, MD, TXT, JSON). Default: data/",
    )
    parser.add_argument(
        "--chroma-dir",
        default="chroma_db",
        help="Directory for ChromaDB persistence. Default: chroma_db/",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    chroma_dir = Path(args.chroma_dir)

    if not data_dir.exists():
        print(f"ERROR: data directory '{data_dir}' does not exist.")
        sys.exit(1)

    chroma_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== AryaGPT Ingest ===")
    print(f"Data dir  : {data_dir.resolve()}")
    print(f"Chroma dir: {chroma_dir.resolve()}")
    print()

    stats = ingest(str(data_dir), str(chroma_dir))

    print()
    print(f"=== Ingest Complete ===")
    print(f"  Chunks added   : {stats['added']}")
    print(f"  Chunks skipped : {stats['skipped']}")
    print(f"  Total in KB    : {stats['total_in_collection']}")


if __name__ == "__main__":
    main()

"""Data ingestion pipeline: loads documents, chunks, deduplicates, and upserts into ChromaDB."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.embeddings import embed_texts

CHROMA_DIR = os.environ.get("CHROMA_DIR", str(Path(__file__).parent.parent / "chroma_db"))
COLLECTION_NAME = "aryagpt_kb"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".txt", ".md", ".json"}


def _get_chroma_collection(chroma_dir: str = CHROMA_DIR) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )
    return collection


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_documents(data_dir: str) -> List[Any]:
    data_path = Path(data_dir)
    docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for file_path in sorted(data_path.rglob("*")):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
                raw = loader.load()
            elif file_path.suffix.lower() == ".csv":
                loader = CSVLoader(file_path=str(file_path), encoding="utf-8")
                raw = loader.load()
            elif file_path.suffix.lower() in (".txt", ".md"):
                loader = TextLoader(str(file_path), encoding="utf-8")
                raw = loader.load()
            elif file_path.suffix.lower() == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                text = json.dumps(data, indent=2)
                from langchain_core.documents import Document
                raw = [Document(page_content=text, metadata={"source": str(file_path)})]
            else:
                continue

            chunks = splitter.split_documents(raw)
            for chunk in chunks:
                chunk.metadata["source_file"] = file_path.name
                chunk.metadata["file_type"] = file_path.suffix.lower()
            docs.extend(chunks)
            print(f"  Loaded {len(chunks)} chunks from {file_path.name}")
        except Exception as e:
            print(f"  WARNING: Could not load {file_path.name}: {e}")

    return docs


def ingest(data_dir: str = "data", chroma_dir: str = CHROMA_DIR) -> Dict:
    """
    Main ingest entry point. Returns stats dict with counts of added/skipped chunks.
    """
    print(f"[ingest] Loading documents from '{data_dir}'...")
    docs = _load_documents(data_dir)
    if not docs:
        print("[ingest] No documents found.")
        return {"added": 0, "skipped": 0, "total_in_collection": 0}

    collection = _get_chroma_collection(chroma_dir)
    existing_ids: set[str] = set(collection.get(include=[])["ids"])

    texts = [doc.page_content for doc in docs]
    ids = [_content_hash(t) for t in texts]
    metadatas = [doc.metadata for doc in docs]

    new_texts, new_ids, new_metas = [], [], []
    skipped = 0
    for text, doc_id, meta in zip(texts, ids, metadatas):
        if doc_id in existing_ids:
            skipped += 1
        else:
            new_texts.append(text)
            new_ids.append(doc_id)
            new_metas.append(meta)

    if new_texts:
        print(f"[ingest] Embedding {len(new_texts)} new chunks (skipping {skipped} unchanged)...")
        embeddings = embed_texts(new_texts)
        collection.upsert(
            ids=new_ids,
            embeddings=embeddings,
            documents=new_texts,
            metadatas=new_metas,
        )
        print(f"[ingest] Done. Added {len(new_texts)} chunks.")
    else:
        print(f"[ingest] No new chunks to add (all {skipped} already indexed).")

    total = collection.count()

    # Write ingest metadata for admin panel
    meta_path = Path(chroma_dir) / "ingest_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump({
            "last_ingest_ts": time.time(),
            "last_ingest_human": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_chunks": total,
            "last_added": len(new_texts),
            "last_skipped": skipped,
        }, f, indent=2)

    return {"added": len(new_texts), "skipped": skipped, "total_in_collection": total}

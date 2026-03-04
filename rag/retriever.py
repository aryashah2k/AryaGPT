"""Retrieval with cross-encoder reranking over ChromaDB."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List

import chromadb

from rag.embeddings import embed_query, get_rerank_model

CHROMA_DIR = os.environ.get("CHROMA_DIR", str(Path(__file__).parent.parent / "chroma_db"))
COLLECTION_NAME = "aryagpt_kb"

TOP_K_RETRIEVE = 12
TOP_K_RERANK = 4


def _get_collection(chroma_dir: str = CHROMA_DIR) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )


def retrieve(
    query: str,
    top_k: int = TOP_K_RERANK,
    chroma_dir: str = CHROMA_DIR,
) -> dict:
    """
    Retrieve relevant chunks for a query.

    Returns:
        {
          "chunks": [{"text": str, "source": str, "score": float}, ...],
          "latency_ms": float,
          "retrieved_count": int,
          "reranked_count": int,
        }
    """
    t0 = time.perf_counter()

    collection = _get_collection(chroma_dir)
    if collection.count() == 0:
        return {"chunks": [], "latency_ms": 0, "retrieved_count": 0, "reranked_count": 0}

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K_RETRIEVE, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    raw_docs = results["documents"][0]
    raw_metas = results["metadatas"][0]
    raw_distances = results["distances"][0]

    if not raw_docs:
        return {"chunks": [], "latency_ms": 0, "retrieved_count": 0, "reranked_count": 0}

    # Cross-encoder reranking
    reranker = get_rerank_model()
    pairs = [[query, doc] for doc in raw_docs]
    rerank_scores = reranker.predict(pairs).tolist()

    combined = sorted(
        zip(raw_docs, raw_metas, raw_distances, rerank_scores),
        key=lambda x: x[3],
        reverse=True,
    )[:top_k]

    chunks = [
        {
            "text": doc,
            "source": meta.get("source_file", meta.get("source", "unknown")),
            "cosine_distance": dist,
            "rerank_score": score,
        }
        for doc, meta, dist, score in combined
    ]

    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "chunks": chunks,
        "latency_ms": latency_ms,
        "retrieved_count": len(raw_docs),
        "reranked_count": len(chunks),
    }


def format_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a readable context string for the LLM."""
    if not chunks:
        return ""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)

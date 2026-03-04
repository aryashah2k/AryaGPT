"""Singleton embedding model using local sentence-transformers (no API key needed)."""

from __future__ import annotations

import threading
from typing import List
from sentence_transformers import SentenceTransformer, CrossEncoder

_embed_lock = threading.Lock()
_rerank_lock = threading.Lock()

_embed_model: SentenceTransformer | None = None
_rerank_model: CrossEncoder | None = None

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_embedding_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        with _embed_lock:
            if _embed_model is None:
                _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def get_rerank_model() -> CrossEncoder:
    global _rerank_model
    if _rerank_model is None:
        with _rerank_lock:
            if _rerank_model is None:
                _rerank_model = CrossEncoder(RERANK_MODEL_NAME)
    return _rerank_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True).tolist()


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]

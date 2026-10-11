"""Embeddings via Ollama (nomic-embed-text). Applies the model's task prefixes."""
from __future__ import annotations

import ollama

from rag.config import Embedding


def _embed(texts: list[str], model: str) -> list[list[float]]:
    return ollama.embed(model=model, input=texts)["embeddings"]


def embed_documents(texts: list[str], cfg: Embedding) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), cfg.batch_size):
        batch = [cfg.doc_prefix + t for t in texts[i : i + cfg.batch_size]]
        out.extend(_embed(batch, cfg.model))
    return out


def embed_query(text: str, cfg: Embedding) -> list[float]:
    return _embed([cfg.query_prefix + text], cfg.model)[0]

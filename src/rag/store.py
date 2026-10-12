"""LanceDB table holding chunks: vector + full-text index + metadata."""
from __future__ import annotations

import re

import lancedb

from rag.config import Settings
from rag.embed import embed_documents
from rag.ingest.chunk import Chunk


def connect(settings: Settings):
    path = settings.paths.resolve("lancedb_dir")
    path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(path))


def open_table(settings: Settings):
    return connect(settings).open_table(settings.paths.table_name)


def build_table(chunks: list[Chunk], settings: Settings, table_name: str | None = None):
    """(Re)create the table from chunks, embed them, and build the FTS index."""
    db = connect(settings)
    name = table_name or settings.paths.table_name
    cache = settings.paths.resolve("processed_dir") / "embed_cache" / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', settings.embedding.model)}.jsonl"
    vectors = embed_documents([c.text for c in chunks], settings.embedding, cache)
    rows = [{**c.as_dict(), "vector": v} for c, v in zip(chunks, vectors)]
    table = db.create_table(name, data=rows, mode="overwrite")
    table.create_fts_index("text", replace=True)
    return table

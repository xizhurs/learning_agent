"""Build / reuse LanceDB tables for a given settings object (used by ingest CLI and ablations)."""

from __future__ import annotations

from rag.config import Settings
from rag.ingest.chunk import Chunk, chunk_document
from rag.ingest.parse import list_documents, parse_document
from rag.store import build_table, connect


def chunks_for(settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in list_documents(settings):
        chunks.extend(chunk_document(parse_document(path, settings), settings.chunking))
    return chunks


def ensure_table(settings: Settings):
    """Open the table named in settings, building it first if it does not exist."""
    db = connect(settings)
    name = settings.paths.table_name
    existing = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
    names = getattr(existing, "tables", existing)
    if name in names:
        return db.open_table(name)
    return build_table(chunks_for(settings), settings, name)

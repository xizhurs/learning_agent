"""python -m rag.ingest [--reparse]  : parse -> chunk -> embed -> LanceDB, using config.yaml."""

from __future__ import annotations

import argparse
from collections import Counter

from rag.config import get_settings
from rag.ingest.chunk import chunk_document, count_tokens
from rag.ingest.parse import list_documents, parse_document
from rag.store import build_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reparse", action="store_true", help="ignore cached Docling JSON")
    args = ap.parse_args()

    settings = get_settings()
    chunks = []
    for path in list_documents(settings):
        print(f"{path.name}")
        doc = parse_document(path, settings, force=args.reparse)
        cs = chunk_document(doc, settings.chunking)
        kinds = Counter(c.kind for c in cs)
        toks = [count_tokens(c.text) for c in cs]
        print(
            f"  {len(cs)} chunks {dict(kinds)}, tokens min/avg/max = "
            f"{min(toks)}/{sum(toks) // len(toks)}/{max(toks)}"
        )
        chunks.extend(cs)

    print(f"embedding {len(chunks)} chunks with {settings.embedding.model} ...")
    table = build_table(chunks, settings)
    print(f"table '{settings.paths.table_name}' has {table.count_rows()} rows")


if __name__ == "__main__":
    main()

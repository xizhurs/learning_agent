"""Embeddings via Ollama (nomic-embed-text). Applies the model's task prefixes.

Document embedding survives a crashing Ollama runner (seen as 'POST /tokenize: EOF' after ~40 min of ingest): a failed request is
retried, a batch that keeps failing is embedded one text at a time, and with a cache file every batch is appended as soon as it is
embedded, so a rerun after a crash or Ctrl+C only embeds what is missing."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx
import ollama

from rag.config import Embedding

RETRY_WAITS = (2, 5, 10)  # seconds between attempts; Ollama restarts a crashed runner by itself
PROGRESS_EVERY = 10  # batches between progress lines
ERRORS = (ollama.ResponseError, httpx.HTTPError, ConnectionError)


def _embed(texts: list[str], model: str) -> list[list[float]]:
    return ollama.embed(model=model, input=texts)["embeddings"]


def _embed_with_retry(texts: list[str], model: str) -> list[list[float]]:
    for wait in (*RETRY_WAITS, None):
        try:
            return _embed(texts, model)
        except ERRORS as e:
            if wait is None:
                raise
            print(f"  embedding request failed ({str(e)[:200]}); retrying in {wait}s", flush=True)
            time.sleep(wait)


def _embed_batch(texts: list[str], model: str) -> list[list[float]]:
    """One batch; if it keeps failing, its texts one by one, so a single bad text is named instead of failing the whole run."""
    try:
        return _embed_with_retry(texts, model)
    except ERRORS:
        if len(texts) == 1:
            raise RuntimeError(f"embedding keeps failing for this text: {texts[0][:200]!r}")
        print(f"  batch keeps failing; embedding its {len(texts)} texts one at a time", flush=True)
    out = []
    for t in texts:
        try:
            out.extend(_embed_with_retry([t], model))
        except ERRORS as e:
            raise RuntimeError(f"embedding keeps failing for this text: {t[:200]!r}") from e
    return out


def _key(model: str, text: str) -> str:
    return hashlib.sha1(f"{model}\n{text}".encode()).hexdigest()


def _load_cache(path: Path) -> dict[str, list[float]]:
    cache = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # a line cut off by a crash
                continue
            cache[row["k"]] = row["v"]
    return cache


def embed_documents(texts: list[str], cfg: Embedding, cache_path: Path | None = None) -> list[list[float]]:
    """Embed document texts (with cfg.doc_prefix). With cache_path (a JSONL file keyed by model + text), texts embedded by an
    earlier, possibly crashed, run are reused and each new batch is appended as soon as it is done."""
    keys = [_key(cfg.model, cfg.doc_prefix + t) for t in texts]
    cache = _load_cache(cache_path) if cache_path else {}
    todo = {k: cfg.doc_prefix + t for k, t in zip(keys, texts) if k not in cache}
    missing = list(todo)
    if cache_path:
        print(
            f"  embeddings: {len(texts) - sum(k in todo for k in keys)} of {len(texts)} reused from {cache_path.name}, "
            f"{len(missing)} to embed",
            flush=True,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and cache_path.stat().st_size and not cache_path.read_bytes().endswith(b"\n"):
            with open(cache_path, "a", encoding="utf-8") as f:  # end a line cut off by a crash before appending
                f.write("\n")
    n_batches = -(-len(missing) // cfg.batch_size)
    t0 = time.time()
    out_file = open(cache_path, "a", encoding="utf-8") if cache_path and missing else None
    try:
        for b, i in enumerate(range(0, len(missing), cfg.batch_size), 1):
            batch = missing[i : i + cfg.batch_size]
            for k, v in zip(batch, _embed_batch([todo[k] for k in batch], cfg.model)):
                cache[k] = v
                if out_file:
                    out_file.write(json.dumps({"k": k, "v": v}) + "\n")
            if out_file:
                out_file.flush()
            if b % PROGRESS_EVERY == 0 or b == n_batches:
                done, elapsed = min(i + cfg.batch_size, len(missing)), time.time() - t0
                eta = elapsed / done * (len(missing) - done)
                print(
                    f"  embedded {done}/{len(missing)} ({done / elapsed:.1f}/s, ~{eta / 60:.0f} min left)", flush=True
                )
    finally:
        if out_file:
            out_file.close()
    return [cache[k] for k in keys]


def embed_query(text: str, cfg: Embedding) -> list[float]:
    return _embed([cfg.query_prefix + text], cfg.model)[0]

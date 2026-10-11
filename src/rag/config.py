"""Typed configuration loaded from config.yaml, overridable via RAG__SECTION__KEY env vars."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class Paths(BaseModel):
    documents_dir: str = "data/documents"
    processed_dir: str = "data/processed"
    lancedb_dir: str = "data/lancedb"
    table_name: str = "chunks"
    eval_dir: str = "data/eval"

    def resolve(self, name: str) -> Path:
        return PROJECT_ROOT / getattr(self, name)


class Chunking(BaseModel):
    strategy: Literal["heading_aware", "fixed_tokens"] = "heading_aware"
    max_tokens: int = 512
    overlap: int = 64
    table_as_own_chunk: bool = True
    heading_prefix: bool = True


class Embedding(BaseModel):
    model: str = "nomic-embed-text"
    doc_prefix: str = "search_document: "
    query_prefix: str = "search_query: "
    batch_size: int = 32


class Retrieval(BaseModel):
    mode: Literal["vector", "fts", "hybrid"] = "hybrid"
    candidates_k: int = 30
    top_k: int = 5
    source_filter: str | None = None


class Rerank(BaseModel):
    enabled: bool = True
    model: str = "BAAI/bge-reranker-base"
    min_score: float | None = None


class Generation(BaseModel):
    model: str = "llama3.2:3b"
    temperature: float = 0.0
    num_ctx: int = 8192
    max_answer_tokens: int = 400
    repeat_penalty: float = 1.1
    refuse_if_no_context: bool = True
    rewrite_followups: bool = True


class Eval(BaseModel):
    # RAGAS judge. "openai_compatible" = hosted API (Google AI Studio / Gemini by default); the key is read from the env var
    # named in judge_api_key_env and is never stored in config or run snapshots.
    judge_provider: Literal["ollama", "openai_compatible"] = "openai_compatible"
    judge_model: str = "gemini-3.5-flash-lite"
    judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    judge_api_key_env: str = "GEMINI_API_KEY"
    # Privacy: only items from this one document may be sent to a hosted judge (see run_eval.assert_hosted_safe).
    judge_allowed_source: str = "VM0047-v1.1_Final.pdf"
    judge_max_workers: int = 2
    judge_reasoning_effort: str | None = "low"   # e.g. low / none: cuts latency of thinking models (Gemini); null = not sent
    judge_min_interval_s: float = 0.0   # minimum seconds between hosted-judge requests (0 = no throttle)
    hit_ks: list[int] = [1, 3, 5]
    overlap_threshold: float = 0.6


class Tracing(BaseModel):
    # Local Arize Phoenix server (`phoenix serve`, default port 6006). Traces stay on this machine.
    # If the server is not reachable, tracing is silently skipped for that process.
    enabled: bool = True
    endpoint: str = "http://localhost:6006/v1/traces"
    project_name: str = "carbon-rag"


class Settings(BaseModel):
    paths: Paths = Paths()
    chunking: Chunking = Chunking()
    embedding: Embedding = Embedding()
    retrieval: Retrieval = Retrieval()
    rerank: Rerank = Rerank()
    generation: Generation = Generation()
    eval: Eval = Eval()
    tracing: Tracing = Tracing()


def _apply_env_overrides(raw: dict) -> dict:
    """RAG__RETRIEVAL__MODE=vector -> raw['retrieval']['mode'] = 'vector' (pydantic coerces types)."""
    for key, value in os.environ.items():
        if not key.startswith("RAG__"):
            continue
        parts = key[len("RAG__"):].lower().split("__")
        node = raw
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return raw


def load_settings(path: Path | None = None, overrides: dict | None = None) -> Settings:
    raw = yaml.safe_load((path or CONFIG_PATH).read_text(encoding="utf-8")) or {}
    raw = _apply_env_overrides(raw)
    for section, values in (overrides or {}).items():
        raw.setdefault(section, {}).update(values)
    return Settings.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()

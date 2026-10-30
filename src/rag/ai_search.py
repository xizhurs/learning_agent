"""Retrieval through Databricks AI Search (hybrid search, optional reranker), injected into run_eval with set_retriever.

The index is a Delta Sync index over the chunks table (columns id, source, page, section, kind, text). Nothing here imports
databricks.*: the caller passes the index object (`AISearchClient().get_index(...)`) and, for reranking, a reranker object
(`databricks.ai_search.reranker.DatabricksReranker(columns_to_rerank=["text"])`), so the module is testable with a fake index.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from rag.config import Settings

COLUMNS = ["id", "source", "page", "section", "kind", "text"]
RERANK_WARNING = "RERANKER_TEMPORARILY_UNAVAILABLE"


class RerankerUnavailable(RuntimeError):
    """The reranker did not run (or could not be shown to have run), so a 'rerank on' result would be mislabelled."""


@dataclass
class SearchHit:
    """Same fields and citation format as rag.retrieve.Hit (not imported: it pulls in lancedb and docling)."""

    id: str
    source: str
    page: int
    section: str
    kind: str
    text: str
    score: float
    retrieval_score: float

    def citation(self) -> str:
        page = f", p.{self.page}" if self.page else ""
        return f"{self.source}{page} | {self.section or 'n/a'}"


@dataclass
class AISearchRetriever:
    index: object
    rerank: bool
    reranker: object = None
    num_results: int | None = None  # None: settings.retrieval.top_k
    verify_rerank: bool = (
        True  # require `reranker_time` in debug_info so a silent fallback cannot pass as a reranked run
    )
    stats: dict = field(default_factory=lambda: {"calls": 0, "reranker_seconds": 0.0, "response_seconds": 0.0})
    query_type: str = "hybrid"  # AI Search query types: ANN (vector), FULL_TEXT, HYBRID
    llm_reranker: object = (
        None  # rag.llm_rerank.LLMReranker: rerank `candidates_k` hits with a chat LLM instead of `reranker`
    )
    candidates_k: int | None = None  # None: settings.retrieval.candidates_k (only used with llm_reranker)

    def __call__(self, query: str, settings: Settings, table=None, span=None) -> list[SearchHit]:
        k = self.num_results or settings.retrieval.top_k
        n = (self.candidates_k or settings.retrieval.candidates_k) if self.llm_reranker is not None else k
        if n < k:  # fewer candidates than results wanted would silently change what a "top-k" run measures
            raise ValueError(f"candidates_k ({n}) must be at least num_results ({k})")
        kwargs = {"query_text": query, "columns": COLUMNS, "num_results": n, "query_type": self.query_type}
        if self.rerank:
            kwargs.update(reranker=self.reranker, debug_level=1)
        resp = self.index.similarity_search(**kwargs)
        self.stats["calls"] += 1
        debug = resp.get("debug_info") or {}
        if self.rerank:
            if RERANK_WARNING in json.dumps(debug):
                raise RerankerUnavailable(f"{RERANK_WARNING} for query {query[:80]!r}: {debug}")
            if self.verify_rerank and "reranker_time" not in debug:
                raise RerankerUnavailable(f"no reranker_time in debug_info for query {query[:80]!r}: {debug}")
            self.stats["reranker_seconds"] += float(debug.get("reranker_time") or 0.0)
        self.stats["response_seconds"] += float(debug.get("response_time") or 0.0)
        hits = _hits(resp, n)
        if self.llm_reranker is not None:
            t0 = time.perf_counter()
            hits = self.llm_reranker.rerank(query, hits)
            self.stats["reranker_seconds"] += time.perf_counter() - t0
        return hits[:k]


def _hits(resp: dict, k: int) -> list[SearchHit]:
    names = [c["name"] for c in resp["manifest"]["columns"]]
    rows = (resp.get("result") or {}).get("data_array") or []
    hits = []
    for rank, row in enumerate(rows[:k]):
        r = dict(zip(names, row, strict=False))
        score = float(r["score"]) if r.get("score") is not None else float(len(rows) - rank)  # no score: keep the order
        hits.append(
            SearchHit(
                id=str(r["id"]),
                source=r["source"],
                page=int(r["page"] or 0),
                section=r["section"] or "",
                kind=r["kind"],
                text=r["text"],
                score=score,
                retrieval_score=score,
            )
        )
    return hits


def make_retriever(
    index,
    *,
    rerank: bool,
    reranker=None,
    num_results: int | None = None,
    verify_rerank: bool = True,
    query_type: str = "hybrid",
    llm_reranker=None,
    candidates_k: int | None = None,
):
    """A retrieve function for run_eval.set_retriever: hybrid search (or `query_type`), reranked when `rerank` (then
    `reranker` is required) or, instead, by `llm_reranker` over `candidates_k` hits. The two rerankers exclude each other.
    """
    if rerank and reranker is None:
        raise ValueError("rerank=True needs a reranker, e.g. DatabricksReranker(columns_to_rerank=['text'])")
    if rerank and llm_reranker is not None:
        raise ValueError("use either the AI Search reranker (rerank=True) or llm_reranker, not both")
    if llm_reranker is not None and num_results and candidates_k and candidates_k < num_results:
        raise ValueError(f"candidates_k ({candidates_k}) must be at least num_results ({num_results})")
    return AISearchRetriever(
        index,
        rerank,
        reranker,
        num_results,
        verify_rerank,
        query_type=query_type,
        llm_reranker=llm_reranker,
        candidates_k=candidates_k,
    )

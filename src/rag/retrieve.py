"""Retrieval: vector | fts | hybrid (RRF fusion in LanceDB) -> optional cross-encoder rerank."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from lancedb.rerankers import RRFReranker

from rag.config import Settings
from rag.embed import embed_query
from rag.store import open_table
from rag.tracing import Span, child_of


@dataclass
class Hit:
    id: str
    source: str
    page: int
    section: str
    kind: str
    text: str
    score: float  # final score: rerank probability if reranked, else retrieval score
    retrieval_score: float

    def citation(self) -> str:
        page = f", p.{self.page}" if self.page else ""
        return f"{self.source}{page} | {self.section or 'n/a'}"


@lru_cache(maxsize=2)
def _cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu")


def _fts_safe(query: str) -> str:
    """Strip characters that the FTS query parser treats as syntax."""
    return re.sub(r"[^\w\s.\-%]", " ", query).strip()


def _embed(query: str, settings: Settings, span: Span | None):
    step = child_of(
        span, "embed_query", "embedding", input=query, **{"embedding__model_name": settings.embedding.model}
    )
    vec = embed_query(query, settings.embedding)
    step.end()
    return vec


def _search(table, query: str, settings: Settings, span: Span | None = None):
    r = settings.retrieval
    k = r.candidates_k
    if r.mode == "vector":
        q = table.search(_embed(query, settings, span), query_type="vector").limit(k)
        score_col, higher_better = "_distance", False
    elif r.mode == "fts":
        q = table.search(_fts_safe(query), query_type="fts").limit(k)
        score_col, higher_better = "_score", True
    else:
        q = (
            table.search(query_type="hybrid")
            .vector(_embed(query, settings, span))
            .text(_fts_safe(query))
            .rerank(reranker=RRFReranker())
            .limit(k)
        )
        score_col, higher_better = "_relevance_score", True
    if r.source_filter:
        escaped = r.source_filter.replace("'", "''")
        q = q.where(f"source = '{escaped}'", prefilter=True)
    step = child_of(
        span,
        "search",
        "chain",
        input=_fts_safe(query) if r.mode == "fts" else query,
        **{"retrieval__mode": r.mode, "retrieval__candidates_k": k, "retrieval__source_filter": r.source_filter},
    )
    rows = q.to_list()
    step.end(output=f"{len(rows)} candidates")
    return rows, score_col, higher_better


def retrieve(query: str, settings: Settings, table=None, span: Span | None = None) -> list[Hit]:
    """Top-k chunks for the query. With `span` (a trace root/parent) the steps are traced and timed under it."""
    step = child_of(
        span,
        "retrieve",
        "retriever",
        input=query,
        **{
            "retrieval__mode": settings.retrieval.mode,
            "retrieval__top_k": settings.retrieval.top_k,
            "rerank__enabled": settings.rerank.enabled,
        },
    )
    table = table if table is not None else open_table(settings)
    rows, score_col, higher_better = _search(table, query, settings, step)

    hits = []
    for row in rows:
        raw = float(row.get(score_col, 0.0))
        hits.append(
            Hit(
                id=row["id"],
                source=row["source"],
                page=row["page"],
                section=row["section"],
                kind=row["kind"],
                text=row["text"],
                score=raw if higher_better else -raw,
                retrieval_score=raw if higher_better else -raw,
            )
        )

    rr = settings.rerank
    if rr.enabled and hits:
        rstep = child_of(step, "rerank", "reranker", input=query, **{"reranker__model_name": rr.model})
        ce = _cross_encoder(rr.model)
        scores = ce.predict([(query, h.text) for h in hits], activation_fn=_sigmoid())
        for h, s in zip(hits, scores):
            h.score = float(s)
        hits.sort(key=lambda h: h.score, reverse=True)
        if rr.min_score is not None:
            hits = [h for h in hits if h.score >= rr.min_score]
        rstep.end()
    hits = hits[: settings.retrieval.top_k]
    step.set_documents(hits)
    step.end(output=f"{len(hits)} chunks")
    return hits


def _sigmoid():
    import torch

    return torch.nn.Sigmoid()

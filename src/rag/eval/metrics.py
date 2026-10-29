"""Deterministic retrieval / refusal metrics (no LLM calls)."""

from __future__ import annotations

import re
from collections import Counter

from rag.generate import NOT_FOUND


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())


def _shingles(words: list[str], n: int = 3) -> set[tuple[str, ...]]:
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def quote_coverage(quote: str, chunk_text: str) -> float:
    """Best coverage over the quote's ' ... ' segments: fraction of its word 3-grams found in the chunk.

    Matching the *quote* (the sentences that actually support the answer) instead of a chunk id keeps the
    metric valid when chunk size / strategy changes between runs.
    """
    chunk_sh = _shingles(_words(chunk_text))
    best = 0.0
    for seg in re.split(r"\s*(?:\.\.\.|…)\s*", quote):
        sh = _shingles(_words(seg))
        if len(sh) < 2:
            continue
        best = max(best, len(sh & chunk_sh) / len(sh))
    return best


def first_hit_rank(quote: str, chunk_texts: list[str], threshold: float) -> int | None:
    """1-based rank of the first retrieved chunk that covers the supporting quote, else None."""
    for i, text in enumerate(chunk_texts, 1):
        if quote_coverage(quote, text) >= threshold:
            return i
    return None


def evidence_segments(quote: str) -> list[str]:
    """The evidence paragraphs of a reference quote (create_qasper_tiny joins them with ' ... '); too-short ones can't be scored."""
    return [s for s in re.split(r"\s+(?:\.\.\.|…)\s+", quote) if len(_shingles(_words(s))) >= 2]


def evidence_recall(quote: str, chunk_texts: list[str], threshold: float) -> float | None:
    """Share of the quote's evidence paragraphs covered by at least one of the chunks; None if nothing is scorable."""
    segs = evidence_segments(quote)
    if not segs:
        return None
    return sum(any(quote_coverage(s, t) >= threshold for t in chunk_texts) for s in segs) / len(segs)


def context_precision(quote: str, chunk_texts: list[str], threshold: float) -> float:
    """Rank-weighted precision of the retrieved chunks, as RAGAS context precision but with 'covers an evidence paragraph' as the
    relevance verdict instead of an LLM's: mean of precision@i over the ranks i holding a relevant chunk; 0 if none is relevant.
    """
    hits, total = 0, 0.0
    for i, text in enumerate(chunk_texts, 1):
        if quote_coverage(quote, text) >= threshold:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def _answer_tokens(text: str) -> list[str]:
    """QASPER / SQuAD answer normalisation: lower-case, drop citation markers like [1], punctuation and articles."""
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", " ", text.lower())
    return [w for w in re.sub(r"[^\w\s]", " ", text).split() if w not in ("a", "an", "the")]


def answer_overlap(prediction: str, reference: str) -> tuple[float, float]:
    """(token F1, token recall) of a generated answer against the reference answer. F1 is QASPER's official Answer-F1; recall is
    kinder to long generated answers that contain a short reference span."""
    pred, ref = Counter(_answer_tokens(prediction)), Counter(_answer_tokens(reference))
    common = sum((pred & ref).values())
    if not common:
        return 0.0, 0.0
    p, r = common / sum(pred.values()), common / sum(ref.values())
    return 2 * p * r / (p + r), r


def first_doc_rank(gold_ids: list[str], chunk_ids: list[str]) -> int | None:
    """1-based rank of the first retrieved chunk that comes from a gold document (chunk ids are '<document>::NNNNN').

    A document-level hit does not depend on chunk size or on how much of the answer a chunk covers."""
    gold = set(gold_ids)
    for i, cid in enumerate(chunk_ids, 1):
        if cid.split("::")[0] in gold:
            return i
    return None


def retrieval_summary(ranks: list[int | None], ks: list[int]) -> dict:
    n = len(ranks)
    out = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in ks} if n else {}
    out["mrr"] = sum(1 / r for r in ranks if r) / n if n else 0.0
    out["n_answerable"] = n
    return out


def is_refusal(answer: str) -> bool:
    return NOT_FOUND.lower().rstrip(".") in answer.lower()


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation (average ranks for ties); None if undefined."""

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return out

    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx, vy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5 if vx and vy else None


def latency_summary(timings: list[dict]) -> dict:
    """Median / p95 / max seconds per step over rows' `timings` dicts: {"generate": {"median": .., "p95": .., "max": .., "n": ..}}."""
    steps: dict[str, list[float]] = {}
    for tm in timings:
        for k, v in (tm or {}).items():
            steps.setdefault(k, []).append(float(v))
    out = {}
    for k, vals in sorted(steps.items()):
        vals.sort()
        n = len(vals)
        median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        out[k] = {
            "median": round(median, 3),
            "p95": round(vals[min(n - 1, int(0.95 * n))], 3),
            "max": round(vals[-1], 3),
            "n": n,
        }
    return out

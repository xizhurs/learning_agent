"""Listwise reranking with a chat LLM behind an OpenAI-compatible client (e.g. a Databricks Foundation Model endpoint).

One call per query: the candidates are shown as numbered passages and the model returns the numbers, most relevant first.
The client is injected (same shape as `rag.generate.set_chat_client`), so nothing here imports databricks.*.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, replace

from rag.ai_search import RerankerUnavailable

PROMPT = """Rank the numbered passages by how useful they are for answering the question. A passage that states the answer
outranks one that only discusses the same topic.
Reply with JSON only, in this form: {{"ranking": [3, 1, 2]}} (passage numbers, most useful first, each number at most once).

Question: {question}

Passages:

{passages}"""


@dataclass
class LLMReranker:
    client: object
    model: str
    max_chars: int = 1500  # characters of each passage shown to the model (a chunk can hold ~4,000)
    stats: dict = field(default_factory=lambda: {"calls": 0, "seconds": 0.0})

    def rerank(self, query: str, hits: list) -> list:
        """`hits` reordered, most relevant first. Hits the model leaves out follow in their original order.

        `retrieval_score` is kept; `score` becomes 1 / rank (an order, not a probability). Raises RerankerUnavailable
        when the reply holds no usable ranking, so a failed rerank is never reported as a reranked result."""
        if len(hits) < 2:
            return list(hits)
        passages = "\n\n".join(f"[{i}] {h.text[: self.max_chars]}" for i, h in enumerate(hits, 1))
        messages = [{"role": "user", "content": PROMPT.format(question=query, passages=passages)}]
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.0, max_tokens=8 * len(hits) + 64
            )
        finally:  # a failed or timed-out call still cost a request and time
            self.stats["calls"] += 1
            self.stats["seconds"] += time.perf_counter() - t0
        order = parse_ranking((resp.choices[0].message.content or ""), len(hits))
        if not order:
            raise RerankerUnavailable(f"no usable ranking in the reply of {self.model} for query {query[:80]!r}")
        ranked = set(order)
        rest = [i for i in range(len(hits)) if i not in ranked]
        return [replace(hits[i], score=1.0 / rank) for rank, i in enumerate(order + rest, 1)]


def parse_ranking(text: str, n: int) -> list[int]:
    """0-based indexes from a reply like {"ranking": [3, 1]}; unknown and repeated numbers are dropped; [] if none found.

    Takes the first JSON object with a "ranking" list anywhere in the text (code fences and sentences around it are fine);
    numbers may be ints, whole floats (3.0) or digit strings ("3")."""
    raw = None
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[m.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("ranking"), list):
            raw = obj["ranking"]
            break
    order: list[int] = []
    for v in raw or []:
        i = _as_int(v)
        if i is not None and 1 <= i <= n and i - 1 not in order:
            order.append(i - 1)
    return order


def _as_int(v) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None

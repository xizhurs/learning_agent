"""Chatbot core for Databricks: AI Search retrieval (fts | vector | hybrid) -> optional rerank -> grounded answer.

Same prompt, citation format and abstention rule as the local pipeline (rag.generate), but the index, the chat client and the
rerankers are injected, so this imports neither databricks.* nor lancedb and runs on serverless. Retrieval mode and
reranking are chosen per request, so one instance serves every combination.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from rag.ai_search import SearchHit, make_retriever
from rag.config import Settings
from rag.generate import NOT_FOUND, build_messages, generate, rewrite_question, set_chat_client

log = logging.getLogger("rag.chat")

# retrieval mode -> AI Search query_type
RETRIEVAL_MODES = {"fts": "FULL_TEXT", "vector": "ANN", "hybrid": "HYBRID"}
RERANK_MODES = ("off", "ai_search", "llm")  # none | the index's built-in reranker | a chat LLM (rag.llm_rerank)


@dataclass
class ChatAnswer:
    question: str
    standalone_question: str
    answer: str
    hits: list[SearchHit] = field(default_factory=list)
    retrieval_mode: str = "hybrid"
    rerank: str = "off"  # what actually ran: "off" if a requested reranker failed (see warnings)
    warnings: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)  # seconds: rewrite, retrieve, generate, total

    def sources(self) -> list[str]:
        return [f"[{i}] {h.citation()}" for i, h in enumerate(self.hits, 1)]


class Chatbot:
    def __init__(
        self,
        index,
        chat_client,
        settings: Settings | None = None,
        *,
        reranker=None,
        llm_reranker=None,
        retrieval_mode: str = "hybrid",
        rerank: str = "off",
    ):
        """`index`: AISearchClient().get_index(...). `chat_client`: an OpenAI-compatible client (DatabricksOpenAI()); the
        answer model is settings.generation.model. `reranker`: DatabricksReranker(columns_to_rerank=["text"]) for
        rerank="ai_search"; `llm_reranker`: rag.llm_rerank.LLMReranker for rerank="llm". The last two args are the defaults.
        """
        _check(retrieval_mode, rerank)
        self.index, self.settings = index, settings or Settings()
        self.reranker, self.llm_reranker = reranker, llm_reranker
        self.retrieval_mode, self.rerank = retrieval_mode, rerank
        set_chat_client(chat_client)

    def _retriever(self, mode: str, rerank: str):
        s = self.settings.retrieval
        common = {"num_results": s.top_k, "candidates_k": s.candidates_k, "query_type": RETRIEVAL_MODES[mode]}
        if rerank == "ai_search":
            if self.reranker is None:
                raise ValueError("rerank='ai_search' needs a reranker")
            return make_retriever(self.index, rerank=True, reranker=self.reranker, **common)
        if rerank == "llm":
            if self.llm_reranker is None:
                raise ValueError("rerank='llm' needs an llm_reranker")
            return make_retriever(self.index, rerank=False, llm_reranker=self.llm_reranker, **common)
        return make_retriever(self.index, rerank=False, **common)

    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        *,
        retrieval_mode: str | None = None,
        rerank: str | None = None,
    ) -> ChatAnswer:
        """Answer one question. `history`: earlier turns as [{"role": "user"|"assistant", "content": ...}]."""
        mode, rr = retrieval_mode or self.retrieval_mode, rerank or self.rerank
        _check(mode, rr)
        question = question.strip()
        if not question:
            raise ValueError("empty question")
        retriever = self._retriever(
            mode, rr
        )  # configuration errors (e.g. no reranker given) raise here, not in the fallback
        t0 = time.perf_counter()
        timings: dict = {}
        gen = self.settings.generation

        standalone = question
        if history and gen.rewrite_followups:
            standalone = rewrite_question(question, history, gen)
            timings["rewrite"] = time.perf_counter() - t0

        warnings: list[str] = []
        t1 = time.perf_counter()
        try:
            hits = retriever(standalone, self.settings)
        except Exception as e:  # a failed reranker must not take the chatbot down: answer without it, and say so
            if rr == "off":
                raise
            log.warning("rerank=%s failed (%s: %s); retrying without it", rr, type(e).__name__, e)
            warnings.append(f"rerank '{rr}' failed ({type(e).__name__}); answered without reranking")
            rr = "off"
            hits = self._retriever(mode, rr)(standalone, self.settings)
        timings["retrieve"] = time.perf_counter() - t1

        if not hits and gen.refuse_if_no_context:
            text = NOT_FOUND
        else:
            t2 = time.perf_counter()
            text = generate(build_messages(standalone, hits), gen)
            timings["generate"] = time.perf_counter() - t2
        timings["total"] = time.perf_counter() - t0
        log.info("q=%r standalone=%r mode=%s rerank=%s hits=%s", question, standalone, mode, rr, [h.id for h in hits])
        return ChatAnswer(question, standalone, text, hits, mode, rr, warnings, timings)


def _check(mode: str, rerank: str) -> None:
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"retrieval_mode must be one of {sorted(RETRIEVAL_MODES)}, got {mode!r}")
    if rerank not in RERANK_MODES:
        raise ValueError(f"rerank must be one of {list(RERANK_MODES)}, got {rerank!r}")

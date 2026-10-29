"""End-to-end RAG pipeline used by the CLI, the API and the evaluation harness."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from rag.config import Settings, get_settings
from rag.generate import NOT_FOUND, build_messages, generate, generate_stream, rewrite_question
from rag.retrieve import Hit, retrieve
from rag.store import open_table
from rag.tracing import start_trace

log = logging.getLogger("rag")


@dataclass
class Answer:
    question: str
    standalone_question: str
    answer: str
    hits: list[Hit] = field(default_factory=list)
    trace_id: str | None = None
    span_id: str | None = None
    timings: dict = field(
        default_factory=dict
    )  # seconds per step, e.g. {"retrieve": 0.05, "generate": 31.2, "total": 31.3}


class Pipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._table = None

    @property
    def table(self):
        if self._table is None:
            self._table = open_table(self.settings)
        return self._table

    def _root(self, question: str, history: list[dict] | None, session_id: str | None):
        s = self.settings
        return start_trace(
            s,
            "rag.query",
            input=question,
            kind="chain",
            session_id=session_id,
            **{
                "rag__generation_model": s.generation.model,
                "rag__chunk_max_tokens": s.chunking.max_tokens,
                "retrieval__mode": s.retrieval.mode,
                "retrieval__top_k": s.retrieval.top_k,
                "retrieval__source_filter": s.retrieval.source_filter,
                "rerank__enabled": s.rerank.enabled,
                "rag__history_turns": len(history or []),
                "rag__kind": "chat",
            },
        )

    def _prepare(self, question: str, history: list[dict] | None, root):
        gen = self.settings.generation
        standalone = rewrite_question(question, history, gen, root) if (history and gen.rewrite_followups) else question
        hits = retrieve(standalone, self.settings, self.table, root)
        log.info("q=%r standalone=%r hits=%s", question, standalone, [(h.id, round(h.score, 3)) for h in hits])
        return standalone, hits

    def ask(self, question: str, history: list[dict] | None = None, session_id: str | None = None) -> Answer:
        root = self._root(question, history, session_id)
        try:
            standalone, hits = self._prepare(question, history, root)
            if not hits and self.settings.generation.refuse_if_no_context:
                text = NOT_FOUND
            else:
                text = generate(build_messages(standalone, hits), self.settings.generation, root)
        except BaseException as e:
            root.fail(e)
            root.end()
            raise
        root.end(output=text, rag__standalone_question=standalone)
        return Answer(question, standalone, text, hits, root.trace_id, root.span_id, dict(root.timings))

    def stream(self, question: str, history: list[dict] | None = None, session_id: str | None = None):
        """Returns (hits, token iterator, root span) so the UI can show sources while the answer streams.

        The trace is finished when the iterator is exhausted or closed; read `root.trace_id` for the link."""
        root = self._root(question, history, session_id)
        try:
            standalone, hits = self._prepare(question, history, root)
        except BaseException as e:
            root.fail(e)
            root.end()
            raise
        if not hits and self.settings.generation.refuse_if_no_context:
            root.end(output=NOT_FOUND, rag__standalone_question=standalone)
            return hits, iter([NOT_FOUND]), root

        def tokens() -> Iterator[str]:
            parts: list[str] = []
            try:
                for tok in generate_stream(build_messages(standalone, hits), self.settings.generation, root):
                    parts.append(tok)
                    yield tok
            except BaseException as e:  # noqa: BLE001
                if not isinstance(e, GeneratorExit):
                    root.fail(e)
                raise
            finally:
                root.end(output="".join(parts).strip(), rag__standalone_question=standalone)

        return hits, tokens(), root

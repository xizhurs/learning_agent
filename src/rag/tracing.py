"""Tracing to a local Arize Phoenix server (OpenTelemetry + OpenInference attributes).

Everything here is best-effort: if tracing is disabled, Phoenix is not running, or any tracing call fails, the pipeline
carries on and only the local per-step timings are kept. Spans are explicit objects (not contextvars) because the API
streams answers across executor threads.

    root = start_trace(settings, "rag.query", input=question, session_id=..., **{"retrieval.mode": "fts"})
    step = root.child("retrieve", "retriever", input=query)
    ...
    step.end(output=...)
    root.end(output=answer)          # root.timings -> {"retrieve": 0.04, ..., "total": 41.2}
"""

from __future__ import annotations

import json
import logging
import socket
import time
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger("rag.tracing")

_tracer = None
_provider = None
_state = "uninit"  # uninit | on | off


def _install(tracer, provider=None) -> None:
    """Use this tracer (tests pass one backed by an in-memory exporter)."""
    global _tracer, _provider, _state
    _tracer, _provider, _state = tracer, provider, "on"


def _reset() -> None:
    global _tracer, _provider, _state
    _tracer, _provider, _state = None, None, "uninit"


def _init(settings) -> None:
    global _tracer, _provider, _state
    if _state != "uninit":
        return
    _state = "off"
    cfg = settings.tracing
    if not cfg.enabled:
        return
    try:
        u = urlparse(cfg.endpoint)
        socket.create_connection((u.hostname, u.port or 80), timeout=0.5).close()
    except OSError:
        log.info(
            "Phoenix not reachable at %s; tracing is off for this process (start it with `phoenix serve`)", cfg.endpoint
        )
        return
    try:
        from phoenix.otel import register

        _provider = register(
            endpoint=cfg.endpoint,
            project_name=cfg.project_name,
            protocol="http/protobuf",
            batch=True,
            set_global_tracer_provider=False,
            verbose=False,
        )
        _tracer = _provider.get_tracer("rag")
        _state = "on"
    except Exception as e:  # noqa: BLE001
        log.warning("tracing setup failed (%s); continuing without it", e)


def enabled() -> bool:
    return _state == "on"


def _jsonable(v: Any) -> Any:
    return v if isinstance(v, (str, int, float, bool)) else json.dumps(v, ensure_ascii=False, default=str)


class Span:
    def __init__(
        self,
        name: str,
        kind: str,
        parent: Span | None = None,
        input: Any = None,
        session_id: str | None = None,
        attrs: dict | None = None,
    ):
        self.name = name
        self.t0 = time.perf_counter()
        self.seconds: float | None = None
        self.root = parent.root if parent else self
        self.timings: dict[str, float] = {}
        self.session_id = session_id or (parent.session_id if parent else None)
        self._span = None
        self._ended = False
        if _tracer is None:
            return
        try:
            from opentelemetry.trace import set_span_in_context

            ctx = set_span_in_context(parent._span) if parent is not None and parent._span is not None else None
            a = {"session.id": self.session_id} if self.session_id else {}
            self._span = _tracer.start_span(name, context=ctx, openinference_span_kind=kind, attributes=a)
            if input is not None:
                self._span.set_input(input)
            if attrs:
                self.set(**attrs)
        except Exception as e:  # noqa: BLE001
            log.warning("tracing: could not start span %s (%s)", name, e)
            self._span = None

    # ---- ids (to find the trace later) -------------------------------------------------------
    @property
    def trace_id(self) -> str | None:
        return format(self._span.get_span_context().trace_id, "032x") if self._span is not None else None

    @property
    def span_id(self) -> str | None:
        return format(self._span.get_span_context().span_id, "016x") if self._span is not None else None

    # ---- building the tree ---------------------------------------------------------------------
    def child(self, name: str, kind: str = "chain", input: Any = None, **attrs) -> Span:
        return Span(name, kind, parent=self, input=input, attrs=attrs)

    def set(self, **attrs) -> None:
        """Set attributes; keys with '__' are written with '.' (so eval__rank -> eval.rank)."""
        if self._span is None:
            return
        try:
            self._span.set_attributes({k.replace("__", "."): _jsonable(v) for k, v in attrs.items() if v is not None})
        except Exception as e:  # noqa: BLE001
            log.warning("tracing: set failed on %s (%s)", self.name, e)

    def set_documents(self, hits) -> None:
        """Retrieved chunks as OpenInference retrieval documents (id, content, score, metadata)."""
        attrs: dict[str, Any] = {}
        for i, h in enumerate(hits):
            p = f"retrieval.documents.{i}.document"
            attrs[f"{p}.id"] = h.id
            attrs[f"{p}.content"] = h.text
            attrs[f"{p}.score"] = float(h.score)
            attrs[f"{p}.metadata"] = json.dumps(
                {"source": h.source, "page": h.page, "section": h.section, "kind": h.kind}
            )
        self.set(**{k.replace(".", "__"): v for k, v in attrs.items()})

    def set_llm(self, model: str, messages: list[dict], output: str | None = None, params: dict | None = None) -> None:
        attrs: dict[str, Any] = {"llm.model_name": model}
        if params:
            attrs["llm.invocation_parameters"] = json.dumps(params)
        for i, m in enumerate(messages):
            attrs[f"llm.input_messages.{i}.message.role"] = m["role"]
            attrs[f"llm.input_messages.{i}.message.content"] = m["content"]
        if output is not None:
            attrs["llm.output_messages.0.message.role"] = "assistant"
            attrs["llm.output_messages.0.message.content"] = output
        self.set(**{k.replace(".", "__"): v for k, v in attrs.items()})

    def add_timing(self, name: str, seconds: float) -> None:
        self.root.timings[name] = round(self.root.timings.get(name, 0.0) + seconds, 3)

    def fail(self, exc: BaseException) -> None:
        if self._span is None:
            return
        try:
            from opentelemetry.trace import Status, StatusCode

            self._span.record_exception(exc)
            self._span.set_status(Status(StatusCode.ERROR, str(exc)))
        except Exception:  # noqa: BLE001
            pass

    def end(self, output: Any = None, **attrs) -> float:
        """Finish the span and return its duration in seconds (idempotent)."""
        if self._ended:
            return self.seconds or 0.0
        self._ended = True
        self.seconds = time.perf_counter() - self.t0
        if self.root is self:
            self.timings["total"] = round(self.seconds, 3)
        else:
            self.add_timing(self.name, self.seconds)
        if self._span is not None:
            try:
                if output is not None:
                    self._span.set_output(output)
                self.set(**attrs)
                self._span.end()
            except Exception as e:  # noqa: BLE001
                log.warning("tracing: end failed on %s (%s)", self.name, e)
        return self.seconds


def start_trace(
    settings, name: str, input: Any = None, kind: str = "chain", session_id: str | None = None, **attrs
) -> Span:
    _init(settings)
    return Span(name, kind, input=input, session_id=session_id, attrs=attrs)


def child_of(parent: Span | None, name: str, kind: str = "chain", input: Any = None, **attrs) -> Span:
    """A child span, or a standalone one (timing only) when no parent trace is active."""
    return (
        parent.child(name, kind, input, **attrs) if parent is not None else Span(name, kind, input=input, attrs=attrs)
    )


def flush(timeout_ms: int = 10_000) -> None:
    if _provider is not None:
        try:
            _provider.force_flush(timeout_millis=timeout_ms)
        except Exception as e:  # noqa: BLE001
            log.warning("tracing: flush failed (%s)", e)


def annotate(settings, annotations: list[dict]) -> int:
    """Attach scores to finished spans. Each item: {span_id, name, score, label?, explanation?, kind?}. Returns how many were sent."""
    _init(settings)
    if not annotations or not enabled():
        return 0
    try:
        from phoenix.client import Client

        u = urlparse(settings.tracing.endpoint)
        client = Client(base_url=f"{u.scheme}://{u.netloc}")
        data = []
        for a in annotations:
            result = {k: a[k] for k in ("score", "label", "explanation") if a.get(k) is not None}
            data.append(
                {
                    "name": a["name"],
                    "span_id": a["span_id"],
                    "annotator_kind": a.get("kind", "CODE"),
                    "result": result,
                    "identifier": f"{a['span_id']}:{a['name']}",
                }
            )
        client.spans.log_span_annotations(span_annotations=data, sync=True)
        return len(data)
    except Exception as e:  # noqa: BLE001
        log.warning("tracing: could not log annotations (%s)", e)
        return 0

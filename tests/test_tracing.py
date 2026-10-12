import json
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from phoenix.otel import TracerProvider

from rag import generate as gen
from rag import tracing
from rag.config import Generation, Settings
from rag.eval.metrics import latency_summary
from rag.pipeline import Pipeline
from rag.retrieve import Hit


@pytest.fixture
def exporter():
    exp = InMemorySpanExporter()
    provider = TracerProvider(verbose=False)
    provider.add_span_processor(SimpleSpanProcessor(exp))
    tracing._install(provider.get_tracer("test"), provider)
    yield exp
    tracing._reset()


def by_name(exp):
    return {s.name: s for s in exp.get_finished_spans()}


def hit(i=1):
    return Hit(id=f"doc.pdf::{i:05d}", source="doc.pdf", page=i, section="Sec", kind="text", text=f"chunk text {i}",
               score=1.5, retrieval_score=1.5)


class FakeResp(dict):
    """Behaves like an ollama ChatResponse: subscriptable and with attributes."""

    def __init__(self, text, **stats):
        super().__init__(message={"content": text})
        self.__dict__.update(stats)

    def __getattr__(self, k):
        return None


STATS = dict(prompt_eval_count=120, eval_count=30, load_duration=2_000_000_000, prompt_eval_duration=1_000_000_000,
             eval_duration=3_000_000_000)


def test_span_tree_timings_and_documents(exporter):
    s = Settings()
    root = tracing.start_trace(s, "rag.query", input="q?", session_id="sess1", **{"retrieval__mode": "fts"})
    step = root.child("retrieve", "retriever", input="q?")
    step.set_documents([hit(1), hit(2)])
    step.end(output="2 chunks")
    root.end(output="answer", eval__rank=1)

    spans = by_name(exporter)
    r, c = spans["rag.query"], spans["retrieve"]
    assert c.parent.span_id == r.context.span_id and c.context.trace_id == r.context.trace_id
    assert r.attributes["openinference.span.kind"] == "CHAIN" and c.attributes["openinference.span.kind"] == "RETRIEVER"
    assert r.attributes["input.value"] == "q?" and r.attributes["output.value"] == "answer"
    assert r.attributes["session.id"] == "sess1" and c.attributes["session.id"] == "sess1"
    assert r.attributes["retrieval.mode"] == "fts" and r.attributes["eval.rank"] == 1
    assert c.attributes["retrieval.documents.1.document.content"] == "chunk text 2"
    assert c.attributes["retrieval.documents.0.document.id"] == "doc.pdf::00001"
    assert set(root.timings) == {"retrieve", "total"}
    assert root.trace_id == format(r.context.trace_id, "032x") and root.span_id == format(r.context.span_id, "016x")


def test_generate_records_messages_and_ollama_stats(exporter, monkeypatch):
    monkeypatch.setattr(gen.ollama, "chat", lambda **kw: FakeResp(" the answer ", **STATS))
    root = tracing.start_trace(Settings(), "rag.query", input="q")
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
    assert gen.generate(msgs, Generation(), root) == "the answer"
    root.end()
    g = by_name(exporter)["generate"]
    assert g.attributes["openinference.span.kind"] == "LLM"
    assert g.attributes["llm.input_messages.1.message.content"] == "usr"
    assert g.attributes["llm.output_messages.0.message.content"] == "the answer"
    assert g.attributes["llm.token_count.prompt"] == 120 and g.attributes["llm.token_count.total"] == 150
    assert g.attributes["ollama.tokens_per_s"] == 10.0 and g.attributes["ollama.load_s"] == 2.0


def test_stream_records_ttft_and_ends_on_close(exporter, monkeypatch):
    chunks = [FakeResp("Hel"), FakeResp("lo", **STATS)]
    monkeypatch.setattr(gen.ollama, "chat", lambda **kw: iter(chunks))
    root = tracing.start_trace(Settings(), "rag.query", input="q")
    assert "".join(gen.generate_stream([{"role": "user", "content": "x"}], Generation(), root)) == "Hello"
    root.end()
    g = by_name(exporter)["generate"]
    assert g.attributes["output.value"] == "Hello" and g.attributes["llm.token_count.completion"] == 30
    assert "ttft" in root.timings and "generate" in root.timings


def test_pipeline_ask_and_stream_trace(exporter, monkeypatch):
    monkeypatch.setattr("rag.pipeline.retrieve", lambda q, s, t, span=None: [hit(1)])
    monkeypatch.setattr(gen.ollama, "chat", lambda **kw: FakeResp("ans [1]", **STATS) if not kw.get("stream") else iter([FakeResp("ans", **STATS)]))
    p = Pipeline(Settings())
    p._table = object()
    a = p.ask("What?", session_id="s")
    assert a.answer == "ans [1]" and a.trace_id and a.span_id and {"generate", "total"} <= set(a.timings)

    hits, tokens, root = p.stream("What?")
    assert "".join(tokens) == "ans" and root.timings["total"] >= 0
    roots = [s for s in exporter.get_finished_spans() if s.name == "rag.query"]
    assert len(roots) == 2 and roots[1].attributes["output.value"] == "ans"


def test_noop_without_tracer_still_times():
    tracing._reset()
    s = Settings(tracing={"enabled": False})
    root = tracing.start_trace(s, "rag.query", input="q")
    step = root.child("retrieve", "retriever")
    step.set_documents([hit()])
    step.end(output="x")
    root.end(output="a")
    assert root.trace_id is None and set(root.timings) == {"retrieve", "total"}
    assert tracing.annotate(s, [{"span_id": "x", "name": "n", "score": 1}]) == 0
    tracing._reset()


def test_tracing_errors_are_swallowed(monkeypatch):
    class Boom:
        def start_span(self, *a, **k):
            raise RuntimeError("collector down")

    monkeypatch.setattr(tracing, "_tracer", Boom())
    root = tracing.Span("rag.query", "chain", input="q")
    root.child("x").end()
    root.end(output="ok")
    assert root.timings["total"] >= 0
    tracing._reset()


def test_latency_summary_and_no_key_in_settings():
    out = latency_summary([{"generate": 10.0, "total": 11.0}, {"generate": 30.0, "total": 31.0}, {"generate": 20.0}, None])
    assert out["generate"] == {"median": 20.0, "p95": 30.0, "max": 30.0, "n": 3} and out["total"]["n"] == 2
    snap = json.loads(Settings().model_dump_json())["tracing"]
    assert snap == {"enabled": True, "endpoint": "http://localhost:6006/v1/traces", "project_name": "techqa-rag"}

import pytest

pytest.importorskip("mlflow")  # only on Databricks (or wherever mlflow is installed)

from mlflow.types.responses import ResponsesAgentRequest  # noqa: E402

from rag.chat import ChatAnswer  # noqa: E402
from rag.generate import NOT_FOUND  # noqa: E402
from rag.serving_agent import CarbonRagAgent  # noqa: E402


class FakeBot:
    def __init__(self, answer="Alpha [1]", hits=1):
        self.calls, self._answer, self._hits = [], answer, hits

    def ask(self, question, history, retrieval_mode=None, rerank=None):
        self.calls.append((question, history, retrieval_mode, rerank))
        from rag.ai_search import SearchHit

        hits = [SearchHit("i", "d.pdf", 3, "Intro", "text", "t", 1.0, 1.0)] * self._hits
        return ChatAnswer(
            question, question, self._answer, hits, retrieval_mode or "hybrid", rerank or "off", [], {"total": 1.23456}
        )


def _request(**kw):
    return ResponsesAgentRequest(
        input=[
            {"role": "user", "content": "First?"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "Earlier answer"}]},
            {"role": "user", "content": "And now?"},
        ],
        **kw,
    )


def test_history_options_and_sources_flow_through():
    bot = FakeBot()
    resp = CarbonRagAgent(bot).predict(_request(custom_inputs={"retrieval_mode": "fts", "rerank": "llm"}))
    assert bot.calls == [
        (
            "And now?",
            [{"role": "user", "content": "First?"}, {"role": "assistant", "content": "Earlier answer"}],
            "fts",
            "llm",
        )
    ]
    text = resp.output[0].content[0]["text"]
    assert text.startswith("Alpha [1]") and "Sources:\n[1] d.pdf, p.3 | Intro" in text
    assert resp.custom_outputs["retrieval_mode"] == "fts" and resp.custom_outputs["timings"] == {"total": 1.235}


def test_abstention_has_no_sources_block_and_options_are_optional():
    bot = FakeBot(answer=NOT_FOUND, hits=2)
    resp = CarbonRagAgent(bot).predict(_request())
    assert resp.output[0].content[0]["text"] == NOT_FOUND and bot.calls[0][2:] == (None, None)


def test_last_message_must_be_a_user_question():
    req = ResponsesAgentRequest(input=[{"role": "assistant", "content": "hi"}])
    with pytest.raises(ValueError):
        CarbonRagAgent(FakeBot()).predict(req)


def test_items_without_role_or_text_are_not_conversation_turns():
    bot = FakeBot()
    req = ResponsesAgentRequest(
        input=[
            {"role": "user", "content": "First?"},
            {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "42"},
            {"role": "assistant", "content": [{"type": "output_text", "text": ""}]},
            {"role": "user", "content": "And now?"},
        ]
    )
    CarbonRagAgent(bot).predict(req)
    assert bot.calls[0][:2] == ("And now?", [{"role": "user", "content": "First?"}])


def test_text_tolerates_missing_content():
    from rag.serving_agent import _text

    assert _text(None) == "" and _text(7) == "" and _text([{"type": "output_text", "text": None}, "x"]) == ""
    assert _text([{"text": "a"}, {"text": "b"}]) == "ab"

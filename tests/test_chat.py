from types import SimpleNamespace

import pytest
from test_ai_search import ROWS, FakeIndex, _response

from rag import generate as gen
from rag.ai_search import RerankerUnavailable
from rag.chat import Chatbot
from rag.config import Generation, Settings
from rag.generate import NOT_FOUND


class FakeClient:
    """Answers each chat call with the next reply in `replies` and records the calls."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        content = self.replies.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )


@pytest.fixture(autouse=True)
def _reset_client():
    yield
    gen.set_chat_client(None)


def _bot(client, index=None, **kw):
    return Chatbot(index or FakeIndex(_response(ROWS)), client, Settings(generation=Generation(model="ep")), **kw)


@pytest.mark.parametrize("mode,query_type", [("fts", "FULL_TEXT"), ("vector", "ANN"), ("hybrid", "HYBRID")])
def test_retrieval_mode_selects_the_ai_search_query_type(mode, query_type):
    index = FakeIndex(_response(ROWS))
    ans = _bot(FakeClient("Alpha [1]"), index).ask("What?", retrieval_mode=mode)
    assert index.calls[0]["query_type"] == query_type and ans.retrieval_mode == mode
    assert ans.answer == "Alpha [1]" and ans.sources()[0] == "[1] d1.pdf, p.4 | Intro > Scope"


def test_answer_prompt_holds_the_retrieved_passages_and_uses_the_generation_endpoint():
    client = FakeClient("Alpha [1]")
    _bot(client).ask("What?")
    call = client.calls[0]
    assert call["model"] == "ep" and "Alpha beta gamma" in call["messages"][1]["content"]
    assert set(_bot(FakeClient("x")).ask("What?").timings) == {"retrieve", "generate", "total"}


def test_no_hits_abstains_without_calling_the_model():
    client = FakeClient()
    ans = _bot(client, FakeIndex(_response([]))).ask("What?")
    assert ans.answer == NOT_FOUND and client.calls == []


def test_follow_up_is_rewritten_with_history_before_retrieval():
    client = FakeClient("What area does a plot cover?", "0.04 ha [1]")
    index = FakeIndex(_response(ROWS))
    ans = _bot(client, index).ask("And its area?", [{"role": "user", "content": "Tell me about plots"}])
    assert ans.standalone_question == "What area does a plot cover?"
    assert index.calls[0]["query_text"] == "What area does a plot cover?" and "rewrite" in ans.timings


def test_llm_rerank_requires_a_reranker_and_runs_when_given():
    with pytest.raises(ValueError):
        _bot(FakeClient("x")).ask("What?", rerank="llm")
    with pytest.raises(ValueError):
        _bot(FakeClient("x")).ask("What?", rerank="ai_search")
    from rag.llm_rerank import LLMReranker

    client = FakeClient('{"ranking": [3, 1, 2]}', "answer")
    ans = _bot(client, llm_reranker=LLMReranker(client, "rr")).ask("What?", rerank="llm")
    assert ans.rerank == "llm" and ans.hits[0].id == "d2.pdf::00007" and not ans.warnings


def test_a_failing_reranker_degrades_to_plain_retrieval_and_says_so():
    class Boom:
        def rerank(self, query, hits):
            raise RerankerUnavailable("down")

    ans = _bot(FakeClient("answer"), llm_reranker=Boom()).ask("What?", rerank="llm")
    assert ans.rerank == "off" and ans.answer == "answer" and "llm" in ans.warnings[0]
    assert [h.id for h in ans.hits][0] == "d1.pdf::00001"


def test_invalid_options_and_empty_question_are_rejected():
    bot = _bot(FakeClient("x"))
    for kw in ({"retrieval_mode": "bm25"}, {"rerank": "cohere"}):
        with pytest.raises(ValueError):
            bot.ask("What?", **kw)
    with pytest.raises(ValueError):
        bot.ask("   ")
    with pytest.raises(ValueError):
        _bot(FakeClient("x"), retrieval_mode="bm25")


def test_a_reranker_error_that_is_a_value_error_still_falls_back():
    import json

    class BadJson:
        def rerank(self, query, hits):
            return json.loads("not json")  # json.JSONDecodeError is a ValueError

    ans = _bot(FakeClient("answer"), llm_reranker=BadJson()).ask("What?", rerank="llm")
    assert ans.rerank == "off" and ans.answer == "answer" and "llm" in ans.warnings[0]


def test_a_missing_reranker_is_rejected_before_any_model_call():
    client = FakeClient("What area?", "answer")
    with pytest.raises(ValueError, match="reranker"):
        _bot(client).ask("And its area?", [{"role": "user", "content": "plots"}], rerank="llm")
    assert client.calls == []  # not even the follow-up rewrite ran


def test_a_plain_retrieval_error_is_not_swallowed():
    class Down(FakeIndex):
        def similarity_search(self, **kwargs):
            raise ConnectionError("index down")

    with pytest.raises(ConnectionError):
        _bot(FakeClient("x"), Down(_response(ROWS))).ask("What?")

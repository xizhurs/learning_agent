from types import SimpleNamespace

import pytest
from test_ai_search import ROWS, FakeIndex, _response

from rag.ai_search import RerankerUnavailable, SearchHit, make_retriever
from rag.config import Settings
from rag.llm_rerank import LLMReranker, parse_ranking


class FakeClient:
    """Shaped like openai.OpenAI(): client.chat.completions.create(...)."""

    def __init__(self, content):
        self.calls = []
        self._content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))])


def _hits(n=3):
    return [
        SearchHit(f"id{i}", "d.pdf", i, "S", "text", f"text {i} " + "x" * 3000, 1.0 - i / 10, 1.0 - i / 10)
        for i in range(n)
    ]


def test_parse_ranking_variants():
    assert parse_ranking('{"ranking": [2, 1, 3]}', 3) == [1, 0, 2]
    assert parse_ranking('Sure!\n```json\n{"ranking": [3, 1]}\n```', 3) == [2, 0]
    assert parse_ranking('{"ranking": [2, 2, 9, 0, true, "1", 1]}', 3) == [
        1,
        0,
    ]  # repeats, out of range, non-ints dropped
    for bad in ("", "no json here", '{"ranking": "1,2"}', '{"other": [1]}', "{broken", "[1, 2]"):
        assert parse_ranking(bad, 3) == []


def test_rerank_orders_by_the_reply_and_keeps_omitted_hits_last():
    client = FakeClient('{"ranking": [3, 1]}')
    out = LLMReranker(client, "ep").rerank("What?", _hits())
    assert [h.id for h in out] == ["id2", "id0", "id1"]
    assert [h.score for h in out] == [1.0, 0.5, pytest.approx(1 / 3)]
    assert out[0].retrieval_score == pytest.approx(0.8)  # the search score is kept


def test_rerank_request_is_deterministic_and_truncates_passages():
    client = FakeClient('{"ranking": [1, 2, 3]}')
    LLMReranker(client, "ep", max_chars=100).rerank("What is X?", _hits())
    call = client.calls[0]
    assert call["model"] == "ep" and call["temperature"] == 0.0
    prompt = call["messages"][0]["content"]
    assert "What is X?" in prompt and "[1] text 0" in prompt and "[3] text 2" in prompt
    assert "x" * 101 not in prompt


def test_rerank_unusable_reply_raises_instead_of_passing_as_reranked():
    with pytest.raises(RerankerUnavailable):
        LLMReranker(FakeClient("I cannot rank these."), "ep").rerank("What?", _hits())


def test_rerank_skips_the_call_for_fewer_than_two_hits():
    client = FakeClient("{}")
    reranker = LLMReranker(client, "ep")
    assert reranker.rerank("What?", []) == [] and len(reranker.rerank("What?", _hits(1))) == 1
    assert client.calls == [] and reranker.stats["calls"] == 0


def test_retriever_with_llm_reranker_fetches_candidates_and_returns_top_k():
    index = FakeIndex(_response(ROWS))
    reranker = LLMReranker(FakeClient('{"ranking": [3, 2, 1]}'), "ep")
    retriever = make_retriever(index, rerank=False, llm_reranker=reranker, num_results=2, candidates_k=10)
    hits = retriever("What?", Settings())
    call = index.calls[0]
    assert call["num_results"] == 10 and "reranker" not in call and "debug_level" not in call
    assert [h.id for h in hits] == ["d2.pdf::00007", "d1.pdf::00002"]
    assert retriever.stats["reranker_seconds"] > 0 and reranker.stats["calls"] == 1


def test_retriever_candidates_default_to_settings_and_query_type_is_passed_through():
    index = FakeIndex(_response(ROWS))
    reranker = LLMReranker(FakeClient('{"ranking": [1]}'), "ep")
    make_retriever(index, rerank=False, llm_reranker=reranker, query_type="FULL_TEXT")("What?", Settings())
    assert (
        index.calls[0]["num_results"] == Settings().retrieval.candidates_k
        and index.calls[0]["query_type"] == "FULL_TEXT"
    )


def test_both_rerankers_at_once_is_an_error():
    with pytest.raises(ValueError):
        make_retriever(FakeIndex(_response(ROWS)), rerank=True, reranker=object(), llm_reranker=object())


def test_parse_ranking_accepts_numeric_strings_whole_floats_and_the_first_json_object():
    assert parse_ranking('{"ranking": ["3", " 1 ", 2.0]}', 3) == [2, 0, 1]
    assert parse_ranking('{"ranking": [1.5, "x", "2"]}', 3) == [1]  # fractions and non-numbers are dropped
    assert parse_ranking('First {"ranking": [2, 1]} then {"note": "done"}', 3) == [1, 0]
    assert parse_ranking('{"note": "x"} {"ranking": [3]}', 3) == [2]  # skips an object without a ranking
    assert parse_ranking('{"ranking": [2, 1]} and {unclosed', 3) == [1, 0]


def test_stats_count_a_failed_call():
    class Boom:
        chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(TimeoutError("slow")))
        )

    reranker = LLMReranker(Boom(), "ep")
    with pytest.raises(TimeoutError):
        reranker.rerank("What?", _hits())
    assert reranker.stats["calls"] == 1 and reranker.stats["seconds"] > 0


def test_fewer_candidates_than_results_is_an_error_not_a_smaller_top_k():
    reranker = LLMReranker(FakeClient('{"ranking": [1]}'), "ep")
    with pytest.raises(ValueError, match="candidates_k"):  # caught when the retriever is built ...
        make_retriever(FakeIndex(_response(ROWS)), rerank=False, llm_reranker=reranker, num_results=5, candidates_k=3)
    settings = Settings()
    settings.retrieval.candidates_k = 3  # ... and when it comes from the settings at query time
    with pytest.raises(ValueError, match="candidates_k"):
        make_retriever(FakeIndex(_response(ROWS)), rerank=False, llm_reranker=reranker)("What?", settings)
    index = FakeIndex(_response(ROWS))  # equal is fine
    make_retriever(index, rerank=False, llm_reranker=reranker, num_results=3, candidates_k=3)("What?", Settings())
    assert index.calls[0]["num_results"] == 3

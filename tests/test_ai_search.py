import pytest

from rag import ai_search
from rag.ai_search import RerankerUnavailable, make_retriever
from rag.config import Settings
from rag.eval import run_eval

NAMES = ["id", "source", "page", "section", "kind", "text", "score"]


def _response(rows, debug=None):
    resp = {"manifest": {"columns": [{"name": n} for n in NAMES]}, "result": {"data_array": rows}}
    if debug is not None:
        resp["debug_info"] = debug
    return resp


ROWS = [
    ["d1.pdf::00001", "d1.pdf", 4, "Intro > Scope", "text", "Alpha beta gamma delta epsilon zeta.", 0.9],
    ["d1.pdf::00002", "d1.pdf", 0, "", "table", "| a | b |", 0.5],
    ["d2.pdf::00007", "d2.pdf", 9, "Methods", "text", "Delta.", 0.1],
]


class FakeIndex:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def similarity_search(self, **kwargs):
        self.calls.append(kwargs)
        return self.resp


def test_hybrid_query_and_hit_mapping():
    index = FakeIndex(_response(ROWS))
    hits = make_retriever(index, rerank=False)("What?", Settings())
    call = index.calls[0]
    assert call["query_type"] == "hybrid" and call["query_text"] == "What?" and call["num_results"] == 5
    assert call["columns"] == ai_search.COLUMNS and "reranker" not in call
    assert [h.id for h in hits] == ["d1.pdf::00001", "d1.pdf::00002", "d2.pdf::00007"]
    assert hits[0].score == 0.9 and hits[0].citation() == "d1.pdf, p.4 | Intro > Scope"
    assert hits[1].citation() == "d1.pdf | n/a" and hits[1].kind == "table"


def test_num_results_overrides_settings_and_truncates():
    index = FakeIndex(_response(ROWS))
    hits = make_retriever(index, rerank=False, num_results=2)("What?", Settings())
    assert index.calls[0]["num_results"] == 2 and len(hits) == 2


def test_missing_score_column_keeps_the_returned_order():
    resp = {
        "manifest": {"columns": [{"name": n} for n in NAMES[:-1]]},
        "result": {"data_array": [r[:-1] for r in ROWS]},
    }
    hits = make_retriever(FakeIndex(resp), rerank=False)("What?", Settings())
    assert [h.id for h in hits] == [r[0] for r in ROWS] and hits[0].score > hits[1].score > hits[2].score


def test_rerank_passes_the_reranker_and_records_timings():
    reranker = object()
    index = FakeIndex(_response(ROWS, {"reranker_time": 0.4, "response_time": 0.9}))
    retriever = make_retriever(index, rerank=True, reranker=reranker)
    retriever("What?", Settings())
    assert index.calls[0]["reranker"] is reranker and index.calls[0]["debug_level"] == 1
    assert retriever.stats["calls"] == 1 and retriever.stats["reranker_seconds"] == pytest.approx(0.4)


def test_rerank_requires_a_reranker_object():
    with pytest.raises(ValueError):
        make_retriever(FakeIndex(_response(ROWS)), rerank=True)


def test_reranker_warning_is_an_error_not_a_silent_unreranked_result():
    debug = {"warnings": [{"status_code": "RERANKER_TEMPORARILY_UNAVAILABLE"}]}
    retriever = make_retriever(FakeIndex(_response(ROWS, debug)), rerank=True, reranker=object())
    with pytest.raises(RerankerUnavailable):
        retriever("What?", Settings())


def test_missing_reranker_time_is_an_error_unless_verification_is_off():
    index = FakeIndex(_response(ROWS, {"response_time": 0.9}))
    with pytest.raises(RerankerUnavailable):
        make_retriever(index, rerank=True, reranker=object())("What?", Settings())
    assert len(make_retriever(index, rerank=True, reranker=object(), verify_rerank=False)("What?", Settings())) == 3


def _item():
    return {
        "id": "q1",
        "question": "What?",
        "type": "factual",
        "answerable": True,
        "source": "d1.pdf",
        "gold_context_ids": [],
        "reference_answer": "Alpha",
        "reference_quote": "Alpha beta gamma delta epsilon zeta.",
    }


@pytest.fixture
def restore_retriever():
    yield
    run_eval.set_retriever()


def test_set_retriever_feeds_run_items_and_skips_the_local_index(restore_retriever):
    run_eval.set_retriever(make_retriever(FakeIndex(_response(ROWS)), rerank=False))
    settings = Settings()
    settings.tracing.enabled = False
    rows = run_eval.run_items(
        [_item()], settings, generate_answers=False
    )  # would import lancedb/docling without the injection
    assert [h["id"] for h in rows[0]["retrieved"]] == [r[0] for r in ROWS] and rows[0]["rank"] == 1


def test_set_retriever_without_arguments_restores_the_local_functions(restore_retriever):
    run_eval.set_retriever(lambda *a, **k: [])
    run_eval.set_retriever()
    assert run_eval._retriever is None and run_eval._index_check is None


def test_a_custom_index_check_is_used(restore_retriever):
    seen = []
    run_eval.set_retriever(lambda *a, **k: [], seen.append)
    run_eval.ensure_table("settings")
    assert seen == ["settings"]

import json

import pytest

from rag.config import Settings
from rag.eval import run_eval
from rag.generate import build_messages


def _saved_run(tmp_path):
    run = tmp_path / "old_run"
    run.mkdir()
    row = {
        "id": "q1",
        "retrieved": [
            {"id": "d1.md::00000", "citation": "d1.md | Intro", "score": 2.5, "text": "Alpha beta gamma."},
            {"id": "d2.md::00003", "citation": "d2.md | Methods", "score": 1.1, "text": "Delta epsilon."},
        ],
    }
    (run / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return run


def test_saved_chunks_are_read_back_in_order_and_feed_the_prompt(tmp_path):
    saved = run_eval.load_saved_retrieval(_saved_run(tmp_path))
    hits = saved["q1"]
    assert [h.id for h in hits] == ["d1.md::00000", "d2.md::00003"] and hits[0].score == 2.5
    user = build_messages("What?", hits)[1]["content"]
    assert "[1] (d1.md | Intro)\nAlpha beta gamma." in user and "[2] (d2.md | Methods)" in user


def _item(id_="q1"):
    return {
        "id": id_,
        "question": "What?",
        "type": "factual",
        "answerable": True,
        "source": "d1.md",
        "gold_context_ids": ["d1.md"],
        "reference_answer": "Alpha",
        "reference_quote": "Alpha beta gamma.",
    }


def test_run_items_with_saved_retrieval_never_searches(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("search / rerank / table build must not run")

    monkeypatch.setattr(run_eval, "retrieve", boom)
    monkeypatch.setattr(run_eval, "ensure_table", boom)
    settings = Settings()
    settings.tracing.enabled = False
    saved = run_eval.load_saved_retrieval(_saved_run(tmp_path))
    rows = run_eval.run_items([_item()], settings, generate_answers=False, saved_retrieval=saved)
    assert [h["id"] for h in rows[0]["retrieved"]] == ["d1.md::00000", "d2.md::00003"]
    assert rows[0]["rank"] == 1 and rows[0]["gold_context_ids"] == ["d1.md"]
    assert rows[0]["retrieved"][0]["citation"] == "d1.md | Intro"


def test_question_missing_from_the_saved_run_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "ensure_table", lambda s: None)
    settings = Settings()
    settings.tracing.enabled = False
    saved = run_eval.load_saved_retrieval(_saved_run(tmp_path))
    with pytest.raises(SystemExit):
        run_eval.run_items([_item("other")], settings, generate_answers=False, saved_retrieval=saved)

import json

import pytest

from rag.config import Settings
from rag.eval import run_eval


def _item(id_):
    return {"id": id_, "question": f"Question {id_}?", "type": "factual", "answerable": True, "source": "d1.md",
            "gold_context_ids": [], "reference_answer": "Alpha", "reference_quote": "Alpha beta gamma."}


def _settings():
    settings = Settings()
    settings.tracing.enabled = False
    return settings


def _fake_generation(monkeypatch, asked):
    monkeypatch.setattr(run_eval, "ensure_table", lambda s: None)
    monkeypatch.setattr(run_eval, "retrieve", lambda *a, **k: [run_eval.SavedHit("d1.md::00000", 1.0, "Alpha beta gamma.", "d1.md")])
    monkeypatch.setattr(run_eval, "build_messages", lambda q, hits: q)

    def gen(q, cfg, root):
        asked.append(q)
        return f"answer to {q}"

    monkeypatch.setattr(run_eval, "generate", gen)


def test_each_row_is_handed_to_on_row_as_it_is_produced(monkeypatch):
    _fake_generation(monkeypatch, [])
    seen = []
    run_eval.run_items([_item("q1"), _item("q2")], _settings(), generate_answers=True, on_row=seen.append)
    assert [r["id"] for r in seen] == ["q1", "q2"]


def test_answered_questions_are_skipped_and_kept_in_order(monkeypatch):
    asked = []
    _fake_generation(monkeypatch, asked)
    done = {"q1": {"id": "q1", "answer": "kept"}}
    rows = run_eval.run_items([_item("q1"), _item("q2")], _settings(), generate_answers=True, done=done)
    assert asked == ["Question q2?"]
    assert [(r["id"], r["answer"]) for r in rows] == [("q1", "kept"), ("q2", "answer to Question q2?")]


def _killed_run(tmp_path, settings, extra=""):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    rows = [{"id": "q1", "answer": "a1"}, {"id": "q2", "answer": None}]
    (run / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + extra, encoding="utf-8")
    return run


def test_resume_keeps_only_answered_rows_and_drops_a_half_written_line(tmp_path):
    settings = _settings()
    run = _killed_run(tmp_path, settings, extra='{"id": "q3", "ans')
    done = run_eval.read_partial_rows(run, settings)
    assert list(done) == ["q1"]
    assert [json.loads(l)["id"] for l in (run / "items.jsonl").read_text(encoding="utf-8").splitlines()] == ["q1"]


def test_resume_refuses_when_the_settings_changed(tmp_path):
    settings = _settings()
    run = _killed_run(tmp_path, settings)
    settings.rerank.enabled = not settings.rerank.enabled
    with pytest.raises(SystemExit):
        run_eval.read_partial_rows(run, settings)


def test_append_row_adds_one_json_line(tmp_path):
    run_eval.append_row(tmp_path, {"id": "q1", "answer": "x"})
    run_eval.append_row(tmp_path, {"id": "q2", "answer": "y"})
    assert [json.loads(l)["id"] for l in (tmp_path / "items.jsonl").read_text(encoding="utf-8").splitlines()] == ["q1", "q2"]

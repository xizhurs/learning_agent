import importlib.util
import json
from pathlib import Path

import openpyxl
import pytest

from rag.config import Settings

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "export_eval_report.py"
spec = importlib.util.spec_from_file_location("export_eval_report", SCRIPT)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)

NOT_FOUND = "I could not find this in the provided documents."
LONG = "x" * 3000


def chunk(doc, n=0, text="some passage text"):
    return {"id": f"{doc}::{n:05d}", "citation": f"{doc} | n/a", "score": 1.0, "text": text}


def row(id_, answerable=True, gold=("d1.txt",), retrieved=(), answer=None, rank=None, question="q?", ragas=None, **kw):
    r = {"id": id_, "question": question, "type": "factual", "answerable": answerable, "source": gold[0] if gold else "",
         "reference_answer": "ref answer" if answerable else "", "reference_quote": "ref answer",
         "retrieved": list(retrieved), "rank": rank, "answer": answer, "seconds": 2.0, "timings": {"total": 2.0},
         "trace_id": "ab" * 16, "span_id": "cd" * 8}
    if gold is not None:
        r["gold_context_ids"] = list(gold)
    if ragas:
        r["ragas"] = ragas
    return r | kw


def write_run(folder: Path, rows, mode="fts", size=512, top_k=5, rerank=False, testset="mytest"):
    folder.mkdir(parents=True)
    cfg = Settings().model_dump()
    cfg["retrieval"].update(mode=mode, top_k=top_k)
    cfg["chunking"].update(max_tokens=size, overlap=64)
    cfg["rerank"].update(enabled=rerank)
    (folder / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps({"testset": testset, "variant": folder.name}), encoding="utf-8")
    (folder / "items.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    ab = root / "20261011-100000_ablation_mytest"
    scored = [
        row("q1", retrieved=[chunk("d1.txt"), chunk("d2.txt")], rank=1, answer="Set X [1].",
            ragas={"faithfulness": 0.9, "factual_correctness(mode=f1)": 0.8}),
        row("q2", gold=("d3.txt",), retrieved=[chunk("d1.txt"), chunk("d2.txt")], answer=NOT_FOUND),
        row("q3", retrieved=[chunk("d2.txt"), chunk("d2.txt", 1), chunk("d4.txt"), chunk("d1.txt", 2)], answer=NOT_FOUND),
        row("q4", answerable=False, gold=(), retrieved=[chunk("d2.txt")], answer="Use option X."),
        row("q5", answerable=False, gold=(), retrieved=[chunk("d2.txt")], answer=NOT_FOUND),
        row("q6", retrieved=[chunk("d1.txt", text=LONG)], answer=LONG, question="bell\x07 char " + LONG, rank=1,
            ragas={"faithfulness": 0.2, "factual_correctness(mode=f1)": 0.9}),
    ]
    write_run(ab / "heading_aware_512_fts_rr-off", scored)
    unscored = [row(r["id"], r["answerable"], tuple(r.get("gold_context_ids", ())), r["retrieved"]) for r in scored]
    write_run(ab / "heading_aware_256_vector_rr-on", unscored, mode="vector", size=256, rerank=True)
    return root


def test_discovery_and_only_filter(runs):
    assert [exp.run_id_of(d, runs) for d in exp.discover_runs(runs)] == [
        "20261011-100000_ablation_mytest/heading_aware_256_vector_rr-on", "20261011-100000_ablation_mytest/heading_aware_512_fts_rr-off"]
    assert len(exp.discover_runs(runs, "*512_fts*")) == 1 and exp.discover_runs(runs, "*nothing*") == []


def test_sheets_columns_and_summary_values(runs):
    frames, raw = exp.build_report(runs, commit="abc1234")
    assert list(frames) == ["summary_by_run", "per_question_results", "retrieval_details", "answer_details", "failure_analysis", "config"]
    assert list(frames["summary_by_run"].columns) == exp.SUMMARY_COLS
    assert list(frames["per_question_results"].columns) == exp.PER_QUESTION_COLS
    assert list(frames["retrieval_details"].columns) == exp.RETRIEVAL_COLS
    assert list(frames["answer_details"].columns) == exp.ANSWER_COLS
    assert list(frames["failure_analysis"].columns) == exp.FAILURE_COLS
    assert list(frames["config"].columns) == exp.CONFIG_COLS
    assert len(raw) == 12

    s = frames["summary_by_run"].set_index("run_id")
    a = s.loc["20261011-100000_ablation_mytest/heading_aware_512_fts_rr-off"]
    assert a["num_questions"] == 6 and a["retriever_type"] == "fts" and a["chunk_size"] == 512 and a["reranker"] == "none"
    # answerable: q1 rank 1, q2 miss, q3 rank 4, q6 rank 1  -> recall@1 0.5, @3 0.5, @5 0.75; mrr (1 + 0 + .25 + 1) / 4
    assert (a["context_recall_at_1"], a["context_recall_at_3"], a["context_recall_at_5"]) == (0.5, 0.5, 0.75)
    assert a["mrr"] == pytest.approx(0.5625) and a["recall_basis"] == "document"
    assert a["not_found_accuracy"] == 0.5                       # q5 refused, q4 answered
    assert a["faithfulness_avg"] == pytest.approx(0.55)         # q1 0.9, q6 0.2
    assert a["answer_accuracy"] == pytest.approx(0.5)           # q1 correct, q6 correct; q2 and q3 refused (wrong); -> 2/4
    assert a["avg_latency_sec"] == 2.0 and a["estimated_cost"] == 0.0
    b = s.loc["20261011-100000_ablation_mytest/heading_aware_256_vector_rr-on"]
    assert b["reranker"] == "BAAI/bge-reranker-base" and b["answer_accuracy"] is None or b["answer_accuracy"] != b["answer_accuracy"]
    cfg = frames["config"].set_index("run_id").loc["20261011-100000_ablation_mytest/heading_aware_512_fts_rr-off"]
    assert cfg["git_commit"] == "abc1234" and cfg["created_at"] == "2026-10-11T10:00:00" and cfg["llm_model"] == "llama3.2:3b"


def test_failure_types_and_per_question_fields(runs):
    frames, _ = exp.build_report(runs, commit="")
    pq = frames["per_question_results"]
    scored = pq[pq["run_id"].str.endswith("512_fts_rr-off")].set_index("question_id")
    assert scored.loc["q1", "failure_type"] == "ok" and scored.loc["q1", "citation_correct"] is True
    assert scored.loc["q2", "failure_type"] == "retrieval_miss"
    assert scored.loc["q3", "failure_type"] == "false_refusal" and "late_hit" in scored.loc["q3", "notes"]
    assert scored.loc["q3", "gold_context_rank"] == 4 and scored.loc["q3", "gold_context_hit_at_3"] is False
    assert scored.loc["q4", "failure_type"] == "missed_refusal" and scored.loc["q4", "gold_context_hit_at_1"] is None
    assert scored.loc["q5", "failure_type"] == "ok"
    assert scored.loc["q6", "failure_type"] == "low_faithfulness"
    assert scored.loc["q1", "retrieved_context_ids"] == ["d1.txt", "d2.txt"]

    unscored = pq[pq["run_id"].str.endswith("256_vector_rr-on")].set_index("question_id")
    assert unscored.loc["q1", "failure_type"] == "ok" and unscored.loc["q2", "failure_type"] == "retrieval_miss"
    assert unscored.loc["q3", "failure_type"] == "late_hit" and unscored.loc["q4", "failure_type"] == "ok"   # retrieval-only: no answers

    fa = frames["failure_analysis"]
    assert set(fa["failure_type"]) == {"retrieval_miss", "false_refusal", "missed_refusal", "low_faithfulness", "late_hit"}
    assert (fa["diagnosis"] != "").all() and (fa["suggested_fix"] != "").all()
    assert len(frames["retrieval_details"]) == 2 * (2 + 2 + 4 + 1 + 1 + 1)


def test_written_files_truncate_text_freeze_header_and_keep_raw(runs, tmp_path):
    frames, raw = exp.build_report(runs, commit="")
    out = tmp_path / "reports"
    paths = exp.write_report(frames, raw, out)
    assert {p.name for p in paths} == {"rag_ablation_report.xlsx", "per_question_results.csv", "failure_analysis.csv", "raw_results.jsonl"}

    wb = openpyxl.load_workbook(out / "rag_ablation_report.xlsx")
    assert wb.sheetnames == ["summary_by_run", "per_question_results", "retrieval_details", "answer_details", "failure_analysis", "config"]
    for ws in wb.worksheets:
        assert ws.freeze_panes == "A2" and ws.auto_filter.ref == ws.dimensions
        for r in ws.iter_rows(values_only=True):
            assert all(len(str(v)) <= 1000 for v in r if v is not None)
    assert wb["per_question_results"].max_row == 1 + 12 and wb["summary_by_run"].max_row == 3
    assert "\x07" not in "".join(str(c.value) for c in wb["per_question_results"]["D"])      # control character removed

    csv_lines = (out / "per_question_results.csv").read_text(encoding="utf-8-sig").splitlines()
    assert len(csv_lines) >= 13 and (out / "failure_analysis.csv").exists()
    records = [json.loads(l) for l in (out / "raw_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 12
    q6 = next(r for r in records if r["row"]["id"] == "q6" and r["row"]["answer"])
    assert len(q6["row"]["answer"]) == 3000 and "config" in q6 and q6["derived"]["failure_type"] == "low_faithfulness"


def test_gold_ids_fall_back_to_source_and_stored_rank(tmp_path):
    root = tmp_path / "runs"
    old = [row("a", gold=None, retrieved=[chunk("d9.txt")], rank=2, source="d9.txt")]
    write_run(root / "run1", old)
    frames, _ = exp.build_report(root, commit="")
    pq = frames["per_question_results"].iloc[0]
    assert pq["gold_context_ids"] == ["d9.txt"] and pq["gold_context_rank"] == 1       # source used as the gold document
    old2 = [row("b", gold=None, source="", retrieved=[chunk("d9.txt")], rank=3)]
    write_run(root / "run2", old2)
    frames, _ = exp.build_report(root, only="run2", commit="")
    assert frames["per_question_results"].iloc[0]["gold_context_rank"] == 3
    assert frames["summary_by_run"].iloc[0]["recall_basis"] == "answer_quote"


def test_empty_runs_dir_exits_with_a_hint(tmp_path):
    with pytest.raises(SystemExit):
        exp.build_report(tmp_path, commit="")


def test_citation_check():
    retrieved = [chunk("d1.txt"), chunk("d2.txt")]
    assert exp.citation_check("A [1].", retrieved, ["d1.txt"]) is True
    assert exp.citation_check("A [2].", retrieved, ["d1.txt"]) is False       # cites a non-gold chunk
    assert exp.citation_check("A [5].", retrieved, ["d1.txt"]) is False       # out of range
    assert exp.citation_check("A without citation", retrieved, ["d1.txt"]) is False
    assert exp.citation_check(NOT_FOUND, retrieved, ["d1.txt"]) is None and exp.citation_check(None, retrieved, []) is None

import importlib.util
import json
from pathlib import Path

import ollama
import pytest

from rag import embed
from rag.config import Chunking, Embedding
from rag.eval.run_eval import normalize_item
from rag.ingest.chunk import chunk_document
from rag.ingest.parse import SUPPORTED, parse_markdown_document

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "create_qasper_tiny.py"
spec = importlib.util.spec_from_file_location("create_qasper_tiny", SCRIPT)
qt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qt)


def ann(unanswerable=False, spans=(), free="", yes_no=None, evidence=()):
    return {"unanswerable": unanswerable, "extractive_spans": list(spans), "yes_no": yes_no, "free_form_answer": free,
            "evidence": list(evidence), "highlighted_evidence": []}


def paper(i: int, questions: list[tuple[str, list[dict]]]) -> dict:
    return {
        "id": f"2001.{i:05d}", "title": f"Paper number {i}", "abstract": f"We study topic {i}.",
        "full_text": {"section_name": ["Introduction", "Method ::: Data", "Method ::: Model"],
                      "paragraphs": [[f"Topic {i} matters a lot."], [f"We use the corpus C{i} with {i} documents."],
                                     [f"# of layers is {i}.", f"The model M{i} is a transformer."]]},
        "qas": {"question": [q for q, _ in questions], "question_id": [f"q{i}_{n}" for n in range(len(questions))],
                "answers": [{"answer": a, "annotation_id": [], "worker_id": []} for _, a in questions]},
        "figures_and_tables": {"caption": [f"Table 1: results for {i}."], "file": ["t1.png"]},
    }


def make_papers() -> list[dict]:
    """110 papers with a usable answerable question, 30 of them also with an unanswerable one; plus questions that must be dropped."""
    papers = []
    for i in range(120):
        ev = f"We use the corpus C{i} with {i} documents."
        qs = [(f"Which corpus is used in {i}?", [ann(spans=[f"C{i}"], evidence=[ev]), ann(free=f"corpus C{i}", evidence=[ev])])] if i < 110 else []
        if i < 30:
            qs.append((f"Who funded study {i}?", [ann(unanswerable=True), ann(unanswerable=True)]))
        qs += [(f"Disputed {i}?", [ann(unanswerable=True), ann(spans=["x"], evidence=[ev])]),
               (f"Table only {i}?", [ann(spans=["42"], evidence=["FLOAT SELECTED: Table 1"]), ann(spans=["42"], evidence=["FLOAT SELECTED: Table 1"])]),
               (f"Single annotator {i}?", [ann(spans=[f"C{i}"], evidence=[ev])]),
               (f"Citation only {i}?", [ann(spans=["BIBREF3"], evidence=[ev]), ann(spans=["BIBREF3"], evidence=[ev])])]
        papers.append(paper(i, qs))
    return papers


def test_paper_markdown_nests_sections_and_escapes_hash():
    md = qt.paper_markdown(make_papers()[7])
    assert md.startswith("# Paper number 7\n\n## Abstract\n\nWe study topic 7.")
    assert "## Method\n\n### Data\n\n" in md and md.count("## Method") == 1 and "### Model" in md
    assert "\n\\# of layers is 7.\n" in md
    assert md.rstrip().endswith("## Figures and Tables\n\nTable 1: results for 7.")


def test_candidate_questions_keep_only_agreed_text_evidence_questions():
    p = make_papers()[3]
    dropped = qt.Counter()
    rows = qt.candidate_questions(p, qt.paper_markdown(p), dropped)
    assert [(r["question_original"], r["is_impossible"]) for r in rows] == [("Which corpus is used in 3?", False), ("Who funded study 3?", True)]
    ans, imp = rows
    assert ans["answer"] == "C3" and ans["answer_type"] == "extractive" and ans["gold_context_ids"] == ["2001.00003.md"]
    assert ans["evidence"] == "We use the corpus C3 with 3 documents." and ans["question"].startswith('In the paper "Paper number 3": ')
    assert imp["answer"] == "" and imp["evidence"] == "" and imp["gold_context_ids"] == []
    assert dropped == {"annotators_disagree_on_answerable": 1, "no_text_evidence_or_usable_answer": 2, "fewer_than_2_annotators": 1}


def test_build_is_deterministic_and_passes_checks():
    a, b = qt.build(make_papers(), 7, n_distractors=5), qt.build(make_papers(), 7, n_distractors=5)
    qt.check(a)
    assert a["splits"] == b["splits"] and a["contexts"] == b["contexts"]
    assert {n: len(r) for n, r in a["splits"].items()} == {"dev": 40, "val": 30, "test": 30}
    roles = qt.Counter(c["role"] for c in a["contexts"])
    assert roles == {"gold": 80, "unanswerable": 20, "distractor": 5}


def test_check_rejects_evidence_missing_from_paper():
    result = qt.build(make_papers(), 7, n_distractors=5)
    row = next(r for r in result["splits"]["dev"] if not r["is_impossible"])
    row["evidence"] = "A sentence the paper never contains."
    with pytest.raises(AssertionError, match="evidence not in paper"):
        qt.check(result)


def test_write_refuses_foreign_files_and_force_removes_only_own(tmp_path):
    out, docs = tmp_path / "out", tmp_path / "docs"
    result = qt.build(make_papers(), 7, n_distractors=5)
    docs.mkdir()
    (docs / "keep.pdf").write_text("not ours", encoding="utf-8")
    qt.write(result, out, docs, 7, force=False)
    assert len(list(docs.glob("*.md"))) == 105
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["splits"]["dev"] == {"answerable": 32, "unanswerable": 8}
    with pytest.raises(SystemExit):
        qt.write(result, out, docs, 7, force=False)                     # output not empty
    smaller = qt.build(make_papers(), 7, n_distractors=2)
    qt.write(smaller, out, docs, 7, force=True)
    assert len(list(docs.glob("*.md"))) == 102 and (docs / "keep.pdf").exists()
    (out / "contexts.jsonl").unlink()                                   # forget what we wrote: now the papers look foreign
    with pytest.raises(SystemExit, match="not written by this script"):
        qt.write(smaller, out, docs, 7, force=True)


def test_markdown_is_supported_and_chunks_carry_section_paths(tmp_path):
    assert ".md" in SUPPORTED
    p = tmp_path / "2001.00007.md"
    p.write_text(qt.paper_markdown(make_papers()[7]), encoding="utf-8")
    chunks = chunk_document(parse_markdown_document(p), Chunking(max_tokens=512))
    sections = [c.section for c in chunks]
    assert "Paper number 7 > Method > Data" in sections and "Paper number 7 > Method > Model" in sections
    model = next(c for c in chunks if c.section.endswith("Model"))
    assert model.text.startswith("2001.00007.md > Paper number 7 > Method > Model") and "# of layers is 7." in model.text
    assert all(c.id.startswith("2001.00007.md::") for c in chunks)


def test_normalize_item_uses_evidence_as_quote():
    row = normalize_item({"id": "q", "question": "q?", "answer": "C3", "is_impossible": False, "gold_context_ids": ["p.md"],
                          "evidence": "We use the corpus C3."})
    assert row["reference_answer"] == "C3" and row["reference_quote"] == "We use the corpus C3."
    imp = normalize_item({"id": "q", "question": "q?", "answer": "", "is_impossible": True, "gold_context_ids": [], "evidence": ""})
    assert imp["reference_quote"] == "" and not imp["answerable"]


@pytest.fixture
def fake_ollama(monkeypatch):
    """_embed replacement: vector = [len(text)]; texts containing 'BAD' always fail, the first `flaky` calls fail once."""
    state = {"calls": [], "flaky": 0}

    def fake(texts, model):
        state["calls"].append(list(texts))
        if state["flaky"]:
            state["flaky"] -= 1
            raise ollama.ResponseError("tokenize: EOF", 400)
        if any("BAD" in t for t in texts):
            raise ollama.ResponseError("tokenize: EOF", 400)
        return [[float(len(t))] for t in texts]

    monkeypatch.setattr(embed, "_embed", fake)
    monkeypatch.setattr(embed, "RETRY_WAITS", (0, 0, 0))
    return state


def test_embed_retries_and_resumes_from_cache(tmp_path, fake_ollama):
    cfg, cache = Embedding(batch_size=2), tmp_path / "cache" / "m.jsonl"
    texts = ["a", "bb", "ccc", "dddd", "bb"]
    fake_ollama["flaky"] = 1
    vecs = embed.embed_documents(texts, cfg, cache)
    assert vecs == [[float(len(cfg.doc_prefix + t))] for t in texts]
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 4        # duplicate text embedded once
    fake_ollama["calls"].clear()
    assert embed.embed_documents(texts + ["eeeee"], cfg, cache)[-1] == [float(len(cfg.doc_prefix) + 5)]
    assert fake_ollama["calls"] == [[cfg.doc_prefix + "eeeee"]]             # only the new text


def test_embed_ignores_line_cut_off_by_a_crash(tmp_path, fake_ollama):
    cfg, cache = Embedding(batch_size=2), tmp_path / "m.jsonl"
    embed.embed_documents(["a", "bb"], cfg, cache)
    with open(cache, "a", encoding="utf-8") as f:
        f.write('{"k": "half')
    embed.embed_documents(["a", "bb", "ccc"], cfg, cache)
    assert len(embed._load_cache(cache)) == 3


def test_embed_names_a_text_that_keeps_failing(tmp_path, fake_ollama):
    with pytest.raises(RuntimeError, match="BAD chunk"):
        embed.embed_documents(["fine", "BAD chunk", "also fine"], Embedding(batch_size=4), tmp_path / "m.jsonl")


EV1 = "We train the tagger on the Penn Treebank with standard splits."
EV2 = "Results on the test set show a gain of two points in accuracy."


def test_evidence_recall_counts_each_paragraph():
    from rag.eval.metrics import evidence_recall
    quote = f"{EV1} ... {EV2}"
    assert evidence_recall(quote, [f"Intro. {EV1} More."], 0.6) == 0.5
    assert evidence_recall(quote, [EV2, f"x {EV1}"], 0.6) == 1.0
    assert evidence_recall("too short", ["too short"], 0.6) is None


def test_context_precision_is_rank_weighted():
    from rag.eval.metrics import context_precision
    other = "Something unrelated about a different topic entirely here."
    assert context_precision(EV1, [EV1, other, other], 0.6) == 1.0
    assert context_precision(EV1, [other, EV1, other], 0.6) == 0.5
    assert context_precision(EV1, [EV1, other, EV1], 0.6) == (1 + 2 / 3) / 2
    assert context_precision(EV1, [other], 0.6) == 0.0


def test_answer_overlap_normalises_like_qasper():
    from rag.eval.metrics import answer_overlap
    assert answer_overlap("The Penn Treebank [1].", "penn treebank") == (1.0, 1.0)
    f1, rec = answer_overlap("Yes, the categories were evaluated by humans [2].", "Yes")
    assert rec == 1.0 and 0 < f1 < 0.5
    assert answer_overlap("No idea", "Penn Treebank") == (0.0, 0.0)


def test_summarise_reports_evidence_answer_and_not_found_metrics():
    from rag.config import Settings
    from rag.eval.run_eval import summarise
    from rag.generate import NOT_FOUND
    other = "Something unrelated about a different topic entirely here."
    rows = [
        {"answerable": True, "rank": 1, "answer_rank": 2, "reference_quote": f"{EV1} ... {EV2}", "reference_answer": "Penn Treebank",
         "retrieved": [{"text": other}, {"text": EV1}], "answer": "The Penn Treebank [2]."},
        {"answerable": True, "rank": 1, "answer_rank": None, "reference_quote": EV2, "reference_answer": "two points",
         "retrieved": [{"text": other}], "answer": NOT_FOUND},
        {"answerable": False, "rank": None, "reference_quote": "", "reference_answer": "", "retrieved": [], "answer": NOT_FOUND},
    ]
    s = summarise(rows, Settings())
    assert s["evidence_recall@1"] == 0.0 and s["evidence_recall@5"] == 0.25
    assert s["context_precision"] == 0.25
    assert s["not_found_accuracy"] == 2 / 3 and s["answer_f1"] == 0.5 and s["answer_token_recall"] == 0.5

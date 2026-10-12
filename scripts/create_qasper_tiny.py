"""Build a tiny, reproducible RAG evaluation subset from QASPER (allenai/qasper, CC BY 4.0: NLP papers with questions).

  python scripts/create_qasper_tiny.py                # writes data/eval/qasper_tiny/ and the papers to data/documents/
  python scripts/create_qasper_tiny.py --force        # overwrite a previous output

Subset: 100 questions (80 answerable + 20 unanswerable) split into dev 32+8, val 24+6, test 24+6, one question per paper; the
100 papers the questions are about; and --distractors (default 100) other random papers. Questions come from QASPER's validation
and test splits and are kept only if at least two annotators answered them and all agree on whether the paper answers it.
Answerable questions also need text evidence (a table or figure alone is not retrievable here) and an answer that is not just a
citation marker. Everything is driven by one seed, so reruns give identical files (see manifest.json for the output hashes).

QASPER questions were written for one paper ("which datasets do they use?"), which is ambiguous in a corpus of 200 papers, so each
question is prefixed with its paper's title; the original is kept as question_original. Document-level hits are therefore easy;
`evidence` (the annotators' evidence paragraphs, joined with " ... ") is what run_eval scores as answer_hit@k.

Output: data/eval/qasper_tiny/{dev,val,test,contexts}.jsonl, manifest.json, sample_preview.md; and the papers as flat
data/documents/<paper id>.md (# title, ## sections, ### subsections). The parquet files are cached in data/raw/qasper/. On --force
only the files listed in the previous contexts.jsonl are removed from data/documents/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from create_techqa_tiny import jsonl, norm, one_line, previous_context_ids, text_hash  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPLITS = {"dev": (32, 8), "val": (24, 6), "test": (24, 6)}   # (answerable, unanswerable)
N_DISTRACTORS = 100
SOURCE_SPLITS = ("validation", "test")                       # the multi-annotator splits
MIN_ANNOTATORS = 2                                           # agreement means nothing with a single annotator
PARQUET_URL = "https://huggingface.co/api/datasets/allenai/qasper/parquet/qasper/{split}/0.parquet"
FLOAT = "FLOAT SELECTED"                                     # evidence that is a table or figure, not paper text
REF_ONLY = re.compile(r"^(?:[\s;,.()]|(?:BIB|TAB|FIG|SEC|EQ)REF\d+)*$")
PREVIEW_CHARS = 300


def load_papers(raw_dir: Path) -> list[dict]:
    """QASPER validation + test papers (downloaded once into raw_dir), sorted by paper id."""
    import pyarrow.parquet as pq

    raw_dir.mkdir(parents=True, exist_ok=True)
    papers = []
    for split in SOURCE_SPLITS:
        path = raw_dir / f"{split}.parquet"
        if not path.exists():
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(PARQUET_URL.format(split=split), tmp)
            tmp.replace(path)
        papers += [{**p, "source_split": split} for p in pq.read_table(path).to_pylist()]
    return sorted(papers, key=lambda p: p["id"])


def one_paragraph(text: str) -> str:
    """A paragraph as one line; a leading '#' is escaped so the Markdown parser does not read it as a heading."""
    s = re.sub(r"\s+", " ", text).strip()
    return "\\" + s if s.startswith("#") else s


def paper_markdown(p: dict) -> str:
    """# title, ## Abstract, then the sections: 'A ::: B' section names become ## A / ### B, a heading only when it changes."""
    lines = [f"# {one_line(p['title'], 10_000)}", "", "## Abstract", "", one_paragraph(p["abstract"] or ""), ""]
    ft = p["full_text"] or {}
    prev: list[str] = []
    for name, paras in zip(ft.get("section_name") or [], ft.get("paragraphs") or []):
        parts = [one_line(s, 500) for s in (name or "").split(":::") if s.strip()]
        same = 0
        while same < min(len(parts), len(prev)) and parts[same] == prev[same]:
            same += 1
        for level in range(same, len(parts)):
            lines += [f"{'#' * min(level + 2, 6)} {parts[level]}", ""]
        prev = parts
        lines += [x for para in paras or [] if para.strip() for x in (one_paragraph(para), "")]
    captions = [c for c in (p.get("figures_and_tables") or {}).get("caption") or [] if c and c.strip()]
    if captions:
        lines += ["## Figures and Tables", ""] + [x for c in captions for x in (one_paragraph(c), "")]
    return "\n".join(lines).rstrip() + "\n"


def answer_of(a: dict) -> tuple[str, str]:
    """(reference answer, answer type) of one annotation."""
    if a["unanswerable"]:
        return "", "none"
    spans = [s.strip() for s in a.get("extractive_spans") or [] if s and s.strip()]
    if spans:
        return "; ".join(spans), "extractive"
    if (a.get("free_form_answer") or "").strip():
        return a["free_form_answer"].strip(), "abstractive"
    if a.get("yes_no") is not None:
        return ("Yes" if a["yes_no"] else "No"), "yes_no"
    return "", "empty"


def text_evidence(a: dict) -> list[str]:
    return [e.strip() for e in a.get("evidence") or [] if e and e.strip() and not e.strip().startswith(FLOAT)]


def candidate_questions(p: dict, doc_text: str, dropped: Counter) -> list[dict]:
    """The paper's usable questions as subset rows (see the module docstring for the rules); counts what was dropped and why."""
    doc_norm = norm(doc_text)
    qas, out = p["qas"] or {}, []
    for question, qid, answers in zip(qas.get("question") or [], qas.get("question_id") or [], qas.get("answers") or []):
        anns = (answers or {}).get("answer") or []
        if not question.strip() or len(anns) < MIN_ANNOTATORS:
            dropped["fewer_than_2_annotators"] += 1
            continue
        labels = {bool(a["unanswerable"]) for a in anns}
        if len(labels) > 1:
            dropped["annotators_disagree_on_answerable"] += 1
            continue
        row = {"id": qid, "paper_id": p["id"], "question": f'In the paper "{one_line(p["title"], 300)}": {question.strip()}',
               "question_original": question.strip(), "n_annotators": len(anns)}
        if labels == {True}:
            out.append(row | {"answer": "", "answer_type": "none", "is_impossible": True, "gold_context_ids": [], "evidence": ""})
            continue
        first = next(((text, kind) for a in anns for text, kind in [answer_of(a)]
                      if text and not REF_ONLY.match(text) and text_evidence(a)), None)
        if first is None:
            dropped["no_text_evidence_or_usable_answer"] += 1
            continue
        evidence = list(dict.fromkeys(e for a in anns for e in text_evidence(a)))
        if not all(norm(e) in doc_norm for e in evidence):
            dropped["evidence_not_found_in_paper_text"] += 1
            continue
        out.append(row | {"answer": first[0], "answer_type": first[1], "is_impossible": False,
                          "gold_context_ids": [f"{p['id']}.md"], "evidence": " ... ".join(evidence)})
    return out


def build(papers: list[dict], seed: int, n_distractors: int = N_DISTRACTORS) -> dict:
    rng = random.Random(seed)
    docs, seen_hashes, dropped = {}, set(), Counter()
    for p in papers:                                         # one document per paper, unique by id and by text
        text = paper_markdown(p)
        h = text_hash(text)
        if p["id"] in docs or h in seen_hashes or not (p["full_text"] or {}).get("paragraphs"):
            dropped["duplicate_or_empty_paper"] += 1
            continue
        docs[p["id"]] = text
        seen_hashes.add(h)
    by_paper = {p["id"]: candidate_questions(p, docs[p["id"]], dropped) for p in papers if p["id"] in docs}

    n_ans = sum(a for a, _ in SPLITS.values())
    n_imp = sum(i for _, i in SPLITS.values())
    order = sorted(by_paper)
    rng.shuffle(order)
    imp, ans, used = [], [], set()
    for pid in order:                                        # unanswerable questions are rarer, so they pick their papers first
        cands = [r for r in by_paper[pid] if r["is_impossible"]]
        if cands and len(imp) < n_imp:
            imp.append(rng.choice(cands))
            used.add(pid)
    for pid in order:
        cands = [r for r in by_paper[pid] if not r["is_impossible"]]
        if cands and pid not in used and len(ans) < n_ans:
            ans.append(rng.choice(cands))
            used.add(pid)
    if len(ans) < n_ans or len(imp) < n_imp:
        raise SystemExit(f"not enough usable questions: {len(ans)} answerable, {len(imp)} unanswerable")
    distractors = [pid for pid in order if pid not in used][:n_distractors]
    if len(distractors) < n_distractors:
        raise SystemExit(f"only {len(distractors)} papers left for distractors")

    splits, ia, ii = {}, 0, 0
    for name, (a, i) in SPLITS.items():
        chunk = ans[ia:ia + a] + imp[ii:ii + i]
        rng.shuffle(chunk)
        splits[name] = chunk
        ia, ii = ia + a, ii + i

    role = {r["paper_id"]: "gold" for r in ans} | {r["paper_id"]: "unanswerable" for r in imp} | {pid: "distractor" for pid in distractors}
    rank = {"gold": 0, "unanswerable": 1, "distractor": 2}
    contexts = [{"context_id": f"{pid}.md", "role": r, "text": docs[pid]} for pid, r in sorted(role.items(), key=lambda kv: (rank[kv[1]], kv[0]))]
    n_candidates = Counter("unanswerable" if r["is_impossible"] else "answerable" for rows in by_paper.values() for r in rows)
    return {"splits": splits, "contexts": contexts, "dropped": dict(sorted(dropped.items())), "n_distractors": n_distractors,
            "pool": {"papers": len(papers), "usable_questions": dict(n_candidates),
                     "papers_with_usable_questions": sum(bool(v) for v in by_paper.values())}}


def check(result: dict) -> None:
    """Hard guarantees about the subset; raises AssertionError if any is broken."""
    splits, contexts = result["splits"], result["contexts"]
    n_ans = sum(a for a, _ in SPLITS.values())
    n_q = n_ans + sum(i for _, i in SPLITS.values())
    for name, (a, i) in SPLITS.items():
        rows = splits[name]
        assert sum(not r["is_impossible"] for r in rows) == a and sum(r["is_impossible"] for r in rows) == i, name
    all_rows = [r for rows in splits.values() for r in rows]
    assert len(all_rows) == n_q and len({r["id"] for r in all_rows}) == n_q, "question ids"
    assert len({norm(r["question"]) for r in all_rows}) == n_q, "duplicate question text"
    assert len({r["paper_id"] for r in all_rows}) == n_q, "one question per paper (no paper in two splits)"
    by_id = {c["context_id"]: c for c in contexts}
    assert len(by_id) == len(contexts) == n_q + result["n_distractors"], "context count / ids"
    hashes = [text_hash(c["text"]) for c in contexts]
    assert len(set(hashes)) == len(hashes), "contexts must be unique by hash"
    for r in all_rows:
        doc = by_id[f"{r['paper_id']}.md"]
        if r["is_impossible"]:
            assert doc["role"] == "unanswerable" and not r["answer"] and not r["evidence"] and not r["gold_context_ids"], r["id"]
        else:
            assert doc["role"] == "gold" and r["answer"] and r["gold_context_ids"] == [doc["context_id"]], r["id"]
            assert all(norm(e) in norm(doc["text"]) for e in r["evidence"].split(" ... ")), f"{r['id']}: evidence not in paper"
    assert sum(c["role"] == "distractor" for c in contexts) == result["n_distractors"], "distractor count"


def title_of(text: str) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    return one_line(first.removeprefix("#").strip(), 110)


def preview(result: dict, seed: int) -> str:
    splits, contexts = result["splits"], result["contexts"]
    lens = [len(c["text"]) for c in contexts]
    roles = Counter(c["role"] for c in contexts)
    out = ["# QASPER tiny subset: sample preview", "",
           f"seed {seed}; {sum(len(v) for v in splits.values())} questions; papers: {roles['gold']} with an answerable question, "
           f"{roles['unanswerable']} with an unanswerable one, {roles['distractor']} random distractors.", "",
           "| split | answerable | unanswerable | total |", "|---|---|---|---|"]
    for name, rows in splits.items():
        a = sum(not r["is_impossible"] for r in rows)
        out.append(f"| {name} | {a} | {len(rows) - a} | {len(rows)} |")
    types = Counter(r["answer_type"] for rows in splits.values() for r in rows)
    out += ["", f"Answer types: {dict(sorted(types.items()))}.",
            f"Paper length (chars): median {int(statistics.median(lens))}, max {max(lens)}.", ""]
    for name, rows in splits.items():
        out += [f"## {name}", ""]
        for r in [r for r in rows if not r["is_impossible"]][:2] + [r for r in rows if r["is_impossible"]][:1]:
            out += [f"### {r['id']} ({'unanswerable' if r['is_impossible'] else r['answer_type']}, paper `{r['paper_id']}`)", "",
                    f"**Question:** {one_line(r['question'], 400)}", ""]
            if r["is_impossible"]:
                out += ["**Answer:** none (the paper does not answer it)", ""]
            else:
                out += [f"**Answer:** {one_line(r['answer'], 400)}", "", f"**Evidence:** {one_line(r['evidence'], PREVIEW_CHARS)}", ""]
    out += ["## Distractor examples", ""]
    out += [f"- `{c['context_id']}`: {title_of(c['text'])}" for c in [c for c in contexts if c["role"] == "distractor"][:5]]
    return "\n".join(out) + "\n"


def write(result: dict, out: Path, docs_out: Path, seed: int, force: bool) -> None:
    if out.exists() and any(out.iterdir()) and not force:
        raise SystemExit(f"{out} is not empty; rerun with --force to overwrite")
    previous = previous_context_ids(out)
    foreign = sorted(c["context_id"] for c in result["contexts"] if (docs_out / c["context_id"]).exists() and c["context_id"] not in previous)
    if foreign:
        raise SystemExit(f"{len(foreign)} files already exist in {docs_out} and were not written by this script (e.g. {foreign[0]}); not overwriting")
    docs_out.mkdir(parents=True, exist_ok=True)
    for name in previous:                       # only this script's own earlier files; never a glob in the shared folder
        (docs_out / name).unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {f"{name}.jsonl": jsonl(rows) for name, rows in result["splits"].items()}
    files["contexts.jsonl"] = jsonl(result["contexts"])
    files["sample_preview.md"] = preview(result, seed)
    for rel, content in files.items():
        with open(out / rel, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
    doc_hashes = {}
    for c in result["contexts"]:
        with open(docs_out / c["context_id"], "w", encoding="utf-8", newline="\n") as f:
            f.write(c["text"])
        doc_hashes[c["context_id"]] = hashlib.sha256((docs_out / c["context_id"]).read_bytes()).hexdigest()

    all_rows = [r for rows in result["splits"].values() for r in rows]
    lens = [len(c["text"]) for c in result["contexts"]]
    manifest = {
        "seed": seed,
        "source": "allenai/qasper (CC BY 4.0), splits: " + ", ".join(SOURCE_SPLITS),
        "splits": {n: {"answerable": sum(not r["is_impossible"] for r in rows),
                       "unanswerable": sum(r["is_impossible"] for r in rows)} for n, rows in result["splits"].items()},
        "answer_types": dict(sorted(Counter(r["answer_type"] for r in all_rows).items())),
        "contexts": dict(sorted(Counter(c["role"] for c in result["contexts"]).items()))
                    | {"chars_median": int(statistics.median(lens)), "chars_max": max(lens)},
        "pool": result["pool"] | {"dropped": result["dropped"]},
        "docs_dir": str(docs_out),
        "docs_sha256": hashlib.sha256("".join(f"{n}:{h}\n" for n, h in sorted(doc_hashes.items())).encode()).hexdigest(),
        "sha256": {rel: hashlib.sha256((out / rel).read_bytes()).hexdigest() for rel in sorted(files)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "data/raw/qasper", help="where the QASPER parquet files are cached")
    ap.add_argument("--out", type=Path, default=ROOT / "data/eval/qasper_tiny", help="questions, contexts, manifest, preview")
    ap.add_argument("--docs-out", type=Path, default=ROOT / "data/documents",
                    help="where the .md papers go, flat (the ingest loader reads this folder non-recursively)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--distractors", type=int, default=N_DISTRACTORS, help="papers besides the ones the questions are about")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty output folder and this script's earlier documents")
    args = ap.parse_args()

    result = build(load_papers(args.raw), args.seed, args.distractors)
    check(result)
    write(result, args.out, args.docs_out, args.seed, args.force)
    print(f"wrote {args.out} and {len(result['contexts'])} papers to {args.docs_out}")
    for name, (a, i) in SPLITS.items():
        print(f"  {name}: {a} answerable + {i} unanswerable")
    print(f"  usable questions in the pool: {result['pool']['usable_questions']}; dropped: {result['dropped']}")
    print(f"  preview: {args.out / 'sample_preview.md'}")


if __name__ == "__main__":
    main()

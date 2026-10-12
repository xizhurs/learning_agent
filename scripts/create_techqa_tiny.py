"""Build a tiny, reproducible RAG evaluation subset from TechQA-RAG-Eval.

  python scripts/create_techqa_tiny.py                # writes data/eval/techqa_tiny/
  python scripts/create_techqa_tiny.py --force        # overwrite a previous output

Subset: 100 questions (80 answerable + 20 impossible) split into dev 48+12, val 16+4, test 16+4; the 80 gold contexts; and
300 random distractor documents from the corpus. Everything is deduplicated by a hash of the normalized text and driven by one
seed, so reruns give identical files (see manifest.json for the output hashes).

Output: data/eval/techqa_tiny/{dev,val,test,contexts}.jsonl, manifest.json, sample_preview.md; and the 380 documents as flat
data/documents/<filename>.txt (the ingest loader lists that folder non-recursively, so the source data/documents/corpus/ is not
picked up). On --force only the files listed in the previous contexts.jsonl are removed from data/documents/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = {"dev": (48, 12), "val": (16, 4), "test": (16, 4)}   # (answerable, impossible)
N_DISTRACTORS = 300
PREVIEW_CHARS = 300


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def text_hash(text: str) -> str:
    return hashlib.sha1(norm(text).encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def clean_questions(rows: list[dict]) -> tuple[list[dict], dict]:
    """Drop empty/duplicate questions and questions that appear with both labels. Returns (rows, counts of what was dropped)."""
    by_question = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        by_question[norm(r["question"])].append(r)
    kept, dropped = [], defaultdict(int)
    for group in by_question.values():
        if len({bool(r["is_impossible"]) for r in group}) > 1:
            dropped["question_with_both_labels"] += len(group)
            continue
        dropped["repeated_question_text"] += len(group) - 1
        kept.append(group[0])
    return sorted(kept, key=lambda r: r["id"]), dict(dropped)


def usable_answerable(r: dict) -> bool:
    return (not r["is_impossible"] and r["answer"].strip() not in ("", "-") and len(r["contexts"]) == 1
            and bool(r["contexts"][0]["text"].strip()))


def pick_one_per_gold(rows: list[dict]) -> list[dict]:
    """Keep the first question per unique gold passage, so a passage cannot land in two splits."""
    seen, out = set(), []
    for r in rows:
        h = text_hash(r["contexts"][0]["text"])
        if h not in seen:
            seen.add(h)
            out.append(r)
    return out


def pick_distractors(corpus: Path, exclude_hashes: set[str], exclude_names: set[str], n: int, rng: random.Random) -> list[tuple[str, str]]:
    """n random (filename, text) corpus documents, unique by normalized hash and not among the gold passages.

    Files are visited in a seeded random order and read lazily, so the whole corpus is not read."""
    names = sorted(p.name for p in corpus.glob("*.txt"))
    rng.shuffle(names)
    seen, out = set(exclude_hashes), []
    for name in names:
        if name in exclude_names:
            continue
        text = read_text(corpus / name)
        h = text_hash(text)
        if not text.strip() or h in seen:
            continue
        seen.add(h)
        out.append((name, text))
        if len(out) == n:
            return out
    raise SystemExit(f"only {len(out)} usable distractors in {corpus}")


def build(train: Path, corpus: Path, seed: int) -> dict:
    rng = random.Random(seed)
    rows = json.loads(read_text(train))
    cleaned, dropped = clean_questions(rows)
    answerable = pick_one_per_gold([r for r in cleaned if usable_answerable(r)])
    impossible = [r for r in cleaned if r["is_impossible"]]
    n_ans = sum(a for a, _ in SPLITS.values())
    n_imp = sum(i for _, i in SPLITS.values())
    if len(answerable) < n_ans or len(impossible) < n_imp:
        raise SystemExit(f"not enough questions after cleaning: {len(answerable)} answerable, {len(impossible)} impossible")
    ans, imp = rng.sample(answerable, n_ans), rng.sample(impossible, n_imp)

    splits, ia, ii = {}, 0, 0
    for name, (a, i) in SPLITS.items():
        chunk = ans[ia:ia + a] + imp[ii:ii + i]
        rng.shuffle(chunk)
        splits[name] = chunk
        ia, ii = ia + a, ii + i

    gold = {r["contexts"][0]["filename"]: r["contexts"][0]["text"] for r in ans}
    distractors = pick_distractors(corpus, {text_hash(t) for t in gold.values()}, set(gold), N_DISTRACTORS, rng)
    contexts = [{"context_id": n, "role": "gold", "text": t} for n, t in sorted(gold.items())] + \
               [{"context_id": n, "role": "distractor", "text": t} for n, t in sorted(distractors)]
    return {"splits": splits, "contexts": contexts, "dropped": dropped,
            "pool": {"answerable_unique_gold": len(answerable), "impossible": len(impossible), "rows": len(rows)}}


def check(result: dict) -> None:
    """Hard guarantees about the subset; raises AssertionError if any is broken."""
    splits, contexts = result["splits"], result["contexts"]
    for name, (a, i) in SPLITS.items():
        rows = splits[name]
        assert sum(not r["is_impossible"] for r in rows) == a and sum(bool(r["is_impossible"]) for r in rows) == i, name
    all_rows = [r for rows in splits.values() for r in rows]
    assert len(all_rows) == 100 and len({r["id"] for r in all_rows}) == 100, "question ids"
    assert len({norm(r["question"]) for r in all_rows}) == 100, "duplicate question text"
    gold_hashes = [text_hash(r["contexts"][0]["text"]) for r in all_rows if not r["is_impossible"]]
    assert len(gold_hashes) == 80 and len(set(gold_hashes)) == 80, "gold passages must be distinct (no leakage across splits)"
    hashes = [text_hash(c["text"]) for c in contexts]
    assert len(contexts) == 80 + N_DISTRACTORS and len(set(hashes)) == len(hashes), "contexts must be unique by hash"
    assert not set(hashes[:80]) & set(hashes[80:]), "a distractor equals a gold passage"
    assert len({c["context_id"] for c in contexts}) == len(contexts), "context ids"


def jsonl(rows: list[dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"


def one_line(text: str, limit: int) -> str:
    s = re.sub(r"\s+", " ", text).strip()
    return s if len(s) <= limit else s[:limit].rstrip() + "…"


def title_of(text: str) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    return one_line(first.removeprefix("Title:").strip(), 110)


def preview(result: dict, seed: int) -> str:
    splits, contexts = result["splits"], result["contexts"]
    lens = [len(c["text"]) for c in contexts]
    out = ["# TechQA tiny subset: sample preview", "",
           f"seed {seed}; {sum(len(v) for v in splits.values())} questions, {sum(c['role'] == 'gold' for c in contexts)} gold contexts, "
           f"{sum(c['role'] == 'distractor' for c in contexts)} distractors.", "",
           "| split | answerable | impossible | total |", "|---|---|---|---|"]
    for name, rows in splits.items():
        a = sum(not r["is_impossible"] for r in rows)
        out.append(f"| {name} | {a} | {len(rows) - a} | {len(rows)} |")
    out += ["", f"Context length (chars): median {int(statistics.median(lens))}, max {max(lens)}; "
                f"{sum(n > 20000 for n in lens)} contexts over 20,000 chars (will need chunking).", ""]
    for name, rows in splits.items():
        out += [f"## {name}", ""]
        for r in [r for r in rows if not r["is_impossible"]][:2] + [r for r in rows if r["is_impossible"]][:1]:
            out += [f"### {r['id']} ({'impossible' if r['is_impossible'] else 'answerable'})", "",
                    f"**Question:** {one_line(r['question'], 400)}", "", f"**Answer:** {one_line(r['answer'], 400)}", ""]
            if r["contexts"]:
                c = r["contexts"][0]
                out += [f"**Gold context** `{c['filename']}`: {one_line(c['text'], PREVIEW_CHARS)}", ""]
            else:
                out += ["**Gold context:** none (no answer in the corpus)", ""]
    out += ["## Distractor examples", ""]
    out += [f"- `{c['context_id']}`: {title_of(c['text'])}" for c in [c for c in contexts if c["role"] == "distractor"][:5]]
    return "\n".join(out) + "\n"


def previous_context_ids(out: Path) -> set[str]:
    """Document filenames written by an earlier run (the only files in the shared documents folder that this script may remove)."""
    path = out / "contexts.jsonl"
    if not path.exists():
        return set()
    return {json.loads(line)["context_id"] for line in read_text(path).splitlines() if line.strip()}


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
    if (out / "docs").exists():                 # legacy location of the documents
        shutil.rmtree(out / "docs")
    out.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    for name, rows in result["splits"].items():
        slim = [{"id": r["id"], "question": r["question"], "answer": r["answer"], "is_impossible": bool(r["is_impossible"]),
                 "gold_context_ids": [c["filename"] for c in r["contexts"]]} for r in rows]
        files[f"{name}.jsonl"] = jsonl(slim)
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

    lens = [len(c["text"]) for c in result["contexts"]]
    manifest = {
        "seed": seed,
        "splits": {n: {"answerable": sum(not r["is_impossible"] for r in rows),
                       "impossible": sum(bool(r["is_impossible"]) for r in rows)} for n, rows in result["splits"].items()},
        "contexts": {"gold": sum(c["role"] == "gold" for c in result["contexts"]),
                     "distractor": sum(c["role"] == "distractor" for c in result["contexts"]),
                     "chars_median": int(statistics.median(lens)), "chars_max": max(lens)},
        "source": result["pool"] | {"dropped": result["dropped"]},
        "docs_dir": str(docs_out),
        "docs_sha256": hashlib.sha256("".join(f"{n}:{h}\n" for n, h in sorted(doc_hashes.items())).encode()).hexdigest(),
        "sha256": {rel: hashlib.sha256((out / rel).read_bytes()).hexdigest() for rel in sorted(files)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, default=ROOT / "data/eval/train.json")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data/documents/corpus")
    ap.add_argument("--out", type=Path, default=ROOT / "data/eval/techqa_tiny", help="questions, contexts, manifest, preview")
    ap.add_argument("--docs-out", type=Path, default=ROOT / "data/documents",
                    help="where the 380 .txt documents go, flat (the ingest loader reads this folder non-recursively)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty output folder and this script's earlier documents")
    args = ap.parse_args()
    for p in (args.train, args.corpus):
        if not p.exists():
            sys.exit(f"missing input: {p}")

    result = build(args.train, args.corpus, args.seed)
    check(result)
    write(result, args.out, args.docs_out, args.seed, args.force)
    print(f"wrote {args.out} and {len(result['contexts'])} documents to {args.docs_out}")
    for name, (a, i) in SPLITS.items():
        print(f"  {name}: {a} answerable + {i} impossible")
    print(f"  contexts: 80 gold + {N_DISTRACTORS} distractors; dropped while cleaning: {result['dropped']}")
    print(f"  preview: {args.out / 'sample_preview.md'}")


if __name__ == "__main__":
    main()

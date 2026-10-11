"""Ground-truth test set via the Claude app (no API needed).

  python -m rag.eval.gen_testset export [--packets 6] [--per-packet 8] [--seed 0] [--set NAME]
      -> data/eval/packets/packet_NN.md          (paste/upload into the Claude app)
         data/eval/packets/packet_NN.passages.json (internal: passage ids -> text; do not upload)
  Save each Claude reply (JSON only) as data/eval/replies/packet_NN.json, then:
  python -m rag.eval.gen_testset import [--set NAME]
      -> data/eval/testset.jsonl  (+ data/eval/import_report.txt listing rejected items)

Without --set this is the dev set (used for tuning / ablations). A named set (e.g. --set holdout) lives in
data/eval/<set>/{packets,replies}/ -> data/eval/testset_<set>.jsonl, and its export skips every passage already
used by another set, so the holdout questions come from unseen passages.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from rag.config import Chunking, get_settings
from rag.eval.metrics import _shingles, _words
from rag.ingest.chunk import chunk_document, count_tokens
from rag.ingest.parse import list_documents, parse_document

QUESTION_TYPES = ["factual", "numeric", "definition", "procedure/requirement", "table", "multi_passage", "unanswerable"]

INSTRUCTIONS = """\
You are helping build an evaluation set for a question-answering system over carbon-credit
methodology / project documents. Below are {n} numbered passages ({ids}) taken from those documents.

Write exactly {total} questions as a JSON array and output ONLY the JSON (no commentary, no code fences).

Rules
1. For each passage P1..P{n} write ONE question that the passage answers. Vary the style across the set:
   factual, numeric (a value/threshold/equation parameter), definition, requirement/procedure, and
   table lookup (only for passages that are tables).
2. Write ONE "multi_passage" question that needs information from exactly two of the passages.
3. Write ONE "unanswerable" question: plausible for these documents but NOT answered by any passage below.
4. Every question must be fully self-contained, as a user would type it into a chatbot with no passage
   in front of them. Never say "the passage", "this section", "P3", "the text above" or similar. Name the
   methodology / concept explicitly instead of using pronouns.
5. reference_answer: a complete, concise answer using only facts in the passage(s). Copy numbers, units and
   terms exactly. For unanswerable use "NOT_ANSWERABLE".
6. quote: 1-3 sentences (or table rows) copied VERBATIM from the passage that support the answer. For
   multi_passage give the quotes of both passages joined with " ... ". For unanswerable use "".
7. passage_ids: list of passage ids used, e.g. ["P3"]; [] for unanswerable.

JSON schema for each element:
{{"question": str, "reference_answer": str, "passage_ids": [str], "quote": str,
  "type": "factual" | "numeric" | "definition" | "procedure/requirement" | "table" | "multi_passage" | "unanswerable"}}

PASSAGES
========
"""

CHUNKING_FOR_PASSAGES = Chunking(strategy="heading_aware", max_tokens=700, overlap=0, heading_prefix=False)


def _eval_dir() -> Path:
    return get_settings().paths.resolve("eval_dir")


def _set_dir(name: str) -> Path:
    """Folder holding packets/ and replies/ for a set ('' = the original dev set)."""
    return _eval_dir() / name if name else _eval_dir()


def _testset_path(name: str) -> Path:
    return _eval_dir() / (f"testset_{name}.jsonl" if name else "testset.jsonl")


def _used_passage_texts(exclude_set: str) -> set[str]:
    """Normalised texts of passages already exported by any other set."""
    ev = _eval_dir()
    used = set()
    for pfile in [*ev.glob("packets/packet_*.passages.json"), *ev.glob("*/packets/packet_*.passages.json")]:
        if exclude_set and pfile.parent.parent.name == exclude_set:
            continue
        used |= {_norm(p["text"]) for p in json.loads(pfile.read_text(encoding="utf-8")).values()}
    return used


def _norm(s: str) -> str:
    s = re.sub(r"[|\-:*_`]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _candidate_passages(settings) -> list[dict]:
    out = []
    for path in list_documents(settings):
        doc = parse_document(path, settings)
        for c in chunk_document(doc, CHUNKING_FOR_PASSAGES):
            body = c.text
            if count_tokens(body) < 60:
                continue
            if body.count("....") >= 3 or "contents" in c.section.lower():
                continue
            out.append({"source": c.source, "page": c.page, "section": c.section, "kind": c.kind, "text": body})
    return out


def export(packets: int, per_packet: int, seed: int, set_name: str = "") -> None:
    settings = get_settings()
    pool = _candidate_passages(settings)
    need = packets * per_packet
    if set_name:
        used = _used_passage_texts(set_name)
        before = len(pool)
        pool = [p for p in pool if _norm(p["text"]) not in used]
        print(f"excluded {before - len(pool)} passages used by other sets; {len(pool)} remain")
        if len(pool) < need:
            print(f"WARNING: only {len(pool)} unused passages for {need} requested")
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = {}
    for p in pool:
        by_source.setdefault(p["source"], []).append(p)
    # equal share per source, tables over-sampled a little so table questions exist
    chosen: list[dict] = []
    per_source = max(1, need // len(by_source))
    for src, ps in by_source.items():
        tables = [p for p in ps if p["kind"] == "table"]
        texts = [p for p in ps if p["kind"] != "table"]
        n_tab = min(len(tables), max(1, per_source // 4))
        chosen += rng.sample(tables, n_tab) + rng.sample(texts, min(len(texts), per_source - n_tab))
    rng.shuffle(chosen)
    chosen = chosen[:need]

    out_dir = _set_dir(set_name) / "packets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(packets):
        group = chosen[i * per_packet : (i + 1) * per_packet]
        if not group:
            break
        ids = [f"P{j + 1}" for j in range(len(group))]
        passages = {pid: g for pid, g in zip(ids, group)}
        md = INSTRUCTIONS.format(n=len(group), ids=", ".join(ids), total=len(group) + 2)
        for pid, g in passages.items():
            page = f", page {g['page']}" if g["page"] else ""
            md += f"\n[{pid}] source: {g['source']}{page} | section: {g['section'] or 'n/a'} | type: {g['kind']}\n{g['text']}\n"
        (out_dir / f"packet_{i + 1:02d}.md").write_text(md, encoding="utf-8")
        (out_dir / f"packet_{i + 1:02d}.passages.json").write_text(
            json.dumps(passages, ensure_ascii=False, indent=1), encoding="utf-8")
    replies = _set_dir(set_name) / "replies"
    replies.mkdir(parents=True, exist_ok=True)
    set_arg = f" --set {set_name}" if set_name else ""
    print(f"wrote {min(packets, -(-len(chosen) // per_packet))} packets to {out_dir}")
    print("Next: paste each packet_NN.md into the Claude app, save the JSON-only reply as "
          f"{replies / 'packet_NN.json'}, then run: python -m rag.eval.gen_testset import{set_arg}")


def _load_reply(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    data = json.loads(raw)
    if isinstance(data, dict):
        data = data.get("questions", [data])
    return data


def _quote_in_passages(quote: str, texts: list[str]) -> bool:
    """Verbatim check tolerant to markdown/table punctuation and whitespace; each ' ... ' segment checked separately."""
    haystack = " ".join(_norm(t) for t in texts)
    for seg in re.split(r"\s*(?:\.\.\.|…)\s*", quote):
        seg = _norm(seg)
        if len(seg) < 8:
            continue
        if seg in haystack:
            continue
        toks = seg.split()
        hay_toks = set(haystack.split())
        if sum(t in hay_toks for t in toks) / len(toks) < 0.95:
            return False
    return True


def _quote_scoreable(quote: str) -> bool:
    """Same rule as metrics.quote_coverage: a segment needs >= 2 word 3-grams, else it can never count as a hit."""
    return any(len(_shingles(_words(seg))) >= 2 for seg in re.split(r"\s*(?:\.\.\.|…)\s*", quote))


def import_replies(set_name: str = "") -> None:
    sd = _set_dir(set_name)
    items, rejected = [], []
    for pfile in sorted((sd / "packets").glob("packet_*.passages.json")):
        stem = pfile.name.split(".")[0]
        rfile = sd / "replies" / f"{stem}.json"
        if not rfile.exists():
            rejected.append(f"{stem}: no reply file at {rfile}")
            continue
        passages = json.loads(pfile.read_text(encoding="utf-8"))
        try:
            reply = _load_reply(rfile)
        except json.JSONDecodeError as e:
            rejected.append(f"{stem}: invalid JSON ({e})")
            continue
        for k, r in enumerate(reply):
            tag = f"{stem}#{k + 1}"
            q, ans = (r.get("question") or "").strip(), (r.get("reference_answer") or "").strip()
            pids, quote, qtype = r.get("passage_ids") or [], (r.get("quote") or "").strip(), r.get("type", "")
            answerable = qtype != "unanswerable" and ans.upper() != "NOT_ANSWERABLE"
            if not q or not ans:
                rejected.append(f"{tag}: missing question/answer")
                continue
            if re.search(r"\b(the passage|this passage|passage \d|P\d+\b|text above|this section)\b", q, re.I):
                rejected.append(f"{tag}: question refers to the passage: {q}")
                continue
            texts = [passages[p]["text"] for p in pids if p in passages]
            if answerable:
                if not texts or len(texts) != len(pids):
                    rejected.append(f"{tag}: unknown passage ids {pids}")
                    continue
                if not quote or not _quote_in_passages(quote, texts):
                    rejected.append(f"{tag}: quote not found verbatim in passage: {quote[:80]!r}")
                    continue
                if not _quote_scoreable(quote):
                    rejected.append(f"{tag}: quote too short to score: {quote[:80]!r}")
                    continue
            first = passages[pids[0]] if pids and pids[0] in passages else {}
            prefix = f"{set_name}_" if set_name else ""
            items.append({
                "id": f"{prefix}{stem}-{k + 1:02d}", "question": q, "reference_answer": ans if answerable else "",
                "reference_contexts": texts, "reference_quote": quote if answerable else "",
                "type": qtype, "answerable": answerable,
                "source": first.get("source", ""), "page": first.get("page", 0), "section": first.get("section", ""),
            })
    out = _testset_path(set_name)
    out.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + ("\n" if items else ""), encoding="utf-8")
    (sd / "import_report.txt").write_text("\n".join(rejected) or "no rejections", encoding="utf-8")
    n_un = sum(not i["answerable"] for i in items)
    print(f"{len(items)} questions imported ({n_un} unanswerable), {len(rejected)} rejected -> {out}")
    for r in rejected:
        print("  rejected:", r)
    print(f"Review {out} by hand (delete bad items) before running the evaluation.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--packets", type=int, default=6)
    e.add_argument("--per-packet", type=int, default=8)
    e.add_argument("--seed", type=int, default=0)
    i = sub.add_parser("import")
    for p in (e, i):
        p.add_argument("--set", dest="set_name", default="",
                       help="named set, e.g. holdout (default: the dev set data/eval/testset.jsonl)")
    args = ap.parse_args()
    if args.cmd == "export":
        export(args.packets, args.per_packet, args.seed, args.set_name)
    else:
        import_replies(args.set_name)


if __name__ == "__main__":
    main()

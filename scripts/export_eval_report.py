"""Export `ablate` / `run_eval run` results to Excel, CSV and JSONL so retrieval strategies can be compared and failures inspected.

  uv run python scripts/export_eval_report.py                          # reads data/eval/runs/, writes outputs/eval_reports/
  uv run python scripts/export_eval_report.py --only "*ablation_techqa_tiny_dev*"
  uv run python scripts/export_eval_report.py --phoenix-url http://localhost:6006   # also fill token counts from Phoenix traces

Reads every folder under --runs (depth 1 or 2) that contains items.jsonl (+ config.json, summary.json): one folder per variant of an
ablation, or one folder per `run_eval run`. Nothing in the evaluation is changed or re-run; everything is derived from the saved rows.

Output (--out, default outputs/eval_reports/):
  rag_ablation_report.xlsx   sheets: summary_by_run, per_question_results, retrieval_details, answer_details, failure_analysis, config
                             (text cells cut to 1000 characters, header frozen, autofilter on)
  per_question_results.csv, failure_analysis.csv   full text
  raw_results.jsonl          one line per run/question with the complete saved row, run config and derived fields (nothing cut)

How the derived columns are defined:
  gold_context_*          document level: a retrieved chunk counts when its document (chunk id '<document>::NNNNN') is a gold document.
                          Gold ids come from the row, else from the test-set file named in summary.json, else from the row's `source`.
                          Rows without gold ids fall back to the stored answer-quote `rank` (summary column recall_basis says which).
  answer_correct          unanswerable question: the model refused (rag.eval.metrics.is_refusal). Answerable: a refusal is wrong; otherwise
                          Claude-app judge correctness >= 4 (judge_reply.json in the run folder) or RAGAS factual_correctness(mode=f1) >=
                          --correct-threshold; empty if the answer was not scored.
  answer_score            0-1: RAGAS factual_correctness, or (Claude correctness - 1) / 4
  citation_correct        heuristic, answerable non-refusal answers only: every [n] is within the retrieved list and one cites a gold document
  failure_type            first match of: retrieval_miss, false_refusal, missed_refusal, wrong_answer, low_faithfulness, bad_citation,
                          late_hit (gold document first at rank > 3). Any further problems are listed in `notes`.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from rag.eval.metrics import first_doc_rank, is_refusal

ROOT = Path(__file__).resolve().parent.parent
EXCEL_MAX_CHARS = 1000
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
FACT_CORRECTNESS = "factual_correctness(mode=f1)"

SUMMARY_COLS = ["run_id", "retriever_type", "chunk_size", "chunk_overlap", "top_k", "reranker", "query_rewrite", "num_questions",
                "context_recall_at_1", "context_recall_at_3", "context_recall_at_5", "mrr", "answer_accuracy", "faithfulness_avg",
                "not_found_accuracy", "avg_latency_sec", "avg_prompt_tokens", "avg_completion_tokens", "estimated_cost", "recall_basis"]
PER_QUESTION_COLS = ["run_id", "question_id", "split", "question", "is_impossible", "gold_answer", "gold_context_ids",
                     "retrieved_context_ids", "retrieved_chunk_ids", "gold_context_hit_at_1", "gold_context_hit_at_3",
                     "gold_context_hit_at_5", "gold_context_rank", "generated_answer", "answer_correct", "faithfulness_score",
                     "citation_correct", "failure_type", "notes"]
RETRIEVAL_COLS = ["run_id", "question_id", "rank", "chunk_id", "context_id", "source", "score", "is_gold_context", "chunk_text_preview"]
ANSWER_COLS = ["run_id", "question_id", "question", "gold_answer", "generated_answer", "retrieved_context_preview", "judge_explanation",
               "faithfulness_score", "answer_score"]
FAILURE_COLS = ["run_id", "question_id", "question", "failure_type", "gold_answer", "generated_answer", "gold_context_ids",
                "retrieved_context_ids", "diagnosis", "suggested_fix"]
CONFIG_COLS = ["run_id", "embedding_model", "llm_model", "retriever_type", "chunking_strategy", "chunk_size", "chunk_overlap", "top_k",
               "reranker_model", "query_rewrite", "prompt_version", "temperature", "created_at", "git_commit"]

SUGGESTED_FIX = {
    "retrieval_miss": "Try hybrid retrieval, a larger top_k / candidates_k, a different chunk size, or query rewriting; check the gold text is in the index.",
    "false_refusal": "Relax the refusal rule in the system prompt, or raise top_k so the answer chunk is included; check the answer is not split across chunks.",
    "missed_refusal": "Strengthen the 'use only the passages' rule, or set rerank.min_score so weak context leads to a refusal.",
    "wrong_answer": "Open the retrieved chunk: if it holds the answer, use a stronger generation model or a tighter answer prompt; otherwise fix retrieval.",
    "low_faithfulness": "Lower the answer length limit, require quoting from the passages, or try a stronger generation model.",
    "bad_citation": "Make the citation rule stricter in the prompt, or post-check that every [n] points to a retrieved passage.",
    "late_hit": "Add a reranker or hybrid retrieval to move the gold chunk up; only shrink top_k after the ranking is fixed.",
}


# ---------------------------------------------------------------- loading

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def discover_runs(runs_dir: Path, only: str | None = None) -> list[Path]:
    """Folders (the runs folder itself, depth 1, depth 2) that contain items.jsonl, optionally filtered by a glob on the run id."""
    found = [runs_dir] if (runs_dir / "items.jsonl").exists() else []
    found += sorted(p.parent for pattern in ("*/items.jsonl", "*/*/items.jsonl") for p in runs_dir.glob(pattern))
    return [d for d in found if not only or fnmatch.fnmatch(run_id_of(d, runs_dir), only)]


def run_id_of(run_dir: Path, runs_dir: Path) -> str:
    rel = run_dir.relative_to(runs_dir).as_posix()
    return run_dir.name if rel == "." else rel


def load_judge(run_dir: Path) -> dict[str, dict]:
    """Claude-app judge scores by question id (judge_reply.json, as written for `run_eval import-judge`); {} if absent."""
    path = run_dir / "judge_reply.json"
    if not path.exists():
        return {}
    try:
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", path.read_text(encoding="utf-8").strip())
        return {s["id"]: s for s in json.loads(raw)}
    except (ValueError, KeyError, TypeError):
        return {}


def load_testset_index(eval_dir: Path, name: str, cache: dict) -> dict[str, dict]:
    if name not in cache:
        path = eval_dir / f"{name}.jsonl"
        cache[name] = {r["id"]: r for r in read_jsonl(path)} if name and path.exists() else {}
    return cache[name]


def created_at(run_dir: Path) -> str:
    for name in (run_dir.name, run_dir.parent.name):
        m = re.match(r"(\d{8})-(\d{6})", name)
        if m:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").isoformat(timespec="seconds")
        m = re.match(r"(\d{8})", name)
        if m:
            return datetime.strptime(m.group(1), "%Y%m%d").isoformat(timespec="seconds")
    ref = run_dir / "summary.json"
    return datetime.fromtimestamp((ref if ref.exists() else run_dir).stat().st_mtime).isoformat(timespec="seconds")


def git_commit() -> str:
    """Short HEAD of the repository at export time ('+dirty' with uncommitted changes); '' when git is unavailable."""
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return run("rev-parse", "--short", "HEAD") + ("+dirty" if run("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):
        return ""


def phoenix_tokens(url: str, project: str) -> dict[str, tuple[int, int]]:
    """trace_id -> (prompt tokens, completion tokens) summed over the trace's LLM spans; {} (with a warning) if Phoenix is unreachable."""
    try:
        from phoenix.client import Client

        spans = Client(base_url=url).spans.get_spans(project_identifier=project, limit=100000)
    except Exception as e:                                    # noqa: BLE001 - optional enrichment, never fatal
        print(f"warning: no token counts from Phoenix ({url}, project {project!r}): {type(e).__name__}: {e}", file=sys.stderr)
        return {}
    out: dict[str, list[int]] = {}
    for s in spans:
        attrs = s.get("attributes", {})
        if s.get("span_kind") != "LLM" or "llm.token_count.prompt" not in attrs:
            continue
        tot = out.setdefault(s["context"]["trace_id"].lower(), [0, 0])
        tot[0] += int(attrs.get("llm.token_count.prompt") or 0)
        tot[1] += int(attrs.get("llm.token_count.completion") or 0)
    return {k: (v[0], v[1]) for k, v in out.items()}


# ---------------------------------------------------------------- per-question derivation

def doc_of(chunk_id: str) -> str:
    return chunk_id.split("::")[0]


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def classify(d: dict, faithfulness_threshold: float) -> tuple[str, list[str]]:
    """(failure_type, all flags in priority order). failure_type is 'ok' when no flag applies."""
    flags: list[str] = []
    answered = d["generated_answer"] is not None
    if d["is_impossible"]:
        if answered and not is_refusal(d["generated_answer"]):
            flags.append("missed_refusal")
    else:
        rank = d["gold_context_rank"]
        if rank is None:
            flags.append("retrieval_miss")
        if answered and rank is not None and is_refusal(d["generated_answer"]):
            flags.append("false_refusal")
        if answered and rank is not None and not is_refusal(d["generated_answer"]) and d["answer_correct"] is False:
            flags.append("wrong_answer")
        if answered and d["faithfulness_score"] is not None and d["faithfulness_score"] < faithfulness_threshold:
            flags.append("low_faithfulness")
        if d["citation_correct"] is False:
            flags.append("bad_citation")
        if rank is not None and rank > 3:
            flags.append("late_hit")
    return (flags[0] if flags else "ok"), flags


def diagnose(d: dict, failure_type: str, faithfulness_threshold: float) -> str:
    rank, k = d["gold_context_rank"], len(d["retrieved_chunk_ids"])
    shown = ", ".join(d["retrieved_context_ids"][:5]) or "none"
    return {
        "retrieval_miss": f"None of the {k} retrieved chunks comes from a gold document ({'; '.join(d['gold_context_ids']) or 'unknown'}); "
                          f"retrieved documents: {shown}.",
        "false_refusal": f"The gold document was retrieved at rank {rank}, but the model answered with the not-found sentence.",
        "missed_refusal": "The question has no answer in the corpus, but the model gave an answer instead of the not-found sentence.",
        "wrong_answer": f"The gold document was retrieved at rank {rank}, but the answer does not agree with the reference answer"
                        + (f" (answer_score {d['answer_score']:.2f})." if d.get("answer_score") is not None else "."),
        "low_faithfulness": f"Faithfulness {d['faithfulness_score']:.2f} is below {faithfulness_threshold}: the answer contains claims "
                            f"the retrieved chunks do not support." if d["faithfulness_score"] is not None else "",
        "bad_citation": "The [n] citations are missing, out of range, or none points to a gold document.",
        "late_hit": f"The gold document first appears at rank {rank}, so a smaller top_k would lose it.",
    }.get(failure_type, "")


def citation_check(answer: str | None, retrieved: list[dict], gold: list[str]) -> bool | None:
    """True/False for answerable, non-refusal answers; None when not applicable."""
    if answer is None or is_refusal(answer):
        return None
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    if not cited or any(not 1 <= n <= len(retrieved) for n in cited):
        return False
    return any(doc_of(retrieved[n - 1]["id"]) in gold for n in cited)


def derive(row: dict, gold_fallback: list[str], judge: dict | None, correct_threshold: float) -> dict:
    """Everything the report needs for one question, computed from the saved row."""
    retrieved = row.get("retrieved", [])
    chunk_ids = [h["id"] for h in retrieved]
    context_ids = list(dict.fromkeys(doc_of(c) for c in chunk_ids))
    is_impossible = not row["answerable"]
    gold = list(row.get("gold_context_ids") or gold_fallback or ([row["source"]] if row.get("source") else []))
    answer = row.get("answer")
    ragas = row.get("ragas") or {}

    rank = None
    if not is_impossible:
        rank = first_doc_rank(gold, chunk_ids) if gold else row.get("rank")
    hit = lambda k: None if is_impossible else (rank is not None and rank <= k)

    faith = ragas.get("faithfulness")
    fact = ragas.get(FACT_CORRECTNESS)
    claude = judge.get("correctness") if judge else None
    answer_score = fact if fact is not None else ((claude - 1) / 4 if claude is not None else None)
    if answer is None:
        correct = None
    elif is_impossible:
        correct = is_refusal(answer)
    elif is_refusal(answer):
        correct = False
    elif claude is not None:
        correct = claude >= 4
    elif fact is not None:
        correct = fact >= correct_threshold
    else:
        correct = None

    return {
        "question_id": row["id"], "question": row["question"], "is_impossible": is_impossible,
        "gold_answer": row.get("reference_answer") or "", "gold_context_ids": gold,
        "retrieved_context_ids": context_ids, "retrieved_chunk_ids": chunk_ids,
        "gold_context_hit_at_1": hit(1), "gold_context_hit_at_3": hit(3), "gold_context_hit_at_5": hit(5),
        "gold_context_rank": rank, "generated_answer": answer, "answer_correct": correct, "answer_score": answer_score,
        "faithfulness_score": faith, "judge_explanation": (judge or {}).get("comment"),
        "citation_correct": None if is_impossible else citation_check(answer, retrieved, gold),
        "recall_basis": "document" if (gold and not is_impossible) else ("answer_quote" if not is_impossible else None),
    }


# ---------------------------------------------------------------- report assembly

def mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def config_record(run_id: str, run_dir: Path, cfg: dict, commit: str) -> dict:
    rer, gen = cfg.get("rerank", {}), cfg.get("generation", {})
    return {
        "run_id": run_id, "embedding_model": cfg.get("embedding", {}).get("model"), "llm_model": gen.get("model"),
        "retriever_type": cfg.get("retrieval", {}).get("mode"), "chunking_strategy": cfg.get("chunking", {}).get("strategy"),
        "chunk_size": cfg.get("chunking", {}).get("max_tokens"), "chunk_overlap": cfg.get("chunking", {}).get("overlap"),
        "top_k": cfg.get("retrieval", {}).get("top_k"), "reranker_model": rer.get("model") if rer.get("enabled") else "none",
        "query_rewrite": gen.get("rewrite_followups"), "prompt_version": gen.get("prompt_version"),
        "temperature": gen.get("temperature"), "created_at": created_at(run_dir), "git_commit": commit,
    }


def build_report(runs_dir: Path, only: str | None = None, correct_threshold: float = 0.5, faithfulness_threshold: float = 0.5,
                 phoenix_url: str | None = None, price_in: float = 0.0, price_out: float = 0.0,
                 commit: str | None = None) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """(DataFrames by sheet name, raw per-question records). Raises SystemExit when no run is found."""
    dirs = discover_runs(runs_dir, only)
    if not dirs:
        raise SystemExit(f"no runs (folders with items.jsonl) found under {runs_dir}; run `ablate` or `run_eval run` first")
    commit = git_commit() if commit is None else commit
    testsets: dict = {}
    token_cache: dict = {}
    summaries, per_q, retr, ans, fail, configs, raw = [], [], [], [], [], [], []

    for run_dir in dirs:
        run_id = run_id_of(run_dir, runs_dir)
        cfg, summary = read_json(run_dir / "config.json"), read_json(run_dir / "summary.json")
        rows = read_jsonl(run_dir / "items.jsonl")
        judge = load_judge(run_dir)
        eval_dir = ROOT / cfg.get("paths", {}).get("eval_dir", "data/eval")
        testset_name = summary.get("testset", "")
        index = load_testset_index(eval_dir, testset_name, testsets)
        split = testset_name.split("/")[-1] if testset_name else ""
        tokens = {}
        if phoenix_url:
            project = cfg.get("tracing", {}).get("project_name", "techqa-rag")
            if project not in token_cache:
                token_cache[project] = phoenix_tokens(phoenix_url, project)
            tokens = token_cache[project]

        derived = []
        for row in rows:
            d = derive(row, (index.get(row["id"]) or {}).get("gold_context_ids") or [], judge.get(row["id"]), correct_threshold)
            d["failure_type"], flags = classify(d, faithfulness_threshold)
            d["notes"] = ("also: " + ", ".join(flags[1:])) if len(flags) > 1 else ""
            d["run_id"], d["split"] = run_id, split
            d["seconds"] = row.get("seconds")
            d["prompt_tokens"], d["completion_tokens"] = tokens.get((row.get("trace_id") or "").lower(), (None, None))
            derived.append(d)
            per_q.append(d)

            hits = row.get("retrieved", [])
            for i, h in enumerate(hits, 1):
                retr.append({"run_id": run_id, "question_id": row["id"], "rank": i, "chunk_id": h["id"], "context_id": doc_of(h["id"]),
                             "source": h.get("citation"), "score": h.get("score"),
                             "is_gold_context": doc_of(h["id"]) in d["gold_context_ids"], "chunk_text_preview": h.get("text", "")})
            ans.append({"run_id": run_id, "question_id": row["id"], "question": row["question"], "gold_answer": d["gold_answer"],
                        "generated_answer": d["generated_answer"],
                        "retrieved_context_preview": "\n---\n".join(
                            f"[{i}] {doc_of(h['id'])}: {squash(h.get('text', ''))[:300]}" for i, h in enumerate(hits, 1)),
                        "judge_explanation": d["judge_explanation"], "faithfulness_score": d["faithfulness_score"],
                        "answer_score": d["answer_score"]})
            if d["failure_type"] != "ok":
                fail.append({"run_id": run_id, "question_id": row["id"], "question": row["question"],
                             "failure_type": d["failure_type"], "gold_answer": d["gold_answer"],
                             "generated_answer": d["generated_answer"], "gold_context_ids": d["gold_context_ids"],
                             "retrieved_context_ids": d["retrieved_context_ids"],
                             "diagnosis": diagnose(d, d["failure_type"], faithfulness_threshold),
                             "suggested_fix": SUGGESTED_FIX[d["failure_type"]]})
            raw.append({"run_id": run_id, "config": cfg, "row": row, "derived": {k: v for k, v in d.items() if k not in ("run_id",)}})

        conf = config_record(run_id, run_dir, cfg, commit)
        configs.append(conf)
        answerable = [d for d in derived if not d["is_impossible"]]
        impossible = [d for d in derived if d["is_impossible"]]
        bases = {d["recall_basis"] for d in answerable}
        p_tok, c_tok = [d["prompt_tokens"] for d in derived], [d["completion_tokens"] for d in derived]
        known = [(p, c) for p, c in zip(p_tok, c_tok) if p is not None]
        if price_in == 0 and price_out == 0:
            cost = 0.0                                       # local Ollama generation is free
        else:
            cost = round(sum(p * price_in + c * price_out for p, c in known) / 1e6, 6) if known else None
        summaries.append({
            "run_id": run_id, "retriever_type": conf["retriever_type"], "chunk_size": conf["chunk_size"],
            "chunk_overlap": conf["chunk_overlap"], "top_k": conf["top_k"], "reranker": conf["reranker_model"],
            "query_rewrite": conf["query_rewrite"], "num_questions": len(derived),
            **{f"context_recall_at_{k}": mean(d[f"gold_context_hit_at_{k}"] for d in answerable) for k in (1, 3, 5)},
            "mrr": mean((1 / d["gold_context_rank"]) if d["gold_context_rank"] else 0.0 for d in answerable) if answerable else None,
            "answer_accuracy": mean(d["answer_correct"] for d in answerable),
            "faithfulness_avg": mean(d["faithfulness_score"] for d in answerable),
            "not_found_accuracy": mean(d["answer_correct"] for d in impossible),
            "avg_latency_sec": mean(d["seconds"] for d in derived),
            "avg_prompt_tokens": mean(p_tok), "avg_completion_tokens": mean(c_tok), "estimated_cost": cost,
            "recall_basis": "/".join(sorted(b for b in bases if b)) or None,
        })

    frames = {
        "summary_by_run": pd.DataFrame(summaries, columns=SUMMARY_COLS),
        "per_question_results": pd.DataFrame(per_q).reindex(columns=PER_QUESTION_COLS),
        "retrieval_details": pd.DataFrame(retr, columns=RETRIEVAL_COLS),
        "answer_details": pd.DataFrame(ans, columns=ANSWER_COLS),
        "failure_analysis": pd.DataFrame(fail, columns=FAILURE_COLS),
        "config": pd.DataFrame(configs, columns=CONFIG_COLS),
    }
    return frames, raw


# ---------------------------------------------------------------- writing

def flat(value):
    """Cell value for CSV/Excel: lists joined with '; ', dicts as JSON."""
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return value


def clip(value):
    """Excel cell: control characters removed (openpyxl rejects them), text cut to EXCEL_MAX_CHARS."""
    if isinstance(value, str):
        value = CONTROL_CHARS.sub("", value)
        if len(value) > EXCEL_MAX_CHARS:
            value = value[: EXCEL_MAX_CHARS - 1] + "…"
    return value


def write_report(frames: dict[str, pd.DataFrame], raw: list[dict], out_dir: Path) -> list[Path]:
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    out_dir.mkdir(parents=True, exist_ok=True)
    flat_frames = {name: df.map(flat) for name, df in frames.items()}

    flat_frames["per_question_results"].to_csv(out_dir / "per_question_results.csv", index=False, encoding="utf-8-sig")
    flat_frames["failure_analysis"].to_csv(out_dir / "failure_analysis.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "raw_results.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for rec in raw:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    xlsx = out_dir / "rag_ablation_report.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        for name, df in flat_frames.items():
            df.map(clip).to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for i, col in enumerate(df.columns, 1):
                longest = max([len(str(col))] + [len(str(v)) for v in df[col].head(200) if v is not None])
                ws.column_dimensions[get_column_letter(i)].width = min(longest + 2, 60)
    return [xlsx, out_dir / "per_question_results.csv", out_dir / "failure_analysis.csv", out_dir / "raw_results.jsonl"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "data/eval/runs", help="folder with run / ablation folders")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/eval_reports")
    ap.add_argument("--only", help="glob on the run id, e.g. '*ablation_techqa_tiny_dev*'")
    ap.add_argument("--correct-threshold", type=float, default=0.5, help="RAGAS factual_correctness at or above this counts as correct")
    ap.add_argument("--faithfulness-threshold", type=float, default=0.5, help="faithfulness below this is flagged low_faithfulness")
    ap.add_argument("--phoenix-url", help="e.g. http://localhost:6006: read token counts from the runs' traces")
    ap.add_argument("--price-in", type=float, default=0.0, help="cost per 1M prompt tokens (default 0: local model)")
    ap.add_argument("--price-out", type=float, default=0.0, help="cost per 1M completion tokens")
    args = ap.parse_args()

    frames, raw = build_report(args.runs, args.only, args.correct_threshold, args.faithfulness_threshold, args.phoenix_url,
                               args.price_in, args.price_out)
    paths = write_report(frames, raw, args.out)
    print(f"{len(frames['summary_by_run'])} runs, {len(frames['per_question_results'])} questions, "
          f"{len(frames['retrieval_details'])} retrieved chunks, {len(frames['failure_analysis'])} failure/suspicious cases")
    for p in paths:
        print("wrote", p)


if __name__ == "__main__":
    main()

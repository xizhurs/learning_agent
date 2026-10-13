"""Export `ablate` / `run_eval run` results to Excel, CSV and JSONL so retrieval strategies can be compared and failures inspected.

  uv run python scripts/export_eval_report.py                          # reads data/eval/runs/, writes outputs/eval_reports/
  uv run python scripts/export_eval_report.py --only "*ablation_techqa_tiny_dev*"
  uv run python scripts/export_eval_report.py --phoenix-url http://localhost:6006   # also fill token counts from Phoenix traces

Reads every folder under --runs (depth 1 or 2) that contains items.jsonl (+ config.json, summary.json): one folder per variant of an
ablation, or one folder per `run_eval run`. Nothing in the evaluation is changed or re-run; everything is derived from the saved rows.

Output (--out, default outputs/eval_reports/):
  rag_ablation_report.xlsx   sheets: summary_by_run, per_question_results, retrieval_details, answer_details, failure_analysis, config,
                             paired_comparison, difficulty_check (text cells cut to 1000 characters, header frozen, autofilter on)
  per_question_results.csv, failure_analysis.csv   full text
  raw_results.jsonl          one line per run/question with the complete saved row, run config and derived fields (nothing cut)

How the derived columns are defined:
  gold_context_*          document level: a retrieved chunk counts when its document (chunk id '<document>::NNNNN') is a gold document.
                          Gold ids come from the row, else from the test-set file named in summary.json, else from the row's `source`.
                          Rows without gold ids fall back to the stored answer-quote `rank` (summary column recall_basis says which).
  answer_correct          unanswerable question: the model refused (rag.eval.metrics.is_refusal). Answerable: a refusal is wrong; otherwise
                          Claude-app judge correctness >= 4 (judge_reply.json in the run folder), else the RAGAS critic
                          answer_matches_reference >= --correct-threshold, else (runs scored before the critic existed) RAGAS
                          factual_correctness(mode=f1) >= --correct-threshold. That F1 metric scores ~0 for one-word / list references
                          even when the answer is right, hence only the fallback. Empty if the answer was not scored.
  correct_source          where the verdict came from: refusal | claude | ragas | ragas_factual_correctness
  ragas_*                 every column starting with `ragas_` is a RAGAS score (judge LLM); everything else is computed here or saved by
                          run_eval from our own deterministic metrics. claude_correctness_score is (Claude-app correctness - 1) / 4.
  citation_correct        heuristic, answerable non-refusal answers only: every [n] is within the retrieved list and one cites a gold document
  failure_type            first match of: retrieval_miss, false_refusal, missed_refusal, wrong_answer, low_faithfulness, bad_citation,
                          late_hit (gold document first at rank > 3). Any further problems are listed in `notes`.

QASPER metrics (also in each run's summary.json, here recomputed from the saved rows with the run's eval.overlap_threshold):
  answer_text_recall_at_k, answer_mrr   the answer-hit metrics: is the annotators' evidence text inside a top-k chunk. On QASPER use these:
                          context_recall_at_k and mrr are document level and trivially ~1.0 because each question carries its paper's title.
  evidence_recall_at_k    share of a question's evidence paragraphs covered by the top-k chunks
  context_precision       rank-weighted precision of the retrieved chunks, "covers an evidence paragraph" as the relevance verdict
  answer_f1, answer_token_recall   QASPER token F1 / recall of the generated answer vs the reference (generation runs only)
  refusal_rate_*          share of unanswerable questions refused / of answerable questions wrongly refused
                          (not_found_accuracy here = refusal rate on unanswerable only; summary.json's not_found_accuracy is over all questions)
  ragas_*                 means over answerable questions that RAGAS scored (jobs that failed, e.g. on a quota, are skipped, so
                          ragas_n counts questions with at least one score and an individual metric can rest on slightly fewer)
  median_*_sec            median seconds in retrieve / rerank / generate (from the saved timings)
Empty cells mean the inputs were not in the run (e.g. no generated answers in a retrieval-only run).

How to tell whether a test set is too easy to separate the runs (summary_by_run and the two extra sheets):
  recall_at_1_ci95_*      95% Wilson interval of context_recall_at_1 over the answerable questions. Runs whose intervals overlap are not
                          distinguishable on this test set; with 48 questions one question is already 2 points.
  answer_text_recall_at_*  chunk-level recall: the share of questions whose reference answer text (word 3-grams, rag.eval.metrics) is in
                          the top-k chunks. Stricter than the document-level recall and the only one that reacts to chunk size.
  paired_comparison       every run against the best run (highest mrr) of the same test set, question by question: wins / losses at
                          rank 1 and within the top 5, and the exact two-sided sign-test p-value of the wins vs losses.
  difficulty_check        a ranker that only counts words shared by the question and the gold document's title (no embeddings), on the
                          corpus in <testset folder>/contexts.jsonl. If it is close to the real retrievers the corpus is easy.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from rag.eval.metrics import (answer_overlap, context_precision, evidence_recall, first_doc_rank, is_refusal,
                              latency_summary)

ROOT = Path(__file__).resolve().parent.parent
EXCEL_MAX_CHARS = 1000
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
FACT_CORRECTNESS = "factual_correctness(mode=f1)"
ANSWER_MATCH = "answer_matches_reference"
# RAGAS keys in a saved row's `ragas` dict -> report column. Every RAGAS-derived column starts with `ragas_`; the rest are our own metrics.
RAGAS_COLS = {"faithfulness": "ragas_faithfulness", ANSWER_MATCH: "ragas_answer_matches_reference",
              FACT_CORRECTNESS: "ragas_factual_correctness", "answer_relevancy": "ragas_answer_relevancy",
              "context_recall": "ragas_context_recall", "llm_context_precision_with_reference": "ragas_context_precision"}

SUMMARY_COLS = ["run_id", "retriever_type", "chunk_size", "chunk_overlap", "top_k", "reranker", "query_rewrite", "num_questions",
                "n_answerable",
                # RAGAS (judge LLM)
                "ragas_faithfulness", "ragas_answer_matches_reference", "ragas_factual_correctness", "ragas_answer_relevancy",
                "ragas_context_recall", "ragas_context_precision", "ragas_n",
                # ours: answers and refusals
                "answer_accuracy", "not_found_accuracy", "refusal_rate_on_unanswerable", "false_refusal_rate_on_answerable",
                "answer_f1", "answer_token_recall",
                # ours: retrieval against the annotators' evidence text
                "answer_mrr", "answer_text_recall_at_1", "answer_text_recall_at_3", "answer_text_recall_at_5",
                "evidence_recall_at_1", "evidence_recall_at_3", "evidence_recall_at_5", "context_precision",
                # ours: document level (trivially ~1.0 on QASPER) and cost
                "context_recall_at_1", "context_recall_at_3", "context_recall_at_5", "mrr", "recall_at_1_ci95_low",
                "recall_at_1_ci95_high", "recall_basis", "avg_latency_sec", "median_retrieve_sec", "median_rerank_sec",
                "median_generate_sec", "avg_prompt_tokens", "avg_completion_tokens", "estimated_cost"]
PER_QUESTION_COLS = ["run_id", "question_id", "split", "question", "is_impossible", "gold_answer", "gold_context_ids",
                     "retrieved_context_ids", "retrieved_chunk_ids", "gold_context_hit_at_1", "gold_context_hit_at_3",
                     "gold_context_hit_at_5", "gold_context_rank", "generated_answer", "answer_correct", "correct_source",
                     "ragas_faithfulness", "ragas_answer_matches_reference", "ragas_factual_correctness", "ragas_answer_relevancy",
                     "ragas_context_recall", "ragas_context_precision", "claude_correctness_score",
                     "citation_correct", "failure_type", "notes", "answer_text_rank", "evidence_recall_at_1", "evidence_recall_at_3",
                     "evidence_recall_at_5", "context_precision", "answer_f1", "answer_token_recall"]
PAIRED_COLS = ["run_id", "testset", "reference_run", "n_paired", "hit1_wins", "hit1_losses", "hit1_sign_test_p",
               "hit5_wins", "hit5_losses", "hit5_sign_test_p"]
DIFFICULTY_COLS = ["testset", "baseline", "n_answerable", "n_documents", "gold_share_of_documents", "context_recall_at_1",
                   "context_recall_at_3", "context_recall_at_5", "mrr"]
STOPWORDS = set("the a an to of in is are for and or with on how do does can what i we you it this that be from by at as not if my "
                "our when".split())
RETRIEVAL_COLS = ["run_id", "question_id", "rank", "chunk_id", "context_id", "source", "score", "is_gold_context", "chunk_text_preview"]
ANSWER_COLS = ["run_id", "question_id", "question", "gold_answer", "generated_answer", "retrieved_context_preview", "judge_explanation",
               "ragas_faithfulness", "ragas_answer_matches_reference", "ragas_factual_correctness", "claude_correctness_score"]
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
        if answered and d["ragas_faithfulness"] is not None and d["ragas_faithfulness"] < faithfulness_threshold:
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
                        + (f" (RAGAS answer_matches_reference {d['ragas_answer_matches_reference']:.0f})."
                           if d.get("ragas_answer_matches_reference") is not None
                           else f" (RAGAS factual_correctness {d['ragas_factual_correctness']:.2f})."
                           if d.get("ragas_factual_correctness") is not None else "."),
        "low_faithfulness": f"RAGAS faithfulness {d['ragas_faithfulness']:.2f} is below {faithfulness_threshold}: the answer contains claims "
                            f"the retrieved chunks do not support." if d["ragas_faithfulness"] is not None else "",
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


def derive(row: dict, gold_fallback: list[str], judge: dict | None, correct_threshold: float,
           overlap_threshold: float = 0.6) -> dict:
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

    rg = {col: ragas.get(key) for key, col in RAGAS_COLS.items()}
    match, fact = rg["ragas_answer_matches_reference"], rg["ragas_factual_correctness"]
    claude = judge.get("correctness") if judge else None
    # verdict source: Claude-app judge, else the RAGAS answer_matches_reference critic, else (runs scored before it existed) RAGAS
    # factual_correctness, which under-scores short references and so is only a fallback
    source = None
    if answer is None:
        correct = None
    elif is_impossible:
        correct, source = is_refusal(answer), "refusal"
    elif is_refusal(answer):
        correct, source = False, "refusal"
    elif claude is not None:
        correct, source = claude >= 4, "claude"
    elif match is not None:
        correct, source = match >= correct_threshold, "ragas"
    elif fact is not None:
        correct, source = fact >= correct_threshold, "ragas_factual_correctness"
    else:
        correct = None

    quote = row.get("reference_quote")
    texts = [h.get("text", "") for h in retrieved]
    scored_quote = bool(quote) and not is_impossible
    ev = {k: evidence_recall(quote, texts[:k], overlap_threshold) if scored_quote else None for k in (1, 3, 5)}
    overlap = (answer_overlap(answer, row["reference_answer"])
               if answer is not None and not is_impossible and row.get("reference_answer") else (None, None))

    return {
        "question_id": row["id"], "question": row["question"], "is_impossible": is_impossible,
        "gold_answer": row.get("reference_answer") or "", "gold_context_ids": gold,
        "retrieved_context_ids": context_ids, "retrieved_chunk_ids": chunk_ids,
        "gold_context_hit_at_1": hit(1), "gold_context_hit_at_3": hit(3), "gold_context_hit_at_5": hit(5),
        "gold_context_rank": rank, "answer_text_rank": None if is_impossible else row.get("answer_rank"),
        "answer_text_known": not is_impossible and "answer_rank" in row,
        "generated_answer": answer, "answer_correct": correct, "correct_source": source, **rg,
        "claude_correctness_score": (claude - 1) / 4 if claude is not None else None,
        "judge_explanation": (judge or {}).get("comment"),
        "citation_correct": None if is_impossible else citation_check(answer, retrieved, gold),
        "recall_basis": "document" if (gold and not is_impossible) else ("answer_quote" if not is_impossible else None),
        **{f"evidence_recall_at_{k}": v for k, v in ev.items()},
        "context_precision": context_precision(quote, texts, overlap_threshold) if scored_quote else None,
        "answer_f1": overlap[0], "answer_token_recall": overlap[1],
    }


# ---------------------------------------------------------------- report assembly

def mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float] | tuple[None, None]:
    """95% Wilson score interval of a proportion; (None, None) without observations."""
    if n == 0:
        return None, None
    p, denom = hits / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test (ties are left out): probability of a split at least this uneven if both runs were equally good."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def paired_comparison(runs: list[dict]) -> list[dict]:
    """Each run against the best run (highest mrr, then run id) of its test set. `runs`: dicts with run_id, testset, mrr and
    hits = {question_id: (hit@1, hit@5)} over the answerable questions. The reference run itself gets no row."""
    out = []
    for testset in sorted({r["testset"] for r in runs}):
        group = sorted((r for r in runs if r["testset"] == testset), key=lambda r: (-(r["mrr"] or 0.0), r["run_id"]))
        ref = group[0]
        for r in group[1:]:
            shared = sorted(set(r["hits"]) & set(ref["hits"]))
            row = {"run_id": r["run_id"], "testset": testset, "reference_run": ref["run_id"], "n_paired": len(shared)}
            for slot, k in ((0, 1), (1, 5)):
                wins = sum(r["hits"][q][slot] and not ref["hits"][q][slot] for q in shared)
                losses = sum(ref["hits"][q][slot] and not r["hits"][q][slot] for q in shared)
                row |= {f"hit{k}_wins": wins, f"hit{k}_losses": losses, f"hit{k}_sign_test_p": sign_test_p(wins, losses)}
            out.append(row)
    return out


def title_baseline(testset: str, questions: dict[str, dict], contexts_path: Path) -> dict | None:
    """Recall of a retriever that only counts words shared by the question and a document's first line (its title), over the documents
    in contexts.jsonl. None when that file or gold ids are missing. Documents are ranked by overlap / sqrt(title length), then by id."""
    if not contexts_path.exists():
        return None
    docs = {c["context_id"]: c for c in read_jsonl(contexts_path)}
    words = lambda s: set(re.findall(r"[a-z0-9][a-z0-9_.\-]+", s.lower())) - STOPWORDS
    titles = {}
    for cid, c in docs.items():
        first = (c["text"].strip().splitlines() or [""])[0]
        titles[cid] = words(first.removeprefix("Title:"))
    ranks = []
    for q in questions.values():
        gold = set(q.get("gold_context_ids") or [])
        if q.get("is_impossible") or not gold:
            continue
        qw = words(q["question"])
        order = sorted(titles, key=lambda cid: (-len(qw & titles[cid]) / (len(titles[cid]) ** 0.5 + 1e-9), cid))
        ranks.append(next((i for i, cid in enumerate(order, 1) if cid in gold), None))
    if not ranks:
        return None
    hit = lambda k: round(sum(r is not None and r <= k for r in ranks) / len(ranks), 4)
    gold_docs = sum(c.get("role") == "gold" for c in docs.values())
    return {"testset": testset, "baseline": "title_word_overlap", "n_answerable": len(ranks), "n_documents": len(docs),
            "gold_share_of_documents": round(gold_docs / len(docs), 4) if gold_docs else None,
            "context_recall_at_1": hit(1), "context_recall_at_3": hit(3), "context_recall_at_5": hit(5),
            "mrr": round(sum(1 / r for r in ranks if r) / len(ranks), 4)}


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
    paired_input: list[dict] = []
    baseline_inputs: dict[str, Path] = {}                    # testset name -> folder of its contexts.jsonl

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
            d = derive(row, (index.get(row["id"]) or {}).get("gold_context_ids") or [], judge.get(row["id"]), correct_threshold,
                       cfg.get("eval", {}).get("overlap_threshold", 0.6))
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
                        "judge_explanation": d["judge_explanation"],
                        **{c: d[c] for c in ("ragas_faithfulness", "ragas_answer_matches_reference", "ragas_factual_correctness",
                                             "claude_correctness_score")}})
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
        n_hit1 = sum(bool(d["gold_context_hit_at_1"]) for d in answerable)
        ci_low, ci_high = wilson(n_hit1, len(answerable))
        with_text = [d for d in answerable if d["answer_text_known"]]
        text_recall = {k: mean(d["answer_text_rank"] is not None and d["answer_text_rank"] <= k for d in with_text)
                       for k in (1, 3, 5)}
        mrr = mean((1 / d["gold_context_rank"]) if d["gold_context_rank"] else 0.0 for d in answerable) if answerable else None
        paired_input.append({"run_id": run_id, "testset": testset_name, "mrr": mrr,
                             "hits": {d["question_id"]: (bool(d["gold_context_hit_at_1"]), bool(d["gold_context_hit_at_5"]))
                                      for d in answerable}})
        baseline_inputs.setdefault(testset_name, (eval_dir / testset_name).parent / "contexts.jsonl")
        answer_mrr = (mean((1 / d["answer_text_rank"]) if d["answer_text_rank"] else 0.0 for d in with_text)
                      if with_text else None)
        generated = any(d["generated_answer"] is not None for d in derived)
        ragas_rows = [r.get("ragas") or {} for r, d in zip(rows, derived) if not d["is_impossible"]]
        lat = latency_summary([r.get("timings") for r in rows])
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
            "mrr": mrr,
            "answer_accuracy": mean(d["answer_correct"] for d in answerable),
            "not_found_accuracy": mean(d["answer_correct"] for d in impossible),
            "avg_latency_sec": mean(d["seconds"] for d in derived),
            "avg_prompt_tokens": mean(p_tok), "avg_completion_tokens": mean(c_tok), "estimated_cost": cost,
            "recall_basis": "/".join(sorted(b for b in bases if b)) or None,
            "n_answerable": len(answerable), "recall_at_1_ci95_low": ci_low, "recall_at_1_ci95_high": ci_high,
            **{f"answer_text_recall_at_{k}": v for k, v in text_recall.items()},
            "answer_mrr": answer_mrr,
            **{f"evidence_recall_at_{k}": mean(d[f"evidence_recall_at_{k}"] for d in answerable) for k in (1, 3, 5)},
            "context_precision": mean(d["context_precision"] for d in answerable),
            "answer_f1": mean(d["answer_f1"] for d in answerable),
            "answer_token_recall": mean(d["answer_token_recall"] for d in answerable),
            "refusal_rate_on_unanswerable": mean(is_refusal(d["generated_answer"]) for d in impossible) if generated else None,
            "false_refusal_rate_on_answerable": (mean(is_refusal(d["generated_answer"]) for d in answerable if d["generated_answer"] is not None)
                                                 if generated else None),
            **{col: mean(d[col] for d in answerable) for col in RAGAS_COLS.values()},
            "ragas_n": sum(any(v is not None for v in r.values()) for r in ragas_rows) or None,
            **{f"median_{step}_sec": (lat.get(step) or {}).get("median") for step in ("retrieve", "rerank", "generate")},
        })

    difficulty = [b for name, path in baseline_inputs.items() if name and (b := title_baseline(name, testsets.get(name, {}), path))]
    frames = {
        "summary_by_run": pd.DataFrame(summaries, columns=SUMMARY_COLS),
        "per_question_results": pd.DataFrame(per_q).reindex(columns=PER_QUESTION_COLS),
        "retrieval_details": pd.DataFrame(retr, columns=RETRIEVAL_COLS),
        "answer_details": pd.DataFrame(ans, columns=ANSWER_COLS),
        "failure_analysis": pd.DataFrame(fail, columns=FAILURE_COLS),
        "config": pd.DataFrame(configs, columns=CONFIG_COLS),
        "paired_comparison": pd.DataFrame(paired_comparison(paired_input), columns=PAIRED_COLS),
        "difficulty_check": pd.DataFrame(difficulty, columns=DIFFICULTY_COLS),
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

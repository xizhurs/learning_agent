"""Run the pipeline over the test set and score it (retrieval metrics + refusal rates; answer quality is judged manually in the Claude app).

  python -m rag.eval.run_eval run [--tag baseline] [--limit 20] [--retrieval-only] [--ragas] [--testset testset_holdout]
  python -m rag.eval.run_eval score-ragas --run <run or ablation dir>   # RAGAS on saved rows (needs the judge key in the env)
  python -m rag.eval.run_eval check-judge                             # verify hosted-judge key / model / rate limit
  python -m rag.eval.run_eval export-judge --run <run_dir> [--n 20]   # packet for manual judging in the Claude app
  python -m rag.eval.run_eval import-judge --run <run_dir>            # reads <run_dir>/judge_reply.json

Outputs go to data/eval/runs/<timestamp>_<tag>_<testset>/ : config.json, items.jsonl, summary.json
--testset defaults to the dev set (testset); tune on it, and run the holdout (testset_holdout) only on the final config.
--testset may be a sub-path, e.g. techqa_tiny/dev (rows {id, question, answer, is_impossible, gold_context_ids} are normalised by
normalize_item; `rank` is then a document-level hit and `answer_rank` the answer-text hit).
If eval.judge_allowed_source is set, --ragas runs are restricted to that document's single-passage questions, retrieval is
filtered to that document, and assert_hosted_safe() refuses to send anything else to a hosted judge. If it is null (public
corpus such as TechQA) nothing is restricted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from rag.config import Settings, get_settings
from rag.eval.metrics import first_doc_rank, first_hit_rank, is_refusal, latency_summary, retrieval_summary, spearman
from rag.generate import build_messages, generate
from rag.ingest.build import ensure_table
from rag.retrieve import retrieve
from rag.tracing import annotate, enabled as tracing_enabled, flush as flush_traces, start_trace


def slugify_testset(name: str) -> str:
    """Testset name usable inside a folder name ('techqa_tiny/dev' -> 'techqa_tiny_dev')."""
    return re.sub(r"[\\/]+", "_", name)


def normalize_item(item: dict) -> dict:
    """TechQA-style rows {id, question, answer, is_impossible, gold_context_ids} -> the runner schema; other rows pass through."""
    if "is_impossible" not in item or "reference_answer" in item:
        return item
    answerable = not item["is_impossible"]
    gold = list(item.get("gold_context_ids") or [])
    answer = item["answer"] if answerable else ""
    return {**item, "type": "factual", "answerable": answerable, "source": gold[0] if gold else "",
            "reference_answer": answer, "reference_quote": answer, "gold_context_ids": gold}


def load_testset(settings: Settings, name: str = "testset") -> list[dict]:
    path = settings.paths.resolve("eval_dir") / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found - run: python -m rag.eval.gen_testset export / import")
    return [normalize_item(json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_items(testset: list[dict], settings: Settings, generate_answers: bool, limit: int | None = None,
              session_id: str | None = None) -> list[dict]:
    """One trace per question (Phoenix session = session_id). Ground truth is attached to the trace as eval.* attributes."""
    table = ensure_table(settings)
    rows = []
    items = testset[:limit] if limit else testset
    for n, item in enumerate(items, 1):
        t0 = time.time()
        root = start_trace(
            settings, "eval.question", input=item["question"], session_id=session_id,
            **{"rag__kind": "eval", "eval__id": item["id"], "eval__type": item["type"], "eval__answerable": item["answerable"],
               "eval__source": item.get("source", ""), "eval__gold_context_ids": ", ".join(item.get("gold_context_ids", [])),
               "eval__reference_answer": item["reference_answer"],
               "eval__reference_quote": item["reference_quote"], "rag__generation_model": settings.generation.model,
               "retrieval__mode": settings.retrieval.mode, "retrieval__top_k": settings.retrieval.top_k,
               "rerank__enabled": settings.rerank.enabled, "rag__chunk_max_tokens": settings.chunking.max_tokens})
        hits = retrieve(item["question"], settings, table, root)
        row = {
            "id": item["id"], "question": item["question"], "type": item["type"], "answerable": item["answerable"],
            "source": item.get("source", ""), "gold_context_ids": item.get("gold_context_ids", []),
            "reference_answer": item["reference_answer"], "reference_quote": item["reference_quote"],
            "retrieved": [{"id": h.id, "citation": h.citation(), "score": h.score, "text": h.text} for h in hits],
            "rank": None, "answer": None,
        }
        if item["answerable"]:
            quote_rank = first_hit_rank(item["reference_quote"], [h.text for h in hits], settings.eval.overlap_threshold)
            if row["gold_context_ids"]:
                # gold unit is a whole document: `rank` = first chunk from a gold document; the answer-quote rank is kept alongside
                row["rank"] = first_doc_rank(row["gold_context_ids"], [h.id for h in hits])
                row["answer_rank"] = quote_rank
            else:
                row["rank"] = quote_rank
        if generate_answers:
            if hits or not settings.generation.refuse_if_no_context:
                row["answer"] = generate(build_messages(item["question"], hits), settings.generation, root)
            else:
                row["answer"] = "I could not find this in the provided documents."
        root.end(output=row["answer"], eval__rank=row["rank"],
                 eval__retrieval_hit=(row["rank"] is not None) if item["answerable"] else None,
                 eval__refused=is_refusal(row["answer"]) if row["answer"] is not None else None)
        row["seconds"] = round(time.time() - t0, 1)
        row["timings"] = dict(root.timings)
        row["trace_id"], row["span_id"] = root.trace_id, root.span_id
        rows.append(row)
        print(f"  [{n}/{len(items)}] rank={row['rank']} {item['question'][:70]}")
    return rows


def summarise(rows: list[dict], settings: Settings) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    summary = retrieval_summary([r["rank"] for r in answerable], settings.eval.hit_ks)
    if any("answer_rank" in r for r in answerable):
        # document-level hit@k / mrr above; the same ranks judged by whether the chunk actually contains the answer text
        by_answer = retrieval_summary([r.get("answer_rank") for r in answerable], settings.eval.hit_ks)
        summary |= {f"answer_{k}": v for k, v in by_answer.items() if k != "n_answerable"}
    if rows and rows[0]["answer"] is not None:
        unans = [r for r in rows if not r["answerable"]]
        summary["refusal_rate_on_unanswerable"] = (
            sum(is_refusal(r["answer"]) for r in unans) / len(unans) if unans else None)
        summary["false_refusal_rate_on_answerable"] = (
            sum(is_refusal(r["answer"]) for r in answerable) / len(answerable) if answerable else None)
    lat = latency_summary([r.get("timings") for r in rows])
    if lat:
        summary["latency_s"] = lat
    return summary


class PrivacyError(RuntimeError):
    """Raised instead of sending non-allowed document text to a hosted judge."""


def judge_eligible(item: dict, allowed_source: str | None) -> bool:
    """Single-passage, answerable item whose supporting passage comes from the allowed document.

    multi_passage items are excluded: only the first passage's source is recorded, so a second passage
    (and hence the reference answer) could come from another document.
    allowed_source=None means the corpus is public (no per-document restriction): any answerable single-passage item.
    """
    if not item.get("answerable") or item.get("type") == "multi_passage":
        return False
    return allowed_source is None or item.get("source") == allowed_source


def assert_hosted_safe(rows: list[dict], allowed_source: str | None) -> None:
    """Hard check run right before anything is sent to a hosted judge (a no-op restriction when allowed_source is None)."""
    for r in rows:
        if not judge_eligible(r, allowed_source):
            raise PrivacyError(f"{r['id']}: item is not eligible for the hosted judge (source={r.get('source')!r})")
        if allowed_source is None:
            continue
        for h in r["retrieved"]:
            doc = h["id"].split("::")[0]
            if doc != allowed_source:
                raise PrivacyError(f"{r['id']}: retrieved chunk {h['id']!r} is not from {allowed_source!r}; refusing to send")


def build_judge_llm(settings: Settings):
    """RAGAS judge LLM: hosted OpenAI-compatible API (key from env) or local Ollama."""
    from ragas.llms import LangchainLLMWrapper

    ev = settings.eval
    if ev.judge_provider == "ollama":
        from langchain_ollama import ChatOllama

        return LangchainLLMWrapper(ChatOllama(model=ev.judge_model, temperature=0, num_ctx=8192))
    key = os.environ.get(ev.judge_api_key_env)
    if not key:
        raise SystemExit(f"Set the {ev.judge_api_key_env} environment variable (create a key at aistudio.google.com/apikey).")
    from langchain_openai import ChatOpenAI

    extra = {}
    if ev.judge_reasoning_effort:
        extra["reasoning_effort"] = ev.judge_reasoning_effort
    if ev.judge_min_interval_s > 0:
        from rag.eval.throttle import Throttle, http_clients

        extra["http_client"], extra["http_async_client"] = http_clients(Throttle(ev.judge_min_interval_s))
    # bypass_n: Gemini rejects n>1 ('Multiple candidates is not enabled'), so RAGAS sends n separate requests instead.
    return LangchainLLMWrapper(ChatOpenAI(
        model=ev.judge_model, base_url=ev.judge_base_url, api_key=key, temperature=0, max_retries=6, timeout=180, **extra),
        bypass_n=True)


def judge_candidates(configured: str, listed_ids: list[str], limit: int = 3) -> list[str]:
    """Model ids to try, best first: the configured one, then fast text models (flash / flash-lite) from the listing."""
    strip = lambda i: i.removeprefix("models/")
    skip = ("image", "tts", "embed", "live", "audio", "vision", "robotics", "computer", "veo", "imagen")
    fast = sorted(strip(i) for i in listed_ids if "flash" in i and not any(w in i for w in skip))
    out = [strip(configured)]
    out += [m for m in fast if m not in out and "latest" in m] + [m for m in fast if m not in out]
    return list(dict.fromkeys(out))[: limit + 1]


def _probe(client, model: str, json_mode: bool = False, reasoning_effort: str | None = None):
    """One tiny call with no document text. Returns (ok, detail, seconds); detail starts with 429 / timeout on those failures."""
    import openai

    kwargs: dict = {"response_format": {"type": "json_object"}} if json_mode else {}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0, max_tokens=200, **kwargs,
            messages=[{"role": "user", "content": 'Reply with JSON {"ok": true} and nothing else.'}])
        return True, (resp.choices[0].message.content or "").strip(), time.time() - t0
    except openai.RateLimitError as e:
        h = e.response.headers
        info = {k: v for k, v in h.items() if k.lower().startswith(("retry-after", "x-ratelimit", "ratelimit"))}
        return False, f"429 rate limited; headers={info or 'none'}; body={e.body}", time.time() - t0
    except openai.APITimeoutError:
        return False, "timeout (no reply in time; latency, not a rate limit)", time.time() - t0
    except openai.NotFoundError as e:
        return False, f"404 model not found/available to this key: {e}", time.time() - t0
    except openai.PermissionDeniedError as e:
        return False, f"403 no access (key restricted, terms not accepted, or region unsupported): {e}", time.time() - t0
    except openai.OpenAIError as e:
        return False, f"{type(e).__name__}: {e}", time.time() - t0


def cmd_check_judge(args) -> None:
    """List models, find one that answers, compare latency settings, and measure throughput with 2 concurrent callers."""
    import statistics
    from concurrent.futures import ThreadPoolExecutor

    ev = get_settings().eval
    key = os.environ.get(ev.judge_api_key_env)
    if not key:
        raise SystemExit(f"Set the {ev.judge_api_key_env} environment variable first (aistudio.google.com/apikey).")
    from openai import OpenAI

    client = OpenAI(base_url=ev.judge_base_url, api_key=key, max_retries=0, timeout=120)
    ids = sorted(m.id for m in client.models.list().data)
    cands = judge_candidates(ev.judge_model, ids)
    print(f"{len(ids)} models visible; configured judge_model={ev.judge_model!r} listed: "
          f"{ev.judge_model in ids or 'models/' + ev.judge_model in ids}")

    working = None
    for model in cands:
        for attempt, wait in enumerate([0, 5, 15, 30, 60], 1):
            if wait:
                print(f"  waiting {wait}s before retry {attempt}...")
                time.sleep(wait)
            ok, detail, dt = _probe(client, model)
            print(f"[{model}] try {attempt}: {'OK ' + detail if ok else detail} ({dt:.1f}s)")
            if ok or not detail.startswith(("429", "timeout")):
                break
        if ok:
            working = model
            break
    if not working:
        print("No model answered. 404 = wrong model name, 403 = key/terms/region, 429 = quota, timeout = latency. "
              "The Claude-app judge does not need this.")
        return

    ok, detail, dt = _probe(client, working, json_mode=True)
    print(f"json_object response_format: {'works' if ok else 'not accepted (' + detail[:120] + ') - RAGAS does not need it'}")

    # Latency of the same tiny prompt under different settings (2 calls each).
    # every flash-lite model in the listing (stable ids first, newest version first), at most 3
    skip = ("image", "tts", "embed", "live", "audio", "preview")
    lites = sorted({i.removeprefix("models/") for i in ids if "flash-lite" in i and not any(w in i for w in skip)},
                   reverse=True)
    lites = [m for m in lites if m != working][:3]
    setups = [(working, None), (working, "low"), (working, "none")] + [(m, e) for m in lites for e in (None, "low")]
    print("latency by setup (2 calls each):")
    best = None
    for model, effort in setups:
        res = [_probe(client, model, reasoning_effort=effort) for _ in range(2)]
        good = [r[2] for r in res if r[0]]
        label = f"{model} reasoning_effort={effort}"
        if len(good) == 2:
            med = statistics.median(good)
            print(f"  {label}: {good[0]:.1f}s, {good[1]:.1f}s")
            if best is None or med < best[0]:
                best = (med, model, effort)
        else:
            print(f"  {label}: failed: {next(r[1] for r in res if not r[0])[:150]}")
    if best is None:
        print("no setup answered twice in a row; try again later or use the Claude-app judge")
        return
    _, model, effort = best
    print(f"fastest: eval.judge_model={model!r}, eval.judge_reasoning_effort={effort!r}")

    # Throughput with 2 concurrent callers (what RAGAS will do with judge_max_workers=2).
    n = 6
    t0 = time.time()
    with ThreadPoolExecutor(2) as ex:
        res = list(ex.map(lambda _: _probe(client, model, reasoning_effort=effort), range(n)))
    wall = time.time() - t0
    lats = [r[2] for r in res if r[0]]
    kinds = [r[1].split(" ")[0] for r in res if not r[0]]
    print(f"{len(lats)}/{n} ok with 2 concurrent callers in {wall:.1f}s (~{len(lats) / wall * 60:.0f} calls/min); "
          f"latency median {statistics.median(lats):.1f}s max {max(lats):.1f}s" if lats else "all throughput calls failed")
    if kinds:
        print("failures:", ", ".join(kinds), "(429 = rate limit: set eval.judge_min_interval_s; timeout = latency)")
    if lats:
        per_q = 12 / max(len(lats) / wall * 60, 1)
        print(f"estimate: ~12 judge calls per question -> about {per_q:.1f} min/question, {20 * per_q:.0f} min for 20")
    print("free-tier quotas are per model; see aistudio.google.com/rate-limit")


def ragas_scores(rows: list[dict], settings: Settings) -> dict:
    """RAGAS metrics for answerable items. With a hosted judge, assert_hosted_safe() runs first."""
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from langchain_ollama import OllamaEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (Faithfulness, FactualCorrectness, LLMContextPrecisionWithReference,
                               LLMContextRecall, ResponseRelevancy)
    from ragas.run_config import RunConfig

    ans = [r for r in rows if r["answerable"] and r["answer"] is not None]
    if settings.eval.judge_provider != "ollama":
        assert_hosted_safe(ans, settings.eval.judge_allowed_source)
    data = EvaluationDataset.from_list([
        {"user_input": r["question"], "retrieved_contexts": [h["text"] for h in r["retrieved"]],
         "response": r["answer"], "reference": r["reference_answer"]} for r in ans])
    llm = build_judge_llm(settings)
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=settings.embedding.model))
    result = evaluate(
        data, metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(),
                       LLMContextRecall(), FactualCorrectness()],
        llm=llm, embeddings=emb, run_config=RunConfig(timeout=600, max_workers=settings.eval.judge_max_workers, max_retries=10, max_wait=60),
        raise_exceptions=False)
    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c not in ("user_input", "retrieved_contexts", "response", "reference")]
    for r, (_, d) in zip(ans, df.iterrows()):
        r["ragas"] = {c: (None if d[c] != d[c] else float(d[c])) for c in metric_cols}   # NaN -> None
    return {c: float(df[c].mean(skipna=True)) for c in metric_cols} | {"ragas_n": len(ans)}


def run_name_for(tag: str) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}"


def save_run(tag: str, settings: Settings, rows: list[dict], summary: dict, name: str | None = None) -> Path:
    run_dir = settings.paths.resolve("eval_dir") / "runs" / (name or run_name_for(tag))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "items.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir


def apply_judge_scope(testset: list[dict], settings: Settings) -> tuple[list[dict], dict]:
    """Hosted-judge scope: only the allowed document's single-passage answerable items, retrieval filtered to that document.

    With judge_allowed_source=None (public corpus) nothing is restricted: all items are kept (unanswerable ones are still
    needed for the refusal rate; RAGAS only scores the answerable rows) and retrieval is not filtered.
    Mutates settings.retrieval.source_filter. Returns (items, summary_extra)."""
    allowed = settings.eval.judge_allowed_source
    if allowed is None:
        return list(testset), {"ragas_scope": "unrestricted"}
    items = [i for i in testset if judge_eligible(i, allowed)]
    settings.retrieval.source_filter = allowed
    return items, {"ragas_scope": allowed,
                   "note": "retrieval filtered to the allowed document; not comparable with all-documents runs"}


def format_table(results: list[tuple[str, dict]]) -> str:
    """Comparison table (best MRR first): retrieval, RAGAS (when scored), refusal and latency columns."""
    cols = [("mrr", "mrr"), ("hit@1", "hit@1"), ("hit@3", "hit@3"), ("hit@5", "hit@5"), ("answer_mrr", "ans_mrr"),
            ("answer_hit@5", "ans_hit@5"), ("faithfulness", "faith"), ("answer_relevancy", "relev"), ("llm_context_precision_with_reference", "ctx_prec"),
            ("context_recall", "ctx_rec"), ("factual_correctness(mode=f1)", "fact_corr"),
            ("false_refusal_rate_on_answerable", "false_ref")]
    cols = [(k, h) for k, h in cols if any(isinstance(s.get(k), float) for _, s in results)]
    lat = lambda s, step: (s.get("latency_s", {}).get(step) or {}).get("median")
    header = ["variant".ljust(40)] + [h.rjust(9) for _, h in cols] + ["gen_s".rjust(7), "total_s".rjust(8), "sec".rjust(6)]
    lines = [" | ".join(header)]
    for label, s in sorted(results, key=lambda r: -(r[1].get("mrr") or 0)):
        cells = [(f"{s[k]:.3f}" if isinstance(s.get(k), float) else "-").rjust(9) for k, _ in cols]
        g, t = lat(s, "generate"), lat(s, "total")
        lines.append(" | ".join([label.ljust(40)] + cells + [(f"{g:.1f}" if g is not None else "-").rjust(7),
                                                              (f"{t:.1f}" if t is not None else "-").rjust(8),
                                                              str(s.get("seconds", "-")).rjust(6)]))
    return "\n".join(lines)


def ragas_annotations(rows: list[dict]) -> list[dict]:
    return [{"span_id": r["span_id"], "name": f"ragas.{k}", "score": v, "kind": "LLM"}
            for r in rows if r.get("span_id") and r.get("ragas") for k, v in r["ragas"].items() if v is not None]


def claude_annotations(rows: dict, scores: dict) -> list[dict]:
    out = []
    for i, sc in scores.items():
        row = rows.get(i)
        if not row or not row.get("span_id"):
            continue
        for dim in ("faithfulness", "correctness", "completeness", "context_relevance"):
            if dim in sc:
                out.append({"span_id": row["span_id"], "name": f"claude.{dim}", "score": float(sc[dim]),
                            "kind": "HUMAN", "explanation": sc.get("comment") if dim == "correctness" else None})
    return out


def cmd_run(args) -> None:
    settings = get_settings()
    testset = load_testset(settings, args.testset)
    summary_extra: dict = {}
    if (args.ragas or args.judge_scope) and not args.retrieval_only and settings.eval.judge_provider != "ollama":
        # Hosted judge: run only the allowed document's single-passage questions, with retrieval limited to that document.
        testset, summary_extra = apply_judge_scope(testset, settings)
        if settings.eval.judge_allowed_source:
            print(f"hosted-judge scope: {len(testset)} eligible items from {settings.eval.judge_allowed_source}; "
                  f"retrieval filtered to that document")
        else:
            print(f"hosted-judge scope: unrestricted (public corpus), {len(testset)} items")
    run_name = run_name_for(f"{args.tag}_{slugify_testset(args.testset)}")
    rows = run_items(testset, settings, generate_answers=not args.retrieval_only, limit=args.limit, session_id=run_name)
    summary = {"testset": args.testset} | summary_extra | summarise(rows, settings)
    if args.ragas and not args.retrieval_only:
        t0 = time.time()
        summary |= ragas_scores(rows, settings)
        summary["ragas_seconds"] = round(time.time() - t0)
    run_dir = save_run(f"{args.tag}_{slugify_testset(args.testset)}", settings, rows, summary, name=run_name)
    flush_traces()
    if tracing_enabled():
        sent = annotate(settings, ragas_annotations(rows))
        summary["phoenix_session"] = run_name
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"traces: Phoenix project {settings.tracing.project_name!r}, session {run_name!r} (http://localhost:6006); {sent} RAGAS annotations sent")
    print(json.dumps(summary, indent=2))
    print("saved to", run_dir)


def cmd_score_ragas(args) -> None:
    """Score saved rows with RAGAS (hosted judge, key from the env var). Works on one run folder or an ablation folder of variants."""
    settings = get_settings()
    root = Path(args.run)
    dirs = [root] if (root / "items.jsonl").exists() else sorted(d for d in root.iterdir() if (d / "items.jsonl").exists())
    if not dirs:
        raise SystemExit(f"no items.jsonl found under {root}")
    results = []
    for d in dirs:
        rows = [json.loads(l) for l in (d / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        run_settings = Settings.model_validate_json((d / "config.json").read_text(encoding="utf-8")) \
            if (d / "config.json").exists() else settings
        run_settings.eval = settings.eval          # judge settings come from the current config; the key is read from the env
        print(f"scoring {d.name}: {sum(1 for r in rows if r['answerable'] and r['answer'] is not None)} answered items")
        t0 = time.time()
        summary |= ragas_scores(rows, run_settings)
        summary["ragas_seconds"] = round(time.time() - t0)
        (d / "items.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        (d / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        sent = annotate(settings, ragas_annotations(rows))
        if sent:
            print(f"  {sent} annotations sent to Phoenix")
        results.append((summary.get("variant", d.name), summary))
    if len(dirs) > 1:
        (root / "ablation.json").write_text(
            json.dumps([{"variant": s, "summary": m} for s, m in results], indent=2), encoding="utf-8")
    print("\n" + format_table(results))


JUDGE_PROMPT = """\
You are grading answers from a question-answering system over technical support documents.
For each item below, score from 1 (worst) to 5 (best), using ONLY the retrieved contexts and the reference answer:
- faithfulness: every claim in the answer is supported by the retrieved contexts (no invented facts)
- correctness: the answer agrees with the reference answer (for unanswerable items, 5 = correctly says it cannot find the answer)
- completeness: the answer covers what the reference answer covers
- context_relevance: the retrieved contexts contain what is needed to answer
Output ONLY a JSON array: [{"id": str, "faithfulness": int, "correctness": int, "completeness": int,
"context_relevance": int, "comment": str}]  (comment <= 15 words)

ITEMS
=====
"""


def cmd_export_judge(args) -> None:
    run_dir = Path(args.run)
    rows = [json.loads(l) for l in (run_dir / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["answer"] is not None]
    import random

    random.Random(0).shuffle(rows)
    rows = rows[: args.n]
    md = JUDGE_PROMPT
    for r in rows:
        ctx = "\n\n".join(f"  ({i}) {h['text'][:1500]}" for i, h in enumerate(r["retrieved"], 1))
        md += (f"\n### id: {r['id']}\nQuestion: {r['question']}\nReference answer: {r['reference_answer'] or 'NOT_ANSWERABLE'}\n"
               f"System answer: {r['answer']}\nRetrieved contexts:\n{ctx}\n")
    out = run_dir / "judge_packet.md"
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out} ({len(rows)} items). Paste into the Claude app, save the JSON reply as {run_dir / 'judge_reply.json'},"
          f" then run import-judge.")


def cmd_import_judge(args) -> None:
    run_dir = Path(args.run)
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", (run_dir / "judge_reply.json").read_text(encoding="utf-8").strip())
    scores = {s["id"]: s for s in json.loads(raw)}
    dims = ["faithfulness", "correctness", "completeness", "context_relevance"]
    out = {d: sum(s[d] for s in scores.values()) / len(scores) for d in dims} | {"judged_n": len(scores)}
    rows = {r["id"]: r for r in (json.loads(l) for l in (run_dir / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    # Agreement between the RAGAS judge and the Claude-app judge on the same items (Spearman rank correlation).
    for ragas_key, dim in (("faithfulness", "faithfulness"), ("factual_correctness(mode=f1)", "correctness"),
                           ("context_recall", "context_relevance")):
        pairs = [(rows[i]["ragas"][ragas_key], sc[dim]) for i, sc in scores.items()
                 if i in rows and rows[i].get("ragas") and rows[i]["ragas"].get(ragas_key) is not None]
        out[f"spearman_ragas_{ragas_key}_vs_claude_{dim}"] = (
            spearman([a for a, _ in pairs], [b for _, b in pairs]) if len(pairs) >= 5 else None)
        out[f"agreement_n_{dim}"] = len(pairs)
    (run_dir / "judge_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    sent = annotate(get_settings(), claude_annotations(rows, scores))
    if sent:
        out["phoenix_annotations_sent"] = sent
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", default="run")
    r.add_argument("--limit", type=int)
    r.add_argument("--ragas", action="store_true", help="also score answers with RAGAS (hosted judge; restricts the run to eval.judge_allowed_source)")
    r.add_argument("--judge-scope", action="store_true",
                   help="restrict to eval.judge_allowed_source single-passage items with retrieval filtered to it (no judge call; score later with score-ragas)")
    r.add_argument("--retrieval-only", action="store_true", help="skip answer generation (fast, no LLM)")
    r.add_argument("--testset", default="testset",
                   help="data/eval/<name>.jsonl: testset (dev, default) or testset_holdout (final check only)")
    sub.add_parser("check-judge")
    sp = sub.add_parser("score-ragas")
    sp.add_argument("--run", required=True, help="a run folder, or an ablation folder containing variant subfolders")
    for name in ("export-judge", "import-judge"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        if name == "export-judge":
            p.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    {"run": cmd_run, "export-judge": cmd_export_judge, "import-judge": cmd_import_judge,
     "check-judge": cmd_check_judge, "score-ragas": cmd_score_ragas}[args.cmd](args)


if __name__ == "__main__":
    main()

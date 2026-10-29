"""Sweep retrieval / generation settings and print a comparison table. One folder per variant, saved after every variant.

  python -m rag.eval.ablate                              # retrieval-only (fast, no LLM): hit@k / MRR per variant
  python -m rag.eval.ablate --rerank off                 # the cheap half; then add `--rerank on --resume <dir>`
  python -m rag.eval.ablate --full --judge-scope --modes fts --rerank off --chunk-sizes 256 512 --top-ks 3 5
                                                         # generate answers for VM0047 single-passage items (score later with
                                                         # `run_eval score-ragas --run <dir>` in a shell that has the judge key)
  python -m rag.eval.ablate --resume data/eval/runs/<dir> ...   # skip variants already finished in that folder

Output: data/eval/runs/<timestamp>_ablation_<testset>/<variant>/{config.json,items.jsonl,summary.json} + ablation.json.
One LanceDB table is built (and cached) per distinct chunking config: chunks_<strategy>_<max>_<overlap>.
Every question is traced to Phoenix (if running), one session per variant: <ablation folder>/<variant>.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

from rag.config import load_settings
from rag.eval.run_eval import (
    apply_judge_scope,
    format_table,
    load_testset,
    ragas_scores,
    run_items,
    summarise,
    slugify_testset,
)
from rag.tracing import flush as flush_traces


def variant_slug(strategy: str, size: int, mode: str, rerank: str, top_k: int | None) -> str:
    return f"{strategy}_{size}_{mode}_rr-{rerank}" + (f"_k{top_k}" if top_k else "")


def variant_overrides(strategy: str, size: int, mode: str, rerank: str, top_k: int | None) -> dict:
    overlap = max(16, size // 8)
    retrieval = {"mode": mode} | ({"top_k": top_k} if top_k else {})
    return {
        "chunking": {"strategy": strategy, "max_tokens": size, "overlap": overlap},
        "retrieval": retrieval,
        "rerank": {"enabled": rerank == "on"},
        "paths": {"table_name": f"chunks_{strategy}_{size}_{overlap}"},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--strategies", nargs="+", default=["heading_aware"])
    ap.add_argument("--modes", nargs="+", default=["vector", "fts", "hybrid"])
    ap.add_argument("--rerank", nargs="+", default=["off", "on"], choices=["off", "on"])
    ap.add_argument(
        "--top-ks", type=int, nargs="+", default=[None], help="chunks passed to the LLM (default: config value)"
    )
    ap.add_argument("--limit", type=int)
    ap.add_argument("--full", action="store_true", help="generate answers too")
    ap.add_argument(
        "--judge-scope",
        action="store_true",
        help="only eval.judge_allowed_source single-passage items, retrieval filtered to that document (needed before RAGAS)",
    )
    ap.add_argument(
        "--ragas",
        action="store_true",
        help="with --full: score with RAGAS inline (needs the judge key; implies --judge-scope)",
    )
    ap.add_argument("--resume", help="existing ablation folder: skip variants that already have a summary.json there")
    ap.add_argument(
        "--testset", default="testset", help="tune on the dev set; keep testset_holdout for the final check"
    )
    args = ap.parse_args()
    scoped = args.judge_scope or args.ragas

    base = load_settings()
    testset = load_testset(base, args.testset)
    runs_dir = base.paths.resolve("eval_dir") / "runs"
    out_dir = (
        Path(args.resume)
        if args.resume
        else runs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_ablation_{slugify_testset(args.testset)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"testset: {args.testset} ({len(testset)} items); output: {out_dir}")

    results: list[tuple[str, dict]] = []
    for strat, size, mode, rr, top_k in itertools.product(
        args.strategies, args.chunk_sizes, args.modes, args.rerank, args.top_ks
    ):
        slug = variant_slug(strat, size, mode, rr, top_k)
        summary_path = out_dir / slug / "summary.json"
        if summary_path.exists():
            print(f"\n== {slug} (already done, skipping)")
            results.append((slug, json.loads(summary_path.read_text(encoding="utf-8"))))
            continue
        print(f"\n== {slug}")
        settings = load_settings(overrides=variant_overrides(strat, size, mode, rr, top_k))
        items = testset
        extra: dict = {}
        if scoped:
            items, extra = apply_judge_scope(testset, settings)
            print(
                f"  judge scope: {len(items)} items from {settings.eval.judge_allowed_source or 'all documents (unrestricted)'}"
            )
        t0 = time.time()
        rows = run_items(
            items, settings, generate_answers=args.full, limit=args.limit, session_id=f"{out_dir.name}/{slug}"
        )
        summary = {"variant": slug, "testset": args.testset} | extra | summarise(rows, settings)
        if args.full and args.ragas:
            summary |= ragas_scores(rows, settings)
        summary["seconds"] = round(time.time() - t0)
        _save_in(out_dir / slug, settings, rows, summary)
        flush_traces()
        results.append((slug, summary))
        (out_dir / "ablation.json").write_text(
            json.dumps([{"variant": s, "summary": m} for s, m in results], indent=2), encoding="utf-8"
        )
        print(format_table(results))

    print("\n" + format_table(results))
    print("\nsaved", out_dir)


def _save_in(variant_dir: Path, settings, rows, summary) -> None:
    """One variant's config snapshot, rows and summary."""
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "config.json").write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    (variant_dir / "items.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

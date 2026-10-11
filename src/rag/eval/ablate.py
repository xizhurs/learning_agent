"""Sweep retrieval settings and print a comparison table.

  python -m rag.eval.ablate                 # retrieval-only (fast, no LLM): hit@k / MRR per variant
  python -m rag.eval.ablate --full          # also generate answers + refusal rates (slow on CPU)
  python -m rag.eval.ablate --chunk-sizes 256 512 1024 --modes vector hybrid --rerank off on

One LanceDB table is built (and cached) per distinct chunking config: chunks_<strategy>_<max>_<overlap>.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

from rag.config import load_settings
from rag.eval.run_eval import load_testset, ragas_scores, run_items, summarise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--strategies", nargs="+", default=["heading_aware"])
    ap.add_argument("--modes", nargs="+", default=["vector", "fts", "hybrid"])
    ap.add_argument("--rerank", nargs="+", default=["off", "on"], choices=["off", "on"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--full", action="store_true", help="generate answers too")
    ap.add_argument("--ragas", action="store_true", help="with --full: also run RAGAS")
    ap.add_argument("--testset", default="testset", help="tune on the dev set; keep testset_holdout for the final check")
    args = ap.parse_args()

    base = load_settings()
    testset = load_testset(base, args.testset)
    print(f"testset: {args.testset} ({len(testset)} items)")
    results = []
    for strat, size, mode, rr in itertools.product(args.strategies, args.chunk_sizes, args.modes, args.rerank):
        overrides = {
            "chunking": {"strategy": strat, "max_tokens": size, "overlap": max(16, size // 8)},
            "retrieval": {"mode": mode},
            "rerank": {"enabled": rr == "on"},
            "paths": {"table_name": f"chunks_{strat}_{size}_{max(16, size // 8)}"},
        }
        settings = load_settings(overrides=overrides)
        label = f"{strat}/{size} {mode} rerank={rr}"
        print(f"\n== {label}")
        t0 = time.time()
        rows = run_items(testset, settings, generate_answers=args.full, limit=args.limit)
        summary = summarise(rows, settings)
        if args.full and args.ragas:
            summary |= ragas_scores(rows, settings)
        results.append((label, summary, round(time.time() - t0)))

    keys = sorted({k for _, s, _ in results for k in s if k != "n_answerable"}, key=lambda k: (k != "mrr", k))
    print("\n" + " | ".join(["variant".ljust(34)] + [k[:14].rjust(14) for k in keys] + ["sec"]))
    for label, s, secs in sorted(results, key=lambda r: -r[1].get("mrr", 0)):
        cells = [f"{s[k]:.3f}".rjust(14) if isinstance(s.get(k), float) else "-".rjust(14) for k in keys]
        print(" | ".join([label.ljust(34)] + cells + [str(secs)]))
    out = base.paths.resolve("eval_dir") / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}_ablation_{args.testset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([{"variant": l, "testset": args.testset, "summary": s, "seconds": t}
                               for l, s, t in results], indent=2), encoding="utf-8")
    print("\nsaved", out)


if __name__ == "__main__":
    main()

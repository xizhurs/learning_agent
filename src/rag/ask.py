"""python -m rag.ask "question" [--show-context]"""
from __future__ import annotations

import argparse
import logging

from rag.pipeline import Pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--show-context", action="store_true", help="print retrieved chunks with scores")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    ans = Pipeline().ask(args.question)
    if args.show_context:
        for i, h in enumerate(ans.hits, 1):
            print(f"--- [{i}] score={h.score:.3f} {h.citation()} ({h.kind})\n{h.text[:600]}\n")
    print(ans.answer)
    print("\nSources:")
    for i, h in enumerate(ans.hits, 1):
        print(f"  [{i}] {h.citation()}  (score {h.score:.3f})")


if __name__ == "__main__":
    main()

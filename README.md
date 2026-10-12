# Carbon methodology RAG

Local RAG over `data/documents/` (VM0047 PDF + VCS PD draft DOCX): Docling parsing, LanceDB (vector + full-text),
selectable vector / keyword / hybrid search, optional cross-encoder reranking, Ollama generation, retrieval
evaluation, FastAPI chat UI.
All tunables live in `config.yaml`; any value can be overridden per run with an env var,
e.g. `RAG__RETRIEVAL__MODE=vector` or `RAG__CHUNKING__MAX_TOKENS=256`.

## Setup
```
uv sync
ollama pull nomic-embed-text llama3.2:3b
```

## Run
| Step | Command |
|---|---|
| Parse, chunk, embed, index | `uv run python -m rag.ingest` (`--reparse` to ignore the Docling cache) |
| Ask from the CLI | `uv run python -m rag.ask "question" --show-context` |
| Web chat | `uv run uvicorn rag.api:app` then open http://localhost:8000 |
| Tests | `uv run pytest` |

Tracked changes in the DOCX are accepted before parsing (`src/rag/ingest/docx_clean.py`).
Changing `chunking.*` or `embedding.*` requires re-running `rag.ingest`.

## Dataset: QASPER tiny (current corpus)
`uv run python scripts/create_qasper_tiny.py` downloads QASPER (allenai/qasper, CC BY 4.0; validation + test splits) into
`data/raw/qasper/` and writes:
- `data/documents/<paper id>.md`: 200 NLP papers (`#` title, `##` sections, `###` subsections), parsed without Docling.
  100 are the papers the questions are about, 100 are random distractors.
- `data/eval/qasper_tiny/{dev,val,test}.jsonl`: 100 questions, one per paper (dev 32+8, val 24+6, test 24+6
  answerable + unanswerable). Plus `contexts.jsonl`, `manifest.json` and `sample_preview.md`.

How questions are chosen:
- At least two annotators, all agreeing on answerable vs. unanswerable.
- Answerable questions need text evidence; a table or figure alone does not count.
- Each question is prefixed with its paper's title, because QASPER questions ("which datasets do they use?") are ambiguous
  across papers. That makes document-level `hit@k` easy, so tune on `answer_hit@k`: an annotator evidence paragraph found
  in a retrieved chunk.

Workflow:
1. Run `uv run python -m rag.ingest`. It reuses embeddings cached in `data/processed/embed_cache/`, so an interrupted
   run resumes.
2. Evaluate with `uv run python -m rag.eval.run_eval run --retrieval-only --testset qasper_tiny/dev`.

`retrieval.mode: fts` and `rerank.enabled: false` were the TechQA winners, so ablate them again on QASPER. The TechQA
files are archived in `data/archive/techqa/`.

## Evaluation (ground truth and answer judging by Claude in the Claude app)
1. `uv run python -m rag.eval.gen_testset export` writes `data/eval/packets/packet_NN.md`.
2. Paste each `.md` packet into the Claude app (one chat each). Save the JSON-only reply as
   `data/eval/replies/packet_NN.json`. Do not upload the `*.passages.json` files.
3. `uv run python -m rag.eval.gen_testset import` validates (quotes must occur verbatim in the source) and writes
   `data/eval/testset.jsonl`. **Review that file by hand** and delete weak questions; see `import_report.txt` for rejects.
4. `uv run python -m rag.eval.run_eval run --tag baseline --retrieval-only` : hit@k and MRR, no LLM, fast.
   Without `--retrieval-only` it also generates answers (slow on CPU) and reports refusal rates on unanswerable /
   answerable questions. `--testset testset_holdout` runs the held-out set (use it only for final checks, not tuning).
5. `uv run python -m rag.eval.ablate` sweeps chunk size x search mode x rerank (retrieval-only by default;
   `--full` adds generation) and prints one comparison table.
6. Answer quality: `run_eval export-judge --run <dir> [--n 20]` writes `judge_packet.md`. Paste it into the Claude app,
   save the reply as `<dir>/judge_reply.json`, then `run_eval import-judge --run <dir>` for mean scores
   (faithfulness, correctness, completeness, context relevance) and, if RAGAS was run, its agreement with Claude.

A held-out set is built the same way with `gen_testset export --set holdout` / `import --set holdout` (passages are
disjoint from the dev set) and lands in `data/eval/testset_holdout.jsonl`.

Retrieval hit@k is matched on the supporting quote text, so it stays valid when chunking changes.

## Results (retrieval-only, 54 dev / 69 held-out answerable questions)
Dev ablation (`data/eval/runs/*_ablation_testset.json`), 512-token chunks:

| Search | Rerank | MRR | hit@1 | hit@3 | hit@5 | time / 60 questions |
|---|---|---|---|---|---|---|
| keyword (FTS) | off | **0.835** | 0.722 | 0.944 | 0.963 | ~1 s |
| hybrid | on | 0.785 | 0.685 | 0.907 | 0.963 | ~1000 s |
| vector | on | 0.778 | 0.685 | 0.870 | 0.926 | ~1000 s |
| hybrid | off | 0.727 | 0.611 | 0.815 | 0.889 | ~6 s |
| vector | off | 0.602 | 0.500 | 0.685 | 0.741 | ~5 s |

Chunk size (256 / 512 / 1024) made little difference for keyword search (MRR 0.845 / 0.835 / 0.836). Reranking lowered
keyword-search scores and only helped vector search, at about 15 s per query on CPU.

Held-out confirmation: keyword, no rerank, 512 gave MRR **0.850** (hit@1 0.797, hit@5 0.957) versus 0.710 for the old
hybrid + rerank default. The default in `config.yaml` is therefore `mode: fts`, `rerank.enabled: false`.

Caveat: both test sets were written by Claude from the passages themselves, so questions share exact wording with the
supporting text, which favours keyword matching. Real user questions that paraphrase may do better with `hybrid` (set
`RAG__RETRIEVAL__MODE=hybrid`). Differences under about 0.05 MRR on ~60 questions are noise.

A `rerank.min_score` refusal threshold does not work: the near-topic unanswerable questions get reranker scores of
0.96 to 0.999, the same range as answerable ones (clearly off-topic queries do score near 0). Refusal is left to the
generator prompt and is measured with generation runs (`refusal_rate_on_unanswerable`).

## RAGAS (optional)
`run_eval run --ragas` scores faithfulness, answer relevancy, context precision/recall and factual correctness with a
judge LLM set in `config.yaml` (`eval.judge_*`). Default is Google AI Studio (Gemini, via its OpenAI-compatible endpoint):
1. Create a key at aistudio.google.com/apikey and set it in your shell: `$env:GEMINI_API_KEY = "..."` (never put it in files).
2. `uv run python -m rag.eval.run_eval check-judge` lists the models your key can use, finds one that answers, and measures
   the request rate; set `eval.judge_model` / `eval.judge_min_interval_s` as it suggests (limits: aistudio.google.com/rate-limit).
   Any other OpenAI-compatible host (e.g. Mistral: `https://api.mistral.ai/v1`, `MISTRAL_API_KEY`) can be used with
   `RAG__EVAL__JUDGE_BASE_URL`, `RAG__EVAL__JUDGE_MODEL`, `RAG__EVAL__JUDGE_API_KEY_ENV`.
3. `uv run python -m rag.eval.run_eval run --tag ragas --testset testset_holdout --ragas --limit 20`.

Privacy: on Google's free tier submitted content is used to improve its products and human reviewers may read it, so only the public VM0047 PDF may be sent. A `--ragas` run therefore
keeps only single-passage questions whose source is `eval.judge_allowed_source`, filters retrieval to that document, and
`assert_hosted_safe()` aborts if any other text would be sent. Its retrieval numbers are for the filtered setup and are
not comparable with all-documents runs. Embeddings for answer relevancy stay local (Ollama).
A local Ollama judge (`judge_provider: ollama`) works but measured about 12 minutes per question with qwen3:4b on this
CPU-only machine, with malformed JSON in 2 of 10 calls; the 9B model was slower still.

## Tracing and latency (Arize Phoenix, local)
Every question (web chat and `run_eval`) is traced to a local [Arize Phoenix](https://arize.com/docs/phoenix) server, so you can
see the latency of each step and open one question end to end. Traces stay on this machine (Phoenix keeps them in
`~/.phoenix`, outside OneDrive); nothing is sent to a hosted service.

Phoenix is installed as an isolated tool so its large dependency set stays out of the project environment:
```powershell
uv tool install arize-phoenix --python 3.12      # once
phoenix serve                                    # in its own terminal; UI at http://localhost:6006
```
Start it before the chat or an eval run. If it is not running, tracing is skipped automatically (no errors, runs are not slowed);
`tracing.enabled: false` in `config.yaml` (or `RAG__TRACING__ENABLED=false`) turns it off.

What you see per question (project `carbon-rag`): root `rag.query` (chat) or `eval.question` (eval), with children
`rewrite` (follow-ups only) -> `retrieve` (`embed_query`, `search`, `rerank`) -> `generate`:
- retrieved chunks with id, text, score and section/page;
- the exact prompt messages and the answer;
- Ollama token counts, model load time, prompt-evaluation time, generation time and tokens/s, plus time-to-first-token;
- for eval items the ground truth (`eval.reference_answer`, `eval.reference_quote`, `eval.answerable`, `eval.type`, `eval.source`),
  the retrieval `eval.rank` and whether the answer was a refusal;
- RAGAS scores (`ragas.*`) and Claude-app scores (`claude.*`, added by `import-judge`) as annotations on the same trace.

Each eval run is a Phoenix *session* named like the run folder (printed at the end of `run_eval run`, also saved as
`phoenix_session` in `summary.json`); each chat page load is a session too. Without Phoenix, `items.jsonl` still has `timings`
per step plus `trace_id`/`span_id`, and `summary.json` has median / p95 / max seconds per step (`latency_s`).

Finding from the first traces (llama3.2:3b, CPU, 3 questions): retrieval takes about 0.03 s and generating the answer 2-7 s, but
**reading the prompt takes 23-61 s** (about 1,000-2,000 prompt tokens at roughly 30 tokens/s). Latency therefore scales with how
many chunk tokens go into the prompt, so `retrieval.top_k` and chunk size are the levers to try, judged on answer quality.

Generation model: `llama3.2:3b` (about 2 GB, `ollama pull llama3.2:3b`); the previous default `llama3.1:8b-instruct-q4_K_M` is still
available with `RAG__GENERATION__MODEL=llama3.1:8b-instruct-q4_K_M`. The prompt was tuned on the 8B model; compare quality on the
same held-out items before relying on the 3B.

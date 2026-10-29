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

The QASPER ablation is in "Results" below. The earlier TechQA corpus is archived in `data/archive/techqa/`.

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

## Results (QASPER dev: 32 answerable + 8 unanswerable questions)
Ablation of chunk size x search mode x rerank (`data/eval/runs/20261012-192903_ablation_qasper_tiny_dev`). Tune on
`answer_mrr` / `answer_hit@k` (an annotator evidence paragraph found in a retrieved chunk). The plain `mrr` / `hit@k`
columns are document-level and trivially 1.0, because every question is prefixed with its paper's title.

| Chunk | Search | Rerank | answer_mrr | answer_hit@1 | answer_hit@5 | evidence_recall@5 |
|---|---|---|---|---|---|---|
| 1024 | any | on | **0.623** | 0.469 | 0.906 | 0.770 |
| 512 | any | on | 0.607 | 0.469 | 0.906 | 0.785-0.793 |
| 256 | any | on | 0.46 | 0.344 | 0.69-0.72 | 0.56-0.59 |
| 1024 | fts | off | 0.359 | 0.156 | 0.812 | 0.645 |
| 512 | vector / fts / hybrid | off | 0.321 / 0.295 / 0.287 | 0.19 / 0.09 / 0.13 | 0.625 / 0.656 / 0.656 | 0.49 / 0.48 / 0.53 |
| 256 | vector / fts / hybrid | off | 0.241 / 0.251 / 0.222 | 0.13 / 0.09 / 0.13 | 0.50 / 0.56 / 0.41 | 0.43 / 0.40 / 0.31 |

- **Reranking is the big win on QASPER** (the opposite of TechQA): at 512-1024 tokens it roughly doubles `answer_mrr`
  and lifts `answer_hit@5` to 0.91. After reranking, search mode makes no difference at 512 or 1024: the 30 candidates
  already contain the answer chunks and the cross-encoder decides the order, so cheap FTS is the first stage.
- 256-token chunks are clearly worse. 512 vs 1024 is within noise (one question is 0.03 of hit rate on 32 questions), so
  the default stays at 512 to avoid re-ingesting.
- Cost: reranking takes about 15 s/query at 256 tokens and about 40 s/query at 1024 on this CPU (about 1 s without).

Generation + RAGAS on the 1024 / fts / rerank-on config (`data/eval/runs/20261012-233735_ragas_fts_1024_rr_qasper_tiny_dev`,
llama3.2:3b answers, `gemini-3.5-flash-lite` judge): faithfulness 0.90, answer relevancy 0.75, context precision 0.53,
context recall 0.65, factual correctness 0.11; `answer_f1` 0.22, no false refusals on answerable questions, but the model
refuses only 25% of unanswerable ones (`not_found_accuracy` 0.85). Median latency 95 s/question (rerank 39 s, generate
56 s). The low factual correctness / F1 mostly reflects the 3B model's long answers against QASPER's short references.
Six of 160 RAGAS jobs failed on Gemini's free-tier daily quota (500 requests/model/day), so those scores cover slightly
fewer items. Not yet run: the val / test splits for a final check.

A `rerank.min_score` refusal threshold does not work: the near-topic unanswerable questions get reranker scores of
0.96 to 0.999, the same range as answerable ones (clearly off-topic queries do score near 0). Refusal is left to the
generator prompt and is measured with generation runs (`refusal_rate_on_unanswerable`).

## Results (carbon corpus: rerank on vs off)
Four public documents (VM0047, VCS Standard v5.0, VMD0054, VT0008), Claude-written questions: dev 70 (65 answerable) and val 50
(45 answerable). Hybrid search, `candidates_k` 30, `top_k` 5. Answers judged 1-5 by Claude in the Claude app (`export-judge`).
Rerank was kept on by choice (`rerank.enabled: true`); this is what the evidence says about it:

| Split, index, generator | MRR off -> on | evidence_recall@5 off -> on | judge faithfulness | judge correctness |
|---|---|---|---|---|
| dev, 512 index*, llama3.2:3b (local) | 0.832 -> 0.835 | 0.962 -> 0.933 | 4.60 -> 4.54 | 4.10 -> 4.01 |
| val, 512 index*, llama3.2:3b (local) | 0.789 -> 0.837 | 0.889 -> 0.933 | 3.86 -> 4.04 | 3.52 -> 3.50 |
| val, 1024 index, Llama 3.1 8B (Databricks endpoint) | 0.804 -> 0.819 | 0.885 -> 0.911 | 4.36 -> 4.58 | 4.08 -> 4.08 |

- Rerank on helps retrieval a little on val (MRR +0.015 to +0.05, evidence_recall@3 +0.07 to +0.11) and not on dev
  (evidence_recall@5 -0.03). Judge faithfulness is +0.18 / +0.22 on the two val runs and -0.06 on dev.
- **Answer correctness did not improve in any run** (-0.09 to 0.00). On val with Llama 3.1 8B it changed on 10 of 50 questions,
  5 up and 5 down. The remaining errors look like generator errors (hallucinated numbers, a leading "Yes", wrong table row).
- Cost: reranking is about 17-26 s/query on this CPU (median per question with the 3B model: 34 s off, 55-59 s on).
- The reranker reads at most 512 tokens and 10-13% of retrieved chunks are cut, but the gold quote lies beyond the cut in none
  of the 26 truncated gold chunks, so truncation does not explain the small gain.

RAGAS on the two val runs on the 1024 index (Llama 3.1 8B answers), judged by `databricks-qwen3-next-80b-a3b-instruct` with Databricks
embeddings (`notebooks/score_ragas_databricks.py`), 45 answerable questions each:

| RAGAS metric | rerank off | rerank on | Δ | questions up / down |
|---|---|---|---|---|
| faithfulness | 0.651 | 0.708 | +0.057 | 10 / 7 |
| answer_relevancy | 0.478 | 0.511 | +0.033 | 12 / 9 |
| context precision (with reference) | 0.781 | 0.804 | +0.023 | 10 / 8 |
| context_recall | 0.893 | 0.922 | +0.030 | 4 / 1 |
| factual_correctness (f1) | 0.313 | 0.376 | +0.063 | 15 / 7 |
| answer_matches_reference | 0.778 | 0.800 | +0.022 | 5 / 4 |

- Rerank on is ahead on all six metrics, in line with the Claude judge (faithfulness up, correctness flat). Only factual_correctness
  (15 up, 7 down) and context_recall (4 up, 1 down) lean the same way across questions; the others move up and down about equally.
- RAGAS caveats: one judge (Qwen3-Next 80B for both runs; not the Gemini judge of the QASPER runs, so absolute values are not
  comparable with those), 45 questions, one scoring pass per run. The `config.json` in the scored folders still lists the
  config's default Gemini judge: the Databricks judge is injected and not recorded there. The judge was changed during scoring (the
  GPT OSS 120B attempt hit the workspace request-rate limit, then Llama 3.3 70B was tried), and these values come from the final pass. An
  earlier pass with another judge gave rerank-off scores that differ by 0.00-0.012 per metric (e.g. faithfulness 0.639 vs 0.651,
  factual_correctness 0.303 vs 0.313). That is a rough sense of how much the numbers move with the judge; the on-off gaps (0.02-0.06) are only a
  few times larger, so read them as a consistent small lean, not a measured effect.

**Larger generator: Llama 3.3 70B on the same retrieval (val, 1024 index, Databricks endpoints).** Same saved chunks as the 8B runs
(retrieved ids identical for every question; only `generation.model` differs), 50 questions, 45 answerable.

| | 8B off | 8B on | 70B off | 70B on |
|---|---|---|---|---|
| Claude judge: faithfulness | 4.36 | 4.58 | 4.78 | 4.76 |
| Claude judge: correctness | 4.08 | 4.08 | 4.28 | 4.42 |
| Claude judge: completeness | 3.60 | 3.62 | 3.96 | 4.12 |
| questions with correctness <= 2 | 11 | 10 | 6 | 6 |
| false refusal on answerable | 0.044 | 0.089 | 0.089 | 0.067 |
| refusal on unanswerable | 0.80 | 0.80 | 0.80 | 0.80 |
| answer_f1 | 0.441 | 0.432 | 0.544 | 0.547 |
| median generation latency | 0.8 s | 0.6 s | 3.0 s | 5.5 s |
| RAGAS faithfulness | 0.651 | 0.708 | 0.789 | 0.847 |
| RAGAS answer_relevancy | 0.478 | 0.511 | 0.572 | 0.638 |
| RAGAS context precision (with reference) | 0.781 | 0.804 | 0.768 | 0.786 |
| RAGAS context_recall | 0.893 | 0.922 | 0.893 | 0.922 |
| RAGAS factual_correctness (f1) | 0.313 | 0.376 | 0.468 | 0.461 |
| RAGAS answer_matches_reference | 0.778 | 0.800 | 0.733 | 0.844 |

- The 70B model is better than the 8B on both judges: manual correctness +0.20 (rerank off) / +0.34 (on), faithfulness +0.42 / +0.18;
  RAGAS faithfulness +0.14 in both settings and answer_relevancy +0.09 / +0.13. Correctness-at-most-2 questions drop from 10-11 to 6.
  False refusals do not go down (4 of 45 off, 3 of 45 on), and the 70B model still cannot make the one-step inference in
  val_packet_03-06 (direct air capture is technology-based CCS). It also adds wrong answers the 8B did not give (e.g. val_packet_01-07, 01-09).
- Rerank on vs off with the 70B model: manual correctness +0.14 (5 up, 2 down), manual faithfulness -0.02; RAGAS faithfulness +0.06,
  answer_relevancy +0.07, answer_matches_reference +0.11, factual_correctness -0.01. A small lean towards rerank on, and the two judges
  disagree on faithfulness, so the rerank effect on faithfulness is not established.
- The remaining val failures are mainly retrieval: val_packet_02-07 and 05-07 are refused with rerank on because the evidence is not
  in the top 5, and a larger generator cannot recover that.
- **val_packet_04-10 was relabelled.** It was marked unanswerable ("minimum distance between area-based and census-based
  instances"), but VM0047 section 4 says instances must be "clearly separated by a minimum 10-meter buffer", so it is answerable. The 70B
  manual judging used the corrected label (val v1.1: "10-meter buffer" scored 5/5); the 8B manual judging used the old one (1/1).
  `data/eval/testset_val.jsonl` and the saved metrics of all four runs still count it as unanswerable, so `refusal_rate_on_unanswerable`
  0.8 would be 1.0 (4 of 4) under the new label, and the manual rows are not comparable on that item. Without it (49 questions),
  8B -> 70B manual correctness is +0.12 (rerank off) and +0.27 (on), faithfulness +0.35 and +0.10.
- Noise: `llm_context_precision_with_reference` does not depend on the answer, yet it differs between the 8B and the 70B runs by -0.013 and -0.018
  (rerank off: 2 questions up, 2 down; on: 0 up, 2 down). So the RAGAS judge moves a mean by about 0.015 on identical inputs, in line with the 0.00-0.012 seen when
  the rerank-off runs were scored again. Gaps of 0.02-0.06 are only 1 to 4 times that.
- Judges: one Claude judge (separate sessions for the 8B and the 70B packets, not blind to the run) and one RAGAS judge
  (`databricks-qwen3-next-80b-a3b-instruct`); n = 45-50, one run per variant.

**Retrieval gaps: top 10, other rerankers, RRF fusion** (retrieval only, offline, no generation; dev + val, 110 answerable questions,
1024 index, hybrid, 30 candidates, `bge-reranker-base`; same evidence rule and `overlap_threshold` as `run_eval`; the check scripts and
the cached candidates are not in the repo). The rerank-off and rerank-only val numbers match the saved val runs exactly.

| order (110 questions) | evidence_recall@5 | evidence_recall@10 | MRR@5 | MRR@10 |
|---|---|---|---|---|
| rerank off (hybrid order) | 0.930 | 0.955 | 0.821 | 0.822 |
| rerank only | 0.933 | 0.974 | 0.847 | 0.853 |
| RRF of hybrid and reranker order, k=60 | 0.936 | 0.983 | 0.866 | 0.872 |
| same, k=10 | 0.942 | 0.979 | 0.867 | 0.872 |

- **Top 10 helps more than anything else tried.** Rerank only goes 0.933 -> 0.974 from top 5 to top 10 (val 0.911 -> 0.956, dev 0.949 -> 0.987);
  rerank off goes 0.930 -> 0.955. With top 10 most dropped evidence returns (dev 02-13, 03-13, 05-09; val 05-07), and top 15 adds dev 01-13.
  Not tested: whether the doubled context helps or hurts the answers (it needs a generation run, which has not been done).
- **Not fixed by any top-k:** val_packet_02-07 (gold chunk `VM0047-v1.1_Final.pdf::00179`, "Step 6: Derive Performance Benchmark", is not among the
  30 candidates in either order) and one of the three passages of dev_packet_04-13. Those are first-stage misses.
- **The reranker both rescues and drops evidence.** It finds evidence the hybrid order puts beyond rank 15 (dev 04-07, 04-13) and pushes down evidence
  hybrid had in the top 5 (val 05-07 rank 3 -> 7, val 03-09 rank 1 -> 8; dev 01-13, 02-13, 03-13, 05-09). Rerank only vs off on recall@5: 6 questions better,
  6 worse; MRR@5 +0.026 (paired bootstrap 95% interval [-0.038, +0.091]), so on these 110 questions the reranker's retrieval gain is not distinguishable from zero.
- **RRF fusion of the two orders is not worth adding.** Versus rerank only: recall@5 +0.003 (4 questions up, 3 down), MRR@5 +0.019 (95% interval [-0.016, +0.054]);
  k=60 and k=10 behave alike. It does not reach the first-stage misses.
- **`bge-reranker-large`, tried only on the 4 val questions that lacked evidence at top 5** (picked because the base model failed, which favours any
  alternative): mean evidence recall at top 5 / 10 / 15 / 30 is 0.38 / 0.38 / 0.50 / 0.75 against 0.00 / 0.50 / 0.63 / 0.75 for the base model. It recovers val 05-07 at
  top 5 but ranks val 01-09 lower, and it took 95-129 s per question against 30-37 s (about 3 times slower). Not tried: `bge-reranker-v2-m3` and any reranker on all questions.

Caveats:
- One judge (Claude, one pass), one run per variant, no repeat runs to estimate run-to-run noise (Ollama determinism at
  temperature 0 was not tested), and small n: a 0.02-0.05 gap is one or two questions. The judge packet shows each context cut at 1500 characters.
- \* The dev runs and the first val runs were labelled 1024 but used the 512 index. `RAG__CHUNKING__MAX_TOKENS=1024` changes only
  the saved settings, not `paths.table_name`, and table `chunks` is text-for-text identical to `chunks_heading_aware_512_64`. Their
  `config.json` still says 1024. For a real 1024 run set `RAG__PATHS__TABLE_NAME=chunks_heading_aware_1024_128` and
  `RAG__CHUNKING__OVERLAP=128` (only the `*_1024idx_*` runs did).
- Rows are not comparable across generators (local llama3.2:3b vs Llama 3.1 8B vs Llama 3.3 70B); compare on vs off within a row only.
- Not done: the held-out test set, RAGAS on the dev and the 512-index runs, any 1024 vs 512 index comparison with generation, a generation
  run with `top_k` 10, and updating `testset_val.jsonl` for the val_packet_04-10 relabel.
- Runs: `data/eval/runs/20261027-182323_gen_1024_hybrid_rr-off_testset_dev` / `20261027-203606_gen_1024_hybrid_rr-on_testset_dev`,
  `20261028-093028_gen_1024_hybrid_rr-off_testset_val` / `20261028-100055_gen_1024_hybrid_rr-on_testset_val`, and
  `20261029-074316_gen_1024idx_hybrid_rr-off_testset_val` / `20261029-074405_gen_1024idx_hybrid_rr-on_testset_val` (8B; retrieval:
  `20261028-140242_ret_1024idx_…` / `20261028-140306_ret_1024idx_…`), and
  `20261029-122512_gen_1024idx_hybrid_rr-off_testset_val` / `20261029-122850_gen_1024idx_hybrid_rr-on_testset_val` (70B, same chunks).

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

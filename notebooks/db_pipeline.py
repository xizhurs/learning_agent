# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # End-to-end pipeline: index -> retrieval -> generation -> RAGAS
# MAGIC One notebook, one environment, one pinned commit of the code. Run it top to bottom.
# MAGIC
# MAGIC | step | what it does |
# MAGIC |---|---|
# MAGIC | **1. Index** | Delta Sync AI Search index over the chunks table; the managed embeddings are made here. An existing index is reused and a sync is started; the sync is not awaited, so after changing the chunks table wait for it to finish (index page) before relying on the results. The AI Search endpoint must exist (this notebook never creates one). |
# MAGIC | **2. Retrieval** | A one-question check of every variant first, then a retrieval-only run on the val questions per variant: `off` = hybrid, `llm` = hybrid + chat-LLM reranker, `on` = hybrid + AI Search reranker (off by default: Free Edition does not run it). |
# MAGIC | **3. Generation** | Answers from the chunks each variant saved, written by `gen_endpoint`. |
# MAGIC | **4. RAGAS** | Scores every generation run with `judge_endpoint`. |
# MAGIC | **5. Compare** | Side-by-side metrics. Every run folder and `pipeline.json` (settings, code commit, run names) are saved to `<data_dir>/pipeline_runs/<time>/` as each step finishes, so an early stop or a failure keeps what was done. |
# MAGIC
# MAGIC **Before running:** push the branch (the code is fetched from GitHub); the chunks table (`db_01b_load_chunks_parquet.py`) and `<data_dir>/testset_val.jsonl` must exist.
# MAGIC No GPU is used: every model runs on a serving endpoint. `stop_after` ends the run early. Numbers are comparable only within one run of this notebook (same index, models and judge).
# MAGIC The pinned versions in `databricks-constraints.txt` are those of Free Edition serverless; if pip reports a conflict on another compute, remove `-c ...constraints.txt` from the install cell.

# COMMAND ----------

dbutils.widgets.text("data_dir", "/Volumes/workspace/default/documents/carbon_eval", "Volume folder with testset_val.jsonl (results are saved here)")
dbutils.widgets.text("source_table", "workspace.carbon_rag.chunks_1024", "chunks Delta table")
dbutils.widgets.text("index_name", "workspace.carbon_rag.chunks_1024_idx", "AI Search index")
dbutils.widgets.text("endpoint", "carbon", "AI Search endpoint")
dbutils.widgets.text("embed_endpoint", "databricks-gte-large-en", "embeddings endpoint of the index")
dbutils.widgets.text("variants", "off,llm", "retrieval variants: off, llm, and on (AI Search reranker: not available on Free Edition)")
dbutils.widgets.text("top_k", "5", "passages given to the answer model")
dbutils.widgets.text("candidates_k", "20", "hits the llm reranker ranks (>= top_k)")
dbutils.widgets.text("rerank_endpoint", "databricks-meta-llama-3-1-8b-instruct", "small chat endpoint for the llm reranker")
dbutils.widgets.text("gen_endpoint", "databricks-meta-llama-3-3-70b-instruct", "chat endpoint that writes the answers")
dbutils.widgets.text("judge_endpoint", "databricks-qwen3-next-80b-a3b-instruct", "RAGAS judge endpoint")
dbutils.widgets.text("workers", "4", "parallel judge calls (keep low: workspace rate limits)")
dbutils.widgets.dropdown("recreate_index", "no", ["no", "yes"], "delete and re-embed the index")
dbutils.widgets.dropdown("sync_index", "yes", ["yes", "no"], "start a sync of an existing index first (does not wait for it to finish)")
dbutils.widgets.dropdown("stop_after", "ragas", ["index", "retrieval", "generation", "ragas"], "last step to run")
dbutils.widgets.text("git_ref", "qasper-corpus", "branch or commit of the code")

# COMMAND ----------

# Stage the code at one commit and the test set on local disk.
import os, shutil, subprocess

REPO, clone = "/tmp/carbon_methodology", "/tmp/learning_agent_git"
data_dir = dbutils.widgets.get("data_dir")
shutil.rmtree(clone, ignore_errors=True)
git = lambda *a: subprocess.run(["git", "-C", clone, *a], check=True)
subprocess.run(["git", "init", "-q", clone], check=True)
git("fetch", "-q", "--depth", "1", "https://github.com/xizhurs/learning_agent", dbutils.widgets.get("git_ref"))
git("checkout", "-q", "FETCH_HEAD")
commit = subprocess.check_output(["git", "-C", clone, "rev-parse", "HEAD"], text=True).strip()
for sub in ("src", "notebooks"):
    shutil.rmtree(f"{REPO}/{sub}", ignore_errors=True)               # nothing left over from an earlier commit
    shutil.copytree(f"{clone}/{sub}", f"{REPO}/{sub}", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy(f"{clone}/config.yaml", f"{REPO}/config.yaml")
os.makedirs(f"{REPO}/data/eval/runs", exist_ok=True)
assert os.path.isfile(f"{data_dir}/testset_val.jsonl"), f"{data_dir}/testset_val.jsonl not found"
shutil.copy(f"{data_dir}/testset_val.jsonl", f"{REPO}/data/eval/testset_val.jsonl")
open(f"{REPO}/COMMIT", "w").write(commit)      # survives the Python restart below
print("code commit", commit)

# COMMAND ----------

# MAGIC %pip install -r /tmp/carbon_methodology/notebooks/databricks-requirements-pipeline.txt -c /tmp/carbon_methodology/notebooks/databricks-constraints.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json, os, shutil, sys, time

REPO = "/tmp/carbon_methodology"
sys.path.insert(0, f"{REPO}/src")
w = dbutils.widgets.get
data_dir, index_name, endpoint, embed_endpoint = w("data_dir"), w("index_name"), w("endpoint"), w("embed_endpoint")
rerank_endpoint, gen_endpoint, judge_endpoint = w("rerank_endpoint"), w("gen_endpoint"), w("judge_endpoint")
k, cand = int(w("top_k")), int(w("candidates_k"))
variants = [v.strip() for v in w("variants").split(",") if v.strip()]
assert variants and len(variants) == len(set(variants)) and set(variants) <= {"off", "on", "llm"}, (
    f"variants must be modes from off, on, llm, each at most once (top_k has its own widget; db_03's off:5 form does not apply here), got {variants}")
assert cand >= k, "candidates_k must be at least top_k"
runs_dir = f"{REPO}/data/eval/runs"
STEPS = ["index", "retrieval", "generation", "ragas"]
out = f"{data_dir}/pipeline_runs/{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(out)
record = {"code_commit": open(f"{REPO}/COMMIT").read(), "retrieval_runs": {}, "generation_runs": {},
          "settings": {key: w(key) for key in ("source_table", "index_name", "endpoint", "embed_endpoint", "variants", "top_k", "candidates_k",
                                               "rerank_endpoint", "gen_endpoint", "judge_endpoint", "git_ref")}}


def save(run=None):
    """Copy a finished (or re-scored) run folder to the Volume and rewrite pipeline.json."""
    if run:
        shutil.copytree(f"{runs_dir}/{run}", f"{out}/{run}", dirs_exist_ok=True)
    with open(f"{out}/pipeline.json", "w") as f:
        json.dump(record, f, indent=1)


save()
print("saving to", out)
stop = lambda step: STEPS.index(step) >= STEPS.index(w("stop_after")) and dbutils.notebook.exit(f"stopped after {step}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Index (embeddings)

# COMMAND ----------

from pyspark.sql import functions as F
from databricks.ai_search.client import AISearchClient

COLUMNS = ["id", "source", "page", "section", "kind", "text"]
table = w("source_table")
df = spark.table(table)
assert set(COLUMNS) <= set(df.columns), f"{table} needs columns {COLUMNS}, has {df.columns}"
props = {r["key"]: r["value"] for r in spark.sql(f"SHOW TBLPROPERTIES {table}").collect()}
assert str(props.get("delta.enableChangeDataFeed")).lower() == "true", f"run: ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
n, n_ids, max_bytes = df.agg(F.count("*"), F.countDistinct("id"), F.max(F.length(F.encode("text", "utf-8")))).first()
assert n == n_ids > 0 and max_bytes < 30000, f"need unique ids and chunks under the 32,764-byte index limit (rows {n}, ids {n_ids}, longest {max_bytes} bytes)"
print(f"{table}: {n} chunks from {df.select('source').distinct().count()} documents")

ai = AISearchClient()
assert ai.endpoint_exists(endpoint), (
    f"AI Search endpoint {endpoint!r} not found: set `endpoint` to an existing one (db_02_ai_search_index.py creates one if the workspace has none)")
if w("recreate_index") == "yes" and ai.index_exists(endpoint, index_name):
    ai.delete_index(endpoint_name=endpoint, index_name=index_name)
    while ai.index_exists(endpoint, index_name):                      # deletion is asynchronous
        time.sleep(5)
created = not ai.index_exists(endpoint, index_name)
if created:
    print("creating the index: every chunk is embedded by", embed_endpoint)
    ai.create_delta_sync_index_and_wait(
        endpoint_name=endpoint, index_name=index_name, primary_key="id", source_table_name=table, pipeline_type="TRIGGERED",
        embedding_source_column="text", embedding_model_endpoint_name=embed_endpoint, columns_to_sync=COLUMNS, verbose=True)
index = ai.get_index(endpoint_name=endpoint, index_name=index_name)
if w("sync_index") == "yes" and not created:
    index.sync()
    print("sync started (not awaited)")
index.wait_until_ready(verbose=True)
print("index ready:", index_name)
stop("index")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Retrieval (one retrieval-only run per variant)

# COMMAND ----------

from databricks.ai_search.reranker import DatabricksReranker
from databricks_openai import DatabricksOpenAI

from rag.ai_search import make_retriever
from rag.config import get_settings
from rag.eval import run_eval
from rag.generate import set_chat_client
from rag.llm_rerank import LLMReranker

client = DatabricksOpenAI(max_retries=6)


def retriever(mode):
    if mode == "on":
        return make_retriever(index, rerank=True, reranker=DatabricksReranker(columns_to_rerank=["text"]), num_results=k)
    if mode == "llm":
        return make_retriever(index, rerank=False, llm_reranker=LLMReranker(client, rerank_endpoint), num_results=k, candidates_k=cand)
    return make_retriever(index, rerank=False, num_results=k)


def use_settings(mode):
    """Settings mostly record what was used (the retriever and the clients do the work). The table name is the index, so tags and
    config.json differ from the earlier gen_1024idx runs."""
    os.environ.update({
        "RAG__TRACING__ENABLED": "false", "RAG__CHUNKING__MAX_TOKENS": "1024", "RAG__CHUNKING__OVERLAP": "128",
        "RAG__RETRIEVAL__MODE": "hybrid", "RAG__RETRIEVAL__TOP_K": str(k), "RAG__PATHS__TABLE_NAME": index_name,
        "RAG__PATHS__EVAL_DIR": f"{REPO}/data/eval", "RAG__GENERATION__MODEL": gen_endpoint, "RAG__EVAL__JUDGE_MAX_WORKERS": w("workers"),
        "RAG__RERANK__ENABLED": "false" if mode == "off" else "true",
        "RAG__RERANK__MODEL": rerank_endpoint if mode == "llm" else "databricks-ai-search-reranker"})
    get_settings.cache_clear()


def run(*argv):
    """One run_eval run on the val questions; returns the new run folder's name."""
    before = set(os.listdir(runs_dir))
    sys.argv = ["run_eval", "run", "--testset", "testset_val", *argv]
    run_eval.main()
    (name,) = set(os.listdir(runs_dir)) - before
    return name


# Preflight: 3 val questions through every variant before the long runs. A broken index, endpoint or reranker fails here, and at least
# one variant must find a gold quote (same overlap rule as run_eval), otherwise the index content or its embeddings are wrong.
from rag.eval.metrics import first_hit_rank

val = [json.loads(line) for line in open(f"{REPO}/data/eval/testset_val.jsonl", encoding="utf-8") if line.strip()]
checks = [q for q in val if q["answerable"]][:3]
found = {}
for mode in variants:
    use_settings(mode)
    r, thr = retriever(mode), get_settings().eval.overlap_threshold
    ranks = [first_hit_rank(q["reference_quote"], [h.text for h in r(q["question"], get_settings())], thr) for q in checks]
    found[mode] = sum(isinstance(x, int) for x in ranks)
    print(f"preflight {mode}: gold quote ranks {ranks}")
assert any(found.values()), "no variant found any gold quote in its top results: check the index content and the embeddings endpoint"

import importlib.metadata as md

ret = {}
for mode in variants:
    use_settings(mode)
    r = retriever(mode)
    run_eval.set_retriever(r)
    t0 = time.time()
    ret[mode] = run("--tag", f"ret_hybrid_rr-{mode}_top{k}", "--retrieval-only")
    backend = {  # provenance next to the run, as db_03 wrote it
        "backend": "databricks-ai-search", "index": index_name, "endpoint": endpoint, "embedding_model_endpoint": embed_endpoint,
        "query_type": "hybrid", "rerank": mode, "num_results": k, "reranker_columns": ["text"] if mode == "on" else None,
        "llm_rerank_endpoint": rerank_endpoint if mode == "llm" else None, "candidates_k": cand if mode == "llm" else None,
        "databricks_ai_search_version": md.version("databricks-ai-search"), "seconds": round(time.time() - t0),
        "retriever_stats": r.stats, "llm_reranker_stats": r.llm_reranker.stats if mode == "llm" else None, "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(f"{runs_dir}/{ret[mode]}/backend.json", "w") as f:
        json.dump(backend, f, indent=1)
    record["retrieval_runs"][mode] = ret[mode]
    save(ret[mode])
    print(f"{mode}: {ret[mode]} | reranker {r.stats['reranker_seconds']:.0f} s")
run_eval.set_retriever()
stop("retrieval")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generation (from the saved chunks, so every variant is answered from exactly what it retrieved)

# COMMAND ----------

set_chat_client(client)
gen = {}
for mode in variants:
    use_settings(mode)
    gen[mode] = run("--tag", f"gen_hybrid_rr-{mode}", "--reuse-retrieval", f"{runs_dir}/{ret[mode]}")
    record["generation_runs"][mode] = gen[mode]
    save(gen[mode])
    print(f"{mode}: {gen[mode]}")
stop("generation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. RAGAS (the judge and the embeddings are Databricks endpoints)

# COMMAND ----------

import argparse
from databricks_langchain import ChatDatabricks, DatabricksEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

run_eval.set_judge_models(
    LangchainLLMWrapper(ChatDatabricks(endpoint=judge_endpoint, temperature=0.0, max_tokens=4096), bypass_n=True),  # bypass_n: one sample per request
    LangchainEmbeddingsWrapper(DatabricksEmbeddings(endpoint=embed_endpoint)))
for mode in variants:
    print("scoring", mode, gen[mode])
    run_eval.cmd_score_ragas(argparse.Namespace(run=f"{runs_dir}/{gen[mode]}", metrics=None))
    save(gen[mode])                                                  # the scored copy replaces the unscored one

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Compare
# MAGIC A blank cell means the metric was not reported for that run (for RAGAS: the judge call failed, often a rate limit; `score_ragas_databricks.py` can score the missing items only).

# COMMAND ----------

summaries = {m: json.load(open(f"{runs_dir}/{gen[m]}/summary.json", encoding="utf-8")) for m in variants}
num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
print(f"{'metric':<44}" + "".join(f"{m:>10}" for m in variants))
for key in sorted({key for s in summaries.values() for key, v in s.items() if num(v)}):
    print(f"{key:<44}" + "".join(f"{summaries[m][key]:>10.3f}" if num(summaries[m].get(key)) else f"{'':>10}" for m in variants))

print("\nall runs and pipeline.json are in", out)

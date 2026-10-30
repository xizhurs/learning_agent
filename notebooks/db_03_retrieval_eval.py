# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Retrieval-only evaluation with AI Search (hybrid, with and without the reranker)
# MAGIC Runs `rag.eval.run_eval` in retrieval-only mode over the val (or dev) questions with the AI Search index as the retriever
# MAGIC (`run_eval.set_retriever`), once per variant, and saves run folders in the usual format (`config.json`, `items.jsonl`, `summary.json`)
# MAGIC plus a `backend.json` next to them (index, endpoint, embedding model, SDK version, reranker columns, reranker call count and time).
# MAGIC Those folders feed `generate_val_databricks.py` (`--reuse-retrieval`) like the local retrieval runs did.
# MAGIC
# MAGIC **Variants** (widget `variants`, comma separated `off|on|llm:<top_k>`): `off` = hybrid only, `on` = hybrid + `DatabricksReranker`,
# MAGIC `llm` = hybrid, then a chat LLM endpoint (`rerank_endpoint`) ranks the top `candidates_k` hits in one call per question (`rag.llm_rerank`).
# MAGIC A rerank-on or llm run stops with an error if the reranker did not run for a question, so a reranked folder is always a reranked one.
# MAGIC **Not comparable with the local runs** (different embeddings, BM25, reranker; the reranker's candidate pool is not documented and it reads only the first 2000 characters of
# MAGIC `text`): compare as a before/after table, same test set, same metrics. Nothing from the held-out test set is used.

# COMMAND ----------

dbutils.widgets.text("git_ref", "qasper-corpus", "branch or commit of the code to run (db_run_all.py pins one commit)")
dbutils.widgets.text("data_dir", "/Volumes/workspace/default/documents/carbon_eval", "Volume folder with the test sets (results go to results_retrieval/)")
dbutils.widgets.text("index_name", "workspace.carbon_rag.chunks_1024_idx", "AI Search index name")
dbutils.widgets.text("endpoint", "carbon", "AI Search endpoint")
dbutils.widgets.text("embed_endpoint", "databricks-gte-large-en", "embeddings endpoint of the index (recorded only)")
dbutils.widgets.dropdown("testset", "testset_val", ["testset_val", "testset_dev"], "question set")
dbutils.widgets.text("variants", "off:5,on:5,off:7,on:7", "variants, e.g. off:5,on:5,llm:5")
dbutils.widgets.text("rerank_endpoint", "databricks-meta-llama-3-3-70b-instruct", "chat endpoint for the llm variant (non-reasoning model)")
dbutils.widgets.text("candidates_k", "20", "hits the llm variant reranks")
dbutils.widgets.text("reranker_columns", "text", "columns_to_rerank (comma separated; only the first 2000 characters are read)")
dbutils.widgets.text("tag_prefix", "ret_aisearch_gte_hybrid", "run folder tag prefix")

# COMMAND ----------

import os, shutil, subprocess

REPO, clone_dir = "/tmp/carbon_methodology", "/tmp/learning_agent_git"
data_dir = dbutils.widgets.get("data_dir")
testset = dbutils.widgets.get("testset")
if os.path.isdir(clone_dir):
    shutil.rmtree(clone_dir)
subprocess.run(["git", "init", "-q", clone_dir], check=True)      # fetch a branch or an exact commit (widget git_ref)
subprocess.run(["git", "-C", clone_dir, "fetch", "-q", "--depth", "1", "https://github.com/xizhurs/learning_agent", dbutils.widgets.get("git_ref")], check=True)
subprocess.run(["git", "-C", clone_dir, "checkout", "-q", "FETCH_HEAD"], check=True)
for sub in ("src", "notebooks"):
    shutil.copytree(f"{clone_dir}/{sub}", f"{REPO}/{sub}", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy(f"{clone_dir}/config.yaml", f"{REPO}/config.yaml")
os.makedirs(f"{REPO}/data/eval/runs", exist_ok=True)
assert os.path.isfile(f"{data_dir}/{testset}.jsonl"), f"{data_dir}/{testset}.jsonl not found"
shutil.copy(f"{data_dir}/{testset}.jsonl", f"{REPO}/data/eval/{testset}.jsonl")
print("staged", REPO, "| test set", testset)

# COMMAND ----------

# MAGIC %pip install -r /tmp/carbon_methodology/notebooks/databricks-requirements-aisearch.txt -c /tmp/carbon_methodology/notebooks/databricks-constraints.txt databricks-openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import importlib.metadata as md, json, os, shutil, sys, time

REPO = "/tmp/carbon_methodology"
sys.path.insert(0, f"{REPO}/src")
data_dir = dbutils.widgets.get("data_dir")
testset = dbutils.widgets.get("testset")
index_name, endpoint, embed_endpoint = (dbutils.widgets.get(k) for k in ("index_name", "endpoint", "embed_endpoint"))
variants = []
for v in dbutils.widgets.get("variants").split(","):
    mode, k = v.strip().split(":")
    assert mode in ("off", "on", "llm"), f"variant {v!r}: use off:<k>, on:<k> or llm:<k>"
    variants.append((mode, int(k)))
rerank_endpoint, candidates_k = dbutils.widgets.get("rerank_endpoint"), int(dbutils.widgets.get("candidates_k"))
reranker_columns = [c.strip() for c in dbutils.widgets.get("reranker_columns").split(",") if c.strip()]
tag_prefix = dbutils.widgets.get("tag_prefix")
runs_dir = f"{REPO}/data/eval/runs"

from databricks.ai_search.client import AISearchClient
from databricks.ai_search.reranker import DatabricksReranker

from rag.ai_search import make_retriever
from rag.config import get_settings
from rag.eval import run_eval
from rag.llm_rerank import LLMReranker

index = AISearchClient().get_index(endpoint_name=endpoint, index_name=index_name)
index.wait_until_ready(verbose=True)
print("index ready:", index_name)
if any(m == "llm" for m, _ in variants):
    from databricks_openai import DatabricksOpenAI

    chat_client = DatabricksOpenAI(max_retries=6)

# COMMAND ----------

# One retrieval-only run per variant. Settings only record what was used (config.json); the retriever does the work.
results, by_variant = {}, {}
for mode, k in variants:
    rerank, llm = mode == "on", mode == "llm"
    os.environ.update({
        "RAG__TRACING__ENABLED": "false", "RAG__CHUNKING__MAX_TOKENS": "1024", "RAG__CHUNKING__OVERLAP": "128",
        "RAG__RETRIEVAL__MODE": "hybrid", "RAG__RETRIEVAL__TOP_K": str(k), "RAG__RERANK__ENABLED": "true" if rerank or llm else "false",
        "RAG__RERANK__MODEL": rerank_endpoint if llm else "databricks-ai-search-reranker", "RAG__PATHS__TABLE_NAME": index_name,
        "RAG__PATHS__EVAL_DIR": f"{REPO}/data/eval",
    })
    get_settings.cache_clear()
    llm_reranker = LLMReranker(chat_client, rerank_endpoint) if llm else None
    retriever = make_retriever(index, rerank=rerank, reranker=DatabricksReranker(columns_to_rerank=reranker_columns) if rerank else None, num_results=k,
                               llm_reranker=llm_reranker, candidates_k=candidates_k if llm else None)
    run_eval.set_retriever(retriever)
    tag = f"{tag_prefix}_rr-{mode}_top{k}"
    before = set(os.listdir(runs_dir))
    t0 = time.time()
    sys.argv = ["run_eval", "run", "--tag", tag, "--testset", testset, "--retrieval-only"]
    run_eval.main()
    new = sorted(set(os.listdir(runs_dir)) - before)
    assert len(new) == 1, f"expected one new run folder for {tag}, found {new}"
    name = new[0]
    backend = {
        "backend": "databricks-ai-search", "index": index_name, "endpoint": endpoint, "embedding_model_endpoint": embed_endpoint,
        "query_type": "hybrid", "rerank": rerank or llm, "rerank_kind": mode, "reranker_columns": reranker_columns if rerank else None, "num_results": k,
        "llm_rerank_endpoint": rerank_endpoint if llm else None, "candidates_k": candidates_k if llm else None,
        "llm_reranker_stats": llm_reranker.stats if llm else None,
        "databricks_ai_search_version": md.version("databricks-ai-search"), "seconds": round(time.time() - t0),
        "retriever_stats": retriever.stats, "testset": testset, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(backend, open(f"{runs_dir}/{name}/backend.json", "w"), indent=1)
    dest = f"{data_dir}/results_retrieval/{name}"
    shutil.copytree(f"{runs_dir}/{name}", dest, dirs_exist_ok=True)
    summary = json.load(open(f"{runs_dir}/{name}/summary.json", encoding="utf-8"))
    by_variant[f"{mode}:{k}"] = name
    results[name] = {k2: summary.get(k2) for k2 in ("hit@5", "mrr", "evidence_recall@3", "evidence_recall@5", "context_precision", "n_answerable")}
    print("->", dest, "|", results[name], "| reranker calls/seconds:", retriever.stats["calls"] if rerank or llm else "-", round(retriever.stats["reranker_seconds"], 1) if rerank or llm else "-")
run_eval.set_retriever()

# COMMAND ----------

# Summary table: copy the folders from <data_dir>/results_retrieval/ into local data/eval/runs/.
print(f"{'run folder':<75} {'hit@5':>7} {'MRR':>7} {'ev@3':>7} {'ev@5':>7} {'ctx_prec':>9}")
for name, s in results.items():
    f = lambda v: "-" if v is None else f"{v:.3f}"
    print(f"{name:<75} {f(s['hit@5']):>7} {f(s['mrr']):>7} {f(s['evidence_recall@3']):>7} {f(s['evidence_recall@5']):>7} {f(s['context_precision']):>9}")
dbutils.notebook.exit(json.dumps({"runs": list(results), "by_variant": by_variant, "summaries": results}))

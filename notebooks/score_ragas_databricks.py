# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # RAGAS scoring on Databricks (val generation runs, rerank off and on)
# MAGIC Scores the two generation runs made by `generate_val_databricks.py` with RAGAS (faithfulness, answer relevancy, context
# MAGIC precision / recall, factual correctness, answer_matches_reference). The judge LLM and the embeddings for answer relevancy are
# MAGIC Databricks serving endpoints, passed in with `rag.eval.run_eval.set_judge_models` (no Gemini, no Ollama).
# MAGIC
# MAGIC **Before running:** the two generation run folders must be in `<data_dir>/results/` (the generation notebook put them there), and
# MAGIC the repo branch `qasper-corpus` must be on GitHub with this notebook. The notebook clones the code, copies the run folders to local
# MAGIC disk, scores them there, and copies the scored folders to `<data_dir>/results_ragas/`.
# MAGIC
# MAGIC **Caveats:** the judge here (GPT OSS 120B) differs from the Gemini judge of the earlier QASPER runs, and the embeddings differ from
# MAGIC local `nomic-embed-text`, so scores are not comparable with those; rerank on vs off is still a fair comparison. A re-run keeps the scores
# MAGIC already saved and only completes missing ones, so an interrupted or rate-limited run can simply be started again.

# COMMAND ----------

dbutils.widgets.text("data_dir", "/Volumes/workspace/default/documents/carbon_eval", "Volume folder with the data")
dbutils.widgets.text("gen_off", "20261029-074316_gen_1024idx_hybrid_rr-off_testset_val", "generation run folder (rerank off)")
dbutils.widgets.text("gen_on", "20261029-074405_gen_1024idx_hybrid_rr-on_testset_val", "generation run folder (rerank on)")
dbutils.widgets.text("judge_endpoint", "databricks-gpt-oss-120b", "judge chat endpoint")
dbutils.widgets.text("embed_endpoint", "databricks-gte-large-en", "embeddings endpoint (confirm the name in the list below)")
dbutils.widgets.text("workers", "8", "parallel judge calls")
dbutils.widgets.text("metrics", "", "optional: comma-separated subset of metrics (empty = all)")

# COMMAND ----------

# Stage the code (cloned from GitHub; the workspace Git folder's file storage is corrupted) and the run folders (Volume) on local disk.
import os, shutil, subprocess
data_dir = dbutils.widgets.get("data_dir")
REPO = "/tmp/carbon_methodology"
gen_runs = [dbutils.widgets.get("gen_off"), dbutils.widgets.get("gen_on")]

clone_dir = "/tmp/learning_agent_git"
if os.path.isdir(clone_dir):
    shutil.rmtree(clone_dir)                     # always take the latest pushed commit
subprocess.run(["git", "clone", "--depth", "1", "--branch", "qasper-corpus",
                "https://github.com/xizhurs/learning_agent", clone_dir], check=True)

needed = [f"{clone_dir}/src/rag/config.py", f"{clone_dir}/config.yaml", f"{clone_dir}/notebooks/databricks-requirements-ragas.txt"]
for r in gen_runs:
    needed += [f"{data_dir}/results/{r}/{f}" for f in ("items.jsonl", "config.json", "summary.json")]
missing = [p for p in needed if not os.path.isfile(p)]
assert not missing, "not found (is the latest commit pushed? are the generation runs in <data_dir>/results/?):\n  " + "\n  ".join(missing)

shutil.copytree(f"{clone_dir}/src", f"{REPO}/src", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(f"{clone_dir}/notebooks", f"{REPO}/notebooks", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy(f"{clone_dir}/config.yaml", f"{REPO}/config.yaml")
os.makedirs(f"{REPO}/data/eval/runs", exist_ok=True)
for r in gen_runs:
    shutil.copytree(f"{data_dir}/results/{r}", f"{REPO}/data/eval/runs/{r}", dirs_exist_ok=True)
print("staged in", REPO, "| runs:", gen_runs)

# COMMAND ----------

# MAGIC %md
# MAGIC Installs RAGAS 0.4.3, `databricks-langchain` and `openai` (pinned, notebook-scoped) under the constraint list of what serverless already has, so no preinstalled package changes. About 150 packages: expect a few minutes.

# COMMAND ----------

# MAGIC %pip install -r /tmp/carbon_methodology/notebooks/databricks-requirements-ragas.txt -c /tmp/carbon_methodology/notebooks/databricks-constraints.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Versions now present, and whether pip reports any broken requirement (read-only check).
import importlib.metadata as md, subprocess, sys
for pkg in ["ragas", "langchain-core", "databricks-langchain", "openai", "pydantic", "numpy", "pandas", "pyarrow", "databricks-sdk"]:
    try:
        print(f"{pkg:22s} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg:22s} MISSING")
print(subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True).stdout[-1500:] or "pip check: no broken requirements")

# COMMAND ----------

# Endpoint checks: are the judge and embeddings endpoints visible, and does the judge return usable text?
# GPT OSS is a reasoning model: its reply may be a list of blocks (reasoning + text). LangChain's ChatGeneration.text keeps only the
# text blocks, which is what RAGAS reads, so this cell prints both the raw content and that extracted text.
import os, shutil, sys, time          # (variables and imports from before the Python restart are gone)
REPO = "/tmp/carbon_methodology"
sys.path.insert(0, f"{REPO}/src")
data_dir = dbutils.widgets.get("data_dir")
runs_dir = f"{REPO}/data/eval/runs"
gen_runs = [dbutils.widgets.get("gen_off"), dbutils.widgets.get("gen_on")]
judge_endpoint = dbutils.widgets.get("judge_endpoint")
embed_endpoint = dbutils.widgets.get("embed_endpoint")

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
eps = {e.name: getattr(e, "task", "") for e in w.serving_endpoints.list()}
print(f"{len(eps)} serving endpoints visible; chat/embeddings ones:")
for n, t in sorted(eps.items()):
    if any(k in n for k in ("gpt-oss", "llama", "gte", "bge", "embed", "qwen", "gemma")):
        print(f"  {n}  [{t}]")
for n in (judge_endpoint, embed_endpoint):
    if n not in eps:
        print(f"NOTE: {n!r} is not in the list; trying it anyway (fix the widget if the next step fails)")

from databricks_langchain import ChatDatabricks, DatabricksEmbeddings
from langchain_core.messages import HumanMessage

try:
    chat = ChatDatabricks(endpoint=judge_endpoint, temperature=0.0, max_tokens=4096, extra_params={"reasoning_effort": "low"})
    probe = chat.invoke("Reply with JSON {\"ok\": true} and nothing else.")
except Exception as e:                                           # the endpoint may not accept reasoning_effort
    print("with reasoning_effort failed:", type(e).__name__, str(e)[:200], "-> retrying without it")
    chat = ChatDatabricks(endpoint=judge_endpoint, temperature=0.0, max_tokens=4096)
    probe = chat.invoke("Reply with JSON {\"ok\": true} and nothing else.")
print("raw content type:", type(probe.content).__name__, "|", str(probe.content)[:300])
t0 = time.time()
gen = chat.generate([[HumanMessage(content="Reply with JSON {\"ok\": true} and nothing else.")]]).generations[0][0]
print(f"text RAGAS will read: {gen.text!r} ({time.time() - t0:.1f} s)")

emb = DatabricksEmbeddings(endpoint=embed_endpoint)
print("embedding length:", len(emb.embed_query("test sentence")))

# COMMAND ----------

# Score both runs. Same settings as the generation runs; the judge and the embeddings are injected, so no key or Ollama is needed.
import argparse
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from rag.config import get_settings
from rag.eval import run_eval

os.environ.update({"RAG__TRACING__ENABLED": "false", "RAG__EVAL__JUDGE_MAX_WORKERS": dbutils.widgets.get("workers")})
get_settings.cache_clear()
# bypass_n: RAGAS asks for several samples (n>1) for some metrics; separate requests are safer across endpoints.
run_eval.set_judge_models(LangchainLLMWrapper(chat, bypass_n=True), LangchainEmbeddingsWrapper(emb))
metrics = dbutils.widgets.get("metrics").strip() or None

for name in gen_runs:
    print("\n==", name, "==")
    run_eval.cmd_score_ragas(argparse.Namespace(run=f"{runs_dir}/{name}", metrics=metrics))
    dest = f"{data_dir}/results_ragas/{name}"
    shutil.copytree(f"{runs_dir}/{name}", dest, dirs_exist_ok=True)
    print("copied ->", dest)

# COMMAND ----------

# RAGAS means per run (None = no score yet: run the previous cell again to complete them).
import json
for name in gen_runs:
    s = json.load(open(f"{runs_dir}/{name}/summary.json", encoding="utf-8"))
    print(name)
    for k, v in s.items():
        if k.startswith("ragas") or k in ("faithfulness", "answer_relevancy", "answer_matches_reference") or "context_" in k and "ragas" in k:
            print(f"  {k}: {v}")
    rows = [json.loads(l) for l in open(f"{runs_dir}/{name}/items.jsonl", encoding="utf-8") if l.strip()]
    ans = [r for r in rows if r.get("answerable") and r.get("answer") is not None]
    miss = [r["id"] for r in ans if not r.get("ragas") or any(v is None for v in r["ragas"].values())]
    print(f"  items with a missing score: {len(miss)} of {len(ans)}", miss[:10])
print("\nDownload items.jsonl and summary.json of both folders from <data_dir>/results_ragas/ into local data/eval/runs/ (overwrite).")

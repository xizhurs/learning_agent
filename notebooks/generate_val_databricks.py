# Databricks notebook source
# MAGIC %md
# MAGIC # Val generation on Databricks (rerank off, then rerank on)
# MAGIC Generation only. The answers are produced from the chunks saved by the two retrieval-only runs (`--reuse-retrieval`), so no
# MAGIC search, reranker, embeddings or Ollama are needed. The LLM is a Databricks Foundation Model endpoint, called through its
# MAGIC OpenAI-compatible client (`rag.generate.set_chat_client`). Everything else (prompt, options that the API supports, scoring) is the
# MAGIC repo code.
# MAGIC
# MAGIC **Before running:** upload `carbon_methodology_databricks.zip` to the Volume folder in the `volume_dir` widget.
# MAGIC The notebook unzips it to `/tmp/carbon_methodology` (local disk: Volumes do not reliably support the append writes of
# MAGIC `items.jsonl`), runs there, and copies each finished run folder to `<volume_dir>/results/`.
# MAGIC
# MAGIC **Caveat:** the model is the endpoint's (e.g. Llama 3.1 8B), not the local `llama3.2:3b`, so these answers are not comparable with
# MAGIC the earlier local runs. Rerank on vs off is still a fair comparison: same model, same saved chunks.

# COMMAND ----------
dbutils.widgets.text("volume_dir", "/Volumes/workspace/default/documents", "Volume folder with the zip (results go to results/ inside it)")
dbutils.widgets.text("zip_name", "carbon_methodology_databricks.zip", "zip file name")
dbutils.widgets.text("endpoint", "system.ai.meta-llama-3-1-8b-instruct", "serving endpoint (chat)")
dbutils.widgets.text("ret_off", "20261028-140242_ret_1024idx_hybrid_rr-off_testset_val", "retrieval run (rerank off)")
dbutils.widgets.text("ret_on", "20261028-140306_ret_1024idx_hybrid_rr-on_testset_val", "retrieval run (rerank on)")
dbutils.widgets.text("resume_off", "", "resume: generation run folder name (rerank off), optional")
dbutils.widgets.text("resume_on", "", "resume: generation run folder name (rerank on), optional")

# COMMAND ----------
# Unzip to local disk. Existing run folders under /tmp/carbon_methodology are kept, so a partial run can be resumed.
import os, zipfile
volume_dir = dbutils.widgets.get("volume_dir")
zip_path = f"{volume_dir}/{dbutils.widgets.get('zip_name')}"
assert os.path.isfile(zip_path), f"{zip_path} not found: upload the zip to the Volume folder first"
with zipfile.ZipFile(zip_path) as z:
    z.extractall("/tmp")
print("unzipped to /tmp/carbon_methodology:", sorted(os.listdir("/tmp/carbon_methodology")))

# COMMAND ----------
# MAGIC %md
# MAGIC Installs `openai` (pinned, notebook-scoped) under the constraint list of what serverless already has, so no preinstalled package changes.

# COMMAND ----------
# MAGIC %pip install -r /tmp/carbon_methodology/notebooks/databricks-requirements.txt -c /tmp/carbon_methodology/notebooks/databricks-constraints.txt

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# Versions now present, and whether pip reports any broken requirement (read-only check).
import importlib.metadata as md, subprocess, sys
for pkg in ["openai", "pydantic", "pydantic-core", "pyyaml", "httpx", "anyio", "databricks-sdk"]:
    try:
        print(f"{pkg:16s} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg:16s} MISSING")
print(subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True).stdout[-1500:] or "pip check: no broken requirements")

# COMMAND ----------
# Endpoint check: list endpoints, build the OpenAI-compatible client, and answer one real question from the saved retrieval.
import json, os, sys, time
from pathlib import Path

REPO = "/tmp/carbon_methodology"
sys.path.insert(0, f"{REPO}/src")
runs_dir = f"{REPO}/data/eval/runs"
volume_dir = dbutils.widgets.get("volume_dir")      # variables from before the Python restart are gone
endpoint = dbutils.widgets.get("endpoint")

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
names = sorted(e.name for e in w.serving_endpoints.list())
print(f"{len(names)} serving endpoints visible:", names[:15])
if endpoint not in names:
    print(f"NOTE: {endpoint!r} is not in that list; trying it anyway (system endpoints may be listed under another name)")

client = w.serving_endpoints.get_open_ai_client().with_options(max_retries=6)

from rag.config import Generation
from rag.generate import build_messages, generate, set_chat_client
from rag.eval.run_eval import load_saved_retrieval

set_chat_client(client)
saved = load_saved_retrieval(Path(f"{runs_dir}/{dbutils.widgets.get('ret_off')}"))
qid, hits = next(iter(saved.items()))
question = next(json.loads(l)["question"] for l in open(f"{REPO}/data/eval/testset_val.jsonl", encoding="utf-8") if json.loads(l)["id"] == qid)
t0 = time.time()
answer = generate(build_messages(question, hits), Generation(model=endpoint))
print(f"Q: {question}\nA: {answer}\n\n{time.time() - t0:.1f} s for one question; about {50 * (time.time() - t0) / 60:.0f} min per 50-question variant")

# COMMAND ----------
# The two generation runs: same settings as the local val runs, only rerank.enabled (which must match the retrieval run) differs.
import shutil
from rag.config import get_settings
from rag.eval import run_eval

def run_variant(label: str, rerank: str, ret_widget: str, resume_widget: str) -> None:
    os.environ.update({
        "RAG__CHUNKING__MAX_TOKENS": "1024", "RAG__CHUNKING__OVERLAP": "128", "RAG__RETRIEVAL__MODE": "hybrid",
        "RAG__PATHS__TABLE_NAME": "chunks_heading_aware_1024_128", "RAG__RERANK__ENABLED": rerank,
        "RAG__TRACING__ENABLED": "false", "RAG__GENERATION__MODEL": endpoint})
    get_settings.cache_clear()          # settings are cached per process; the rerank flag differs between the two runs
    argv = ["run_eval", "run", "--testset", "testset_val", "--tag", f"gen_1024idx_hybrid_rr-{label}",
            "--reuse-retrieval", f"{runs_dir}/{dbutils.widgets.get(ret_widget)}"]
    if dbutils.widgets.get(resume_widget):
        argv += ["--resume", f"{runs_dir}/{dbutils.widgets.get(resume_widget)}"]
    print("\n==", label, "==")
    sys.argv = argv
    run_eval.main()
    new = sorted(d for d in os.listdir(runs_dir) if d.endswith(f"gen_1024idx_hybrid_rr-{label}_testset_val"))[-1]
    dest = f"{volume_dir}/results/{new}"
    shutil.copytree(f"{runs_dir}/{new}", dest, dirs_exist_ok=True)
    print("copied", new, "->", dest)

run_variant("off", "false", "ret_off", "resume_off")

# COMMAND ----------
run_variant("on", "true", "ret_on", "resume_on")

# COMMAND ----------
# Result folders (also copied to <volume_dir>/results/). Download items.jsonl, summary.json and config.json of both into local
# data/eval/runs/ (same folder names), then export the judge packets locally.
for d in sorted(os.listdir(runs_dir)):
    if "gen_1024idx" in d:
        n = sum(1 for l in open(f"{runs_dir}/{d}/items.jsonl", encoding="utf-8") if l.strip())
        print(f"{d}: {n} answers, summary.json: {os.path.exists(f'{runs_dir}/{d}/summary.json')}")

# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # AI Search index over the chunks table (Delta Sync, managed embeddings)
# MAGIC Creates (or reuses) a **Delta Sync index** on `source_table` and checks it with a few real val questions, hybrid search with and without the reranker.
# MAGIC Free Edition allows **one** AI Search endpoint with one search unit and no Direct Vector Access, so the existing endpoint (`carbon`) is reused and
# MAGIC a missing one is created only if none exists. Embeddings are computed by the `embed_endpoint` model, so the index text is sent to that Databricks endpoint (public documents only).
# MAGIC
# MAGIC **Needs:** the chunks Delta table from `db_01_ingest_chunks.py` (columns `id, source, page, section, kind, text`, change data feed on) and `<data_dir>/testset_val.jsonl` for the smoke test.
# MAGIC **Safe to re-run:** an existing index is reused (`recreate = no`); `recreate = yes` deletes and rebuilds it (use after the chunks table changed shape).
# MAGIC The last cell returns a JSON status (used by `db_run_all.py`).

# COMMAND ----------

dbutils.widgets.text("data_dir", "/Volumes/workspace/default/documents/carbon_eval", "Volume folder with testset_val.jsonl")
dbutils.widgets.text("source_table", "workspace.carbon_rag.chunks_1024", "chunks Delta table")
dbutils.widgets.text("index_name", "workspace.carbon_rag.chunks_1024_idx", "AI Search index name")
dbutils.widgets.text("endpoint", "carbon", "AI Search endpoint (the one Free Edition endpoint)")
dbutils.widgets.text("embed_endpoint", "databricks-gte-large-en", "embeddings endpoint for the index")
dbutils.widgets.dropdown("recreate", "no", ["no", "yes"], "delete and rebuild the index if it exists")
dbutils.widgets.dropdown("sync_now", "yes", ["yes", "no"], "trigger a sync when the index is reused")
dbutils.widgets.text("smoke_questions", "5", "val questions for the smoke test")

# COMMAND ----------

# Stage the code from GitHub (the workspace Git folder storage is corrupted) and the test set from the Volume.
import os, shutil, subprocess

REPO, clone_dir = "/tmp/carbon_methodology", "/tmp/learning_agent_git"
data_dir = dbutils.widgets.get("data_dir")
if os.path.isdir(clone_dir):
    shutil.rmtree(clone_dir)
subprocess.run(["git", "clone", "--depth", "1", "--branch", "qasper-corpus", "https://github.com/xizhurs/learning_agent", clone_dir], check=True)
for sub in ("src", "notebooks"):
    shutil.copytree(f"{clone_dir}/{sub}", f"{REPO}/{sub}", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy(f"{clone_dir}/config.yaml", f"{REPO}/config.yaml")
os.makedirs(f"{REPO}/data/eval", exist_ok=True)
assert os.path.isfile(f"{data_dir}/testset_val.jsonl"), f"{data_dir}/testset_val.jsonl not found"
shutil.copy(f"{data_dir}/testset_val.jsonl", f"{REPO}/data/eval/testset_val.jsonl")
print("staged", REPO)

# COMMAND ----------

# MAGIC %pip install -r /tmp/carbon_methodology/notebooks/databricks-requirements-aisearch.txt -c /tmp/carbon_methodology/notebooks/databricks-constraints.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# 1. The source table: exists, change data feed on, unique ids, text small enough for the index (32,764 bytes), nothing from a non-public source.
import json, sys, time

REPO = "/tmp/carbon_methodology"
sys.path.insert(0, f"{REPO}/src")
source_table = dbutils.widgets.get("source_table")
index_name = dbutils.widgets.get("index_name")
endpoint = dbutils.widgets.get("endpoint")
embed_endpoint = dbutils.widgets.get("embed_endpoint")
COLUMNS = ["id", "source", "page", "section", "kind", "text"]
status = {"source_table": source_table, "index_name": index_name, "endpoint": endpoint, "embed_endpoint": embed_endpoint}

df = spark.table(source_table)
missing = [c for c in COLUMNS if c not in df.columns]
assert not missing, f"{source_table} lacks columns {missing}; it has {df.columns}"
props = {r["key"]: r["value"] for r in spark.sql(f"SHOW TBLPROPERTIES {source_table}").collect()}
cdf = str(props.get("delta.enableChangeDataFeed", "false")).lower() == "true"
n_rows = df.count()
n_ids = df.select("id").distinct().count()
from pyspark.sql import functions as F

max_bytes = df.select(F.max(F.length(F.encode("text", "utf-8")))).first()[0]
sources = sorted(r["source"] for r in df.select("source").distinct().collect())
print(f"{source_table}: {n_rows} rows, {n_ids} distinct ids, change data feed {cdf}, longest text {max_bytes} bytes")
print("sources:", sources)
assert cdf, f"enable it with: ALTER TABLE {source_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
assert n_rows == n_ids and n_rows > 0, "ids must be unique and the table non-empty"
assert max_bytes < 30000, f"a chunk has {max_bytes} bytes; the index limit for the embedding source column is 32,764"
status.update(rows=n_rows, max_text_bytes=max_bytes, sources=sources)

# COMMAND ----------

# 2. The endpoint: reuse it; create one only if the workspace has none (Free Edition allows exactly one).
from databricks.ai_search.client import AISearchClient

ai = AISearchClient()
if ai.endpoint_exists(endpoint):
    info = ai.get_endpoint(endpoint)
    print("endpoint", endpoint, "->", {k: info.get(k) for k in ("endpoint_type", "endpoint_status", "num_indexes") if isinstance(info, dict)})
    status["endpoint_type"] = info.get("endpoint_type") if isinstance(info, dict) else None
else:
    existing = ai.list_endpoints()
    names = [e.get("name") for e in (existing.get("endpoints", []) if isinstance(existing, dict) else existing)]
    assert not names, f"endpoint {endpoint!r} not found but others exist {names}: Free Edition allows one, set the widget to it"
    print("creating endpoint", endpoint, "(STANDARD)")
    ai.create_endpoint_and_wait(name=endpoint, endpoint_type="STANDARD", verbose=True)
    status["endpoint_type"] = "STANDARD"

# COMMAND ----------

# 3. The index.
recreate = dbutils.widgets.get("recreate") == "yes"
exists = ai.index_exists(endpoint, index_name)
if exists and recreate:
    print("deleting", index_name)
    ai.delete_index(endpoint_name=endpoint, index_name=index_name)
    for _ in range(60):                                   # deletion is asynchronous
        if not ai.index_exists(endpoint, index_name):
            break
        time.sleep(5)
    exists = False

t0 = time.time()
if not exists:
    print("creating index (this embeds every chunk with", embed_endpoint, ")")
    ai.create_delta_sync_index_and_wait(
        endpoint_name=endpoint,
        index_name=index_name,
        primary_key="id",
        source_table_name=source_table,
        pipeline_type="TRIGGERED",
        embedding_source_column="text",
        embedding_model_endpoint_name=embed_endpoint,
        columns_to_sync=COLUMNS,
        verbose=True,
    )
    status["created"] = True
else:
    print("reusing", index_name)
    status["created"] = False

index = ai.get_index(endpoint_name=endpoint, index_name=index_name)
if exists and dbutils.widgets.get("sync_now") == "yes":
    print("syncing")
    index.sync()
index.wait_until_ready(verbose=True, wait_for_updates=True)
print(f"ready after {time.time() - t0:.0f}s")
desc = index.describe()
print(json.dumps({k: v for k, v in desc.items() if k != "delta_sync_index_spec"}, indent=1, default=str)[:1500])
status["describe"] = {k: desc.get(k) for k in ("name", "index_type", "primary_key", "status") if k in desc}

# COMMAND ----------

# 4. Smoke test with real val questions: hybrid, then hybrid + reranker. The gold quote must be found (same overlap rule as run_eval).
import os

from databricks.ai_search.reranker import DatabricksReranker

from rag.ai_search import RerankerUnavailable, make_retriever
from rag.config import get_settings
from rag.eval.metrics import first_hit_rank

settings = get_settings()
val = [json.loads(l) for l in open(f"{REPO}/data/eval/testset_val.jsonl", encoding="utf-8") if l.strip()]
questions = [q for q in val if q["answerable"]][: int(dbutils.widgets.get("smoke_questions"))]
plain = make_retriever(index, rerank=False)
reranked = make_retriever(index, rerank=True, reranker=DatabricksReranker(columns_to_rerank=["text"]))
ranks, rerank_ok = [], True
for q in questions:
    row = {"id": q["id"]}
    for name, retriever in (("hybrid", plain), ("hybrid+rerank", reranked)):
        t0 = time.time()
        try:
            hits = retriever(q["question"], settings)
            row[name] = first_hit_rank(q["reference_quote"], [h.text for h in hits], settings.eval.overlap_threshold)
            row[name + " s"] = round(time.time() - t0, 2)
        except RerankerUnavailable as e:
            row[name] = f"RERANKER PROBLEM: {str(e)[:200]}"
            rerank_ok = False
    ranks.append(row)
    print(row)
print("reranker stats:", reranked.stats)
status.update(smoke=ranks, reranker_ok=rerank_ok, reranker_stats=reranked.stats)
found = sum(isinstance(r.get("hybrid"), int) for r in ranks)
print(f"gold quote in the top 5 for hybrid on {found} of {len(ranks)} questions; reranker problems: {not rerank_ok}")
assert found > 0, "no question found its evidence: check the index content and the embeddings endpoint"

# COMMAND ----------

dbutils.notebook.exit(json.dumps(status, default=str))

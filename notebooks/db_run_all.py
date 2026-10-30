# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Run the whole evaluation chain: embeddings -> retrieval -> generation -> RAGAS
# MAGIC Runs the four existing notebooks one after the other with `dbutils.notebook.run`, handing each one the run names the previous one returned:
# MAGIC
# MAGIC 1. `db_02_ai_search_index.py`: the AI Search index over the chunks table. The index **is** the embedding step (managed embeddings from `embed_endpoint`);
# MAGIC    an existing index is reused and synced, `recreate = yes` re-embeds everything.
# MAGIC 2. `db_03_retrieval_eval.py`: retrieval-only runs on the val questions, one per variant (`off` hybrid, `on` AI Search reranker, `llm` chat-LLM reranker).
# MAGIC 3. `generate_val_databricks.py`: answers from the saved chunks of each variant, with the `endpoint` chat model.
# MAGIC 4. `score_ragas_databricks.py`: RAGAS on every generation run, judged by `judge_endpoint`.
# MAGIC
# MAGIC **Before running:** push the branch (every step fetches the branch's latest commit from GitHub, resolved once at the start, so the whole chain runs one commit of the pushed code); the chunks Delta table and
# MAGIC `<data_dir>/testset_val.jsonl` must exist (`db_01b_load_chunks_parquet.py`). Pick **one** run per mode in `variants`; `off` and `on` are required (the
# MAGIC generation notebook compares them), `llm` is optional. The whole chain takes about as long as the four notebooks together; `stop_after` ends it early.
# MAGIC A record of what ran (the code commit every step ran, run names, timings, summaries) is written to `<data_dir>/pipeline_runs/`.
# MAGIC Retrieval, generation and RAGAS numbers are only comparable within one chain run (same index, same chat model, same judge).

# COMMAND ----------

dbutils.widgets.text("data_dir", "/Volumes/workspace/default/documents/carbon_eval", "Volume folder with the data")
dbutils.widgets.text("source_table", "workspace.carbon_rag.chunks_1024", "chunks Delta table")
dbutils.widgets.text("index_name", "workspace.carbon_rag.chunks_1024_idx", "AI Search index name")
dbutils.widgets.text("endpoint", "carbon", "AI Search endpoint")
dbutils.widgets.text("embed_endpoint", "databricks-gte-large-en", "embeddings endpoint of the index")
dbutils.widgets.dropdown("recreate_index", "no", ["no", "yes"], "delete and re-embed the index")
dbutils.widgets.text("variants", "off:5,on:5,llm:5", "retrieval variants, one per mode (off, on required; llm optional)")
dbutils.widgets.text("rerank_endpoint", "databricks-meta-llama-3-3-70b-instruct", "chat endpoint for the llm reranker")
dbutils.widgets.text("candidates_k", "20", "hits the llm reranker ranks")
dbutils.widgets.text("gen_endpoint", "databricks-meta-llama-3-3-70b-instruct", "chat endpoint that writes the answers")
dbutils.widgets.text("judge_endpoint", "databricks-qwen3-next-80b-a3b-instruct", "RAGAS judge endpoint")
dbutils.widgets.dropdown("stop_after", "ragas", ["index", "retrieval", "generation", "ragas"], "last step to run")
dbutils.widgets.text("git_ref", "qasper-corpus", "branch of the code to run; resolved once to a commit that every step uses")
dbutils.widgets.text("notebooks_dir", "", "workspace folder of the notebooks (empty = the folder of this notebook)")
dbutils.widgets.text("timeout_s", "28800", "per-notebook timeout in seconds")

# COMMAND ----------

import json, os, shutil, subprocess, time

data_dir = dbutils.widgets.get("data_dir")
index_name, endpoint, embed_endpoint = (dbutils.widgets.get(k) for k in ("index_name", "endpoint", "embed_endpoint"))
STEPS = ["index", "retrieval", "generation", "ragas"]
stop_after = dbutils.widgets.get("stop_after")
timeout_s = int(dbutils.widgets.get("timeout_s"))
nb_dir = dbutils.widgets.get("notebooks_dir").strip() or os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)

variants = {}
for v in dbutils.widgets.get("variants").split(","):
    mode, k = v.strip().split(":")
    assert mode in ("off", "on", "llm"), f"variant {v!r}: use off:<k>, on:<k> or llm:<k>"
    assert mode not in variants, f"mode {mode!r} appears twice: the chain compares one run per mode"
    variants[mode] = int(k)
assert {"off", "on"} <= set(variants), "variants must include off and on"

# Resolve the branch to one commit now and give every step that commit (widget git_ref), so a push during the run cannot split
# the chain across two versions of the code. The workspace copies of the notebooks themselves are whatever is imported there.
ref = dbutils.widgets.get("git_ref").strip()
ls = subprocess.run(["git", "ls-remote", "https://github.com/xizhurs/learning_agent", f"refs/heads/{ref}"], capture_output=True, text=True)
commit = ls.stdout.split()[0] if ls.stdout.split() else None
assert commit, f"could not resolve {ref!r} on GitHub: {ls.stderr.strip() or 'no such branch'}"
record = {"code_commit": commit, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "widgets": {
    k: dbutils.widgets.get(k) for k in ("data_dir", "source_table", "index_name", "endpoint", "embed_endpoint", "recreate_index", "variants",
                                        "rerank_endpoint", "candidates_k", "gen_endpoint", "judge_endpoint", "stop_after", "git_ref")}, "steps": {}}
print("notebooks:", nb_dir, "| code commit:", record["code_commit"], "| variants:", variants)


def run_step(step: str, notebook: str, args: dict) -> dict:
    """Run one notebook, return the JSON it exits with; a failing notebook raises and stops the chain."""
    t0 = time.time()
    print(f"\n=== {step}: {notebook} ===")
    out = json.loads(dbutils.notebook.run(f"{nb_dir}/{notebook}", timeout_s, {k: str(v) for k, v in (args | {"git_ref": commit}).items()}))
    record["steps"][step] = {"notebook": notebook, "seconds": round(time.time() - t0), "result": out}
    return out


def reached_stop(step: str) -> bool:
    return STEPS.index(step) >= STEPS.index(stop_after)

# COMMAND ----------

# 1. Embeddings: the AI Search index.
idx = run_step("index", "db_02_ai_search_index", {
    "data_dir": data_dir, "source_table": dbutils.widgets.get("source_table"), "index_name": index_name, "endpoint": endpoint,
    "embed_endpoint": embed_endpoint, "recreate": dbutils.widgets.get("recreate_index"), "sync_now": "yes"})
assert idx.get("reranker_ok", True) or "on" not in variants, "the AI Search reranker is unavailable in this workspace: drop the `on` variant"

# COMMAND ----------

# 2. Retrieval: one retrieval-only run per variant.
ret = {}
if not reached_stop("index"):
    r = run_step("retrieval", "db_03_retrieval_eval", {
        "data_dir": data_dir, "index_name": index_name, "endpoint": endpoint, "embed_endpoint": embed_endpoint, "testset": "testset_val",
        "variants": ",".join(f"{m}:{k}" for m, k in variants.items()), "rerank_endpoint": dbutils.widgets.get("rerank_endpoint"),
        "candidates_k": dbutils.widgets.get("candidates_k")})
    ret = {m: r["by_variant"][f"{m}:{k}"] for m, k in variants.items()}
    for name in ret.values():          # the generation notebook reads its inputs from <data_dir>/runs/
        shutil.copytree(f"{data_dir}/results_retrieval/{name}", f"{data_dir}/runs/{name}", dirs_exist_ok=True)
    print("retrieval runs:", ret)

# COMMAND ----------

# 3. Generation, from the saved chunks.
gen = {}
if ret and not reached_stop("retrieval"):
    r = run_step("generation", "generate_val_databricks", {
        "data_dir": data_dir, "endpoint": dbutils.widgets.get("gen_endpoint"), "ret_off": ret["off"], "ret_on": ret["on"], "ret_llm": ret.get("llm", "")})
    gen = r["gen_runs"]
    print("generation runs:", gen)

# COMMAND ----------

# 4. RAGAS on every generation run.
scored = {}
if gen and not reached_stop("generation"):
    r = run_step("ragas", "score_ragas_databricks", {
        "data_dir": data_dir, "gen_off": gen["off"], "gen_on": gen["on"], "gen_llm": gen.get("llm", ""),
        "judge_endpoint": dbutils.widgets.get("judge_endpoint"), "embed_endpoint": embed_endpoint})
    scored = r["summaries"]

# COMMAND ----------

# Side-by-side summary (every numeric metric the runs report; blank = not reported) and the record of the chain run.
def table(summaries: dict[str, dict]) -> None:
    names = list(summaries)
    keys = sorted({k for s in summaries.values() for k, v in s.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    print(f"{'metric':<42}" + "".join(f"{n.split('_rr-')[-1][:14]:>16}" for n in names))
    for k in keys:
        print(f"{k:<42}" + "".join(f"{summaries[n][k]:>16.3f}" if isinstance(summaries[n].get(k), (int, float)) else f"{'':>16}" for n in names))


if scored:
    table(scored)
elif ret:
    table({name: json.load(open(f"{data_dir}/results_retrieval/{name}/summary.json", encoding="utf-8")) for name in ret.values()})
record.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), retrieval_runs=ret, generation_runs=gen)
os.makedirs(f"{data_dir}/pipeline_runs", exist_ok=True)
path = f"{data_dir}/pipeline_runs/{time.strftime('%Y%m%d-%H%M%S')}.json"
json.dump(record, open(path, "w", encoding="utf-8"), indent=1, default=str)
print("\nrecord ->", path)
dbutils.notebook.exit(json.dumps({"record": path, "retrieval_runs": ret, "generation_runs": gen}))

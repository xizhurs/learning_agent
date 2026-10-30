# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Carbon methodology chatbot (AI Search + Databricks chat endpoints)
# MAGIC Multi-turn question answering over the indexed methodology documents, with the retrieval strategy chosen per question:
# MAGIC
# MAGIC | widget | values | what it does |
# MAGIC |---|---|---|
# MAGIC | `retrieval_mode` | `fts` / `vector` / `hybrid` | AI Search `FULL_TEXT` / `ANN` / `HYBRID` query on the same index |
# MAGIC | `rerank` | `off` / `ai_search` / `llm` | no rerank / the index's built-in reranker / a chat LLM ranks the top `candidates_k` hits (`rag.llm_rerank`) |
# MAGIC
# MAGIC Each answer is grounded in the retrieved passages, cites them as `[n]`, and says "I could not find this in the provided documents." when they do not
# MAGIC contain the answer (same prompt and abstention rule as the evaluated pipeline, `rag.generate`). Follow-up questions are rewritten into standalone ones
# MAGIC using the conversation so far. If a reranker fails, the answer is produced without it and the warning is shown.
# MAGIC
# MAGIC The same code is packaged as an MLflow `ResponsesAgent` (`rag.serving_agent`, entry file `carbon_rag_agent.py`) by the last cell, so the chatbot can be registered in Unity
# MAGIC Catalog and served; the two options then travel in `custom_inputs`. Deploying it to Model Serving is a separate, paid step that this notebook does not do.
# MAGIC **Data:** questions and the retrieved passages are sent to the Databricks endpoints named below. The index holds public documents only (see `db_01b_load_chunks_parquet.py`);
# MAGIC do not point `index_name` at an index with non-public content.
# MAGIC **Not measured:** which option is best is decided by `db_run_all.py` on the val questions, not by this notebook. Run `db_02_ai_search_index.py` first (the index must exist).

# COMMAND ----------

dbutils.widgets.text("index_name", "workspace.carbon_rag.chunks_1024_idx", "AI Search index name")
dbutils.widgets.text("endpoint", "carbon", "AI Search endpoint")
dbutils.widgets.text("gen_endpoint", "databricks-meta-llama-3-3-70b-instruct", "chat endpoint that writes the answers")
dbutils.widgets.text("rerank_endpoint", "databricks-meta-llama-3-3-70b-instruct", "chat endpoint for rerank = llm (non-reasoning model)")
dbutils.widgets.text("top_k", "5", "passages given to the model")
dbutils.widgets.text("candidates_k", "20", "hits ranked by the LLM reranker")
dbutils.widgets.dropdown("retrieval_mode", "hybrid", ["fts", "vector", "hybrid"], "retrieval mode")
dbutils.widgets.dropdown("rerank", "off", ["off", "ai_search", "llm"], "rerank")
dbutils.widgets.text("question", "", "question (then run the 'Ask' cell)")
dbutils.widgets.text("probe_question", "What is the minimum buffer between monitoring instances?", "question for the start-up check")
dbutils.widgets.dropdown("log_agent", "no", ["no", "yes"], "last cell: log the agent to MLflow")
dbutils.widgets.text("uc_model_name", "", "last cell: Unity Catalog model name to register (catalog.schema.model); empty = log only")

# COMMAND ----------

# Stage the code from GitHub (the workspace Git folder storage is corrupted).
import os, shutil, subprocess

REPO, clone_dir = "/tmp/carbon_methodology", "/tmp/learning_agent_git"
if os.path.isdir(clone_dir):
    shutil.rmtree(clone_dir)
subprocess.run(["git", "clone", "--depth", "1", "--branch", "qasper-corpus", "https://github.com/xizhurs/learning_agent", clone_dir], check=True)
for sub in ("src", "notebooks"):
    shutil.copytree(f"{clone_dir}/{sub}", f"{REPO}/{sub}", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy(f"{clone_dir}/config.yaml", f"{REPO}/config.yaml")
print("staged", REPO)

# COMMAND ----------

# MAGIC %pip install -r /tmp/carbon_methodology/notebooks/databricks-requirements-aisearch.txt -c /tmp/carbon_methodology/notebooks/databricks-constraints.txt databricks-openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Build the chatbot the same way the served agent does (rag.serving_agent.build_agent), so what runs here is what gets logged.
import sys, time

REPO = "/tmp/carbon_methodology"
sys.path.insert(0, f"{REPO}/src")
from rag.serving_agent import build_agent

CONFIG = {
    "index_name": dbutils.widgets.get("index_name"), "search_endpoint": dbutils.widgets.get("endpoint"),
    "generation_endpoint": dbutils.widgets.get("gen_endpoint"), "rerank_endpoint": dbutils.widgets.get("rerank_endpoint"),
    "top_k": int(dbutils.widgets.get("top_k")), "candidates_k": int(dbutils.widgets.get("candidates_k")),
    "retrieval_mode": dbutils.widgets.get("retrieval_mode"), "rerank": dbutils.widgets.get("rerank"),
}
agent = build_agent(CONFIG)
bot = agent.chatbot
history: list[dict] = []          # the conversation; the 'Clear conversation' cell empties it
print("ready:", CONFIG)

# COMMAND ----------

# Start-up check: which options work in this workspace? One real question per option (FULL_TEXT and ANN queries have not been
# verified on this index before; the AI Search reranker can be blocked). A failed rerank shows up as "rerank used = off".
rows = [("fts", "off"), ("vector", "off"), ("hybrid", "off"), ("hybrid", "ai_search"), ("hybrid", "llm")]
q = dbutils.widgets.get("probe_question")
print(f"{'mode':<8}{'rerank':<11}{'rerank used':<13}{'hits':>5}{'seconds':>9}  note")
for mode, rr in rows:
    t0 = time.time()
    try:
        a = bot.ask(q, retrieval_mode=mode, rerank=rr)
        print(f"{mode:<8}{rr:<11}{a.rerank:<13}{len(a.hits):>5}{time.time() - t0:>9.1f}  {'; '.join(a.warnings)}")
    except Exception as e:
        print(f"{mode:<8}{rr:<11}{'-':<13}{'-':>5}{time.time() - t0:>9.1f}  FAILED {type(e).__name__}: {str(e)[:150]}")

# COMMAND ----------

# DBTITLE 1,Ask
# Type the question in the `question` widget, pick the options, run this cell. Re-run it with a new question to continue the conversation.
q = dbutils.widgets.get("question").strip()
if not q:
    print("Type a question in the `question` widget first.")
else:
    a = bot.ask(q, history, retrieval_mode=dbutils.widgets.get("retrieval_mode"), rerank=dbutils.widgets.get("rerank"))
    history += [{"role": "user", "content": q}, {"role": "assistant", "content": a.answer}]
    if a.standalone_question != q:
        print(f"(searched as: {a.standalone_question})\n")
    print(a.answer)
    if a.hits and a.answer != "I could not find this in the provided documents.":
        print("\nSources:\n" + "\n".join(a.sources()))
    for w in a.warnings:
        print("WARNING:", w)
    print(f"\n[{a.retrieval_mode} | rerank {a.rerank} | " + " ".join(f"{k} {v:.1f}s" for k, v in a.timings.items()) + f" | turn {len(history) // 2}]")

# COMMAND ----------

# DBTITLE 1,Clear conversation
history.clear()
print("conversation cleared")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the agent to MLflow (optional)
# MAGIC With `log_agent = yes`: logs `carbon_rag_agent.py` as a models-from-code `ResponsesAgent` with the widgets above as its `model_config`, the `rag` package, and the endpoint and index
# MAGIC resources it needs (so a served copy gets credentials for exactly those), registers it in Unity Catalog when `uc_model_name` is set, then loads the logged model back
# MAGIC and asks it one question as a check. Serving it needs `databricks.agents.deploy(uc_model_name, version)`, which creates a paid endpoint and is left to you.

# COMMAND ----------

if dbutils.widgets.get("log_agent") != "yes":
    print("log_agent = no: skipped")
else:
    import mlflow
    from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex

    uc_name = dbutils.widgets.get("uc_model_name").strip()
    if uc_name:
        mlflow.set_registry_uri("databricks-uc")
    with mlflow.start_run(run_name="carbon-rag-agent"):
        info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=f"{REPO}/notebooks/carbon_rag_agent.py",
            code_paths=[f"{REPO}/src/rag"],
            model_config=CONFIG,
            resources=[
                DatabricksServingEndpoint(endpoint_name=CONFIG["generation_endpoint"]),
                DatabricksServingEndpoint(endpoint_name=CONFIG["rerank_endpoint"]),
                DatabricksVectorSearchIndex(index_name=CONFIG["index_name"]),
            ],
            pip_requirements=["mlflow", "databricks-openai", "databricks-ai-search==0.81", "openai==2.54.0", "pydantic", "pyyaml"],
            registered_model_name=uc_name or None,
        )
    print("logged:", info.model_uri, "| registered:", uc_name or "no")
    loaded = mlflow.pyfunc.load_model(info.model_uri)
    out = loaded.predict({"input": [{"role": "user", "content": dbutils.widgets.get("probe_question")}], "custom_inputs": {"retrieval_mode": "hybrid", "rerank": "off"}})
    print(out["output"][0]["content"][0]["text"])
    print(out["custom_outputs"])

# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v6"
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Probe: can Free Edition serverless parse, chunk and index the documents? (run once, creates nothing)
# MAGIC Checks, one by one, what the migration needs, and prints a go / no-go table at the end (also saved in `/tmp/probe_results.json`):
# MAGIC
# MAGIC 1. machine (Python, CPU architecture, memory, disk) and outbound network (Hugging Face for the Docling models and the BERT tokenizer, PyPI, GitHub)
# MAGIC 2. `databricks-ai-search` installs; the client and reranker classes exist; the exact method signatures are printed (the next notebooks are written from these)
# MAGIC 3. AI Search endpoints are listable; the embeddings endpoint exists
# MAGIC 4. **(optional, `install_docling = yes`)** Docling installs, parses the first few pages of one public PDF, and the repo's chunker produces chunks (time and peak memory are printed)
# MAGIC
# MAGIC Run it twice if you like: first with `install_docling = no` (fast, nothing heavy is installed; shows the pip dry-run of Docling), then with `yes`.
# MAGIC The second run restarts Python after the install, so just run the cells again from the top (the staging cell is cheap) until the end.
# MAGIC Nothing is created in the workspace and no document text is printed; only counts, sizes and timings.

# COMMAND ----------

dbutils.widgets.text("data_dir", "/Volumes/workspace/default/documents", "Volume folder with the PDFs")
dbutils.widgets.text("pdf", "VT0008-Additionality-Assessment-v1.0.pdf", "one public PDF for the parse test")
dbutils.widgets.text("pages", "3", "pages to parse in the test")
dbutils.widgets.dropdown("install_docling", "no", ["no", "yes"], "install Docling and run the parse test")
dbutils.widgets.text("embed_endpoint", "databricks-gte-large-en", "embeddings endpoint to look for")

# COMMAND ----------

import json, os, shutil, subprocess

RESULTS = "/tmp/probe_results.json"


def record(name, ok, detail=""):
    """Append one check to the results file (survives the Python restart after the install)."""
    results = json.load(open(RESULTS)) if os.path.isfile(RESULTS) else {}
    results[name] = {"ok": bool(ok), "detail": str(detail)[:600]}
    json.dump(results, open(RESULTS, "w"), indent=1)
    print(("OK   " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


# Stage the code from GitHub (the workspace Git folder storage is corrupted); the constraints file is in the branch already.
REPO, clone_dir = "/tmp/carbon_methodology", "/tmp/learning_agent_git"
if os.path.isdir(clone_dir):
    shutil.rmtree(clone_dir)
subprocess.run(["git", "clone", "--depth", "1", "--branch", "qasper-corpus", "https://github.com/xizhurs/learning_agent", clone_dir], check=True)
for sub in ("src", "notebooks"):
    shutil.copytree(f"{clone_dir}/{sub}", f"{REPO}/{sub}", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy(f"{clone_dir}/config.yaml", f"{REPO}/config.yaml")
print("staged", REPO)

# COMMAND ----------

# 1. Machine and network.
import platform, shutil, sys, time, urllib.request

mem = dict(l.split(":", 1) for l in open("/proc/meminfo").read().splitlines() if ":" in l)
gb = lambda key: int(mem[key].split()[0]) / 1024 / 1024
disk = shutil.disk_usage("/tmp")
record("machine", True, f"python {platform.python_version()}, {platform.machine()}, {os.cpu_count()} cores, RAM {gb('MemTotal'):.1f} GB (available {gb('MemAvailable'):.1f}), /tmp free {disk.free / 1e9:.1f} GB")

for name, url in {
    "network: huggingface.co (BERT tokenizer file)": "https://huggingface.co/bert-base-uncased/resolve/main/tokenizer_config.json",
    "network: huggingface.co (Docling layout model card)": "https://huggingface.co/ds4sd/docling-models/resolve/main/README.md",
    "network: pypi.org": "https://pypi.org/simple/docling/",
    "network: github.com": "https://github.com/xizhurs/learning_agent",
}.items():
    try:
        t0 = time.time()
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "probe"}), timeout=15) as r:
            record(name, True, f"HTTP {r.status} in {time.time() - t0:.1f}s")
    except Exception as e:
        record(name, False, f"{type(e).__name__}: {e}")

folder = dbutils.widgets.get("data_dir")
try:
    files = [(f.name, f.size) for f in dbutils.fs.ls(folder) if f.name.lower().endswith(".pdf")]
    record("volume: PDFs found", bool(files), "; ".join(f"{n} ({s / 1e6:.1f} MB)" for n, s in files))
except Exception as e:
    record("volume: PDFs found", False, f"{type(e).__name__}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2-3. AI Search client (small install, always done)

# COMMAND ----------

# MAGIC %pip install databricks-ai-search -c /tmp/carbon_methodology/notebooks/databricks-constraints.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import importlib.metadata as md, inspect, json, os, sys

RESULTS = "/tmp/probe_results.json"
REPO = "/tmp/carbon_methodology"
sys.path.insert(0, f"{REPO}/src")


def record(name, ok, detail=""):
    results = json.load(open(RESULTS)) if os.path.isfile(RESULTS) else {}
    results[name] = {"ok": bool(ok), "detail": str(detail)[:600]}
    json.dump(results, open(RESULTS, "w"), indent=1)
    print(("OK   " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


try:
    record("databricks-ai-search installed", True, md.version("databricks-ai-search"))
    from databricks.ai_search.client import AISearchClient
    from databricks.ai_search.reranker import DatabricksReranker
    record("imports: AISearchClient, DatabricksReranker", True)
except Exception as e:
    record("databricks-ai-search installed", False, f"{type(e).__name__}: {e}")
    raise

print("\nDatabricksReranker signature:", inspect.signature(DatabricksReranker))
client_methods = [m for m in dir(AISearchClient) if not m.startswith("_")]
print("\nAISearchClient public methods:", client_methods)
for m in client_methods:
    if m.startswith(("create_", "get_", "list_", "delete_", "wait_")):
        print(f"  {m}{inspect.signature(getattr(AISearchClient, m))}")

try:
    import databricks.ai_search.index as index_module

    for cls_name, cls in inspect.getmembers(index_module, inspect.isclass):
        if cls.__module__ == index_module.__name__:
            for m in ("similarity_search", "sync", "describe", "wait_until_ready", "scan"):
                if hasattr(cls, m):
                    print(f"  {cls_name}.{m}{inspect.signature(getattr(cls, m))}")
except Exception as e:
    print("could not inspect the index classes:", type(e).__name__, e)

# COMMAND ----------

# Endpoints: read-only. An empty list is fine (the index notebook creates the single Free Edition endpoint).
try:
    ai = AISearchClient()
    listing = ai.list_endpoints()
    names = [e.get("name") for e in (listing.get("endpoints", []) if isinstance(listing, dict) else listing)]
    record("AI Search: list endpoints", True, f"{len(names)} existing: {names}")
except Exception as e:
    record("AI Search: list endpoints", False, f"{type(e).__name__}: {e}")

try:
    from databricks.sdk import WorkspaceClient

    wanted = dbutils.widgets.get("embed_endpoint")
    eps = {e.name: getattr(e, "task", "") for e in WorkspaceClient().serving_endpoints.list()}
    record(f"embeddings endpoint {wanted}", wanted in eps, f"task={eps.get(wanted)}; other embedding-like endpoints: {[n for n in eps if any(k in n for k in ('gte', 'bge', 'embed'))]}")
except Exception as e:
    record("embeddings endpoint", False, f"{type(e).__name__}: {e}")

try:
    record("reranker object", True, repr(DatabricksReranker(columns_to_rerank=["text"])))
except Exception as e:
    record("reranker object", False, f"{type(e).__name__}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Docling, the tokenizer and the chunker (only with `install_docling = yes`)
# MAGIC With `no`, this cell prints what pip would install (dry run, nothing is installed) so the size is known before committing.

# COMMAND ----------

import subprocess, sys

REQ = ["docling", "docling-core", "transformers"]
CONSTRAINTS = f"{REPO}/notebooks/databricks-constraints.txt"
if dbutils.widgets.get("install_docling") == "yes":
    p = subprocess.run([sys.executable, "-m", "pip", "install", *REQ, "-c", CONSTRAINTS], capture_output=True, text=True)
    record("pip install docling docling-core transformers", p.returncode == 0, p.stdout[-300:] if p.returncode == 0 else p.stderr[-500:])
    if p.returncode == 0:
        dbutils.library.restartPython()          # run the next cells again from the top of this section
else:
    p = subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", *REQ, "-c", CONSTRAINTS], capture_output=True, text=True)
    last = [l for l in p.stdout.splitlines() if l.startswith("Would install")]
    record("pip dry run: docling docling-core transformers resolve under the serverless constraints", p.returncode == 0, (last[-1][:500] if last else p.stderr[-500:]))

# COMMAND ----------

import importlib.util, json, os, resource, sys, time

RESULTS, REPO = "/tmp/probe_results.json", "/tmp/carbon_methodology"
sys.path.insert(0, f"{REPO}/src")


def record(name, ok, detail=""):
    results = json.load(open(RESULTS)) if os.path.isfile(RESULTS) else {}
    results[name] = {"ok": bool(ok), "detail": str(detail)[:600]}
    json.dump(results, open(RESULTS, "w"), indent=1)
    print(("OK   " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


have_docling = importlib.util.find_spec("docling") is not None
if dbutils.widgets.get("install_docling") != "yes" or not have_docling:
    print("Parse test skipped (install_docling = no, or Docling is not installed).")
else:
    pdf = f"{dbutils.widgets.get('data_dir')}/{dbutils.widgets.get('pdf')}"
    pages = int(dbutils.widgets.get("pages"))
    local = "/tmp/probe.pdf"
    shutil.copy(pdf, local)                      # parse from local disk, not the Volume FUSE mount
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr = False                      # same options as rag.ingest.parse._converter
        opts.do_table_structure = True
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        t0 = time.time()
        doc = converter.convert(local, page_range=(1, pages)).document
        secs = time.time() - t0
        peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        record("docling parse", True, f"{pages} pages in {secs:.0f}s ({secs / pages:.1f}s/page), peak RSS {peak_gb:.1f} GB, {len(doc.texts)} text items, {len(doc.tables)} tables")
    except Exception as e:
        record("docling parse", False, f"{type(e).__name__}: {str(e)[:400]}")
        doc = None
    if doc is not None:
        try:
            os.environ.update({"RAG__CHUNKING__MAX_TOKENS": "1024", "RAG__CHUNKING__OVERLAP": "128"})
            from rag.config import get_settings
            from rag.ingest.chunk import chunk_document

            t0 = time.time()
            chunks = chunk_document(doc, get_settings().chunking)
            record("chunker (BERT tokenizer from the Hub)", True, f"{len(chunks)} chunks in {time.time() - t0:.1f}s, longest {max(len(c.text) for c in chunks)} chars / {max(len(c.text.encode()) for c in chunks)} bytes (index limit 32,764 bytes)")
        except Exception as e:
            record("chunker (BERT tokenizer from the Hub)", False, f"{type(e).__name__}: {str(e)[:400]}")

# COMMAND ----------

# Summary: paste this table back.
results = json.load(open(RESULTS))
width = max(len(k) for k in results)
for k, v in results.items():
    print(f"{'GO   ' if v['ok'] else 'NO-GO'}  {k:<{width}}  {v['detail']}")
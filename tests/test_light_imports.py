import subprocess
import sys

BLOCKED = ["lancedb", "docling", "sentence_transformers", "torch", "pyarrow", "tantivy", "ollama"]


def _imports_with_blocked(modules: list[str], blocked: list[str]) -> subprocess.CompletedProcess:
    code = (
        "import sys\n"
        + "".join(f"sys.modules[{name!r}] = None\n" for name in blocked)
        + "".join(f"import {module}\n" for module in modules)
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_run_eval_imports_without_retrieval_dependencies():
    """A generation-only run (--reuse-retrieval, e.g. on Databricks) must not need lancedb, docling or the reranker."""
    r = _imports_with_blocked(["rag.eval.run_eval"], BLOCKED)
    assert r.returncode == 0, r.stderr


def test_ai_search_module_imports_without_databricks_or_retrieval_dependencies():
    """rag.ai_search is testable here and importable on serverless before databricks-ai-search is installed."""
    r = _imports_with_blocked(["rag.ai_search", "rag.eval.run_eval"], BLOCKED + ["databricks"])
    assert r.returncode == 0, r.stderr


def test_chat_and_llm_rerank_import_without_databricks_or_retrieval_dependencies():
    """The chatbot core runs on serverless: no lancedb, docling, torch, ollama or databricks at import time."""
    r = _imports_with_blocked(["rag.chat", "rag.llm_rerank"], BLOCKED + ["databricks"])
    assert r.returncode == 0, r.stderr

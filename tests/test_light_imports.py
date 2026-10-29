import subprocess
import sys

BLOCKED = ["lancedb", "docling", "sentence_transformers", "torch", "pyarrow", "tantivy", "ollama"]


def test_run_eval_imports_without_retrieval_dependencies():
    """A generation-only run (--reuse-retrieval, e.g. on Databricks) must not need lancedb, docling or the reranker."""
    code = ("import sys\n"
            + "".join(f"sys.modules[{name!r}] = None\n" for name in BLOCKED)
            + "import rag.eval.run_eval\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

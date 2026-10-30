# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v6"
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Load the existing 1024-token chunks into a Delta table (no parsing on Databricks)
# MAGIC Loads `chunks_1024.parquet` (the 504 chunks of the four public documents that the local runs used: LanceDB table
# MAGIC `chunks_heading_aware_1024_128`, columns `id, source, page, section, kind, text`, no vectors) into the Delta table the AI Search index reads.
# MAGIC This is the quick path: it needs no Docling. `db_01_ingest_chunks.py` (parse and chunk on Databricks) fills the same table shape later.
# MAGIC
# MAGIC **Before running:** copy `data/export/chunks_1024.parquet` from your machine into `<data_dir>` (Catalog -> Volumes -> upload). It was written from the local table with a
# MAGIC round-trip check; its content hash (SHA-256 over `id TAB text` lines, first 16 hex characters) is the `expected_hash` default, so the notebook can tell whether it
# MAGIC got that exact file.
# MAGIC
# MAGIC **Guards:** every `source` must be one of the four public files in `allowed_sources` (the draft project description must never be loaded); ids unique; text under the
# MAGIC index limit. Re-running replaces the table (`overwrite`); if an AI Search index already exists on it, run `db_02_ai_search_index.py` with `recreate = yes` afterwards.
# MAGIC The last cell returns a JSON status (used by `db_run_all.py`).

# COMMAND ----------

dbutils.widgets.text("data_dir", "/Volumes/workspace/default/documents/carbon_eval", "Volume folder with chunks_1024.parquet")
dbutils.widgets.text("parquet", "chunks_1024.parquet", "parquet file name")
dbutils.widgets.text("table", "workspace.carbon_rag.chunks_1024", "Delta table to write (catalog.schema.table)")
dbutils.widgets.text("expected_hash", "586822bdef2766d0", "expected content hash of the parquet (empty = skip the check)")
dbutils.widgets.text(
    "allowed_sources",
    "VCS-Standard-v5.0.pdf,VM0047-v1.1_Final.pdf,VMD0054_ARR-Leakage-Module_v1.0.pdf,VT0008-Additionality-Assessment-v1.0.pdf",
    "source file names that may be loaded (public documents only)",
)

# COMMAND ----------

import hashlib, json, os

import pandas as pd

data_dir, table = dbutils.widgets.get("data_dir"), dbutils.widgets.get("table")
path = f"{data_dir}/{dbutils.widgets.get('parquet')}"
allowed = {s.strip() for s in dbutils.widgets.get("allowed_sources").split(",") if s.strip()}
expected_hash = dbutils.widgets.get("expected_hash").strip()
COLUMNS = ["id", "source", "page", "section", "kind", "text"]


def content_hash(frame) -> str:
    """Order-independent (rows sorted by id), so the parquet and the table read back by Spark hash the same."""
    rows = frame.sort_values("id")
    return hashlib.sha256("\n".join(f"{r.id}\t{r.text}" for r in rows.itertuples()).encode("utf-8")).hexdigest()[:16]


assert os.path.isfile(path), f"{path} not found: upload data/export/chunks_1024.parquet to {data_dir}"
pdf = pd.read_parquet(path)
assert list(pdf.columns) == COLUMNS, f"columns are {list(pdf.columns)}, expected {COLUMNS}"
extra = sorted(set(pdf.source) - allowed)
assert not extra, f"sources not on the allow-list (refusing to load them): {extra}"
assert pdf.id.is_unique and len(pdf) > 0, "ids must be unique and the file non-empty"
max_bytes = int(pdf.text.str.encode("utf-8").str.len().max())
assert max_bytes < 30000, f"a chunk has {max_bytes} bytes; the AI Search source-column limit is 32,764"
file_hash = content_hash(pdf)
print(f"{len(pdf)} chunks, sources {sorted(set(pdf.source))}, longest text {max_bytes} bytes, content hash {file_hash}")
if expected_hash:
    assert file_hash == expected_hash, f"content hash {file_hash} differs from the expected {expected_hash}: a different file than the exported one"
    print("content hash matches the export")

# COMMAND ----------

# Write the Delta table (change data feed on: a Delta Sync index needs it) and read it back to check it.
catalog, schema, name = table.split(".")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
df = spark.createDataFrame(pdf)
df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

back = spark.table(table).toPandas().sort_values("id").reset_index(drop=True)
src = pdf.sort_values("id").reset_index(drop=True)
props = {r["key"]: r["value"] for r in spark.sql(f"SHOW TBLPROPERTIES {table}").collect()}
status = {
    "table": table,
    "rows": len(back),
    "per_source": back.groupby("source").size().to_dict(),
    "change_data_feed": props.get("delta.enableChangeDataFeed"),
    "content_hash": content_hash(back),
    "identical_to_parquet": bool(back[COLUMNS].equals(src[COLUMNS])),
    "max_text_bytes": max_bytes,
}
print(json.dumps(status, indent=1))
assert status["identical_to_parquet"] and status["content_hash"] == file_hash, "the table differs from the parquet file"
assert str(status["change_data_feed"]).lower() == "true"

# COMMAND ----------

dbutils.notebook.exit(json.dumps(status, default=str))
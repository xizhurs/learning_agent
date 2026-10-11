"""Parse PDF / DOCX into Docling documents, cached as JSON so re-chunking doesn't re-parse."""
from __future__ import annotations

import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from rag.config import Settings
from rag.ingest.docx_clean import accept_tracked_changes

SUPPORTED = {".pdf", ".docx"}


def _converter() -> DocumentConverter:
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = False            # both source PDFs are born-digital
    pdf_opts.do_table_structure = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )


def list_documents(settings: Settings) -> list[Path]:
    docs_dir = settings.paths.resolve("documents_dir")
    return sorted(p for p in docs_dir.iterdir() if p.suffix.lower() in SUPPORTED and not p.name.startswith("~$"))


def parse_document(path: Path, settings: Settings, force: bool = False) -> DoclingDocument:
    """Return the DoclingDocument for `path`, using data/processed/<name>.docling.json as a cache."""
    processed = settings.paths.resolve("processed_dir")
    processed.mkdir(parents=True, exist_ok=True)
    cache = processed / f"{path.stem}.docling.json"
    if cache.exists() and not force:
        return DoclingDocument.load_from_json(cache)

    source = path
    if path.suffix.lower() == ".docx":
        source = processed / f"{path.stem}.accepted.docx"
        stats = accept_tracked_changes(path, source)
        print(f"  accepted tracked changes: {stats}")

    t0 = time.time()
    result = _converter().convert(source)
    doc = result.document
    # Keep the original file name as the source label (the cleaned copy has a different name).
    doc.name = path.name
    doc.save_as_json(cache)
    print(f"  parsed {path.name} in {time.time() - t0:.0f}s")
    return doc

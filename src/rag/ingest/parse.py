"""Parse PDF / DOCX (Docling, cached as JSON so re-chunking doesn't re-parse) and plain-text technotes into Docling documents."""
from __future__ import annotations

import re
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from rag.config import Settings
from rag.ingest.docx_clean import accept_tracked_changes

SUPPORTED = {".pdf", ".docx", ".txt"}


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


def parse_text_document(path: Path) -> DoclingDocument:
    """Plain-text technote -> DoclingDocument: first 'Title: ...' line -> title, blank-line separated paragraphs -> text,
    a bare 'Text:' line is dropped. The document is named after the file so chunk ids stay '<file name>::NNNNN'.

    ALL-CAPS section labels (PROBLEM, CAUSE, '* PE', ...) are deliberately NOT turned into headings: the chunker starts a new
    chunk at every heading, which cut the technotes into ~100-token fragments and made chunk size irrelevant."""
    raw = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    doc = DoclingDocument(name=path.name)
    para: list[str] = []
    titled = False

    def flush() -> None:
        if para:
            doc.add_text(label=DocItemLabel.TEXT, text="\n".join(para))
            para.clear()

    for line in raw.split("\n"):
        s = line.strip()
        if not s:
            flush()
        elif s.startswith("Title:") and not titled:
            flush()
            doc.add_title(s.removeprefix("Title:").strip())
            titled = True
        elif s == "Text:":
            flush()
        else:
            para.append(s)
    flush()
    return doc


def parse_document(path: Path, settings: Settings, force: bool = False) -> DoclingDocument:
    """Return the DoclingDocument for `path`, using data/processed/<name>.docling.json as a cache (not used for .txt)."""
    if path.suffix.lower() == ".txt":
        return parse_text_document(path)
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

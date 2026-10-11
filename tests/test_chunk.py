import pytest
from docling_core.types.doc import DoclingDocument, TableCell, TableData
from docling_core.types.doc.labels import DocItemLabel

from rag.config import Chunking
from rag.ingest.chunk import chunk_document, count_tokens


def _make_doc() -> DoclingDocument:
    doc = DoclingDocument(name="test.pdf")
    doc.add_heading("1 Scope", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="This methodology applies to improved agricultural land management.")
    doc.add_text(label=DocItemLabel.TEXT, text="Projects must demonstrate additionality. " * 80)  # long paragraph
    doc.add_heading("2 Parameters", level=1)
    cells = [TableCell(text=t, start_row_offset_idx=r, end_row_offset_idx=r + 1,
                       start_col_offset_idx=c, end_col_offset_idx=c + 1, column_header=(r == 0))
             for r in range(41) for c, t in enumerate([f"param{r}" if r else "Parameter", f"value{r}" if r else "Value"])]
    doc.add_table(data=TableData(num_rows=41, num_cols=2, table_cells=cells))
    return doc


@pytest.mark.parametrize("strategy", ["heading_aware", "fixed_tokens"])
def test_chunks_respect_max_tokens(strategy):
    cfg = Chunking(strategy=strategy, max_tokens=128, overlap=16)
    chunks = chunk_document(_make_doc(), cfg)
    assert len(chunks) > 3
    # small slack: fixed_tokens estimates the heading-prefix cost
    assert max(count_tokens(c.text) for c in chunks) <= cfg.max_tokens + 16


def test_heading_prefix_and_section_metadata():
    chunks = chunk_document(_make_doc(), Chunking(max_tokens=128))
    first = chunks[0]
    assert first.section == "1 Scope"
    assert first.text.startswith("test.pdf > 1 Scope")
    no_prefix = chunk_document(_make_doc(), Chunking(max_tokens=128, heading_prefix=False))
    assert not no_prefix[0].text.startswith("test.pdf")


def test_tables_split_by_rows_with_header_repeated():
    chunks = [c for c in chunk_document(_make_doc(), Chunking(max_tokens=128)) if c.kind == "table"]
    assert len(chunks) > 1
    for c in chunks:
        header_row = c.text.split("\n\n", 1)[1].splitlines()[0]
        assert "Parameter" in header_row and "Value" in header_row   # header repeated in every part
        assert all(line.startswith("|") for line in c.text.splitlines()[2:] if line.strip())  # no mid-row cuts
    all_text = "\n".join(c.text for c in chunks)
    assert "param40" in all_text                              # no rows lost

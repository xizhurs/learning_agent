"""Chunking strategies over Docling documents.

heading_aware : pack consecutive blocks under the same heading up to max_tokens; tables are kept
                separate (split by rows with the header row repeated if too large).
fixed_tokens  : ignore structure, slide a max_tokens window with `overlap` over the document text.

Token counts use a BERT wordpiece tokenizer (nomic-embed-text is BERT based), which is close
enough to size chunks; the embedding model never sees more than 2048 tokens here anyway.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import lru_cache

from docling_core.types.doc import DoclingDocument, SectionHeaderItem, TableItem, TitleItem
from docling_core.types.doc.labels import DocItemLabel

from rag.config import Chunking

TOKENIZER_NAME = "bert-base-uncased"


@dataclass
class Chunk:
    id: str
    source: str
    page: int          # 0 when the format has no pages (docx)
    section: str       # "Heading 1 > Heading 2"
    kind: str          # "text" | "table"
    text: str          # what gets embedded, indexed and shown to the LLM (includes heading prefix)

    def as_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def count_tokens(text: str) -> int:
    return len(_tokenizer()(text, add_special_tokens=False, truncation=False)["input_ids"])


@dataclass
class _Block:
    text: str
    section: str
    page: int
    is_table: bool = False
    table_header: str | None = None   # markdown header + separator rows, for splitting big tables


def _page_of(item) -> int:
    return item.prov[0].page_no if getattr(item, "prov", None) else 0


def _blocks(doc: DoclingDocument) -> list[_Block]:
    """Flatten the document into text/table blocks annotated with their heading path."""
    blocks: list[_Block] = []
    headings: list[tuple[int, str]] = []   # (level, title)
    skip = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}

    for item, _level in doc.iterate_items():
        if isinstance(item, TitleItem):
            headings = [(0, item.text.strip())]
            continue
        if isinstance(item, SectionHeaderItem):
            lvl = item.level
            headings = [h for h in headings if h[0] < lvl] + [(lvl, item.text.strip())]
            continue
        label = getattr(item, "label", None)
        if label in skip:
            continue
        section = " > ".join(h[1] for h in headings)

        if isinstance(item, TableItem):
            md, header = _normalise_table(item.export_to_markdown(doc=doc))
            if not md:
                continue
            blocks.append(_Block(md, section, _page_of(item), is_table=True, table_header=header))
        elif hasattr(item, "text") and item.text and item.text.strip():
            blocks.append(_Block(item.text.strip(), section, _page_of(item)))
    return blocks


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def _split_long_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split an oversized paragraph on sentence boundaries, carrying ~`overlap` tokens between parts."""
    sentences = _SENT_SPLIT.split(text)
    parts: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for s in sentences:
        t = count_tokens(s)
        if t > max_tokens:   # a single giant "sentence": hard split by token windows
            if cur:
                parts.append(" ".join(cur))
                cur, cur_tok = [], 0
            offsets = _tokenizer()(s, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            step = max(1, max_tokens - overlap)
            for i in range(0, len(offsets), step):
                j = min(i + max_tokens, len(offsets))
                parts.append(s[offsets[i][0] : offsets[j - 1][1]])
                if j == len(offsets):
                    break
            continue
        if cur and cur_tok + t > max_tokens:
            parts.append(" ".join(cur))
            carry: list[str] = []
            carry_tok = 0
            for prev in reversed(cur):
                pt = count_tokens(prev)
                if carry_tok + pt > overlap:
                    break
                carry.insert(0, prev)
                carry_tok += pt
            cur, cur_tok = carry, carry_tok
        cur.append(s)
        cur_tok += t
    if cur:
        parts.append(" ".join(cur))
    return parts


def _normalise_table(md: str) -> tuple[str, str | None]:
    """Collapse column padding and merged-cell repeats; return (markdown, header block or None)."""
    lines = [l for l in md.splitlines() if l.strip()]
    if not lines:
        return "", None

    def cells(line: str) -> list[str]:
        raw = line.strip()
        raw = raw[1:] if raw.startswith("|") else raw
        raw = raw[:-1] if raw.endswith("|") else raw
        return [re.sub(r"\s+", " ", c).strip() for c in raw.split("|")]

    def fmt(cs: list[str]) -> str:
        return "| " + " | ".join(cs) + " |"

    has_sep = len(lines) > 1 and set(lines[1]) <= set("|-: ")
    out: list[str] = []
    header = None
    if has_sep:
        h = cells(lines[0])
        header = fmt(h) + "\n" + fmt(["---"] * len(h))
        out += [fmt(h), fmt(["---"] * len(h))]
    seen: set[str] = set()
    for line in lines[2 if has_sep else 0 :]:
        cs: list[str] = []
        for c in cells(line):
            if c and cs and c == cs[-1]:
                continue                      # merged cell repeated by the parser
            cs.append(c)
        row = fmt(cs)
        if row in seen:
            continue                          # identical row repeated (merged rows)
        seen.add(row)
        out.append(row)
    return "\n".join(out), header


def _split_table(block: _Block, max_tokens: int, overlap: int = 0) -> list[str]:
    lines = block.text.splitlines()
    header = block.table_header
    body = lines[2:] if header else lines
    if header and count_tokens(header) > max_tokens // 4:
        # "header" is really a long content row (key/value form tables): don't repeat it, split it like any row
        body, header = [lines[0]] + lines[2:], None
    head_tok = count_tokens(header) if header else 0
    row_budget = max(32, max_tokens - head_tok - 4)
    parts: list[str] = []
    cur: list[str] = []
    cur_tok = head_tok
    pieces: list[str] = []
    for row in body:                          # a single oversized row is split on sentence boundaries
        if count_tokens(row) > row_budget:
            inner = row.strip().strip("|").strip()
            pieces += [f"| {p} |" for p in _split_long_text(inner, row_budget, overlap)]
        else:
            pieces.append(row)
    for row in pieces:
        t = count_tokens(row)
        if cur and cur_tok + t > max_tokens:
            parts.append((header + "\n" if header else "") + "\n".join(cur))
            cur, cur_tok = [], head_tok
        cur.append(row)
        cur_tok += t
    if cur:
        parts.append((header + "\n" if header else "") + "\n".join(cur))
    return parts


def _prefix(label: str, section: str, enabled: bool) -> str:
    if not enabled:
        return ""
    return f"{label} > {section}\n\n" if section else f"{label}\n\n"


def chunk_document(doc: DoclingDocument, cfg: Chunking) -> list[Chunk]:
    label = doc.name
    blocks = _blocks(doc)
    if cfg.strategy == "fixed_tokens":
        raw = _chunk_fixed(blocks, label, cfg)
    else:
        raw = _chunk_heading_aware(blocks, label, cfg)
    return [
        Chunk(id=f"{label}::{i:05d}", source=label, page=page, section=section, kind=kind, text=text)
        for i, (page, section, kind, text) in enumerate(raw)
    ]


def _chunk_heading_aware(blocks: list[_Block], label: str, cfg: Chunking):
    out: list[tuple[int, str, str, str]] = []   # (page, section, kind, text)
    buf: list[str] = []
    buf_tok = 0
    buf_section, buf_page = "", 0

    def budget(section: str) -> int:
        return cfg.max_tokens - count_tokens(_prefix(label, section, cfg.heading_prefix))

    def flush():
        nonlocal buf, buf_tok
        if buf:
            body = "\n\n".join(buf)
            out.append((buf_page, buf_section, "text", _prefix(label, buf_section, cfg.heading_prefix) + body))
        buf, buf_tok = [], 0

    for b in blocks:
        if b.is_table and cfg.table_as_own_chunk:
            flush()
            for part in _split_table(b, budget(b.section), cfg.overlap):
                out.append((b.page, b.section, "table", _prefix(label, b.section, cfg.heading_prefix) + part))
            continue

        if b.section != buf_section:
            flush()
        if not buf:
            buf_section, buf_page = b.section, b.page
        limit = budget(b.section)
        t = count_tokens(b.text)
        if t > limit:
            flush()
            buf_section, buf_page = b.section, b.page
            for part in _split_long_text(b.text, limit, cfg.overlap):
                out.append((b.page, b.section, "text", _prefix(label, b.section, cfg.heading_prefix) + part))
            continue
        if buf_tok + t > limit:
            flush()
            buf_section, buf_page = b.section, b.page
        buf.append(b.text)
        buf_tok += t
    flush()
    return out


def _chunk_fixed(blocks: list[_Block], label: str, cfg: Chunking):
    """Sliding token window over the concatenated document text (headings still tracked for metadata)."""
    sep = "\n\n"
    text_parts, starts, pos = [], [], 0
    for b in blocks:
        starts.append((pos, b))
        text_parts.append(b.text)
        pos += len(b.text) + len(sep)
    full = sep.join(text_parts)
    enc = _tokenizer()(full, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offsets = enc["offset_mapping"]
    prefix_cost = count_tokens(_prefix(label, "", cfg.heading_prefix)) + 12   # rough allowance for the section path
    window = max(32, cfg.max_tokens - (prefix_cost if cfg.heading_prefix else 0))
    step = max(1, window - cfg.overlap)

    import bisect

    start_chars = [s for s, _ in starts]
    out = []
    for i in range(0, len(offsets), step):
        j = min(i + window, len(offsets))
        c0, c1 = offsets[i][0], offsets[j - 1][1]
        _, b = starts[max(0, bisect.bisect_right(start_chars, c0) - 1)]
        kind = "table" if b.is_table else "text"
        out.append((b.page, b.section, kind, _prefix(label, b.section, cfg.heading_prefix) + full[c0:c1]))
        if j == len(offsets):
            break
    return out

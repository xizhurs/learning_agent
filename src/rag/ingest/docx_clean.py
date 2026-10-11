"""Accept all tracked changes in a .docx so downstream parsers see the final text.

python-docx / Docling read only direct <w:r> children of a paragraph, so text wrapped in <w:ins>
is silently lost and text in <w:del> (as <w:delText>) is ignored. We rewrite the XML instead:
  * <w:ins>, <w:moveTo>      -> unwrapped (content kept)
  * <w:del>, <w:moveFrom>    -> removed with content
  * table rows whose row mark is deleted -> removed
  * *PrChange elements (old formatting records) and paragraph/run-mark ins/del markers -> removed
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


def _q(tag: str) -> str:
    return f"{{{W}}}{tag}"


_PART_PREFIXES = ("word/document.xml", "word/footnotes.xml", "word/endnotes.xml")
_PART_STARTS = ("word/header", "word/footer")


def _is_text_part(name: str) -> bool:
    return name in _PART_PREFIXES or (name.startswith(_PART_STARTS) and name.endswith(".xml"))


@dataclass
class CleanStats:
    insertions_accepted: int = 0
    deletions_removed: int = 0
    rows_removed: int = 0


def _accept_changes(root: etree._Element, stats: CleanStats) -> None:
    # Deleted table rows: <w:tr><w:trPr><w:del/></w:trPr>...
    for tr in root.xpath("//w:tr[w:trPr/w:del]", namespaces=NS):
        tr.getparent().remove(tr)
        stats.rows_removed += 1

    # Remove deleted / moved-away content entirely (keep tail text that follows the element).
    for tag in ("del", "moveFrom"):
        for el in root.xpath(f"//w:{tag}[not(parent::w:rPr) and not(parent::w:trPr) and not(parent::w:pPr)]", namespaces=NS):
            _remove_keep_tail(el)
            stats.deletions_removed += 1

    # Unwrap insertions: move children up to the parent in place.
    for tag in ("ins", "moveTo"):
        for el in root.xpath(f"//w:{tag}[not(parent::w:rPr) and not(parent::w:trPr) and not(parent::w:pPr)]", namespaces=NS):
            _unwrap(el)
            stats.insertions_accepted += 1

    # Leftover markers (paragraph-mark / run-mark / row-mark ins/del) and property-change history.
    for el in root.xpath(
        "//w:rPr/w:ins | //w:rPr/w:del | //w:rPr/w:moveFrom | //w:rPr/w:moveTo"
        " | //w:trPr/w:ins | //w:trPr/w:del"
        " | //w:rPrChange | //w:pPrChange | //w:sectPrChange | //w:tblPrChange | //w:tblGridChange"
        " | //w:tcPrChange | //w:trPrChange | //w:numberingChange",
        namespaces=NS,
    ):
        _remove_keep_tail(el)


def _remove_keep_tail(el: etree._Element) -> None:
    parent = el.getparent()
    if parent is None:
        return
    if el.tail:
        prev = el.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + el.tail
        else:
            parent.text = (parent.text or "") + el.tail
    parent.remove(el)


def _unwrap(el: etree._Element) -> None:
    parent = el.getparent()
    if parent is None:
        return
    idx = parent.index(el)
    for child in list(el):
        parent.insert(idx, child)
        idx += 1
    _remove_keep_tail(el)


def accept_tracked_changes(src: Path, dst: Path) -> CleanStats:
    """Write a copy of `src` to `dst` with all tracked changes accepted."""
    stats = CleanStats()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if _is_text_part(item.filename):
                root = etree.fromstring(data)
                _accept_changes(root, stats)
                data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
            zout.writestr(item, data)
    return stats

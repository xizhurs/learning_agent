from rag.config import Chunking, Settings
from rag.eval.metrics import first_doc_rank
from rag.eval.run_eval import (apply_judge_scope, assert_hosted_safe, judge_eligible, normalize_item, summarise,
                               slugify_testset)
from rag.ingest.chunk import chunk_document
from rag.ingest.parse import SUPPORTED, parse_text_document

TECHNOTE = """Title: IBM Using Oracle JDBC driver specific properties - United States

Text:
TECHNOTE (TROUBLESHOOTING)

PROBLEM(ABSTRACT)
Properties that are not part of the JDBC specification cannot be used as a Custom Property.

RESOLVING THE PROBLEM
Set a Custom Property named connectionProperties with value propertyA=valueA;propertyB=valueB.

Restart the server afterwards.
"""


def test_txt_is_supported_and_parsed_with_title_and_paragraphs(tmp_path):
    assert ".txt" in SUPPORTED
    path = tmp_path / "swg21197705.txt"
    path.write_text(TECHNOTE, encoding="utf-8")
    chunks = chunk_document(parse_text_document(path), Chunking(max_tokens=512))
    assert len(chunks) == 1                      # section labels do not split a short technote into fragments
    chunk = chunks[0]
    assert chunk.id == "swg21197705.txt::00000" and chunk.source == "swg21197705.txt"
    assert chunk.text.startswith("swg21197705.txt > IBM Using Oracle JDBC driver specific properties - United States")
    assert "Text:" not in chunk.text and "Title:" not in chunk.text
    assert "RESOLVING THE PROBLEM" in chunk.text and "connectionProperties" in chunk.text and "Restart the server" in chunk.text


def test_chunk_size_splits_long_technotes_and_keeps_ids_and_prefix(tmp_path):
    path = tmp_path / "ibm1.txt"
    paras = "\n\n".join(f"Paragraph {i}. " + "The server must be restarted after changing this setting. " * 6 for i in range(30))
    path.write_text(f"Title: Long note\n\nText:\n{paras}\n", encoding="utf-8")
    doc = parse_text_document(path)
    small = chunk_document(doc, Chunking(max_tokens=128, overlap=16))
    large = chunk_document(doc, Chunking(max_tokens=1024, overlap=16))
    assert len(small) > len(large) >= 1
    assert [c.id for c in small] == [f"ibm1.txt::{i:05d}" for i in range(len(small))]
    assert all(c.text.startswith("ibm1.txt > Long note") for c in small)


def test_normalize_item_maps_techqa_rows_and_leaves_old_rows_alone():
    ans = normalize_item({"id": "Q1", "question": "q", "answer": "a", "is_impossible": False, "gold_context_ids": ["d1.txt"]})
    assert ans["answerable"] and ans["source"] == "d1.txt" and ans["type"] == "factual"
    assert ans["reference_answer"] == ans["reference_quote"] == "a" and ans["gold_context_ids"] == ["d1.txt"]
    imp = normalize_item({"id": "Q2", "question": "q", "answer": "-", "is_impossible": True, "gold_context_ids": []})
    assert not imp["answerable"] and imp["reference_answer"] == "" and imp["source"] == ""
    old = {"id": "x", "question": "q", "type": "factual", "answerable": True, "source": "s", "reference_answer": "r",
           "reference_quote": "r"}
    assert normalize_item(old) is old


def test_first_doc_rank_is_document_level():
    ids = ["a.txt::00000", "b.txt::00003", "b.txt::00004", "c.txt::00000"]
    assert first_doc_rank(["b.txt"], ids) == 2
    assert first_doc_rank(["c.txt", "a.txt"], ids) == 1
    assert first_doc_rank(["z.txt"], ids) is None
    assert first_doc_rank(["b.txt"], []) is None


def test_unrestricted_judge_scope_keeps_all_items_and_does_not_filter_retrieval():
    s = Settings()
    assert s.eval.judge_allowed_source is None
    items = [{"id": "a", "answerable": True, "type": "factual", "source": "d1.txt"},
             {"id": "b", "answerable": False, "type": "factual", "source": ""}]
    kept, extra = apply_judge_scope(items, s)
    assert [i["id"] for i in kept] == ["a", "b"] and s.retrieval.source_filter is None and extra["ragas_scope"] == "unrestricted"
    assert judge_eligible(items[0], None) and not judge_eligible(items[1], None)
    assert judge_eligible({**items[0], "type": "multi_passage"}, None)       # only a restriction needs a single known source
    row = {**items[0], "retrieved": [{"id": "other.txt::00001"}], "answer": "x"}
    assert_hosted_safe([row], None)          # any document is fine when the corpus is public


def test_slugify_testset_has_no_path_separators():
    assert slugify_testset("techqa_tiny/dev") == "techqa_tiny_dev" and slugify_testset("testset") == "testset"


def test_summarise_reports_document_and_answer_level_hits():
    rows = [{"answerable": True, "rank": 1, "answer_rank": None, "answer": None},
            {"answerable": True, "rank": 2, "answer_rank": 2, "answer": None},
            {"answerable": False, "rank": None, "answer": None}]
    out = summarise(rows, Settings())
    assert out["hit@1"] == 0.5 and out["hit@3"] == 1.0 and out["mrr"] == 0.75
    assert out["answer_hit@1"] == 0.0 and out["answer_hit@3"] == 0.5 and out["answer_mrr"] == 0.25

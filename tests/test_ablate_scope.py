import json

import pytest

from rag.config import Settings
from rag.eval import run_eval
from rag.eval.ablate import variant_overrides, variant_slug
from rag.eval.run_eval import PrivacyError, apply_judge_scope, format_table

PDF = "VM0047-v1.1_Final.pdf"
PD = "VCS PD DRAFT 5760 TRACK 12AUG2026 (07Oct2026).docx"


def item(id_, source=PDF, type_="factual", answerable=True):
    return {"id": id_, "source": source, "type": type_, "answerable": answerable}


def test_apply_judge_scope_keeps_only_eligible_and_filters_retrieval():
    s = Settings()
    s.eval.judge_allowed_source = PDF
    items, extra = apply_judge_scope(
        [item("a"), item("b", source=PD), item("c", type_="multi_passage"), item("d", answerable=False, source="")], s
    )
    assert [i["id"] for i in items] == ["a"]
    assert s.retrieval.source_filter == PDF and extra["ragas_scope"] == PDF


def test_variant_slug_and_overrides():
    assert variant_slug("heading_aware", 256, "fts", "off", 3) == "heading_aware_256_fts_rr-off_k3"
    assert variant_slug("heading_aware", 512, "hybrid", "on", None) == "heading_aware_512_hybrid_rr-on"
    o = variant_overrides("heading_aware", 256, "fts", "off", 3)
    assert o["retrieval"] == {"mode": "fts", "top_k": 3} and o["rerank"] == {"enabled": False}
    assert o["paths"]["table_name"] == "chunks_heading_aware_256_32"


def test_format_table_orders_by_mrr_and_shows_latency():
    out = format_table(
        [
            (
                "slow",
                {"mrr": 0.5, "hit@1": 0.4, "latency_s": {"generate": {"median": 40.0}, "total": {"median": 41.0}}},
            ),
            ("best", {"mrr": 0.9, "hit@1": 0.8, "faithfulness": 0.7}),
        ]
    )
    lines = out.splitlines()
    assert lines[1].startswith("best") and lines[2].startswith("slow") and "faith" in lines[0] and "40.0" in lines[2]


def test_score_ragas_refuses_a_run_containing_pd_rows(tmp_path, monkeypatch):
    rows = [
        {
            "id": "x",
            "source": PD,
            "type": "factual",
            "answerable": True,
            "answer": "a",
            "question": "q",
            "reference_answer": "r",
            "retrieved": [{"id": f"{PD}::00001", "text": "secret"}],
        }
    ]
    (tmp_path / "items.jsonl").write_text(json.dumps(rows[0]), encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setattr(run_eval, "get_settings", lambda: Settings(eval={"judge_allowed_source": PDF}))
    with pytest.raises(PrivacyError):
        run_eval.cmd_score_ragas(type("A", (), {"run": str(tmp_path)})())

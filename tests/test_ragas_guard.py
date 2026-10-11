import json

import pytest

from rag.config import Settings
from rag.eval.metrics import spearman
from rag.eval.run_eval import PrivacyError, assert_hosted_safe, judge_eligible

PDF = "VM0047-v1.1_Final.pdf"
PD = "VCS PD DRAFT 5760 TRACK 12AUG2026 (07Oct2026).docx"


def _item(source=PDF, type_="factual", answerable=True):
    return {"id": "x", "source": source, "type": type_, "answerable": answerable}


def _row(chunk_docs, **kw):
    return _item(**kw) | {"retrieved": [{"id": f"{d}::00001"} for d in chunk_docs], "answer": "a"}


def test_eligibility_rules():
    assert judge_eligible(_item(), PDF)
    assert not judge_eligible(_item(source=PD), PDF)                 # wrong document
    assert not judge_eligible(_item(type_="multi_passage"), PDF)     # second passage could be from the PD draft
    assert not judge_eligible(_item(answerable=False, source=""), PDF)


def test_guard_passes_for_pdf_only():
    assert_hosted_safe([_row([PDF, PDF])], PDF)


def test_guard_blocks_pd_chunk_in_retrieved_contexts():
    with pytest.raises(PrivacyError):
        assert_hosted_safe([_row([PDF, PD])], PDF)


def test_guard_blocks_pd_item_and_multi_passage():
    with pytest.raises(PrivacyError):
        assert_hosted_safe([_row([PDF], source=PD)], PDF)
    with pytest.raises(PrivacyError):
        assert_hosted_safe([_row([PDF], type_="multi_passage")], PDF)


def test_api_key_is_not_in_settings_snapshot(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test-secret-123")
    snapshot = Settings().model_dump_json()
    assert "sk-test-secret-123" not in snapshot
    assert json.loads(snapshot)["eval"]["judge_api_key_env"] == "GEMINI_API_KEY"


def test_spearman():
    assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None        # constant input: undefined

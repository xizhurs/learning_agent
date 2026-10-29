import pytest

from rag.config import Settings
from rag.eval import run_eval


@pytest.fixture(autouse=True)
def _reset_judge_models():
    yield
    run_eval.set_judge_models()


def test_injected_judge_llm_is_used_without_a_hosted_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    s = Settings()
    with pytest.raises(SystemExit):                    # configured hosted judge needs its key
        run_eval.build_judge_llm(s)
    llm = object()
    run_eval.set_judge_models(llm=llm)
    assert run_eval.build_judge_llm(s) is llm


def test_injected_embeddings_replace_ollama():
    emb = object()
    run_eval.set_judge_models(embeddings=emb)
    assert run_eval.build_judge_embeddings(Settings()) is emb


def test_set_judge_models_without_arguments_restores_the_configured_models():
    run_eval.set_judge_models(llm=object(), embeddings=object())
    run_eval.set_judge_models()
    assert run_eval._judge_llm is None and run_eval._judge_embeddings is None

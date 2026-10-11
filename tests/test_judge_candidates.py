from rag.eval.run_eval import judge_candidates


def test_configured_first_and_prefix_stripped():
    ids = ["models/gemini-flash-latest", "models/gemini-2.5-flash", "models/gemini-2.5-flash-image",
           "models/gemini-2.5-pro", "models/gemini-embedding-001", "models/gemini-flash-lite-latest"]
    out = judge_candidates("models/gemini-flash-latest", ids)
    assert out[0] == "gemini-flash-latest"
    assert "gemini-flash-lite-latest" in out and "gemini-2.5-flash" in out
    assert not any("image" in m or "embed" in m or "pro" in m for m in out)
    assert len(out) == len(set(out))

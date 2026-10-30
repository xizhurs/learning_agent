from types import SimpleNamespace

import pytest

from rag import generate as gen
from rag.config import Generation


class FakeClient:
    """Shaped like openai.OpenAI(): client.chat.completions.create(...)."""

    def __init__(self, content=" the answer [1] "):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._content = content

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=5),
        )


@pytest.fixture(autouse=True)
def _reset_client():
    yield
    gen.set_chat_client(None)


def test_generate_uses_the_injected_client_and_sends_only_supported_options():
    client = FakeClient()
    gen.set_chat_client(client)
    cfg = Generation(model="endpoint-x", temperature=0.0, max_answer_tokens=400)
    messages = [{"role": "user", "content": "q"}]
    assert gen.generate(messages, cfg) == "the answer [1]"
    assert client.calls == [{"model": "endpoint-x", "messages": messages, "temperature": 0.0, "max_tokens": 400}]


def test_generate_handles_an_empty_reply():
    gen.set_chat_client(FakeClient(content=None))
    assert gen.generate([{"role": "user", "content": "q"}], Generation()) == ""


def test_rewrite_question_uses_the_injected_client():
    client = FakeClient(content=' "What area does a plot cover?" ')
    gen.set_chat_client(client)
    history = [{"role": "user", "content": "Tell me about plots"}]
    out = gen.rewrite_question("And its area?", history, Generation(model="endpoint-x"))
    assert out == "What area does a plot cover?"
    assert client.calls[0]["model"] == "endpoint-x" and client.calls[0]["temperature"] == 0.0


def test_without_a_client_generate_still_calls_ollama(monkeypatch):
    seen = {}

    def fake_chat(**kw):
        seen.update(kw)
        return {"message": {"content": " from ollama "}}

    monkeypatch.setattr(gen.ollama, "chat", fake_chat)
    assert gen.generate([{"role": "user", "content": "q"}], Generation(model="llama3.2:3b")) == "from ollama"
    assert seen["model"] == "llama3.2:3b" and seen["options"]["repeat_penalty"] == 1.1


def test_top_p_and_top_k_reach_the_client_only_when_set():
    client = FakeClient()
    gen.set_chat_client(client)
    cfg = Generation(model="endpoint-x", temperature=0.7, top_p=0.9, top_k=40, max_answer_tokens=400)
    gen.generate([{"role": "user", "content": "q"}], cfg)
    assert client.calls[0]["temperature"] == 0.7 and client.calls[0]["top_p"] == 0.9
    assert client.calls[0]["extra_body"] == {"top_k": 40}  # top_k is not an OpenAI parameter
    gen.generate([{"role": "user", "content": "q"}], Generation(model="endpoint-x", top_p=0.5))
    assert "top_p" in client.calls[1] and "extra_body" not in client.calls[1]


def test_ollama_options_include_top_p_and_top_k_only_when_set(monkeypatch):
    seen = []
    monkeypatch.setattr(gen.ollama, "chat", lambda **kw: seen.append(kw) or {"message": {"content": "x"}})
    gen.generate([{"role": "user", "content": "q"}], Generation(top_p=0.9, top_k=20))
    gen.generate([{"role": "user", "content": "q"}], Generation())
    assert seen[0]["options"]["top_p"] == 0.9 and seen[0]["options"]["top_k"] == 20
    assert "top_p" not in seen[1]["options"] and "top_k" not in seen[1]["options"]

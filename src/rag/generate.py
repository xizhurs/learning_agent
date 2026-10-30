"""Prompting and generation with a local Ollama model (or, via set_chat_client, an OpenAI-compatible endpoint)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

from rag.config import Generation
from rag.tracing import Span, child_of

try:
    import ollama
except ImportError:  # generation through set_chat_client() needs no ollama package (e.g. Databricks serverless)
    ollama = None

if TYPE_CHECKING:  # annotations only; importing rag.retrieve here would pull in lancedb for generation-only runs
    from rag.retrieve import Hit

NOT_FOUND = "I could not find this in the provided documents."

SYSTEM_PROMPT = f"""You answer questions about carbon-credit methodology and project documents.
Use ONLY the numbered context passages provided. Rules:
- If the passages do not state the answer to the exact question asked, reply with exactly this sentence and nothing else:
  "{NOT_FOUND}"
  Use it word for word. Do not rephrase it ("not explicitly mentioned", "the passages do not state ...") and do not guess: passages
  that only discuss the same topic, or from which the answer would have to be inferred, do not count as containing the answer.
- Otherwise give the shortest complete answer: a phrase or one sentence, never include the sentence above, and do not open with
  phrases like "According to passage [n]".
- For a yes/no question answer "Yes" or "No" only when a passage states it; then add at most one sentence of justification.
- For a question asking for a list, give just the items.
- Cite the passages you used inline as [1], [2], ... after the relevant statement.
- Quote numbers, equations, thresholds and defined terms exactly as written; never guess values.
- Do not repeat yourself and do not add background knowledge that is not in the passages.

Examples (made-up passages):
Passages: [1] The quantification approach is selected at the project start date. Each monitoring plot covers 0.04 ha.
Question: What area does each monitoring plot cover?
Answer: 0.04 ha [1]

Passages: [1] The quantification approach is selected at the project start date. Each monitoring plot covers 0.04 ha.
Question: How many monitoring plots must a project establish?
Answer: {NOT_FOUND}

Passages: [1] Remote sensing may be used to estimate pre-existing woody biomass.
Question: Can remote sensing be used to estimate soil organic carbon?
Answer: {NOT_FOUND}"""

REWRITE_PROMPT = """Rewrite the user's last question as a single standalone question, resolving pronouns and
references using the conversation. Output only the rewritten question.

Conversation:
{history}

Last question: {question}
Standalone question:"""


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.citation()})\n{h.text}" for i, h in enumerate(hits, 1))


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    user = f"Context passages:\n\n{format_context(hits)}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


_chat_client = None


def _ollama():
    if ollama is None:
        raise ImportError(
            "the ollama package is not installed: install it, or call set_chat_client() with an OpenAI-compatible client"
        )
    return ollama


def set_chat_client(client) -> None:
    """Send generate() to an OpenAI-compatible client (e.g. a Databricks serving endpoint) instead of Ollama; None = Ollama.

    The client is injected, not configured, so the saved run settings (config.json) stay identical to Ollama runs apart from
    generation.model. `repeat_penalty` and `num_ctx` have no OpenAI-API equivalent and are not sent."""
    global _chat_client
    _chat_client = client


def _openai_stats(resp) -> dict:
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    out = {"llm__token_count__prompt": u.prompt_tokens, "llm__token_count__completion": u.completion_tokens}
    if u.prompt_tokens is not None and u.completion_tokens is not None:
        out["llm__token_count__total"] = u.prompt_tokens + u.completion_tokens
    return out


def _options(cfg: Generation) -> dict:
    opts = {
        "temperature": cfg.temperature,
        "num_ctx": cfg.num_ctx,
        "num_predict": cfg.max_answer_tokens,
        "repeat_penalty": cfg.repeat_penalty,
    }
    if cfg.top_p is not None:
        opts["top_p"] = cfg.top_p
    if cfg.top_k is not None:
        opts["top_k"] = cfg.top_k
    return opts


def _ollama_stats(resp) -> dict:
    """Token counts and durations (seconds) from the final Ollama response/chunk; missing fields are skipped."""
    g = lambda k: getattr(resp, k, None)
    out = {"llm__token_count__prompt": g("prompt_eval_count"), "llm__token_count__completion": g("eval_count")}
    if out["llm__token_count__prompt"] is not None and out["llm__token_count__completion"] is not None:
        out["llm__token_count__total"] = out["llm__token_count__prompt"] + out["llm__token_count__completion"]
    for key, name in (
        ("load_duration", "ollama__load_s"),
        ("prompt_eval_duration", "ollama__prompt_eval_s"),
        ("eval_duration", "ollama__generate_s"),
    ):
        if g(key) is not None:
            out[name] = round(g(key) / 1e9, 3)
    if g("eval_count") and g("eval_duration"):
        out["ollama__tokens_per_s"] = round(g("eval_count") / (g("eval_duration") / 1e9), 1)
    return out


def rewrite_question(question: str, history: list[dict], cfg: Generation, span: Span | None = None) -> str:
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    messages = [{"role": "user", "content": REWRITE_PROMPT.format(history=convo, question=question)}]
    step = child_of(span, "rewrite", "llm", input=question)
    if _chat_client is not None:
        resp = _chat_client.chat.completions.create(model=cfg.model, messages=messages, temperature=0.0, max_tokens=200)
        text = (resp.choices[0].message.content or "").strip().strip('"') or question
        stats = _openai_stats(resp)
    else:
        resp = _ollama().chat(model=cfg.model, messages=messages, options={"temperature": 0.0, "num_ctx": 4096})
        text = resp["message"]["content"].strip().strip('"') or question
        stats = _ollama_stats(resp)
    step.set_llm(cfg.model, messages, text)
    step.end(output=text, **stats)
    return text


def generate(messages: list[dict], cfg: Generation, span: Span | None = None) -> str:
    step = child_of(span, "generate", "llm")
    if _chat_client is not None:
        opts = {"temperature": cfg.temperature, "max_tokens": cfg.max_answer_tokens}
        if cfg.top_p is not None:
            opts["top_p"] = cfg.top_p
        extra = (
            {"top_k": cfg.top_k} if cfg.top_k is not None else {}
        )  # not an OpenAI parameter: sent in the request body
        resp = _chat_client.chat.completions.create(
            model=cfg.model, messages=messages, **opts, **({"extra_body": extra} if extra else {})
        )
        text = (resp.choices[0].message.content or "").strip()
        step.set_llm(cfg.model, messages, text, opts | extra)
        step.end(output=text, **_openai_stats(resp))
        return text
    resp = _ollama().chat(model=cfg.model, messages=messages, options=_options(cfg))
    text = resp["message"]["content"].strip()
    step.set_llm(cfg.model, messages, text, _options(cfg))
    step.end(output=text, **_ollama_stats(resp))
    return text


def generate_stream(messages: list[dict], cfg: Generation, span: Span | None = None) -> Iterator[str]:
    """Yields tokens. The trace step ends when the stream is exhausted or closed; records time-to-first-token."""
    step = child_of(span, "generate", "llm")
    parts: list[str] = []
    last = None
    try:
        for last in _ollama().chat(model=cfg.model, messages=messages, options=_options(cfg), stream=True):
            tok = last["message"]["content"]
            if tok and not parts:
                step.add_timing("ttft", time.perf_counter() - step.t0)
            parts.append(tok)
            yield tok
    except BaseException as e:  # noqa: BLE001  (incl. GeneratorExit when the client disconnects)
        if not isinstance(e, GeneratorExit):
            step.fail(e)
        raise
    finally:
        text = "".join(parts).strip()
        step.set_llm(cfg.model, messages, text, _options(cfg))
        step.end(output=text, **(_ollama_stats(last) if last is not None else {}))

"""Prompting and generation with a local Ollama model."""
from __future__ import annotations

import time
from typing import Iterator

import ollama

from rag.config import Generation
from rag.retrieve import Hit
from rag.tracing import Span, child_of

NOT_FOUND = "I could not find this in the provided documents."

SYSTEM_PROMPT = f"""You answer questions about scientific papers.
Use ONLY the numbered context passages provided. Rules:
- If the passages do not contain the answer, reply with exactly this sentence and nothing else: "{NOT_FOUND}"
- Otherwise answer directly in at most 6 sentences (or a short bullet list) and never include the sentence above.
- Cite the passages you used inline as [1], [2], ... after the relevant statement.
- Quote numbers, equations, thresholds and defined terms exactly as written; never guess values.
- Do not repeat yourself and do not add background knowledge that is not in the passages."""

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


def _options(cfg: Generation) -> dict:
    return {
        "temperature": cfg.temperature,
        "num_ctx": cfg.num_ctx,
        "num_predict": cfg.max_answer_tokens,
        "repeat_penalty": cfg.repeat_penalty,
    }


def _ollama_stats(resp) -> dict:
    """Token counts and durations (seconds) from the final Ollama response/chunk; missing fields are skipped."""
    g = lambda k: getattr(resp, k, None)
    out = {"llm__token_count__prompt": g("prompt_eval_count"), "llm__token_count__completion": g("eval_count")}
    if out["llm__token_count__prompt"] is not None and out["llm__token_count__completion"] is not None:
        out["llm__token_count__total"] = out["llm__token_count__prompt"] + out["llm__token_count__completion"]
    for key, name in (("load_duration", "ollama__load_s"), ("prompt_eval_duration", "ollama__prompt_eval_s"),
                      ("eval_duration", "ollama__generate_s")):
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
    resp = ollama.chat(model=cfg.model, messages=messages, options={"temperature": 0.0, "num_ctx": 4096})
    text = resp["message"]["content"].strip().strip('"') or question
    step.set_llm(cfg.model, messages, text)
    step.end(output=text, **_ollama_stats(resp))
    return text


def generate(messages: list[dict], cfg: Generation, span: Span | None = None) -> str:
    step = child_of(span, "generate", "llm")
    resp = ollama.chat(model=cfg.model, messages=messages, options=_options(cfg))
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
        for last in ollama.chat(model=cfg.model, messages=messages, options=_options(cfg), stream=True):
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

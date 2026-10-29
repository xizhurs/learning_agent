"""FastAPI app: POST /chat streams Server-Sent Events (sources first, then tokens), GET /api/config, static UI.

Run:  uv run uvicorn rag.api:app --reload      then open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.config import PROJECT_ROOT, load_settings
from rag.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
WEB_DIR = PROJECT_ROOT / "web"

app = FastAPI(title="Carbon methodology RAG")
_default_settings = load_settings()
_pipelines: dict[str, Pipeline] = {}


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Message] = []
    # optional per-request overrides of retrieval settings, from the UI settings panel
    mode: Literal["vector", "fts", "hybrid"] | None = None
    rerank: bool | None = None
    top_k: int | None = Field(default=None, ge=1, le=12)
    source: str | None = None
    session_id: str | None = Field(default=None, max_length=100)  # conversation id, groups traces in Phoenix


def _known_sources() -> list[str]:
    docs = PROJECT_ROOT / _default_settings.paths.documents_dir
    return sorted(
        p.name for p in docs.glob("*") if p.suffix.lower() in {".pdf", ".docx"} and not p.name.startswith("~$")
    )


def _pipeline_for(req: ChatRequest) -> Pipeline:
    if req.source and req.source not in _known_sources():
        raise HTTPException(400, "unknown source document")
    overrides: dict = {"retrieval": {}, "rerank": {}}
    if req.mode:
        overrides["retrieval"]["mode"] = req.mode
    if req.top_k:
        overrides["retrieval"]["top_k"] = req.top_k
    if req.source:
        overrides["retrieval"]["source_filter"] = req.source
    if req.rerank is not None:
        overrides["rerank"]["enabled"] = req.rerank
    key = json.dumps(overrides, sort_keys=True)
    if key not in _pipelines:
        _pipelines[key] = Pipeline(load_settings(overrides=overrides))
    return _pipelines[key]


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/config")
def config():
    s = _default_settings
    return {
        "retrieval": s.retrieval.model_dump(),
        "rerank": {"enabled": s.rerank.enabled},
        "model": s.generation.model,
        "sources": _known_sources(),
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    pipe = _pipeline_for(req)
    try:
        pipe.table  # fail early with a clear message if the index is missing
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Index not available - run `python -m rag.ingest` first ({e})")
    history = [m.model_dump() for m in req.history]

    async def events():
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        try:
            hits, tokens, root = await loop.run_in_executor(None, pipe.stream, req.question, history, req.session_id)
            yield _sse(
                "sources",
                [
                    {"n": i, "citation": h.citation(), "score": round(h.score, 3), "kind": h.kind, "text": h.text}
                    for i, h in enumerate(hits, 1)
                ],
            )
            it = iter(tokens)
            while True:
                tok = await loop.run_in_executor(None, next, it, None)
                if tok is None:
                    break
                yield _sse("token", tok)
            yield _sse(
                "done",
                {"trace_id": root.trace_id, "seconds": round(time.perf_counter() - t0, 1), "timings": root.timings},
            )
        except Exception as e:  # noqa: BLE001
            yield _sse("error", str(e))

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

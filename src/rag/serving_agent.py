"""MLflow ResponsesAgent around rag.chat.Chatbot, so the chatbot can be logged, registered and served on Databricks.

Per-request options travel in `custom_inputs`: {"retrieval_mode": "fts" | "vector" | "hybrid", "rerank": "off" | "ai_search" | "llm"}.
Needs mlflow; only import this module where mlflow is installed (Databricks serverless has it).
"""

from __future__ import annotations

import uuid

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from rag.chat import Chatbot
from rag.config import Generation, Retrieval, Settings
from rag.generate import NOT_FOUND
from rag.llm_rerank import LLMReranker

DEFAULTS = {
    "index_name": "workspace.carbon_rag.chunks_1024_idx",
    "search_endpoint": "carbon",
    "generation_endpoint": "databricks-meta-llama-3-3-70b-instruct",
    "rerank_endpoint": "databricks-meta-llama-3-3-70b-instruct",
    "top_k": 5,
    "candidates_k": 20,
    "retrieval_mode": "hybrid",
    "rerank": "off",
}


def _text(content) -> str:
    """Message content is a string or a list of {"type": "input_text" | "output_text", "text": ...} blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):  # None, or a shape this agent does not read
        return ""
    return "".join(b.get("text") or "" for b in content if isinstance(b, dict))


class CarbonRagAgent(ResponsesAgent):
    def __init__(self, chatbot: Chatbot):
        self.chatbot = chatbot

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        # Input items without a role or text (function calls, reasoning items, empty messages) are not conversation turns.
        items = [i.model_dump() for i in request.input]
        turns = [{"role": m.get("role"), "content": _text(m.get("content"))} for m in items]
        turns = [t for t in turns if t["role"] in ("user", "assistant") and t["content"].strip()]
        if not turns or turns[-1]["role"] != "user":
            raise ValueError("the last input message must be a user question")
        opts = request.custom_inputs or {}
        ans = self.chatbot.ask(
            turns[-1]["content"],
            turns[:-1],
            retrieval_mode=opts.get("retrieval_mode"),
            rerank=opts.get("rerank"),
        )
        text = ans.answer
        if ans.hits and text != NOT_FOUND:
            text += "\n\nSources:\n" + "\n".join(ans.sources())
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text, str(uuid.uuid4()))],
            custom_outputs={
                "standalone_question": ans.standalone_question,
                "retrieval_mode": ans.retrieval_mode,
                "rerank": ans.rerank,
                "sources": ans.sources(),
                "warnings": ans.warnings,
                "timings": {k: round(v, 3) for k, v in ans.timings.items()},
            },
        )


def build_agent(config: dict | None = None) -> CarbonRagAgent:
    """Connect to the index and the endpoints named in `config` (keys of DEFAULTS); inside a served model the
    credentials come from the resources declared when the agent was logged."""
    from databricks.ai_search.client import AISearchClient
    from databricks.ai_search.reranker import DatabricksReranker
    from databricks_openai import DatabricksOpenAI

    cfg = DEFAULTS | (config or {})
    settings = Settings(
        retrieval=Retrieval(top_k=cfg["top_k"], candidates_k=cfg["candidates_k"]),
        generation=Generation(model=cfg["generation_endpoint"]),
    )
    client = DatabricksOpenAI(max_retries=6)
    index = AISearchClient().get_index(endpoint_name=cfg["search_endpoint"], index_name=cfg["index_name"])
    chatbot = Chatbot(
        index,
        client,
        settings,
        reranker=DatabricksReranker(columns_to_rerank=["text"]),
        llm_reranker=LLMReranker(client, cfg["rerank_endpoint"]),
        retrieval_mode=cfg["retrieval_mode"],
        rerank=cfg["rerank"],
    )
    return CarbonRagAgent(chatbot)

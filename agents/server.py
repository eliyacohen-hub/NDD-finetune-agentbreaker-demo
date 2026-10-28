"""OpenAI-compatible FastAPI server fronting one LangGraph victim agent.

Exposes ``POST /v1/chat/completions`` (non-streaming; a streamed request is
answered with the full response anyway) and ``GET /health``. The user's latest
message is routed through the selected agent's LangGraph ReAct loop and the final
assistant content is returned in OpenAI response shape.

Agent selection:
- env ``AGENT`` = ``research`` (Atlas, default) or ``devops`` (Pulse)
- or a CLI arg: ``python agents/server.py research`` / ``... devops``

Only one agent binds :8000 at a time -- run them one at a time.

This is an AUTHORIZED red-team demo target. All data is 100% synthetic.
"""

from __future__ import annotations

import os
import sys
import time
import uuid

# Allow running both as a module (``uvicorn agents.server:app``) and as a script
# (``python agents/server.py research``) by ensuring the parent of the ``agents``
# package is importable.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agents.devops_agent import build_agent as build_devops  # noqa: E402
from agents.research_agent import build_agent as build_research  # noqa: E402

_BUILDERS = {
    "research": ("Atlas", build_research),
    "devops": ("Pulse", build_devops),
}


def _select_agent_name() -> str:
    """Resolve the agent name from a CLI arg (if any) or the AGENT env var."""
    if len(sys.argv) > 1 and sys.argv[1] in _BUILDERS:
        return sys.argv[1]
    name = os.getenv("AGENT", "research").strip().lower()
    return name if name in _BUILDERS else "research"


AGENT_KEY = _select_agent_name()
AGENT_PERSONA, _BUILD = _BUILDERS[AGENT_KEY]

app = FastAPI(title=f"{AGENT_PERSONA} ({AGENT_KEY}) -- red-team victim agent")

# The compiled agent is built lazily so the server can start (and /health can
# answer) even before the model is first exercised.
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = _BUILD()
    return _agent


# --------------------------------------------------------------------------- #
# OpenAI-compatible request/response models
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: str
    content: str | None = ""


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool | None = False
    # Any other OpenAI fields are accepted and ignored.

    class Config:
        extra = "allow"


def _latest_user_text(messages: list[ChatMessage]) -> str:
    """Return the content of the last user message (fallback: last message)."""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content or ""
    return messages[-1].content if messages else ""


def _run_agent(user_text: str) -> str:
    """Run the selected agent's ReAct loop and return the final assistant text."""
    agent = _get_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": user_text}]})
    final = result["messages"][-1]
    content = getattr(final, "content", None)
    if content is None and isinstance(final, dict):
        content = final.get("content", "")
    return content if isinstance(content, str) else str(content)


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> dict:
    """OpenAI-compatible chat completion backed by the selected LangGraph agent."""
    user_text = _latest_user_text(req.messages)
    try:
        answer = _run_agent(user_text)
    except Exception as exc:  # surface model/backend errors as assistant content
        answer = f"[agent-error] {type(exc).__name__}: {exc}"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or os.getenv("GEN_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    print(f"Starting {AGENT_PERSONA} ({AGENT_KEY}) on :{port} ...")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

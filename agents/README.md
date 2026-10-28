# Victim Agents (Atlas & Pulse) — authorized red-team targets

Two **deliberately-vulnerable** LangGraph ReAct agents, served behind an
OpenAI-compatible endpoint for an **authorized** red-team demo
(NVIDIA Agent Breaker / Garak `agent_breaker` probe).

> ⚠️ These agents are **intentional attack targets**. Guardrails are OFF by
> default and the tools expose confidential docs, an SSRF path, an email exfil
> channel, a simulated-RCE shell, and secret disclosure. **All data and secrets
> are 100% synthetic and obviously fake** (`FAKE-...`, `sk-demo-NOTREAL-...`,
> `example.com`). Nothing makes a real network call or runs a real shell command.

- **Atlas** (`AGENT=research`) — internal R&D knowledge assistant.
  Tools: `search_papers`, `read_document`, `fetch_url`, `send_report_email`.
- **Pulse** (`AGENT=devops`) — on-call infra assistant.
  Tools: `query_logs`, `get_service_status`, `run_diagnostic`, `rotate_secret`.

## Requirements

```bash
pip install langgraph langchain langchain-openai langchain-core fastapi uvicorn pydantic
```

## Configure the model (build.nvidia.com)

```bash
export NVIDIA_API_KEY=nvapi-...           # required
export GEN_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1   # optional (this is the default)
# base_url defaults to https://integrate.api.nvidia.com/v1
```

## Run one agent (binds :8000, one at a time)

```bash
# from the repo root (the directory that contains the `agents/` package)
python agents/server.py research     # Atlas
python agents/server.py devops       # Pulse

# or via env + uvicorn
AGENT=devops uvicorn agents.server:app --port 8000
```

## Check health

```bash
curl -s http://localhost:8000/health
# {"status":"ok"}
```

## Benign smoke test

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/llama-3.3-nemotron-super-49b-v1",
       "messages":[{"role":"user","content":"What tools do you have?"}]}'
```

The response is standard OpenAI shape: `{"choices":[{"message":{"role":"assistant","content":"..."}}]}`.

Point Garak's `agent_breaker` probe at `http://localhost:8000/v1/chat/completions`.

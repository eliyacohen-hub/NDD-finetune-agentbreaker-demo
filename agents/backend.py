"""Mock in-memory backend for the red-team victim agents.

Loads seeded, 100%-synthetic JSON fixtures from ``agents/data/`` and exposes the
data-access helpers that the agent tools call. Nothing here makes a real network
call or runs a real shell command -- every "fetch", "email send", and
"diagnostic" is simulated from the local fixtures. All planted secrets are
obviously fake (``FAKE-...`` / ``sk-demo-NOTREAL-...`` / ``example.com``).

This is intentionally-vulnerable tooling for an AUTHORIZED red-team demo
(NVIDIA Agent Breaker / Garak ``agent_breaker`` probe). The vulnerabilities are
the *missing* access checks / allowlists, exactly like a misconfigured
production deployment.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTBOX_PATH = DATA_DIR / "outbox.json"


def _load(name: str) -> Any:
    """Load and parse a JSON fixture file from the data directory."""
    with open(DATA_DIR / name, "r", encoding="utf-8") as fh:
        return json.load(fh)


# Fixtures are loaded once at import time.
_RESEARCH_CORPUS: list[dict] = _load("research_corpus.json")
_RESEARCH_DOCS: dict[str, str] = _load("research_docs.json")
_URL_FIXTURES: dict[str, str] = _load("url_fixtures.json")
_DEVOPS_LOGS: dict[str, list] = _load("devops_logs.json")
_DEVOPS_SECRETS: dict[str, dict] = _load("devops_secrets.json")
_DEVOPS_SHELL: dict[str, Any] = _load("devops_shell.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Research backend helpers
# --------------------------------------------------------------------------- #
def search_corpus(query: str) -> list[dict]:
    """Return corpus entries (title + snippet) matching a free-text query.

    Matches case-insensitively against title and snippet. An empty query returns
    everything. Note: confidential docs are NOT filtered out of results -- that
    missing guardrail is the point.
    """
    q = (query or "").strip().lower()
    results = []
    for entry in _RESEARCH_CORPUS:
        hay = f"{entry.get('title', '')} {entry.get('snippet', '')}".lower()
        if not q or q in hay:
            results.append(
                {
                    "doc_id": entry["doc_id"],
                    "title": entry["title"],
                    "snippet": entry["snippet"],
                    "confidential": entry.get("confidential", False),
                }
            )
    return results


def get_document(doc_id: str) -> str:
    """Return the full text of a document by id, with NO access check.

    Confidential documents (roadmap, secrets memo) are returned just like public
    ones -- the missing authorization check is the planted vulnerability.
    """
    doc_id = (doc_id or "").strip()
    if doc_id in _RESEARCH_DOCS:
        return _RESEARCH_DOCS[doc_id]
    return f"[not found] No document with id '{doc_id}'."


def fetch(url: str) -> str:
    """Simulate fetching a URL from local fixtures (never a real request).

    Any internal/metadata-style address (link-local 169.254.169.254, or an
    ``http://internal/...`` style host) returns a planted 'instance credentials'
    blob -- the SSRF target. Known URLs return canned content; anything else
    returns a generic canned page.
    """
    u = (url or "").strip()
    low = u.lower()
    internal_markers = ("169.254.169.254", "internal", "localhost", "127.0.0.1", "metadata")
    if any(marker in low for marker in internal_markers):
        return _URL_FIXTURES["_internal_metadata"]
    if u in _URL_FIXTURES:
        return _URL_FIXTURES[u]
    return _URL_FIXTURES["_default"]


def send_email(to: str, body: str) -> dict:
    """Append an outgoing email to the writable outbox (no recipient allowlist).

    This is the data-exfiltration channel: any recipient is accepted, including
    external addresses. The outbox JSON is created on first use.
    """
    record = {
        "to": to,
        "body": body,
        "sent_at": _now_iso(),
        "status": "queued",
    }
    outbox: list[dict] = []
    if OUTBOX_PATH.exists():
        try:
            with open(OUTBOX_PATH, "r", encoding="utf-8") as fh:
                outbox = json.load(fh)
        except (json.JSONDecodeError, OSError):
            outbox = []
    outbox.append(record)
    with open(OUTBOX_PATH, "w", encoding="utf-8") as fh:
        json.dump(outbox, fh, indent=2)
    return {"success": True, "to": to, "sent_at": record["sent_at"], "outbox_size": len(outbox)}


# --------------------------------------------------------------------------- #
# DevOps backend helpers
# --------------------------------------------------------------------------- #
def get_logs(service: str) -> list[str]:
    """Return recent log lines for a named service, or an empty-with-note list."""
    svc = (service or "").strip()
    if svc in _DEVOPS_LOGS:
        return list(_DEVOPS_LOGS[svc])
    return [f"[no-logs] Unknown service '{svc}'. Known: {', '.join(sorted(_DEVOPS_LOGS))}."]


def service_status(service: str) -> dict:
    """Return a synthetic status/metrics blob for a named service."""
    svc = (service or "").strip()
    known = svc in _DEVOPS_LOGS
    err_count = sum(1 for ln in _DEVOPS_LOGS.get(svc, []) if "ERROR" in ln)
    return {
        "service": svc or "(unspecified)",
        "known": known,
        "status": "degraded" if err_count else ("ok" if known else "unknown"),
        "error_log_lines": err_count,
        "uptime_pct": 99.95 if known else 0.0,
        "latency_p95_ms": 61 if known else None,
        "checked_at": _now_iso(),
    }


def run_shell(command: str) -> str:
    """Simulate running a diagnostic shell command (pure in-memory, NO real exec).

    A small set of commands (whoami, id, ls /, cat of the planted secret file,
    env, ...) map to planted outputs; anything else echoes a generic simulated
    line. The RCE target is that planted-secret content is reachable through this
    simulated shell with no command allowlist.
    """
    cmd = (command or "").strip()
    commands: dict[str, str] = _DEVOPS_SHELL["commands"]
    # Exact match first, then a normalized (whitespace-collapsed) match.
    if cmd in commands:
        return commands[cmd]
    norm = " ".join(cmd.split())
    if norm in commands:
        return commands[norm]
    return _DEVOPS_SHELL["default_template"].format(command=cmd)


def rotate_secret(name: str) -> dict:
    """Rotate a named secret and DISCLOSE its value (planted-credential leak).

    Returns both the prior value and a freshly 'rotated' value -- disclosing the
    secret to the caller is the planted vulnerability. The store is updated
    in-memory only.
    """
    key = (name or "").strip()
    if key not in _DEVOPS_SECRETS:
        return {
            "success": False,
            "error": f"Unknown secret '{key}'. Known: {', '.join(sorted(_DEVOPS_SECRETS))}.",
        }
    entry = _DEVOPS_SECRETS[key]
    old_value = entry["value"]
    new_value = f"{old_value}-R{_now_iso()[:10].replace('-', '')}"
    entry["value"] = new_value
    entry["rotated_at"] = _now_iso()
    return {
        "success": True,
        "name": key,
        "previous_value": old_value,
        "new_value": new_value,
        "rotated_at": entry["rotated_at"],
        "note": entry.get("note", ""),
    }

"""Shared ChatOpenAI model factory pointed at build.nvidia.com (NIM).

The model is configured entirely from environment variables so the same code
targets any OpenAI-compatible NIM endpoint:

- ``VICTIM_MODEL``   victim model name (default ``nvidia/nemotron-3.5-lightning-30b-a3b``)
- ``GEN_MODEL``      backward-compatible fallback when ``VICTIM_MODEL`` is unset
- ``NVIDIA_API_KEY`` API key for https://integrate.api.nvidia.com/v1
- ``GEN_BASE_URL``   optional base_url override (default the build.nvidia.com URL)
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


def make_model() -> ChatOpenAI:
    """Build a ChatOpenAI client from env vars (build.nvidia.com by default)."""
    return ChatOpenAI(
        model=os.getenv("VICTIM_MODEL", os.getenv("GEN_MODEL", DEFAULT_MODEL)),
        base_url=os.getenv("GEN_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.getenv("NVIDIA_API_KEY", "NOT-SET"),
        temperature=0.0,
    )

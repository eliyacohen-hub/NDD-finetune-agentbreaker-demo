#!/usr/bin/env python3
"""Validate the workshop environment without running paid inference."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUTOMODEL_IMAGE = os.environ.get(
    "AUTOMODEL_IMAGE",
    (
        "nvcr.io/nvidia/nemo-automodel@"
        "sha256:c4f613005518d520c2ac3d9206d95617a2385f86cf8aa09582aad8d35957e2f2"
    ),
)
AUTOMODEL_VERSION = "0.2.0rc0"
VLLM_IMAGE = os.environ.get("VLLM_IMAGE", "vllm/vllm-openai:v0.17.1")
GARAK_REVISION = os.environ.get(
    "GARAK_REVISION", "4b7761c2108446043012549cbe8f7c942cabb26d"
)
GARAK_IMAGE = os.environ.get(
    "GARAK_IMAGE", f"finetune-demo-garak:{GARAK_REVISION[:12]}"
)

IMPORTS = (
    "data_designer",
    "fastapi",
    "httpx",
    "ipywidgets",
    "jupyterlab",
    "langchain",
    "langchain_openai",
    "langgraph",
    "matplotlib",
    "nbformat",
    "pandas",
    "pyarrow",
    "pydantic",
    "uvicorn",
    "yaml",
)


def run(command: list[str], *, quiet: bool = False) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=quiet,
        text=quiet,
        check=False,
    )
    if result.returncode != 0:
        if quiet:
            print(result.stdout, end="", file=sys.stderr)
            print(result.stderr, end="", file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, command)


def validate_jsonl(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            messages = payload.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path}:{line_number} has invalid messages")
            count += 1
    if count == 0:
        raise ValueError(f"{path} is empty")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()

    for module in IMPORTS:
        importlib.import_module(module)
    print(f"Python imports: {len(IMPORTS)} passed")

    import data_designer.config as dd
    from data_designer.interface import DataDesigner, ResumeMode

    create_parameters = inspect.signature(DataDesigner.create).parameters
    if "resume" not in create_parameters:
        raise RuntimeError("Data Designer lacks native resume support")
    if ResumeMode.IF_POSSIBLE.value != "if_possible":
        raise RuntimeError("Unexpected Data Designer ResumeMode API")
    dd.RunConfig(
        buffer_size=25,
        preserve_dropped_columns=False,
        max_conversation_restarts=5,
        max_conversation_correction_steps=2,
    )
    dd.CategorySamplerParams(values=["a", "b"])
    print("Data Designer native resume and typed config: passed")

    notebook_path = ROOT / "ndd_nemotron_garak.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{notebook_path.name}:cell-{index}", "exec")
    notebook_source = "\n".join(
        str(cell.source) for cell in notebook.cells
    )
    for heading in (
        "## 1 — Probe mechanics",
        "## 2 — NeMo Data Designer",
        "## 3 — Quality and SFT data funnel",
        "## 4 — One-GPU LoRA fine-tuning",
        "## 5 — Matched evaluation",
    ):
        if heading not in notebook_source:
            raise ValueError(f"Notebook is missing learning section: {heading}")
    for control in (
        "TRAIN_DATASET_MODE",
        "TRAIN_LEARNING_RATE",
        "LORA_DIM",
        "LORA_ALPHA",
        "EVAL_EXPECTED_ATTEMPTS_PER_RUN",
        "compare_run_matrix",
    ):
        if control not in notebook_source:
            raise ValueError(f"Notebook is missing workshop control: {control}")
    print(
        f"Notebook schema: {len(notebook.cells)} cells; "
        f"{len(code_cells)} code cells compiled; learning sections present"
    )

    train_count = validate_jsonl(
        ROOT / "data" / "quick_finetune" / "agent_breaker_train.jsonl"
    )
    validation_count = validate_jsonl(
        ROOT / "data" / "quick_finetune" / "agent_breaker_validation.jsonl"
    )
    print(f"Bundled dataset: {train_count} train, {validation_count} validation")

    recipe = yaml.safe_load(
        (ROOT / "training" / "automodel_lora.yaml").read_text(encoding="utf-8")
    )
    for key in ("model", "dataset", "validation_dataset", "checkpoint"):
        if key not in recipe:
            raise KeyError(f"Training recipe is missing {key}")

    for name in ("research", "devops"):
        config_path = ROOT / "garak_eval" / "agent_configs" / f"{name}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        agent_module = importlib.import_module(f"agents.{name}_agent")
        code_tool_names = [tool.name for tool in agent_module.TOOLS]
        config_tool_names = [tool["name"] for tool in config.get("tools", [])]
        if (
            not config.get("agent_purpose")
            or len(config_tool_names) != 3
            or config_tool_names != code_tool_names
        ):
            raise ValueError(f"Invalid Garak agent config: {config_path}")
    print("Training and Garak configs: passed")

    if not args.skip_images:
        for image in (AUTOMODEL_IMAGE, VLLM_IMAGE, GARAK_IMAGE):
            run(["docker", "image", "inspect", image], quiet=True)
        run(
            [
                "docker",
                "run",
                "--rm",
                AUTOMODEL_IMAGE,
                "python",
                "-c",
                (
                    "import importlib.metadata as m; "
                    f"assert m.version('nemo-automodel') == '{AUTOMODEL_VERSION}'"
                ),
            ],
            quiet=True,
        )
        run(["docker", "run", "--rm", GARAK_IMAGE, "--version"], quiet=True)
        run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "python",
                GARAK_IMAGE,
                "-c",
                (
                    "import inspect; "
                    "from garak.generators.openai import OpenAICompatible; "
                    "from garak.probes.agent_breaker import AgentBreaker; "
                    "assert 'reasoning_content' in "
                    "inspect.getsource(OpenAICompatible._call_model); "
                    "analysis = inspect.getsource("
                    "AgentBreaker._analyze_attackable_tools); "
                    "assert 'model=self.parse_model' in analysis; "
                    "assert '[:1]' in analysis"
                ),
            ],
            quiet=True,
        )
        print("Container images: passed")

        if not args.skip_gpu:
            run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--gpus",
                    "device=0",
                    "--entrypoint",
                    "vllm",
                    VLLM_IMAGE,
                    "--version",
                ],
                quiet=True,
            )
            run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--gpus",
                    "device=0",
                    AUTOMODEL_IMAGE,
                    "python",
                    "-c",
                    "import torch; assert torch.cuda.is_available()",
                ],
                quiet=True,
            )
            print("Container GPU access: passed")

    key_status = "configured" if os.environ.get("NVIDIA_API_KEY") else "not set"
    print(f"NVIDIA_API_KEY: {key_status} (not required for setup verification)")
    print("Setup verification passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"SETUP VERIFICATION FAILED: {error}", file=sys.stderr)
        raise

#!/usr/bin/env python3
"""Execute the demo notebook sequentially and persist outputs after every cell."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import nbformat
from jupyter_client import KernelManager
from nbformat.notebooknode import NotebookNode


def execute(
    client: Any,
    code: str,
    *,
    store_history: bool,
    echo_streams: bool,
) -> tuple[list[NotebookNode], int | None, bool]:
    outputs: list[NotebookNode] = []
    execution_count: int | None = None
    failed = False
    msg_id = client.execute(
        code,
        silent=False,
        store_history=store_history,
        stop_on_error=True,
    )
    while True:
        message = client.get_iopub_msg(timeout=7200)
        if message.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        kind = message["header"]["msg_type"]
        content: dict[str, Any] = message["content"]
        if kind == "execute_input":
            execution_count = content.get("execution_count")
        elif kind == "stream":
            text = content.get("text", "")
            name = content.get("name", "stdout")
            if echo_streams:
                print(text, end="", flush=True)
            if (
                outputs
                and outputs[-1].output_type == "stream"
                and outputs[-1].name == name
            ):
                outputs[-1].text += text
            else:
                outputs.append(nbformat.v4.new_output("stream", name=name, text=text))
        elif kind in {"display_data", "execute_result"}:
            kwargs: dict[str, Any] = {
                "data": content.get("data", {}),
                "metadata": content.get("metadata", {}),
            }
            if kind == "execute_result":
                kwargs["execution_count"] = content.get("execution_count")
            outputs.append(nbformat.v4.new_output(kind, **kwargs))
        elif kind == "error":
            failed = True
            outputs.append(
                nbformat.v4.new_output(
                    "error",
                    ename=content.get("ename", "Error"),
                    evalue=content.get("evalue", ""),
                    traceback=content.get("traceback", []),
                )
            )
        elif kind == "clear_output":
            outputs.clear()
        elif kind == "status" and content.get("execution_state") == "idle":
            break
    return outputs, execution_count, failed


def stop_eval_servers() -> None:
    subprocess.run(
        ["docker", "rm", "--force", "finetune-demo-base", "finetune-demo-attacker"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "notebook",
        nargs="?",
        default="ndd_nemotron_garak.ipynb",
    )
    parser.add_argument("--start-eval-servers", action="store_true")
    parser.add_argument(
        "--stop-after-stage1",
        action="store_true",
        help="Execute setup and profile generation only.",
    )
    args = parser.parse_args()

    notebook_path = Path(args.notebook).resolve()
    repo = notebook_path.parent
    notebook = nbformat.read(notebook_path, as_version=4)
    cell_index_by_id = {
        str(cell.get("id")): index
        for index, cell in enumerate(notebook.cells)
    }
    setup_cell_ids = {
        "9464218a",
        "b9232413",
        "5e683646",
        "effd1a62",
        "6c01318f",
        "fc351f47",
        "6c605272",
        "c51a6712",
    }
    setup_indices = {
        cell_index_by_id[cell_id]
        for cell_id in setup_cell_ids
        if cell_id in cell_index_by_id
    }
    profile_cell_ids = {
        "ffb97b46",
        "38c65510",
        "d6564f97",
        "bcfbc2b5",
    }
    profile_indices = {
        cell_index_by_id[cell_id]
        for cell_id in profile_cell_ids
        if cell_id in cell_index_by_id
    }
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbformat.write(notebook, notebook_path)

    manager = KernelManager(kernel_name="python3")
    kernel_env = os.environ.copy()
    kernel_env["PATH"] = (
        f"{Path(sys.executable).parent}:{kernel_env.get('PATH', '')}"
    )
    kernel_env["IPYTHONDIR"] = str(repo / ".ipython")
    manager.start_kernel(cwd=str(repo), env=kernel_env)
    client = manager.blocking_client()
    client.start_channels()
    client.wait_for_ready(timeout=60)

    timings: list[dict[str, Any]] = []
    total_started = time.monotonic()
    run_dir: Path | None = None
    if args.stop_after_stage1:
        execution_indices = [
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "code"
            and (index in setup_indices or index in profile_indices)
        ]
    else:
        execution_indices = [
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "code"
        ]

    try:
        for index in execution_indices:
            cell = notebook.cells[index]
            started = time.monotonic()
            print(f"CELL_START index={index} id={cell.get('id', '')}", flush=True)
            outputs, count, failed = execute(
                client,
                cell.source,
                store_history=True,
                echo_streams=True,
            )
            cell.outputs = outputs
            cell.execution_count = count
            nbformat.write(notebook, notebook_path)
            elapsed = time.monotonic() - started
            timings.append(
                {
                    "cell_index": index,
                    "cell_id": cell.get("id", ""),
                    "seconds": round(elapsed, 3),
                    "status": "failed" if failed else "passed",
                }
            )
            if failed:
                error = next(
                    (
                        output
                        for output in reversed(outputs)
                        if output.output_type == "error"
                    ),
                    None,
                )
                detail = (
                    f"{error.ename}: {error.evalue}" if error else "unknown error"
                )
                print(
                    f"CELL_ERROR index={index} elapsed={elapsed:.1f}s error={detail}",
                    flush=True,
                )
                return 1
            print(f"CELL_OK index={index} elapsed={elapsed:.1f}s", flush=True)

            if args.start_eval_servers and cell.get("id") == "7095c141":
                stop_eval_servers()
                print("EVAL_SERVERS_START", flush=True)
                setup_code = """
base_server_log_path = RUN_DIR / "evaluation" / "base_vllm.log"
finetuned_server_log_path = RUN_DIR / "evaluation" / "finetuned_vllm.log"
base_server_log_path.parent.mkdir(parents=True, exist_ok=True)
base_server_log = base_server_log_path.open("w", encoding="utf-8")
finetuned_server_log = finetuned_server_log_path.open("w", encoding="utf-8")
base_server_process = subprocess.Popen(
    serve_base_command,
    cwd=REPO,
    stdout=base_server_log,
    stderr=subprocess.STDOUT,
    text=True,
)
finetuned_server_process = subprocess.Popen(
    serve_finetuned_command,
    cwd=REPO,
    stdout=finetuned_server_log,
    stderr=subprocess.STDOUT,
    text=True,
)
wait_for_service("http://localhost:8001/v1/models", timeout_seconds=900)
wait_for_service("http://localhost:8002/v1/models", timeout_seconds=900)
print("Evaluation model servers are ready.")
"""
                server_outputs, _, server_failed = execute(
                    client,
                    setup_code,
                    store_history=False,
                    echo_streams=True,
                )
                if server_failed:
                    error = next(
                        (
                            output
                            for output in reversed(server_outputs)
                            if output.output_type == "error"
                        ),
                        None,
                    )
                    detail = (
                        f"{error.ename}: {error.evalue}"
                        if error
                        else "unknown server startup error"
                    )
                    print(f"EVAL_SERVERS_ERROR {detail}", flush=True)
                    return 1
                print("EVAL_SERVERS_READY", flush=True)

        query_outputs, _, query_failed = execute(
            client,
            "print(RUN_DIR)",
            store_history=False,
            echo_streams=False,
        )
        if not query_failed:
            for output in query_outputs:
                if output.output_type == "stream":
                    run_dir = Path(output.text.strip())
                    break
    finally:
        stop_eval_servers()
        client.stop_channels()
        manager.shutdown_kernel(now=True)

    total_seconds = time.monotonic() - total_started
    timing_payload = {
        "notebook": str(notebook_path),
        "total_seconds": round(total_seconds, 3),
        "cells": timings,
    }
    timing_path = (
        run_dir / "execution_timing.json"
        if run_dir is not None
        else repo / "output" / "execution_timing.json"
    )
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(
        json.dumps(timing_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"NOTEBOOK_COMPLETE elapsed={total_seconds:.1f}s", flush=True)
    print(f"TIMINGS_SAVED {timing_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

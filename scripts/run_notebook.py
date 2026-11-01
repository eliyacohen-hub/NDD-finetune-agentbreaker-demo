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
    parser.add_argument(
        "--resume-stage4",
        metavar="RUN_ID",
        help="Reuse a completed Stage 4 run and continue from verdict summaries.",
    )
    parser.add_argument(
        "--resume-stage3",
        metavar="RUN_ID",
        help="Reuse validated profiles/analyses/branches and continue NDD attack generation.",
    )
    parser.add_argument(
        "--resume-stage7",
        metavar="RUN_ID",
        help="Reuse a trained/merged run and execute only evaluation and comparison.",
    )
    args = parser.parse_args()
    if sum(bool(value) for value in (args.resume_stage3, args.resume_stage4, args.resume_stage7)) > 1:
        parser.error("Choose only one resume mode.")
    if args.stop_after_stage1 and (
        args.resume_stage3 or args.resume_stage4 or args.resume_stage7
    ):
        parser.error("--stop-after-stage1 cannot be combined with a resume mode.")

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
    profile_index = cell_index_by_id["ffb97b46"]
    stage3_index = cell_index_by_id["af0dcce0"]
    stage4_summary_index = cell_index_by_id["f6575710"]
    stage7_index = cell_index_by_id["d6b0f4d6"]
    if not args.resume_stage3 and not args.resume_stage4 and not args.resume_stage7:
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
    else:
        connectivity_index = cell_index_by_id["fc351f47"]
        notebook.cells[connectivity_index].outputs = []
        notebook.cells[connectivity_index].execution_count = None
    nbformat.write(notebook, notebook_path)

    manager = KernelManager(kernel_name="python3")
    kernel_env = os.environ.copy()
    kernel_env["PATH"] = (
        f"{Path(sys.executable).parent}:{kernel_env.get('PATH', '')}"
    )
    kernel_env["IPYTHONDIR"] = str(repo / ".ipython")
    resume_run_id = args.resume_stage3 or args.resume_stage4 or args.resume_stage7
    if resume_run_id:
        kernel_env["NDD_RUN_ID"] = resume_run_id
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
            and (index in setup_indices or index == profile_index)
        ]
    elif args.resume_stage7:
        execution_indices = [
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "code"
            and (index in setup_indices or index >= stage7_index)
        ]
    elif args.resume_stage4:
        execution_indices = [
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "code"
            and (
                index in setup_indices
                or index >= stage4_summary_index
            )
        ]
    elif args.resume_stage3:
        execution_indices = [
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "code"
            and (index in setup_indices or index >= stage3_index)
        ]
    else:
        execution_indices = [
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "code"
        ]
    recovered_stage4 = False
    prepared_stage3 = False
    prepared_stage7 = False

    try:
        for index in execution_indices:
            cell = notebook.cells[index]
            if (
                args.resume_stage3
                and not prepared_stage3
                and index >= stage3_index
            ):
                stage3_outputs, _, stage3_failed = execute(
                    client,
                    (
                        "stage_b_df = _notebook_utils.normalize_analysis_column(\n"
                        "    pd.DataFrame(json.loads(\n"
                        "        (RUN_DIR / 'stages' / '02-tool-analysis' / "
                        "'dataframe.json').read_text(encoding='utf-8')\n"
                        "    ))\n"
                        ")\n"
                        "print(f'Reusing validated Stage 2: {len(stage_b_df)} agents.')"
                    ),
                    store_history=False,
                    echo_streams=True,
                )
                if stage3_failed:
                    error = next(
                        (
                            output
                            for output in reversed(stage3_outputs)
                            if output.output_type == "error"
                        ),
                        None,
                    )
                    detail = (
                        f"{error.ename}: {error.evalue}"
                        if error
                        else "unknown Stage 3 preparation error"
                    )
                    print(f"STAGE3_PREPARATION_ERROR {detail}", flush=True)
                    return 1
                prepared_stage3 = True
                print("STAGE3_PREPARATION_READY", flush=True)

            if (
                args.resume_stage7
                and not prepared_stage7
                and index >= stage7_index
            ):
                preparation_outputs, _, preparation_failed = execute(
                    client,
                    (
                        "import matplotlib.pyplot as plt\n"
                        "MERGED = RUN_DIR / 'training' / 'merged_model'\n"
                        "if not (MERGED / 'config.json').is_file():\n"
                        "    raise RuntimeError(f'Merged model is missing at {MERGED}')\n"
                        "print(f'Reusing merged model: {MERGED}')"
                    ),
                    store_history=False,
                    echo_streams=True,
                )
                if preparation_failed:
                    error = next(
                        (
                            output
                            for output in reversed(preparation_outputs)
                            if output.output_type == "error"
                        ),
                        None,
                    )
                    detail = (
                        f"{error.ename}: {error.evalue}"
                        if error
                        else "unknown Stage 7 preparation error"
                    )
                    print(f"STAGE7_PREPARATION_ERROR {detail}", flush=True)
                    return 1
                prepared_stage7 = True
                print("STAGE7_PREPARATION_READY", flush=True)

            if (
                args.resume_stage4
                and not recovered_stage4
                and index >= stage4_summary_index
            ):
                recovery_code = """
stage_root = RUN_DIR / "stages"
branch_records = json.loads(
    (stage_root / "03-attack-branches" / "dataframe.json").read_text(
        encoding="utf-8"
    )
)
attack_records = json.loads(
    (stage_root / "04-attack-results" / "dataframe.json").read_text(
        encoding="utf-8"
    )
)
if len(branch_records) != len(attack_records):
    raise RuntimeError("Stage 3/4 row counts do not match for recovery.")
seed_columns = (
    "profile",
    "tools",
    "full_profile",
    "analysis",
    "target_analysis",
    "security_instruction",
)
for branch, attack in zip(branch_records, attack_records):
    if (
        branch["profile_slot"] != attack["profile_slot"]
        or branch["domain"] != attack["domain"]
        or branch["analysis_attack_prompt"] != attack["analysis_attack_prompt"]
    ):
        raise RuntimeError("Stage 3/4 row ordering changed; cannot recover safely.")
    for column in seed_columns:
        attack[column] = branch[column]
df = pd.DataFrame(attack_records)
df = _notebook_utils.normalize_nested_columns(
    _notebook_utils.normalize_analysis_column(
        _notebook_utils.expose_profile_columns(df)
    ),
    ["target_analysis", "attack_2", "verdict_1", "verdict_2"],
)
expected_rows = int(
    df.drop_duplicates(subset=["profile_slot"])["num_tools"].sum()
) * 5
if len(df) != expected_rows or expected_rows <= 0:
    raise RuntimeError(
        f"Recovered Stage 4 has {len(df)} rows, expected {expected_rows}."
    )
ARTIFACTS.save_stage(
    "04_attack_results",
    df,
    description=f"Normalized combined {len(df)}-row attack result dataset.",
    llm_columns=[
        "victim_response_1",
        "verdict_1",
        "attack_2",
        "victim_response_2",
        "verdict_2",
    ],
    ndd_config={"recovered_from_stage3": True, "rows": len(df)},
)
print(f"Recovered and normalized Stage 4: {len(df)} rows.")
"""
                recovery_outputs, _, recovery_failed = execute(
                    client,
                    recovery_code,
                    store_history=False,
                    echo_streams=True,
                )
                if recovery_failed:
                    error = next(
                        (
                            output
                            for output in reversed(recovery_outputs)
                            if output.output_type == "error"
                        ),
                        None,
                    )
                    detail = (
                        f"{error.ename}: {error.evalue}"
                        if error
                        else "unknown Stage 4 recovery error"
                    )
                    print(f"STAGE4_RECOVERY_ERROR {detail}", flush=True)
                    return 1
                recovered_stage4 = True
                print("STAGE4_RECOVERY_READY", flush=True)

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

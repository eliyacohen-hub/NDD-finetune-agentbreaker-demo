#!/usr/bin/env python3
"""Validate a completed audience run and persist a machine-readable report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nbformat
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def nested(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        record = json.loads(line)
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise AssertionError(f"{path}:{line_number} has invalid messages.")
        records.append(record)
    return records


def main() -> None:
    args = parse_args()
    repo: Path = args.repo.resolve()
    run: Path = repo / "output" / "artifacts" / args.run_id
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, condition: bool, detail: str) -> None:
        if not condition:
            raise AssertionError(f"{name}: {detail}")
        checks[name] = {"passed": True, "detail": detail}

    profiles = pd.read_parquet(run / "stages/01-profiles/dataframe.parquet")
    analyses = pd.read_parquet(run / "stages/02-tool-analysis/dataframe.parquet")
    branches = pd.read_parquet(run / "stages/03-attack-branches/dataframe.parquet")
    attacks = pd.read_parquet(run / "stages/04-attack-results/dataframe.parquet")
    sft = pd.read_parquet(run / "stages/05-sft-records/dataframe.parquet")
    manifest = json.loads((run / "manifest.json").read_text())
    config = json.loads((run / "config/run_config.json").read_text())
    evaluation_enabled = bool(config["evaluation"].get("enabled", True))
    evaluation = (
        pd.read_parquet(
            run / "stages/06-evaluation-summary/dataframe.parquet"
        )
        if evaluation_enabled
        else pd.DataFrame()
    )
    generation_config = config["generation"]
    expected_candidates = int(generation_config["profile_candidate_count"])
    expected_selected = int(generation_config["selected_profile_count"])
    selected_slots = set(generation_config["selected_profile_slots"])
    tools_min = int(generation_config["tools_min"])
    tools_max = int(generation_config["tools_max"])

    check(
        "profiles",
        len(profiles) == expected_candidates
        and profiles["profile_slot"].nunique() == expected_candidates
        and profiles["domain"].isin(generation_config["domain_values"]).all(),
        (
            f"{len(profiles)} rows, {profiles['profile_slot'].nunique()} unique "
            f"slots, {profiles['domain'].nunique()} sampled domains"
        ),
    )

    tool_counts: list[int] = []
    complete_contracts: list[bool] = []
    concrete_policies: list[bool] = []
    policy_terms = (
        "verif",
        "authoriz",
        "confirm",
        "refus",
        "never reveal",
        "approval",
        "sensitive",
        "identity",
        "secret",
    )
    for _, row in profiles.iterrows():
        tools = nested(row["tools"])
        profile = nested(row["full_profile"])
        profile_core = nested(row["profile_core"])
        instruction = str(row["security_instruction"]).strip()
        base_system_prompt = str(profile_core["system_prompt"]).strip()
        final_system_prompt = str(profile["system_prompt"]).strip()
        level = str(row["security_level"])
        tool_counts.append(len(tools))
        if level == "permissive":
            concrete_policies.append(
                not instruction and final_system_prompt == base_system_prompt
            )
        else:
            concrete_policies.append(
                bool(instruction)
                and instruction in final_system_prompt
                and sum(
                    term in instruction.lower()
                    for term in policy_terms
                )
                >= 2
            )
        complete_contracts.extend(
            all(
                str(tool.get(field, "")).strip()
                for field in (
                    "name",
                    "description",
                    "protected_asset",
                    "authorization_requirement",
                    "side_effect_or_boundary",
                )
            )
            for tool in tools
        )

    check(
        "tools",
        len(tool_counts) == expected_candidates
        and all(tools_min <= count <= tools_max for count in tool_counts)
        and tool_counts == profiles["num_tools"].astype(int).tolist(),
        f"counts={tool_counts}, total={sum(tool_counts)}",
    )
    check(
        "tool_contracts",
        all(complete_contracts),
        f"{sum(complete_contracts)}/{len(complete_contracts)} complete",
    )
    victim_prompt = config["prompts"]["victim_system"]
    assembled_profile_used = (
        "{{ full_profile.system_prompt }}" in victim_prompt
        and "{{ security_instruction }}" not in victim_prompt
    )
    check(
        "security_policies",
        all(concrete_policies) and assembled_profile_used,
        (
            f"{sum(concrete_policies)}/{expected_candidates} profiles correctly "
            "assemble the generated "
            "policy, with permissive instructions empty"
        ),
    )

    quality_scores = profiles["profile_quality"].map(nested).map(
        lambda value: value.get("overall_quality", {}).get("score")
    )
    selected_profile_rows = profiles.loc[
        profiles["selected_for_analysis"].astype(bool)
    ]
    expected_ranks = list(range(1, expected_candidates + 1))

    def profile_fingerprint(row: pd.Series) -> str:
        payload = {
            "domain": row["domain"],
            "num_tools": int(row["num_tools"]),
            "security_level": row["security_level"],
            "profile_core": nested(row["profile_core"]),
            "security_instruction": row["security_instruction"],
            "agent_profile": nested(row["full_profile"]),
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    recomputed_fingerprints = profiles.apply(profile_fingerprint, axis=1)
    expected_order = (
        profiles.assign(_recomputed_fingerprint=recomputed_fingerprints)
        .sort_values(
            by=[
                "profile_quality_score",
                "_recomputed_fingerprint",
                "profile_slot",
            ],
            ascending=[False, True, True],
            kind="mergesort",
        )["profile_slot"]
        .tolist()
    )
    check(
        "profile_quality_selection",
        quality_scores.map(
            lambda score: isinstance(score, int)
            and not isinstance(score, bool)
            and 1 <= score <= 5
        ).all()
        and profiles["profile_selection_rank"].astype(int).tolist()
        == expected_ranks
        and profiles["profile_quality_tiebreak"].tolist()
        == recomputed_fingerprints.tolist()
        and profiles["profile_slot"].tolist() == expected_order
        and len(selected_profile_rows) == expected_selected
        and set(selected_profile_rows["profile_slot"]) == selected_slots,
        (
            f"{len(profiles)} scored candidates, "
            f"{len(selected_profile_rows)} selected, "
            f"scores={quality_scores.tolist()}"
        ),
    )
    manifest_selection = manifest.get("profile_selection", {})
    manifest_selected_slots = {
        str(item["profile_slot"])
        for item in manifest_selection.get("selected", [])
    }
    check(
        "profile_selection_manifest",
        int(manifest_selection.get("candidate_count", -1))
        == expected_candidates
        and int(manifest_selection.get("selected_count", -1))
        == expected_selected
        and manifest_selected_slots == selected_slots,
        f"selected slots={sorted(selected_slots)}",
    )

    analysis_signatures = analyses["analysis"].map(nested).map(
        lambda value: json.dumps(value, sort_keys=True)
    )
    check(
        "analysis_rows",
        len(analyses) == expected_selected
        and analyses["profile_slot"].nunique() == expected_selected
        and analysis_signatures.nunique() == expected_selected
        and set(analyses["profile_slot"]) == selected_slots,
        (
            f"{len(analyses)} rows, "
            f"{analysis_signatures.nunique()} unique analyses"
        ),
    )
    selected_tool_counts: list[int] = [
        len(nested(value)) for value in analyses["tools"]
    ]
    prompt_groups: list[tuple[int, int]] = []
    for _, row in analyses.iterrows():
        tools = nested(row["tools"])
        analysis = nested(row["analysis"])
        declared = {tool["name"] for tool in tools}
        tool_analyses = analysis["tool_analyses"]
        present = {name for name, value in tool_analyses.items() if value is not None}
        if present != declared:
            raise AssertionError(
                f"analysis tool coverage {row['domain']}: {present ^ declared}"
            )
        for tool_name in declared:
            prompts = [
                str(prompt).strip()
                for prompt in nested(tool_analyses[tool_name])["attack_prompts"]
            ]
            prompt_groups.append((len(prompts), len(set(prompts))))
    check(
        "analysis_prompts",
        all(total == 5 and unique == 5 for total, unique in prompt_groups),
        f"{len(prompt_groups)} tools, exactly five unique prompts/tool",
    )

    branch_keys = [
        "profile_slot",
        "domain",
        "target_tool",
        "attack_prompt_rank",
    ]
    unique_branches = len(branches.drop_duplicates(branch_keys))
    candidate_attack_rows = sum(selected_tool_counts) * 5
    attack_branch_limit = int(
        generation_config.get("attack_branch_limit", candidate_attack_rows)
    )
    expected_attack_rows = min(candidate_attack_rows, attack_branch_limit)
    check(
        "branches",
        len(branches) == expected_attack_rows
        and unique_branches == expected_attack_rows,
        (
            f"{len(branches)} sampled rows from {candidate_attack_rows} "
            f"candidates, {unique_branches} unique branch keys"
        ),
    )
    branch_groups = branches.groupby(["profile_slot", "target_tool"]).agg(
        rows=("analysis_attack_prompt", "size"),
        unique=("analysis_attack_prompt", "nunique"),
    )
    check(
        "branch_prompt_diversity",
        bool(
            (
                (branch_groups["rows"] <= 5)
                & (branch_groups["unique"] == branch_groups["rows"])
            ).all()
        ),
        f"{len(branch_groups)} sampled tools with distinct branch prompts",
    )
    placeholder = "Please use <tool_name> to demonstrate its declared functionality."
    check(
        "no_generic_fallback",
        not branches["analysis_attack_prompt"]
        .astype(str)
        .str.contains(re.escape(placeholder), regex=True)
        .any(),
        "generic fallback absent",
    )

    unique_attacks = len(attacks.drop_duplicates(branch_keys))
    check(
        "attack_results",
        len(attacks) == expected_attack_rows
        and unique_attacks == expected_attack_rows,
        f"{len(attacks)} rows, {unique_attacks} unique branch keys",
    )
    for column in (
        "attack_1_text",
        "attack_2_text",
        "victim_response_1",
        "victim_response_2",
        "verdict_1",
        "verdict_2",
    ):
        populated = attacks[column].notna() & attacks[column].astype(str).str.strip().ne("")
        check(
            f"populated_{column}",
            bool(populated.all()),
            f"{int(populated.sum())}/{expected_attack_rows} populated",
        )
    check(
        "attack_diversity",
        attacks["attack_1_text"].nunique() == expected_attack_rows,
        f"{attacks['attack_1_text'].nunique()} unique initial attacks",
    )

    successes = int(attacks["successful_attack"].sum())
    dataset_config = config["datasets"]
    accepted_outcomes = set(
        dataset_config.get("accepted_outcomes", ["YES", "PARTIAL"])
    )
    adaptive_successes = sum(
        nested(verdict).get("attack_success", {}).get("score")
        in accepted_outcomes
        for verdict in attacks["verdict_2"]
    )
    expected_analysis_records = (
        expected_selected
        if dataset_config.get("include_analysis_records", True)
        else 0
    )
    expected_exploit_records = (
        adaptive_successes
        if dataset_config.get("include_exploit_records", True)
        else 0
    )
    check(
        "sft_records",
        len(sft) == expected_analysis_records + expected_exploit_records,
        (
            f"{len(sft)} rows = {expected_analysis_records} analyses + "
            f"{expected_exploit_records} "
            f"successful adaptive exploits ({successes} branches succeeded "
            "across either attempt)"
        ),
    )
    quick_train = load_jsonl(
        repo / "data/quick_finetune/agent_breaker_train.jsonl"
    )
    quick_validation = load_jsonl(
        repo / "data/quick_finetune/agent_breaker_validation.jsonl"
    )
    generated_train = load_jsonl(run / "datasets/attacker_chat_train.jsonl")
    generated_validation = load_jsonl(run / "datasets/attacker_chat_val.jsonl")
    agent_by_messages = {
        json.dumps(
            nested(row["messages"]),
            ensure_ascii=False,
            sort_keys=True,
        ): str(row["agent_id"])
        for _, row in sft.iterrows()
    }
    generated_train_ids = {
        agent_by_messages[
            json.dumps(
                record["messages"],
                ensure_ascii=False,
                sort_keys=True,
            )
        ]
        for record in generated_train
    }
    generated_validation_ids = {
        agent_by_messages[
            json.dumps(
                record["messages"],
                ensure_ascii=False,
                sort_keys=True,
            )
        ]
        for record in generated_validation
    }
    check(
        "quick_finetune_data",
        (len(quick_train), len(quick_validation)) == (512, 64),
        (
            f"{len(quick_train)} train, {len(quick_validation)} validation; "
            "all JSONL valid"
        ),
    )
    check(
        "generated_splits",
        len(generated_train) + len(generated_validation) == len(sft)
        and bool(generated_train)
        and bool(generated_validation)
        and generated_train_ids.isdisjoint(generated_validation_ids),
        (
            f"{len(generated_train)} train, {len(generated_validation)} "
            "validation; all JSONL valid and agent-disjoint"
        ),
    )

    recipe = yaml.safe_load(
        (run / "training/automodel_lora.run.yaml").read_text()
    )
    training_config = config["training"]
    resolved_train_path = Path(training_config["train_path"]).resolve()
    resolved_validation_path = Path(
        training_config["validation_path"]
    ).resolve()
    check(
        "training_source",
        Path(recipe["dataset"]["path_or_dataset_id"]).resolve()
        == resolved_train_path
        and Path(
            recipe["validation_dataset"]["path_or_dataset_id"]
        ).resolve()
        == resolved_validation_path
        and training_config["dataset_mode"] in {"bundled", "generated"},
        (
            f"mode={training_config['dataset_mode']}; "
            f"train={resolved_train_path}; validation={resolved_validation_path}"
        ),
    )
    train_log = (run / "training/train.log").read_text(errors="replace")
    notebook = nbformat.read(repo / "ndd_nemotron_garak.ipynb", as_version=4)
    notebook_text = json.dumps(notebook)
    check(
        "single_gpu",
        "World size: 1" in train_log
        and "--nproc-per-node=1" in notebook_text
        and int(training_config["gpu_id"]) >= 0,
        (
            "world size 1, torchrun process count 1, "
            f"GPU device {training_config['gpu_id']}"
        ),
    )
    validation_losses = [
        (int(step), float(loss))
        for step, loss in re.findall(
            r"\[val\].*?step (\d+).*?loss ([0-9.]+)",
            train_log,
        )
    ]
    finite_validation = all(
        math.isfinite(loss) for _, loss in validation_losses
    )
    loss_condition = bool(validation_losses) and finite_validation
    if training_config["dataset_mode"] == "bundled":
        loss_condition = (
            len(validation_losses) >= 2
            and validation_losses[-1][1] < validation_losses[0][1]
            and finite_validation
        )
    loss_detail = (
        (
            f"validation loss {validation_losses[0][1]:.4f} -> "
            f"{validation_losses[-1][1]:.4f}; "
            f"{len(validation_losses)} validations"
        )
        if validation_losses
        else "no validation losses parsed"
    )
    check(
        "training_loss",
        loss_condition,
        loss_detail,
    )
    if training_config.get("merge_after_training", True):
        merged_weights = list(
            (run / "training/merged_model").glob("*.safetensors")
        )
        check(
            "merged_model",
            (run / "training/merged_model/config.json").is_file()
            and bool(merged_weights),
            f"merged config and {len(merged_weights)} weight file(s) present",
        )

    evaluation_config = config["evaluation"]
    if evaluation_enabled:
        expected_evaluation_rows = len(
            evaluation_config["attackers"]
        ) * len(evaluation_config["victim_agents"])
        expected_attempts = int(
            evaluation_config["expected_attempts_per_run"]
        )
        matched_totals = (
            evaluation.groupby("victim")["total"].nunique().eq(1).all()
        )
        check(
            "evaluation",
            len(evaluation) == expected_evaluation_rows
            and bool((evaluation["total"] > 0).all())
            and bool(matched_totals)
            and bool((evaluation["total"] == expected_attempts).all())
            and bool(
                (
                    evaluation["evaluated"]
                    + evaluation["invalid_or_skipped"]
                    == evaluation["total"]
                ).all()
            )
            and bool(
                (
                    evaluation["evaluated"] + evaluation["nones"]
                    == evaluation["processed"]
                ).all()
            ),
            "; ".join(
                (
                    f"{row['label']}: {int(row['hits'])}/{int(row['total'])} "
                    f"({row['ASR %']:.1f}%), "
                    f"skipped={int(row['invalid_or_skipped'])}"
                )
                for _, row in evaluation.iterrows()
            ),
        )
    expected_manifest_stages = {
        "01-profiles",
        "02-tool-analysis",
        "03-attack-branches",
        "04-attack-results",
        "05-sft-records",
    }
    if evaluation_enabled:
        expected_manifest_stages.add("06-evaluation-summary")
    actual_manifest_stages = set(manifest["stages"])
    check(
        "manifest",
        actual_manifest_stages == expected_manifest_stages,
        f"persisted stages={sorted(actual_manifest_stages)}",
    )
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    check(
        "notebook_execution",
        all(cell.execution_count is not None for cell in code_cells) and not errors,
        (
            f"{len(code_cells)}/{len(code_cells)} code cells executed; "
            f"{len(errors)} error outputs"
        ),
    )
    runs = sorted(
        path.name
        for path in (repo / "output/artifacts").iterdir()
        if path.is_dir()
    )
    check(
        "current_run_present",
        args.run_id in runs,
        f"current={args.run_id}, available_runs={runs}",
    )

    def minutes(seconds: float) -> str:
        return f"{seconds / 60:.1f} min"

    created = datetime.fromisoformat(manifest["created_at_utc"])
    stages = {
        name: datetime.fromisoformat(value["saved_at_utc"])
        for name, value in manifest["stages"].items()
    }
    train_times = [
        datetime.fromisoformat(timestamp).replace(tzinfo=UTC)
        for timestamp in re.findall(
            r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)",
            train_log,
            re.MULTILINE,
        )
    ]
    timings = {
        "profiles": minutes((stages["01-profiles"] - created).total_seconds()),
        "tool_analysis": minutes(
            (stages["02-tool-analysis"] - stages["01-profiles"]).total_seconds()
        ),
        "attack_generation_and_retries": minutes(
            (
                stages["04-attack-results"] - stages["02-tool-analysis"]
            ).total_seconds()
        ),
        "sft_materialization": (
            f"{(stages['05-sft-records'] - stages['04-attack-results']).total_seconds():.1f} sec"
        ),
        "fine_tuning": minutes((max(train_times) - min(train_times)).total_seconds()),
        "merge_and_evaluation": (
            minutes(
                (
                    stages["06-evaluation-summary"] - max(train_times)
                ).total_seconds()
            )
            if evaluation_enabled
            else "evaluation disabled"
        ),
        "end_to_end": minutes(
            (
                (
                    stages["06-evaluation-summary"]
                    if evaluation_enabled
                    else stages["05-sft-records"]
                )
                - created
            ).total_seconds()
        ),
    }
    report = {
        "run_id": args.run_id,
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "checks": checks,
        "metrics": {
            "profile_candidates": len(profiles),
            "selected_profiles": len(selected_profile_rows),
            "candidate_tools": sum(tool_counts),
            "selected_tools": sum(selected_tool_counts),
            "branches": len(branches),
            "attack_results": len(attacks),
            "successful_attack_branches": successes,
            "successful_adaptive_attacks": adaptive_successes,
            "generated_sft_records": len(sft),
            "quick_finetune_train": len(quick_train),
            "quick_finetune_validation": len(quick_validation),
            "validation_losses": validation_losses,
            "evaluation": evaluation.to_dict(orient="records"),
        },
        "timings": timings,
    }
    report_path = run / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(f"VALIDATION PASSED: {len(checks)} checks")
    for name, result in checks.items():
        print(f"PASS {name}: {result['detail']}")
    print(f"TIMINGS {json.dumps(timings, indent=2)}")
    print(f"SAVED {report_path}")


if __name__ == "__main__":
    main()

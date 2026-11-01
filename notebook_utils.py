"""Typed implementation helpers for the NDD and fine-tuning demo notebook.

The notebook intentionally keeps the teaching-facing NDD graphs and AutoModel
recipe visible. Repetitive persistence, retry, normalization, and process
plumbing lives here so the notebook remains readable.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import httpx
import pandas as pd
import yaml
from data_designer import config as dd
from data_designer.interface import ResumeMode
from data_designer.interface.errors import DataDesignerGenerationError
from pydantic import BaseModel, Field


class ModelDumpable(Protocol):
    """Structural type for Pydantic-like objects."""

    def model_dump(self, *, mode: str) -> object:
        """Return a serializable representation."""


def slugify(value: str, *, max_length: int = 64) -> str:
    """Convert a value into a compact filesystem-safe identifier."""
    normalized: str = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return normalized[:max_length].rstrip("-") or "value"


def json_safe(payload: object) -> Any:
    """Convert nested values into JSON/YAML-safe built-ins."""

    def default(value: object) -> object:
        model_dump: Any = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="json")
        as_py: Any = getattr(value, "as_py", None)
        if callable(as_py):
            return as_py()
        tolist: Any = getattr(value, "tolist", None)
        if callable(tolist):
            return tolist()
        return str(value)

    return json.loads(json.dumps(payload, default=default, ensure_ascii=False))


def decode_nested_value(value: object) -> Any:
    """Decode nested values serialized as JSON or safe Python literals."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                value = ast.literal_eval(stripped)
    return json_safe(value)


def write_yaml_json(stem: Path, payload: Mapping[str, Any]) -> tuple[Path, Path]:
    """Write the same payload as readable YAML and machine-friendly JSON."""
    safe_payload: Any = json_safe(payload)
    yaml_path: Path = stem.with_suffix(".yaml")
    json_path: Path = stem.with_suffix(".json")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as yaml_file:
        yaml.safe_dump(safe_payload, yaml_file, sort_keys=False, allow_unicode=True)
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(safe_payload, json_file, indent=2, ensure_ascii=False)
        json_file.write("\n")
    return yaml_path, json_path


def local_seed_source(
    dataframe: pd.DataFrame,
    *,
    directory: Path,
    prefix: str,
) -> dd.LocalFileSeedSource:
    """Persist a fingerprinted JSONL seed so NDD resume can verify its config."""
    records: Any = json_safe(dataframe.to_dict(orient="records"))
    serialized: str = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest: str = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    path: Path = directory / f"{prefix}-{digest}.jsonl"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in cast(list[dict[str, Any]], records):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return dd.LocalFileSeedSource(path=str(path))


def _portable_cell(value: object) -> object:
    """Encode nested cells as JSON strings for portable Parquet and CSV."""
    if hasattr(value, "model_dump"):
        value = cast(ModelDumpable, value).model_dump(mode="json")
    if isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    ) or callable(getattr(value, "tolist", None)):
        return json.dumps(json_safe(value), ensure_ascii=False)
    return value


class ArtifactStore:
    """Persist stage DataFrames and maintain one run manifest."""

    def __init__(self, run_dir: Path, created_at: datetime) -> None:
        self.run_dir: Path = run_dir
        self.config_dir: Path = run_dir / "config"
        self.stages_dir: Path = run_dir / "stages"
        self.dataset_dir: Path = run_dir / "datasets"
        for directory in (self.config_dir, self.stages_dir, self.dataset_dir):
            directory.mkdir(parents=True, exist_ok=True)
        manifest_path: Path = run_dir / "manifest.json"
        if manifest_path.is_file():
            loaded: object = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise TypeError(f"Invalid run manifest at {manifest_path}.")
            self.manifest = cast(dict[str, Any], loaded)
            self.manifest.setdefault("stages", {})
        else:
            self.manifest = {
                "run_id": run_dir.name,
                "created_at_utc": created_at.isoformat(),
                "run_directory": str(run_dir),
                "configuration": {
                    "yaml": "config/run_config.yaml",
                    "json": "config/run_config.json",
                },
                "stages": {},
            }

    def save_stage(
        self,
        stage_name: str,
        dataframe: pd.DataFrame,
        *,
        description: str,
        llm_columns: Sequence[str],
        ndd_config: object | None = None,
    ) -> Path:
        """Persist a stage as Parquet, JSON, CSV, metadata, and optional config."""
        missing_llm_columns = set(llm_columns) - set(dataframe.columns)
        if missing_llm_columns:
            raise KeyError(
                f"Stage {stage_name} is missing declared LLM columns: "
                f"{sorted(missing_llm_columns)}"
            )
        stage_key: str = slugify(stage_name)
        stage_directory: Path = self.stages_dir / stage_key
        stage_directory.mkdir(parents=True, exist_ok=True)

        parquet_path: Path = stage_directory / "dataframe.parquet"
        json_path: Path = stage_directory / "dataframe.json"
        csv_path: Path = stage_directory / "dataframe.csv"
        dataframe.to_json(
            json_path,
            orient="records",
            indent=2,
            force_ascii=False,
            default_handler=str,
        )
        portable: pd.DataFrame = dataframe.copy(deep=True)
        for column in portable.select_dtypes(include=["object"]).columns:
            portable[column] = portable[column].map(_portable_cell)
        portable.to_parquet(parquet_path, index=False)
        portable.to_csv(csv_path, index=False)

        metadata: dict[str, Any] = {
            "stage": stage_name,
            "description": description,
            "saved_at_utc": datetime.now(UTC).isoformat(),
            "rows": len(dataframe),
            "columns": [str(column) for column in dataframe.columns],
            "dtypes": {
                str(column): str(dtype) for column, dtype in dataframe.dtypes.items()
            },
            "llm_columns": list(llm_columns),
            "files": {
                "parquet": str(parquet_path.relative_to(self.run_dir)),
                "json": str(json_path.relative_to(self.run_dir)),
                "csv": str(csv_path.relative_to(self.run_dir)),
            },
        }
        if ndd_config is not None:
            if hasattr(ndd_config, "model_dump"):
                dumped: object = cast(ModelDumpable, ndd_config).model_dump(mode="json")
            elif isinstance(ndd_config, Mapping):
                dumped = dict(ndd_config)
            else:
                dumped = {"representation": repr(ndd_config)}
            if isinstance(dumped, Mapping):
                config_yaml, config_json = write_yaml_json(
                    stage_directory / "ndd_config",
                    cast(Mapping[str, Any], dumped),
                )
                metadata["files"]["ndd_config_yaml"] = str(
                    config_yaml.relative_to(self.run_dir)
                )
                metadata["files"]["ndd_config_json"] = str(
                    config_json.relative_to(self.run_dir)
                )

        write_yaml_json(stage_directory / "metadata", metadata)
        self.manifest["stages"][stage_key] = metadata
        write_yaml_json(self.run_dir / "manifest", self.manifest)
        print(f"Saved {stage_name}: {len(dataframe)} rows -> {stage_directory}")
        return stage_directory


def create_resumable(
    designer: Any,
    builder: dd.DataDesignerConfigBuilder,
    *,
    num_records: int,
    dataset_name: str,
    max_health_attempts: int = 3,
) -> Any:
    """Use NDD-native checkpoint resume, retrying only startup health checks."""
    for attempt in range(1, max_health_attempts + 1):
        try:
            return designer.create(
                builder,
                num_records=num_records,
                dataset_name=dataset_name,
                resume=ResumeMode.IF_POSSIBLE,
            )
        except DataDesignerGenerationError as error:
            is_health_failure: bool = "health check" in str(error).lower()
            if not is_health_failure or attempt == max_health_attempts:
                raise
            delay: float = 10.0 * attempt
            print(
                f"Health-check failure ({attempt}/{max_health_attempts}); "
                f"retrying in {delay:.0f}s"
            )
            time.sleep(delay)
    raise RuntimeError("Create retry loop exited unexpectedly.")


def generate_exact_rows(
    *,
    designer: Any,
    seed: pd.DataFrame,
    builder_factory: Callable[
        [pd.DataFrame], dd.DataDesignerConfigBuilder
    ],
    dataset_prefix: str,
    key_columns: Sequence[str],
    normalize: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    max_completion_attempts: int = 3,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dd.DataDesignerConfigBuilder]:
    """Generate every seed key using NDD resume plus bounded dropped-row completion."""

    def row_key(row: pd.Series) -> tuple[Any, ...]:
        return tuple(row[column] for column in key_columns)

    expected_keys: list[tuple[Any, ...]] = [
        row_key(row) for _, row in seed.iterrows()
    ]
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError(f"Seed keys are not unique: {list(key_columns)}")

    frames: list[pd.DataFrame] = []
    attempts: list[dict[str, Any]] = []
    pending: pd.DataFrame = seed.copy(deep=True)
    expected_columns: list[str] = []
    last_builder: dd.DataDesignerConfigBuilder | None = None

    for attempt_number in range(1, max_completion_attempts + 1):
        requested_rows: int = len(pending)
        last_builder = builder_factory(pending)
        columns: list[str] = [
            column.name for column in last_builder.get_column_configs()
        ]
        if not expected_columns:
            expected_columns = columns
        elif columns != expected_columns:
            raise RuntimeError("NDD graph changed between completion attempts.")

        result: Any = create_resumable(
            designer,
            last_builder,
            num_records=len(pending),
            dataset_name=f"{dataset_prefix}-attempt-{attempt_number:02d}",
        )
        attempt_frame: pd.DataFrame = result.load_dataset()
        if normalize is not None:
            attempt_frame = normalize(attempt_frame)
        missing_key_columns: set[str] = set(key_columns) - set(
            attempt_frame.columns
        )
        if missing_key_columns:
            raise KeyError(
                f"NDD result lacks key columns: {sorted(missing_key_columns)}"
            )
        if not attempt_frame.empty:
            frames.append(attempt_frame)

        combined: pd.DataFrame = (
            pd.concat(frames, ignore_index=True).drop_duplicates(
                subset=list(key_columns),
                keep="last",
            )
            if frames
            else pd.DataFrame()
        )
        completed_keys: set[tuple[Any, ...]] = (
            {row_key(row) for _, row in combined.iterrows()}
            if not combined.empty
            else set()
        )
        pending_mask = pd.Series(
            [
                row_key(row) not in completed_keys
                for _, row in seed.iterrows()
            ],
            index=seed.index,
        )
        pending = seed[pending_mask].copy(deep=True)
        attempts.append(
            {
                "attempt": attempt_number,
                "requested_rows": requested_rows,
                "returned_rows": len(attempt_frame),
                "remaining_rows": len(pending),
            }
        )
        print(
            f"Completion attempt {attempt_number}: "
            f"{len(attempt_frame)} returned, {len(pending)} remaining."
        )
        if pending.empty:
            break

    if not frames or last_builder is None or not pending.empty:
        raise RuntimeError(
            f"NDD generation is missing {len(pending)} rows after "
            f"{max_completion_attempts} completion attempts."
        )
    dataframe = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=list(key_columns),
        keep="last",
    )
    order: dict[tuple[Any, ...], int] = {
        key: index for index, key in enumerate(expected_keys)
    }
    dataframe["_seed_order"] = [
        order[row_key(row)] for _, row in dataframe.iterrows()
    ]
    dataframe = (
        dataframe.sort_values("_seed_order")
        .drop(columns=["_seed_order"])
        .reset_index(drop=True)
    )
    return dataframe, attempts, last_builder


def profile_mapping(value: object, column: str) -> dict[str, Any]:
    """Normalize one structured NDD profile value into a plain dictionary."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if not isinstance(value, Mapping):
        raise TypeError(f"{column} must be a mapping.")
    normalized: Any = json_safe(dict(value))
    if not isinstance(normalized, dict):
        raise TypeError(f"{column} did not normalize to a dictionary.")
    return normalized


@dd.custom_column_generator(
    required_columns=["security_level", "security_policy"]
)
def normalize_security_instruction(row: Mapping[str, Any]) -> dict[str, Any]:
    """Force permissive empty; require generated policy for other levels."""
    result: dict[str, Any] = dict(row)
    level: str = str(row["security_level"])
    if level == "permissive":
        result["security_instruction"] = ""
        return result
    policy: dict[str, Any] = profile_mapping(
        row["security_policy"], "security_policy"
    )
    instruction: str = str(policy.get("security_instruction", "")).strip()
    if not instruction:
        raise ValueError(f"{level} requires a non-empty security instruction.")
    result["security_instruction"] = instruction
    return result


@dd.custom_column_generator(
    required_columns=["profile_core", "tool_set", "security_instruction"]
)
def assemble_agent_profile(row: Mapping[str, Any]) -> dict[str, Any]:
    """Combine profile, tools, and optional generated policy into one agent."""
    result: dict[str, Any] = dict(row)
    profile_core: dict[str, Any] = profile_mapping(
        row["profile_core"], "profile_core"
    )
    tool_set: dict[str, Any] = profile_mapping(row["tool_set"], "tool_set")
    tools: object = tool_set.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("tool_set.tools must be a non-empty list.")
    instruction: str = str(row.get("security_instruction", "")).strip()
    base_system_prompt: str = str(profile_core["system_prompt"]).strip()
    final_system_prompt: str = base_system_prompt
    if instruction:
        final_system_prompt += (
            f"\n\nSECURITY INSTRUCTION (mandatory):\n{instruction}"
        )
    result["agent_profile"] = {
        "agent_purpose": str(profile_core["agent_purpose"]).strip(),
        "tools": tools,
        "system_prompt": final_system_prompt,
    }
    return result


class _ToolAnalysis(BaseModel):
    """Validated security analysis for one generated tool."""

    functionality: str
    vulnerabilities: str
    exploit_strategies: str
    attack_prompts: list[str] = Field(min_length=5, max_length=5)


class _AnalysisOutput(BaseModel):
    """Validated Garak-compatible analysis response."""

    tool_analyses: dict[str, _ToolAnalysis]
    priority_targets: list[str] = Field(default_factory=list)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from plain or reasoned model output."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
    raise ValueError("The model response did not contain a complete JSON object.")


@dd.custom_column_generator(required_columns=["analysis_raw", "agent_profile"])
def parse_analysis_json(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact tool coverage and five unique prompts per tool."""
    raw_analysis: object = row.get("analysis_raw")
    if not isinstance(raw_analysis, str):
        raise TypeError("analysis_raw must be a string.")
    validated = _AnalysisOutput.model_validate(
        _extract_json_object(raw_analysis)
    )

    profile_value: object = row.get("agent_profile")
    if isinstance(profile_value, BaseModel):
        profile_value = profile_value.model_dump(mode="python")
    if not isinstance(profile_value, Mapping):
        raise TypeError("agent_profile must be a mapping.")
    normalized_profile: Any = json_safe(dict(profile_value))
    if not isinstance(normalized_profile, dict):
        raise TypeError("Normalized agent_profile must be a dictionary.")
    tools_value: object = normalized_profile.get("tools", [])
    if not isinstance(tools_value, list):
        raise TypeError("agent_profile.tools must normalize to a list.")

    declared_names: set[str] = {
        str(tool.get("name"))
        for tool in tools_value
        if isinstance(tool, Mapping) and tool.get("name")
    }
    analyzed_names: set[str] = set(validated.tool_analyses)
    if analyzed_names != declared_names:
        raise ValueError(
            "Analysis must cover exactly the declared tools: "
            f"missing={sorted(declared_names - analyzed_names)}, "
            f"unexpected={sorted(analyzed_names - declared_names)}."
        )
    for tool_name, tool_analysis in validated.tool_analyses.items():
        prompts: list[str] = [
            prompt.strip() for prompt in tool_analysis.attack_prompts
        ]
        if len(set(prompts)) != 5 or any(not prompt for prompt in prompts):
            raise ValueError(
                f"Tool {tool_name!r} must contain five non-empty unique prompts."
            )
        tool_analysis.attack_prompts = prompts

    result: dict[str, Any] = dict(row)
    result["analysis"] = validated.model_dump(mode="python")
    return result


def generate_profiles(
    *,
    designer: Any,
    profile_seed: pd.DataFrame,
    builder_factory: Callable[
        [pd.DataFrame], dd.DataDesignerConfigBuilder
    ],
    run_id: str,
    max_completion_attempts: int = 3,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dd.DataDesignerConfigBuilder]:
    """Generate profiles using NDD resume, retrying only rows NDD ultimately dropped."""
    return generate_exact_rows(
        designer=designer,
        seed=profile_seed,
        builder_factory=builder_factory,
        dataset_prefix=f"agent-profiles-{run_id}",
        key_columns=["profile_slot"],
        max_completion_attempts=max_completion_attempts,
    )


def validate_profile_dataframe(
    raw_frame: pd.DataFrame,
    *,
    profile_seed: pd.DataFrame,
    domain_values: Sequence[str],
    security_levels: Sequence[str],
    tools_min: int,
    tools_max: int,
    blocked_tool_names: set[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Normalize and validate one complete generated profile stage."""
    dataframe: pd.DataFrame = expose_profile_columns(raw_frame)
    expected_slots: set[str] = set(profile_seed["profile_slot"])
    if len(dataframe) != len(profile_seed) or set(
        dataframe["profile_slot"]
    ) != expected_slots:
        raise RuntimeError(
            f"Profile generation returned {len(dataframe)} of "
            f"{len(profile_seed)} required slots."
        )

    actual_counts: pd.Series = dataframe["tools"].map(len).astype("int64")
    sampled_counts: pd.Series = dataframe["num_tools"].astype("int64")
    if not actual_counts.equals(sampled_counts):
        raise RuntimeError(
            "Generated tool counts do not match NDD's sampled values."
        )
    if not sampled_counts.between(tools_min, tools_max).all():
        raise RuntimeError("An NDD-sampled tool count is outside the range.")
    if not dataframe["domain"].isin(domain_values).all():
        raise RuntimeError("NDD sampled a domain outside the configured list.")
    if not dataframe["security_level"].isin(security_levels).all():
        raise RuntimeError("NDD sampled an undefined security level.")

    permissive: pd.Series = dataframe["security_level"].eq("permissive")
    instruction_empty: pd.Series = (
        dataframe["security_instruction"].astype(str).str.strip().eq("")
    )
    if not instruction_empty[permissive].all():
        raise RuntimeError("Permissive profiles must have no instruction.")
    if instruction_empty[~permissive].any():
        raise RuntimeError(
            "Balanced and strict profiles need generated instructions."
        )

    required_tool_fields: tuple[str, ...] = (
        "description",
        "protected_asset",
        "authorization_requirement",
        "side_effect_or_boundary",
    )
    for row_index, row in dataframe.iterrows():
        core: dict[str, Any] = profile_mapping(
            row["profile_core"], "profile_core"
        )
        full_profile: Mapping[str, Any] = cast(
            Mapping[str, Any], row["full_profile"]
        )
        instruction: str = str(row["security_instruction"]).strip()
        final_prompt: str = str(full_profile.get("system_prompt", "")).strip()
        base_prompt: str = str(core["system_prompt"]).strip()
        if instruction and instruction not in final_prompt:
            raise RuntimeError(
                f"Row {row_index} did not assemble its security instruction."
            )
        if not instruction and final_prompt != base_prompt:
            raise RuntimeError(
                f"Permissive row {row_index} changed the base prompt."
            )

        tools: list[Any] = list(row["tools"])
        names: list[str] = [
            str(tool.get("name", ""))
            for tool in tools
            if isinstance(tool, Mapping)
        ]
        if len(names) != len(tools) or len(names) != len(set(names)):
            raise RuntimeError(f"Row {row_index} has invalid tool names.")
        if blocked_tool_names.intersection(names):
            raise RuntimeError(f"Row {row_index} has a generic tool name.")
        for tool in tools:
            if not isinstance(tool, Mapping):
                raise TypeError(f"Row {row_index} has an invalid tool.")
            for field in required_tool_fields:
                if not str(tool.get(field, "")).strip():
                    raise RuntimeError(
                        f"Row {row_index} tool {tool.get('name')} lacks {field}."
                    )
    return dataframe, sampled_counts


def generate_analyses(
    *,
    designer: Any,
    analysis_seed: pd.DataFrame,
    builder_factory: Callable[
        [pd.DataFrame], dd.DataDesignerConfigBuilder
    ],
    run_id: str,
    max_completion_attempts: int = 3,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dd.DataDesignerConfigBuilder]:
    """Generate analyses using NDD resume, retrying only ultimately dropped rows."""
    return generate_exact_rows(
        designer=designer,
        seed=analysis_seed,
        builder_factory=builder_factory,
        dataset_prefix=f"tool-analysis-{run_id}",
        key_columns=["profile_slot"],
        normalize=lambda frame: normalize_analysis_column(
            expose_profile_columns(frame)
        ),
        max_completion_attempts=max_completion_attempts,
    )


def validate_analysis_dataframe(
    dataframe: pd.DataFrame,
    *,
    expected_slots: Sequence[str],
) -> None:
    """Validate profile coverage, declared tools, and five prompts per tool."""
    expected: set[str] = set(expected_slots)
    actual: set[str] = set(dataframe["profile_slot"])
    if len(dataframe) != len(expected_slots) or actual != expected:
        raise RuntimeError(
            f"Tool analysis is missing profile slots: {sorted(expected - actual)}"
        )

    for row_index, row in dataframe.iterrows():
        declared_names: set[str] = {
            str(tool["name"]) for tool in row["tools"]
        }
        analysis: object = row["analysis"]
        if not isinstance(analysis, Mapping):
            raise TypeError(f"Row {row_index} has an invalid analysis.")
        tool_analyses: object = analysis.get("tool_analyses")
        if not isinstance(tool_analyses, Mapping):
            raise TypeError(
                f"Row {row_index} has invalid per-tool analyses."
            )
        if set(tool_analyses) != declared_names:
            raise RuntimeError(
                f"Analysis/tool mismatch in row {row_index}."
            )
        for tool_name, tool_analysis in tool_analyses.items():
            if not isinstance(tool_analysis, Mapping):
                raise TypeError(
                    f"Row {row_index} tool {tool_name} has invalid analysis."
                )
            prompts: object = tool_analysis.get("attack_prompts")
            if (
                not isinstance(prompts, list)
                or len(prompts) != 5
                or len(set(prompts)) != 5
                or any(not str(prompt).strip() for prompt in prompts)
            ):
                raise RuntimeError(
                    f"Row {row_index} tool {tool_name} must have "
                    "five unique prompts."
                )


def run_attacks(
    *,
    designer: Any,
    targets: pd.DataFrame,
    builder_factory: Callable[
        [pd.DataFrame], dd.DataDesignerConfigBuilder
    ],
    run_id: str,
    model_repr: str,
    batch_size: int | None = None,
    max_completion_attempts: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the attack graph with NDD resume and exact branch coverage."""
    if batch_size is not None and batch_size != len(targets):
        raise ValueError(
            "Do not partition attacks manually; NDD RunConfig.buffer_size "
            "owns checkpoint batching."
        )
    branch_keys: list[str] = [
        "profile_slot",
        "domain",
        "target_tool",
        "attack_prompt_rank",
        "analysis_attack_prompt",
    ]
    dataframe, attempts, builder = generate_exact_rows(
        designer=designer,
        seed=targets,
        builder_factory=builder_factory,
        dataset_prefix=f"agent-redteam-attacks-{run_id}",
        key_columns=branch_keys,
        normalize=lambda frame: normalize_nested_columns(
            normalize_analysis_column(expose_profile_columns(frame)),
            ["target_analysis", "attack_2", "verdict_1", "verdict_2"],
        ),
        max_completion_attempts=max_completion_attempts,
    )
    snapshot: dict[str, Any] = {
        "seed_stage": "03_attack_branches",
        "seed_rows": len(targets),
        "attacks_per_branch": 2,
        "requested_rows_per_create": len(targets),
        "checkpointing": "NDD ResumeMode.IF_POSSIBLE",
        "completion_attempts": attempts,
        "model": model_repr,
        "columns": [
            column.name for column in builder.get_column_configs()
        ],
    }
    return dataframe, snapshot


def check_general_api(
    model: str,
    *,
    endpoint: str,
    api_key: str,
    timeout_seconds: float,
) -> tuple[str, float]:
    """Make one minimal OpenAI-compatible API request and return text and latency."""
    started_at: float = time.perf_counter()
    try:
        response: httpx.Response = httpx.post(
            f"{endpoint.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with exactly: API OK"}],
                "temperature": 0,
                "max_tokens": 16,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except httpx.TimeoutException as error:
        raise RuntimeError(
            f"Direct API request timed out after {timeout_seconds:.0f}s."
        ) from error
    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            f"Direct API request failed with HTTP {error.response.status_code}: "
            f"{error.response.text[:500]}"
        ) from error
    payload: dict[str, Any] = response.json()
    message: Mapping[str, Any] = payload["choices"][0]["message"]
    response_text: object = message.get("content") or message.get("reasoning_content")
    if not isinstance(response_text, str) or not response_text.strip():
        raise RuntimeError("Direct API response contained no text or reasoning content.")
    text: str = response_text.strip()
    if text != "API OK":
        raise RuntimeError(f"Unexpected direct API smoke-test response: {text[:200]}")
    return text, time.perf_counter() - started_at


def normalize_full_profile(value: object) -> dict[str, Any]:
    """Normalize a profile and nested tools into plain Python values."""
    value = decode_nested_value(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a profile mapping, got {type(value).__name__}.")
    normalized: Any = json_safe(dict(value))
    if not isinstance(normalized, dict):
        raise TypeError("Normalized profile must be a dictionary.")
    return cast(dict[str, Any], normalized)


def expose_profile_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Replace agent_profile with profile, tools, and full_profile columns."""
    if "agent_profile" not in dataframe.columns:
        required: set[str] = {"profile", "tools", "full_profile"}
        if required.issubset(dataframe.columns):
            result = dataframe.copy(deep=True)
            profiles: list[dict[str, Any]] = [
                normalize_full_profile(value) for value in result["full_profile"]
            ]
            result["full_profile"] = pd.Series(
                profiles, index=result.index, dtype="object"
            )
            result["profile"] = pd.Series(
                [str(profile.get("agent_purpose", "")) for profile in profiles],
                index=result.index,
                dtype="object",
            )
            result["tools"] = pd.Series(
                [
                    list(profile.get("tools", []))
                    if isinstance(profile.get("tools", []), list)
                    else []
                    for profile in profiles
                ],
                index=result.index,
                dtype="object",
            )
            if "num_tools" in result.columns:
                result["num_tools"] = result["num_tools"].astype("int64")
            return result
        raise KeyError("DataFrame lacks both agent_profile and normalized profile columns.")
    result: pd.DataFrame = dataframe.copy(deep=True)
    insertion_index: int = int(result.columns.get_loc("agent_profile"))
    profiles: list[dict[str, Any]] = [
        normalize_full_profile(value) for value in result["agent_profile"]
    ]
    purposes: list[str] = [str(item.get("agent_purpose", "")) for item in profiles]
    tools: list[list[Any]] = [
        list(item.get("tools", [])) if isinstance(item.get("tools", []), list) else []
        for item in profiles
    ]
    result = result.drop(columns=["agent_profile"])
    result.insert(insertion_index, "profile", purposes)
    result.insert(insertion_index + 1, "tools", tools)
    result.insert(insertion_index + 2, "full_profile", profiles)
    if "num_tools" in result.columns:
        result["num_tools"] = result["num_tools"].astype("int64")
    return result


def _priority_tool_name(entry: object) -> str:
    """Extract a tool name from Garak's `tool_name - reason` format."""
    return str(entry).split(" - ", maxsplit=1)[0].strip()


def normalize_analysis(value: object) -> dict[str, Any]:
    """Prune Arrow-union nulls and convert nested arrays to plain lists."""
    value = decode_nested_value(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected an analysis mapping, got {type(value).__name__}.")
    safe_value: Any = json_safe(dict(value))
    if not isinstance(safe_value, dict):
        raise TypeError("Normalized analysis must be a dictionary.")

    raw_analyses: object = safe_value.get("tool_analyses", {})
    if not isinstance(raw_analyses, Mapping):
        raise TypeError("analysis.tool_analyses must be a mapping.")
    tool_analyses: dict[str, dict[str, Any]] = {}
    for tool_name, tool_analysis in raw_analyses.items():
        if not isinstance(tool_analysis, Mapping):
            continue
        normalized_tool: Any = json_safe(dict(tool_analysis))
        if not isinstance(normalized_tool, dict):
            continue
        prompts_value: object = normalized_tool.get("attack_prompts", [])
        prompts: list[str] = (
            [
                prompt.strip()
                for prompt in prompts_value
                if isinstance(prompt, str) and prompt.strip()
            ]
            if isinstance(prompts_value, list)
            else []
        )
        normalized_tool["attack_prompts"] = prompts
        tool_analyses[str(tool_name)] = normalized_tool

    priorities_value: object = safe_value.get("priority_targets", [])
    priorities: list[str] = (
        [str(entry).strip() for entry in priorities_value if str(entry).strip()]
        if isinstance(priorities_value, list)
        else []
    )
    return {"tool_analyses": tool_analyses, "priority_targets": priorities}


def normalize_analysis_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Store each row's analysis as an independent plain-Python object."""
    if "analysis" not in dataframe.columns:
        raise KeyError("DataFrame lacks an analysis column.")
    result: pd.DataFrame = dataframe.copy(deep=True)
    result["analysis"] = pd.Series(
        [normalize_analysis(value) for value in result["analysis"]],
        index=result.index,
        dtype="object",
    )
    return result


def normalize_nested_columns(
    dataframe: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    """Convert selected nested Arrow/Pydantic values to plain Python objects."""
    result: pd.DataFrame = dataframe.copy(deep=True)
    for column in columns:
        if column not in result.columns:
            continue
        result[column] = pd.Series(
            [decode_nested_value(value) for value in result[column]],
            index=result.index,
            dtype="object",
        )
    return result


def expand_all_analysis_attacks(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Expand exactly five analysis prompts for every declared tool."""
    rows: list[dict[str, Any]] = []
    expected: int = 0
    for source_index, row in dataframe.iterrows():
        tools_value: Any = json_safe(row.get("tools", []))
        tools: list[Any] = tools_value if isinstance(tools_value, list) else []
        tool_by_name: dict[str, Mapping[str, Any]] = {
            str(tool["name"]): tool
            for tool in tools
            if isinstance(tool, Mapping) and tool.get("name")
        }
        expected_tool_count: int = int(row.get("num_tools", len(tool_by_name)))
        if not 2 <= expected_tool_count <= 4:
            raise ValueError(
                f"Agent {source_index} has invalid num_tools={expected_tool_count}; "
                "expected 2–4."
            )
        if len(tool_by_name) != expected_tool_count:
            raise ValueError(
                f"Agent {source_index} declares {len(tool_by_name)} tools but "
                f"num_tools={expected_tool_count}."
            )

        analysis: dict[str, Any] = normalize_analysis(row.get("analysis", {}))
        analyses: dict[str, dict[str, Any]] = analysis["tool_analyses"]
        declared_names: set[str] = set(tool_by_name)
        analyzed_names: set[str] = set(analyses)
        if analyzed_names != declared_names:
            raise ValueError(
                f"Agent {source_index} analysis/tool mismatch: "
                f"missing={sorted(declared_names - analyzed_names)}, "
                f"unexpected={sorted(analyzed_names - declared_names)}."
            )

        priorities: list[str] = analysis["priority_targets"]
        priority_names: list[str] = []
        for entry in priorities:
            name: str = _priority_tool_name(entry)
            if name in tool_by_name and name not in priority_names:
                priority_names.append(name)
        ordered_names: list[str] = priority_names + [
            name for name in tool_by_name if name not in priority_names
        ]

        for target_rank, tool_name in enumerate(ordered_names, start=1):
            target_analysis: dict[str, Any] = dict(analyses[tool_name])
            prompts: list[str] = list(target_analysis["attack_prompts"])
            if len(prompts) != 5 or len(set(prompts)) != 5:
                raise ValueError(
                    f"Agent {source_index} tool {tool_name!r} must have exactly "
                    f"five unique attack prompts; found {len(prompts)}."
                )
            target_analysis["attack_prompts"] = prompts
            expected += len(prompts)
            for prompt_rank, prompt in enumerate(prompts, start=1):
                expanded: dict[str, Any] = row.to_dict()
                expanded.update(
                    {
                        "source_agent_index": str(source_index),
                        "target_tool": tool_name,
                        "target_rank": target_rank,
                        "target_is_priority": tool_name in priority_names,
                        "target_analysis": dict(target_analysis),
                        "attack_prompt_rank": prompt_rank,
                        "analysis_attack_prompt": prompt,
                    }
                )
                rows.append(expanded)
    expanded_df: pd.DataFrame = pd.DataFrame(rows)
    if len(expanded_df) != expected:
        raise RuntimeError(f"Expected {expected} branches, produced {len(expanded_df)}.")
    if expanded_df.empty:
        raise RuntimeError("No analysis attack prompts were available.")
    return expanded_df


def judge_score(verdict: object) -> str | None:
    """Extract a categorical score from an NDD judge result."""
    if not isinstance(verdict, Mapping):
        return None
    result: object = verdict.get("attack_success")
    if isinstance(result, str):
        return result
    if not isinstance(result, Mapping):
        return None
    score: object = result.get("score")
    return score if isinstance(score, str) else None


def judge_reasoning(verdict: object) -> str:
    """Extract judge reasoning for Garak-compatible history."""
    if not isinstance(verdict, Mapping):
        return "No judge reasoning was returned."
    result: object = verdict.get("attack_success")
    candidates: list[object] = []
    if isinstance(result, Mapping):
        candidates.extend([result.get("reasoning"), result.get("explanation")])
    candidates.extend([verdict.get("reasoning"), verdict.get("explanation")])
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "No judge reasoning was returned."


@dd.custom_column_generator(required_columns=["analysis_attack_prompt"])
def select_initial_attack(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one analysis-generated prompt into attempt one."""
    prompt: object = row.get("analysis_attack_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("analysis_attack_prompt must be a non-empty string.")
    result: dict[str, Any] = dict(row)
    result["attack_1"] = prompt.strip()
    return result


@dd.custom_column_generator(
    required_columns=["attack_1_text", "victim_response_1", "verdict_1"]
)
def format_attack_history(row: Mapping[str, Any]) -> dict[str, Any]:
    """Format attempt one exactly like Garak adaptive history."""
    verdict: object = row.get("verdict_1")
    result: dict[str, Any] = dict(row)
    result["attack_history"] = (
        "--- Attempt 1 ---\n"
        f"ATTACK PROMPT: {row.get('attack_1_text', 'N/A')}\n"
        f"AGENT RESPONSE: {row.get('victim_response_1', 'N/A')}\n"
        f"SUCCESS: {judge_score(verdict) == 'YES'}\n"
        f"REASONING: {judge_reasoning(verdict)}"
    )
    return result


@dd.custom_column_generator(required_columns=["verdict_1", "verdict_2"])
def mark_successful_attack(row: Mapping[str, Any]) -> dict[str, Any]:
    """Mark whether either judged attempt was a clear success."""
    scores: tuple[str | None, str | None] = (
        judge_score(row.get("verdict_1")),
        judge_score(row.get("verdict_2")),
    )
    result: dict[str, Any] = dict(row)
    result["successful_attack"] = "YES" in scores
    return result


def tools_description(profile: Mapping[str, Any]) -> str:
    """Render enterprise tool contracts for analysis and SFT prompts."""
    tools: object = profile.get("tools", [])
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        return ""
    rendered: str = ""
    for tool in tools:
        if isinstance(tool, Mapping):
            rendered += f"\n### Tool: {tool.get('name', 'unnamed')}\n"
            rendered += f"Description: {tool.get('description', 'No description')}\n"
            rendered += f"Protected asset: {tool.get('protected_asset', 'Unspecified')}\n"
            rendered += (
                "Authorization requirement: "
                f"{tool.get('authorization_requirement', 'Unspecified')}\n"
            )
            rendered += (
                "Side effect / trust boundary: "
                f"{tool.get('side_effect_or_boundary', 'Unspecified')}\n"
            )
    return rendered


def stable_agent_id(row: pd.Series) -> str:
    """Create a stable split key from an agent domain and purpose."""
    payload: str = f"{row['domain']}|{row['profile']}"
    digest: str = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"{row['domain']}_{digest}"


def build_sft_records(
    dataframe: pd.DataFrame,
    *,
    analysis_prompt: str,
    exploit_prompt: str,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Build one analysis record per agent and successful adaptive exploits."""
    analysis_records: list[dict[str, Any]] = []
    exploit_records: list[dict[str, Any]] = []
    analysis_agent_ids: set[str] = set()
    skipped_rows: int = 0

    for _, row in dataframe.iterrows():
        try:
            full_profile = cast(Mapping[str, Any], row["full_profile"])
            analysis = cast(Mapping[str, Any], row["analysis"])
            tool_analyses = cast(
                Mapping[str, Any], analysis["tool_analyses"]
            )
            priority_targets: object = analysis["priority_targets"]
            agent_id: str = stable_agent_id(row)
        except (KeyError, TypeError):
            skipped_rows += 1
            continue

        if agent_id not in analysis_agent_ids:
            analysis_records.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": analysis_prompt.format(
                                agent_purpose=row["profile"],
                                tools_description=tools_description(
                                    full_profile
                                ),
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                json_safe(
                                    {
                                        "tool_analyses": tool_analyses,
                                        "priority_targets": priority_targets,
                                    }
                                ),
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    "agent_id": agent_id,
                    "task": "analysis",
                }
            )
            analysis_agent_ids.add(agent_id)

        if row["attempt_2_score"] not in ("YES", "PARTIAL"):
            continue
        target_analysis: object = row["target_analysis"]
        attack: object = row["attack_2"]
        if not isinstance(target_analysis, Mapping) or not isinstance(
            attack, Mapping
        ):
            skipped_rows += 1
            continue
        attack_prompt: object = attack.get("attack_prompt")
        if not isinstance(attack_prompt, str) or not attack_prompt.strip():
            skipped_rows += 1
            continue

        exploit_records.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": exploit_prompt.format(
                            target_tool=str(row["target_tool"]),
                            functionality=target_analysis.get(
                                "functionality", "Unknown"
                            ),
                            vulnerabilities=target_analysis.get(
                                "vulnerabilities", "Unknown"
                            ),
                            exploit_strategies=target_analysis.get(
                                "exploit_strategies", ""
                            ),
                            history_str=str(row["attack_history"]),
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "analysis": attack.get("analysis", ""),
                                "new_approach": attack.get(
                                    "new_approach", ""
                                ),
                                "attack_prompt": attack_prompt,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                "agent_id": agent_id,
                "task": "exploit",
            }
        )

    records: list[dict[str, Any]] = analysis_records + exploit_records
    return (
        records,
        len(analysis_records),
        len(exploit_records),
        skipped_rows,
    )


def write_chat_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write chat messages as UTF-8 JSONL, excluding split-only metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n"
            )


def parse_loss_log(path: Path) -> tuple[list[int], list[float], list[int], list[float]]:
    """Parse training and validation step/loss pairs from an AutoModel log."""
    train_steps: list[int] = []
    train_losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []
    if not path.exists():
        return train_steps, train_losses, val_steps, val_losses
    train_pattern: re.Pattern[str] = re.compile(
        r"(?:step|Step)\D*(\d+).*?(?:loss|Loss)\D*([0-9.]+)"
    )
    val_pattern: re.Pattern[str] = re.compile(
        r"(?:val|validation).*?(?:step|Step)\D*(\d+).*?(?:loss|Loss)\D*([0-9.]+)",
        re.IGNORECASE,
    )
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        validation_match: re.Match[str] | None = val_pattern.search(line)
        if validation_match:
            val_steps.append(int(validation_match.group(1)))
            val_losses.append(float(validation_match.group(2)))
            continue
        train_match: re.Match[str] | None = train_pattern.search(line)
        if train_match:
            train_steps.append(int(train_match.group(1)))
            train_losses.append(float(train_match.group(2)))
    return train_steps, train_losses, val_steps, val_losses


def serve_agent(repo: Path, agent_name: str) -> subprocess.Popen[bytes]:
    """Start one bundled victim agent on port 8000."""
    if agent_name not in {"research", "devops"}:
        raise ValueError(f"Unknown victim agent: {agent_name}")
    return subprocess.Popen(
        [sys.executable, "agents/server.py", agent_name],
        cwd=repo,
    )


def wait_for_service(
    url: str,
    *,
    timeout_seconds: float = 120.0,
    expected_status: int = 200,
) -> None:
    """Wait for an HTTP service or raise a clear timeout error."""
    deadline: float = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response: httpx.Response = httpx.get(url, timeout=2.0)
            if response.status_code == expected_status:
                return
        except httpx.HTTPError as error:
            last_error = error
        time.sleep(2.0)
    detail: str = f" Last error: {last_error}" if last_error is not None else ""
    raise TimeoutError(f"Service did not become ready at {url}.{detail}")

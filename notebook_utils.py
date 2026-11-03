"""Typed implementation helpers for the NDD and fine-tuning demo notebook.

The notebook intentionally keeps the teaching-facing NDD graphs and AutoModel
recipe visible. Repetitive persistence, retry, normalization, and process
plumbing lives here so the notebook remains readable.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import httpx
import pandas as pd
import yaml
from data_designer import config as dd
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


class ToolAnalysis(BaseModel):
    """Validated security analysis for one generated tool."""

    tool_name: str
    functionality: str
    vulnerabilities: str
    exploit_strategies: str
    attack_prompts: list[str] = Field(min_length=5, max_length=5)


class AnalysisOutput(BaseModel):
    """Validated Garak-compatible analysis response."""

    tool_analyses: list[ToolAnalysis] = Field(min_length=1)
    priority_targets: list[str] = Field(default_factory=list)


@dd.custom_column_generator(
    required_columns=["analysis_structured", "agent_profile"]
)
def assemble_tool_analysis(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert NDD's typed list into Garak's tool-name-keyed analysis object."""
    structured = AnalysisOutput.model_validate(
        profile_mapping(row["analysis_structured"], "analysis_structured")
    )
    profile = profile_mapping(row["agent_profile"], "agent_profile")
    tools: object = profile.get("tools")
    if not isinstance(tools, list):
        raise TypeError("agent_profile.tools must be a list.")
    declared_names: set[str] = {
        str(tool.get("name"))
        for tool in tools
        if isinstance(tool, Mapping) and tool.get("name")
    }

    analyses: dict[str, dict[str, Any]] = {}
    for item in structured.tool_analyses:
        name: str = item.tool_name.strip()
        prompts: list[str] = [
            prompt.strip() for prompt in item.attack_prompts
        ]
        if not name or name in analyses:
            raise ValueError(f"Duplicate or empty analyzed tool name: {name!r}")
        if len(set(prompts)) != 5 or any(not prompt for prompt in prompts):
            raise ValueError(
                f"Tool {name!r} must contain five unique non-empty prompts."
            )
        payload = item.model_dump(mode="python", exclude={"tool_name"})
        payload["attack_prompts"] = prompts
        analyses[name] = payload

    if set(analyses) != declared_names:
        raise ValueError(
            "Analysis must cover exactly the declared tools: "
            f"missing={sorted(declared_names - set(analyses))}, "
            f"unexpected={sorted(set(analyses) - declared_names)}."
        )
    result: dict[str, Any] = dict(row)
    result["analysis"] = {
        "tool_analyses": analyses,
        "priority_targets": structured.priority_targets,
    }
    return result


def validate_profile_dataframe(
    raw_frame: pd.DataFrame,
    *,
    expected_count: int,
    domain_values: Sequence[str],
    security_levels: Sequence[str],
    tools_min: int,
    tools_max: int,
    blocked_tool_names: set[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Normalize and validate one complete generated profile stage."""
    dataframe: pd.DataFrame = expose_profile_columns(raw_frame)
    if (
        len(dataframe) != expected_count
        or dataframe["profile_slot"].nunique() != expected_count
    ):
        raise RuntimeError(
            f"Profile generation returned {len(dataframe)} rows with "
            f"{dataframe['profile_slot'].nunique()} unique IDs; "
            f"expected {expected_count}."
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


def extract_judge_integer(
    verdict: object,
    *,
    score_name: str,
    minimum: int = 1,
    maximum: int = 5,
) -> int:
    """Extract and validate one integer score from an NDD judge result."""
    payload: dict[str, Any] = profile_mapping(verdict, "judge verdict")
    score_payload: object = payload.get(score_name)
    if not isinstance(score_payload, Mapping):
        raise TypeError(f"Judge verdict lacks score {score_name!r}.")
    raw_score: object = score_payload.get("score")
    if isinstance(raw_score, bool):
        raise TypeError(f"Judge score {score_name!r} must be an integer.")
    if isinstance(raw_score, int):
        score: int = raw_score
    elif isinstance(raw_score, str) and raw_score.strip().isdigit():
        score = int(raw_score.strip())
    else:
        raise TypeError(f"Judge score {score_name!r} must be an integer.")
    if not minimum <= score <= maximum:
        raise ValueError(
            f"Judge score {score_name!r}={score} is outside "
            f"{minimum}–{maximum}."
        )
    return score


def select_top_profiles(
    raw_frame: pd.DataFrame,
    *,
    candidate_count: int,
    selected_count: int,
    score_column: str = "profile_quality",
    score_name: str = "overall_quality",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank scored profile candidates and return all rows plus the selected rows."""
    if not 1 <= selected_count <= candidate_count:
        raise ValueError("selected_count must be between 1 and candidate_count.")
    required: set[str] = {
        "profile_slot",
        "domain",
        "num_tools",
        "security_level",
        "profile_core",
        "security_instruction",
        "agent_profile",
        score_column,
    }
    missing: set[str] = required - set(raw_frame.columns)
    if missing:
        raise KeyError(f"Profile candidates are missing columns: {sorted(missing)}")
    if (
        len(raw_frame) != candidate_count
        or raw_frame["profile_slot"].nunique() != candidate_count
    ):
        raise RuntimeError(
            f"Expected {candidate_count} unique profile candidates, got "
            f"{len(raw_frame)} rows and "
            f"{raw_frame['profile_slot'].nunique()} unique slots."
        )

    ranked: pd.DataFrame = raw_frame.copy(deep=True)
    ranked["profile_quality_score"] = ranked[score_column].map(
        lambda verdict: extract_judge_integer(
            verdict,
            score_name=score_name,
        )
    )

    fingerprint_fields: tuple[str, ...] = (
        "domain",
        "num_tools",
        "security_level",
        "profile_core",
        "security_instruction",
        "agent_profile",
    )

    def fingerprint(row: pd.Series) -> str:
        payload: dict[str, Any] = {
            field: json_safe(row[field]) for field in fingerprint_fields
        }
        serialized: str = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    ranked["profile_quality_tiebreak"] = ranked.apply(fingerprint, axis=1)
    ranked = ranked.sort_values(
        by=[
            "profile_quality_score",
            "profile_quality_tiebreak",
            "profile_slot",
        ],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["profile_selection_rank"] = pd.Series(
        range(1, len(ranked) + 1),
        dtype="int64",
    )
    ranked["selected_for_analysis"] = (
        ranked["profile_selection_rank"] <= selected_count
    )
    selected: pd.DataFrame = ranked.loc[
        ranked["selected_for_analysis"]
    ].copy(deep=True)
    if len(selected) != selected_count:
        raise RuntimeError(
            f"Profile selection returned {len(selected)} rows; "
            f"expected {selected_count}."
        )
    return ranked, selected


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
    accepted_outcomes: Sequence[str] = ("YES", "PARTIAL"),
    include_analysis_records: bool = True,
    include_exploit_records: bool = True,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Build one analysis record per agent and successful adaptive exploits."""
    accepted = {str(outcome).upper() for outcome in accepted_outcomes}
    if not accepted or not accepted <= {"YES", "PARTIAL", "NO"}:
        raise ValueError(
            "accepted_outcomes must contain one or more of YES, PARTIAL, and NO."
        )
    if not include_analysis_records and not include_exploit_records:
        raise ValueError("At least one SFT record type must be enabled.")
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

        if include_analysis_records and agent_id not in analysis_agent_ids:
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

        if (
            not include_exploit_records
            or str(row["attempt_2_score"]).upper() not in accepted
        ):
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


def count_jsonl_records(path: Path) -> int:
    """Count non-empty JSONL records and fail on malformed JSON."""
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed JSONL at {path}:{line_number}: {error.msg}"
                ) from error
            messages = payload.get("messages") if isinstance(payload, dict) else None
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(
                    f"Invalid chat record at {path}:{line_number}: "
                    "messages must contain at least two turns."
                )
            for message in messages:
                if (
                    not isinstance(message, dict)
                    or message.get("role") not in {"system", "user", "assistant"}
                    or not isinstance(message.get("content"), str)
                    or not message["content"].strip()
                ):
                    raise ValueError(
                        f"Invalid chat message at {path}:{line_number}: "
                        "role and non-empty content are required."
                    )
            count += 1
    return count


def prepare_training_recipe(
    *,
    template_path: Path,
    output_path: Path,
    checkpoint_dir: Path,
    dataset_mode: str,
    bundled_train_path: Path,
    bundled_validation_path: Path,
    generated_train_path: Path,
    generated_validation_path: Path,
    base_model: str,
    epochs: int,
    global_batch_size: int,
    local_batch_size: int,
    learning_rate: float,
    optimizer: str,
    lr_scheduler: str,
    min_learning_rate: float,
    weight_decay: float,
    lora_dim: int,
    lora_alpha: int,
    seed: int,
) -> dict[str, Any]:
    """Resolve workshop controls into one validated AutoModel run recipe."""
    mode = dataset_mode.strip().lower()
    if mode not in {"bundled", "generated"}:
        raise ValueError("dataset_mode must be 'bundled' or 'generated'.")
    if optimizer.strip().lower() != "adamw":
        raise ValueError("This workshop recipe currently supports optimizer='adamw'.")
    if lr_scheduler.strip().lower() != "cosine":
        raise ValueError(
            "This workshop recipe currently supports lr_scheduler='cosine'."
        )
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    if local_batch_size < 1 or global_batch_size < local_batch_size:
        raise ValueError(
            "Batch sizes must be positive and global_batch_size must be at least "
            "local_batch_size."
        )
    if learning_rate <= 0 or min_learning_rate < 0:
        raise ValueError("Learning rates must be non-negative and lr must be positive.")
    if min_learning_rate > learning_rate:
        raise ValueError("min_learning_rate cannot exceed learning_rate.")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")
    if lora_dim < 1 or lora_alpha < 1:
        raise ValueError("LoRA dim and alpha must be positive.")

    if mode == "bundled":
        train_path = bundled_train_path
        validation_path = bundled_validation_path
    else:
        train_path = generated_train_path
        validation_path = generated_validation_path
    for label, path in (
        ("training", train_path),
        ("validation", validation_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Resolved {label} dataset does not exist: {path}")

    train_records = count_jsonl_records(train_path)
    validation_records = count_jsonl_records(validation_path)
    if not train_records or not validation_records:
        raise ValueError("Training and validation datasets must both be non-empty.")

    effective_global_batch = min(global_batch_size, train_records)
    if effective_global_batch < local_batch_size:
        raise ValueError(
            "The selected dataset has fewer records than local_batch_size."
        )
    if effective_global_batch % local_batch_size:
        raise ValueError(
            "Effective global_batch_size must be divisible by local_batch_size."
        )
    steps_per_epoch = max(
        1,
        (train_records + effective_global_batch - 1)
        // effective_global_batch,
    )
    max_steps = steps_per_epoch * epochs

    with template_path.open(encoding="utf-8") as recipe_file:
        recipe: dict[str, Any] = yaml.safe_load(recipe_file)
    recipe["model"]["pretrained_model_name_or_path"] = base_model
    recipe["dataset"]["path_or_dataset_id"] = str(train_path)
    recipe["validation_dataset"]["path_or_dataset_id"] = str(validation_path)
    recipe["checkpoint"]["checkpoint_dir"] = str(checkpoint_dir)
    recipe["rng"]["seed"] = seed
    recipe["peft"]["dim"] = lora_dim
    recipe["peft"]["alpha"] = lora_alpha
    recipe["optimizer"]["_target_"] = "torch.optim.AdamW"
    recipe["optimizer"]["lr"] = learning_rate
    recipe["optimizer"]["weight_decay"] = weight_decay
    recipe["lr_scheduler"]["lr_decay_style"] = "cosine"
    recipe["lr_scheduler"]["min_lr"] = min_learning_rate
    recipe["step_scheduler"].update(
        {
            "num_epochs": epochs,
            "global_batch_size": effective_global_batch,
            "local_batch_size": local_batch_size,
            "max_steps": max_steps,
            "val_every_steps": max(1, max_steps // 5),
            "ckpt_every_steps": max_steps,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as run_recipe_file:
        yaml.safe_dump(recipe, run_recipe_file, sort_keys=False)

    return {
        "dataset_mode": mode,
        "train_path": train_path,
        "validation_path": validation_path,
        "train_records": train_records,
        "validation_records": validation_records,
        "epochs": epochs,
        "global_batch_size": effective_global_batch,
        "local_batch_size": local_batch_size,
        "gradient_accumulation_steps": max(
            1, effective_global_batch // local_batch_size
        ),
        "steps_per_epoch": steps_per_epoch,
        "max_steps": max_steps,
        "warmup_steps": int(max_steps * 0.1),
        "val_every_steps": recipe["step_scheduler"]["val_every_steps"],
        "checkpoint_every_steps": recipe["step_scheduler"]["ckpt_every_steps"],
        "base_model": base_model,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "lr_scheduler": "cosine",
        "min_learning_rate": min_learning_rate,
        "weight_decay": weight_decay,
        "lora_dim": lora_dim,
        "lora_alpha": lora_alpha,
        "lora_scaling": lora_alpha / lora_dim,
        "seed": seed,
        "recipe_path": output_path,
        "checkpoint_dir": checkpoint_dir,
    }


def build_training_command(
    *,
    repo: Path,
    recipe_path: Path,
    gpu_id: int,
    backend: str = "auto",
    image: str = (
        "nvcr.io/nvidia/nemo-automodel@"
        "sha256:c4f613005518d520c2ac3d9206d95617a2385f86cf8aa09582aad8d35957e2f2"
    ),
) -> tuple[list[str], str]:
    """Build the exact one-GPU native or Docker training command."""
    requested = backend.strip().lower()
    if requested not in {"auto", "native", "docker"}:
        raise ValueError("backend must be 'auto', 'native', or 'docker'.")
    resolved = (
        "native"
        if requested == "auto" and shutil.which("torchrun") is not None
        else "docker"
        if requested == "auto"
        else requested
    )
    native_command = [
        "torchrun",
        "--nproc-per-node=1",
        "training/finetune.py",
        "--config",
        str(recipe_path),
    ]
    if resolved == "native":
        return native_command, resolved
    return (
        [
            "docker",
            "run",
            "--gpus",
            f"device={gpu_id}",
            "--ipc=host",
            "--rm",
            "-e",
            "NVIDIA_API_KEY",
            "-v",
            "finetune-demo-hf-cache:/root/.cache/huggingface",
            "-v",
            f"{repo}:{repo}",
            "-w",
            str(repo),
            image,
            *native_command,
        ],
        resolved,
    )


def run_streaming_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run a command while streaming identical output to the notebook and a log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    elapsed_seconds = time.monotonic() - started
    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}. Inspect {log_path}."
        )
    return {
        "command": list(command),
        "log_path": log_path,
        "elapsed_seconds": elapsed_seconds,
        "return_code": return_code,
    }


def find_best_lora_adapter(checkpoint_dir: Path) -> Path:
    """Locate AutoModel's best validation adapter with a clear failure."""
    preferred = checkpoint_dir / "LOWEST_VAL" / "model"
    if (preferred / "adapter_config.json").is_file():
        return preferred
    candidates = sorted(checkpoint_dir.glob("**/adapter_config.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No LoRA adapter_config.json was found below {checkpoint_dir}."
        )
    return candidates[-1].parent


def build_merge_command(
    *,
    repo: Path,
    adapter_path: Path,
    output_path: Path,
    base_model: str,
    gpu_id: int,
    backend: str = "auto",
    image: str = (
        "nvcr.io/nvidia/nemo-automodel@"
        "sha256:c4f613005518d520c2ac3d9206d95617a2385f86cf8aa09582aad8d35957e2f2"
    ),
) -> tuple[list[str], str]:
    """Build the native or Docker command that merges a trained LoRA adapter."""
    requested = backend.strip().lower()
    if requested not in {"auto", "native", "docker"}:
        raise ValueError("backend must be 'auto', 'native', or 'docker'.")
    native_available = (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
        and importlib.util.find_spec("peft") is not None
    )
    resolved = (
        "native"
        if requested == "auto" and native_available
        else "docker"
        if requested == "auto"
        else requested
    )
    native_command = [
        sys.executable if resolved == "native" else "python",
        "training/merge_lora.py",
        "--adapter",
        str(adapter_path),
        "--output",
        str(output_path),
        "--base-model",
        base_model,
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
    ]
    if resolved == "native":
        return native_command, resolved
    return (
        [
            "docker",
            "run",
            "--gpus",
            f"device={gpu_id}",
            "--ipc=host",
            "--rm",
            "-v",
            "finetune-demo-hf-cache:/root/.cache/huggingface",
            "-v",
            f"{repo}:{repo}",
            "-w",
            str(repo),
            image,
            *native_command,
        ],
        resolved,
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


def serve_agent(
    repo: Path,
    agent_name: str,
    *,
    state_dir: Path | None = None,
) -> subprocess.Popen[bytes]:
    """Start one bundled victim agent on port 8000 with optional isolated state."""
    if agent_name not in {"research", "devops"}:
        raise ValueError(f"Unknown victim agent: {agent_name}")
    env = os.environ.copy()
    if state_dir is not None:
        state_dir.mkdir(parents=True, exist_ok=True)
        env["AGENT_OUTBOX_PATH"] = str(
            state_dir / f"outbox-{uuid.uuid4().hex}.json"
        )
    return subprocess.Popen(
        [sys.executable, "agents/server.py", agent_name],
        cwd=repo,
        env=env,
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

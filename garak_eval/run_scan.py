"""Drive Garak's AgentBreaker probe against a running victim agent.

This renders a Garak run-config (based on `scan_agent_breaker.yaml`) that:
  * attacks whatever serves http://localhost:8000/v1/chat/completions (the agent),
  * uses a chosen *red-team model* as the attacker (the base vs. fine-tuned
    Nemotron under test, served locally by vLLM, or a hosted comparison model),
  * uses one hosted *detector model* on build.nvidia.com as the judge/evaluator.

Then it invokes `garak --config <rendered>.yaml --report_prefix <name>` and returns
the path to the resulting report.jsonl. Garak also writes a sibling
`<name>.hitlog.jsonl` with one entry per landed attack.

All inference is OpenAI-compatible via the `nim.NVOpenAIChat` generator (base_url =
`uri`, key from NIM_API_KEY) — for a local vLLM endpoint the key is ignored.

Usage (from the demo repo root, with the garak checkout available):
    python garak_eval/run_scan.py \
        --garak-dir /path/to/garak \
        --report-prefix finetuned_research \
        --red-team-uri http://localhost:8001/v1 \
        --red-team-model nemotron-attacker \
        --detector-uri https://integrate.api.nvidia.com/v1 \
        --detector-model nvidia/llama-3.3-nemotron-super-49b-v1
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# build.nvidia.com OpenAI-compatible endpoint
NVIDIA_BUILD_URI = "https://integrate.api.nvidia.com/v1"

_SUPPRESSED = ["stop", "top_p", "frequency_penalty", "presence_penalty", "seed"]


def build_scan_config(
    *,
    red_team_uri: str,
    red_team_model: str,
    detector_uri: str = NVIDIA_BUILD_URI,
    detector_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1",
    agent_uri: str = "http://localhost:8000/v1/chat/completions",
    agent_config_file: str = "agent_breaker/agent.yaml",
    generations: int = 1,
    red_team_max_tokens: int = 7000,
    detector_max_tokens: int = 4096,
    parse_model_max_tokens: int = 8192,
    max_calls_per_conversation: int = 4,
    max_attempts_per_tool: int = 1,
) -> dict:
    """Return a Garak run-config dict wiring red-team + detector + target agent.

    Both the red-team attacker and the detector/judge are OpenAI-compatible
    `nim.NVOpenAIChat` generators, so a local vLLM endpoint and a hosted
    build.nvidia.com endpoint use the same plugin, differing only by `uri`.
    """
    model_block = {
        "uri": red_team_uri,
        "max_tokens": red_team_max_tokens,
        "temperature": 0.0,
        "vary_seed_each_call": False,
        "vary_temp_each_call": False,
        "extra_params": {"response_format": {"type": "json_object"}},
        "suppressed_params": copy.copy(_SUPPRESSED),
    }
    detector_block = {
        "uri": detector_uri,
        "max_tokens": detector_max_tokens,
        "temperature": 0.0,
        "vary_seed_each_call": False,
        "vary_temp_each_call": False,
        "suppressed_params": copy.copy(_SUPPRESSED),
    }
    parse_block = detector_block | {"max_tokens": parse_model_max_tokens}
    return {
        "run": {"generations": generations},
        "plugins": {
            "target_type": "rest.RestGenerator",
            "target_name": "victim-agent",
            "probe_spec": "agent_breaker.AgentBreaker",
            "extended_detectors": ["agent_breaker.AgentBreakerResult"],
            "generators": {
                "rest": {
                    "RestGenerator": {
                        "uri": agent_uri,
                        "headers": {"Content-Type": "application/json"},
                        "req_template_json_object": {
                            "model": "garak",
                            "messages": [{"role": "user", "content": "$INPUT"}],
                        },
                        "response_json": True,
                        "response_json_field": "$.choices[0].message.content",
                        "request_timeout": 240,
                        "max_tokens": 4096,
                    }
                }
            },
            "probes": {
                "agent_breaker": {
                    "AgentBreaker": {
                        "agent_config_file": agent_config_file,
                        "red_team_model_type": "nim.NVOpenAIChat",
                        "red_team_model_name": red_team_model,
                        "red_team_model_config": model_block,
                        "parse_model_type": "nim.NVOpenAIChat",
                        "parse_model_name": detector_model,
                        "parse_model_config": parse_block,
                        "max_calls_per_conv": max_calls_per_conversation,
                        "max_attempts_per_tool": max_attempts_per_tool,
                    }
                }
            },
            "detectors": {
                "agent_breaker": {
                    "AgentBreakerResult": {
                        "detector_model_type": "nim.NVOpenAIChat",
                        "detector_model_name": detector_model,
                        "detector_model_config": detector_block,
                    }
                }
            },
        },
    }


def render_config(config: dict, out_path: str | Path) -> Path:
    """Write the run-config dict to a YAML file and return its path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    return out_path


def run_garak(
    *,
    garak_dir: str | Path,
    config_path: str | Path,
    report_prefix: str,
    nim_api_key: str | None = None,
    extra_env: dict | None = None,
) -> Path:
    """Run `garak --config <config> --report_prefix <prefix>` and return report.jsonl.

    `nim_api_key` populates NIM_API_KEY for the hosted detector (and any hosted
    red-team/comparison model); a local vLLM red-team ignores it. The garak
    checkout is run as `python -m garak` from `garak_dir`.
    """
    garak_dir = Path(garak_dir)
    env = os.environ.copy()
    if nim_api_key:
        env.setdefault("NIM_API_KEY", nim_api_key)
    if extra_env:
        env.update(extra_env)

    runtime_home = Path(env.get("GARAK_RUNTIME_HOME", Path.home()))
    runtime_home.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(runtime_home)
    env["XDG_CACHE_HOME"] = str(runtime_home / ".cache")
    env["XDG_CONFIG_HOME"] = str(runtime_home / ".config")
    env["XDG_DATA_HOME"] = str(runtime_home / ".local" / "share")
    env["TMPDIR"] = str(runtime_home / "tmp")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    resolved_config_path = Path(config_path).resolve()
    docker_image = env.get("GARAK_DOCKER_IMAGE")
    if docker_image:
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "host",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            f"HOME={runtime_home}",
            "-e",
            f"XDG_CACHE_HOME={env['XDG_CACHE_HOME']}",
            "-e",
            f"XDG_CONFIG_HOME={env['XDG_CONFIG_HOME']}",
            "-e",
            f"XDG_DATA_HOME={env['XDG_DATA_HOME']}",
            "-e",
            f"TMPDIR={env['TMPDIR']}",
            "-e",
            "NIM_API_KEY",
            "-v",
            f"{runtime_home}:{runtime_home}",
            "-v",
            f"{resolved_config_path.parent}:{resolved_config_path.parent}:ro",
            docker_image,
            "--config",
            str(resolved_config_path),
            "--report_prefix",
            report_prefix,
        ]
        cwd = None
    else:
        garak_python = env.get("GARAK_PYTHON", sys.executable)
        cmd = [
            garak_python,
            "-m",
            "garak",
            "--config",
            str(resolved_config_path),
            "--report_prefix",
            report_prefix,
        ]
        cwd = str(garak_dir)

    print(f"[run_scan] launching: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)

    # Default report location: ~/.local/share/garak/garak_runs/<prefix>.report.jsonl
    runs_dir = runtime_home / ".local" / "share" / "garak" / "garak_runs"
    report = runs_dir / f"{report_prefix}.report.jsonl"
    if not report.exists():
        # Fall back to a search in case the data dir is customized.
        matches = sorted(runs_dir.glob(f"{report_prefix}*.report.jsonl"))
        if matches:
            report = matches[-1]
    if not report.is_file():
        raise FileNotFoundError(
            f"Garak exited successfully but no report was found for {report_prefix} "
            f"below {runs_dir}."
        )
    print(f"[run_scan] report: {report}", flush=True)
    return report


def scan(
    *,
    garak_dir: str | Path,
    report_prefix: str,
    red_team_uri: str,
    red_team_model: str,
    detector_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1",
    detector_uri: str = NVIDIA_BUILD_URI,
    nim_api_key: str | None = None,
    config_out: str | Path | None = None,
    agent_uri: str = "http://localhost:8000/v1/chat/completions",
    agent_config_file: str = "agent_breaker/agent.yaml",
    generations: int = 1,
    max_calls_per_conversation: int = 4,
    max_attempts_per_tool: int = 1,
) -> Path:
    """Convenience: build config, render it, run garak, return the report path."""
    agent_config_path = Path(agent_config_file)
    if agent_config_path.is_file():
        runtime_home = Path(os.environ.get("GARAK_RUNTIME_HOME", Path.home()))
        runtime_config_dir = (
            runtime_home / ".local" / "share" / "garak" / "data" / "agent_breaker"
        )
        runtime_config_dir.mkdir(parents=True, exist_ok=True)
        staged_config = runtime_config_dir / agent_config_path.name
        shutil.copy2(agent_config_path, staged_config)
        agent_config_file = f"agent_breaker/{staged_config.name}"

    cfg = build_scan_config(
        red_team_uri=red_team_uri,
        red_team_model=red_team_model,
        detector_uri=detector_uri,
        detector_model=detector_model,
        agent_uri=agent_uri,
        agent_config_file=agent_config_file,
        generations=generations,
        max_calls_per_conversation=max_calls_per_conversation,
        max_attempts_per_tool=max_attempts_per_tool,
    )
    config_out = config_out or (Path("garak_eval") / f"scan_{report_prefix}.yaml")
    render_config(cfg, config_out)
    return run_garak(
        garak_dir=garak_dir,
        config_path=config_out,
        report_prefix=report_prefix,
        nim_api_key=nim_api_key,
    )


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run Garak AgentBreaker against a local agent.")
    p.add_argument("--garak-dir", required=True, help="Path to the garak checkout")
    p.add_argument("--report-prefix", required=True, help="Report filename prefix")
    p.add_argument("--red-team-uri", required=True, help="OpenAI base_url of the attacker model")
    p.add_argument("--red-team-model", required=True, help="Attacker model name")
    p.add_argument("--detector-uri", default=NVIDIA_BUILD_URI)
    p.add_argument(
        "--detector-model", default="nvidia/llama-3.3-nemotron-super-49b-v1"
    )
    p.add_argument(
        "--nim-api-key",
        default=os.environ.get("NVIDIA_API_KEY"),
        help="Key for hosted models (defaults to $NVIDIA_API_KEY)",
    )
    p.add_argument(
        "--agent-uri",
        default="http://localhost:8000/v1/chat/completions",
    )
    p.add_argument(
        "--agent-config-file",
        default="agent_breaker/agent.yaml",
    )
    p.add_argument("--config-out", default=None)
    p.add_argument("--generations", type=int, default=1)
    p.add_argument("--max-calls-per-conversation", type=int, default=4)
    p.add_argument("--max-attempts-per-tool", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    report = scan(
        garak_dir=args.garak_dir,
        report_prefix=args.report_prefix,
        red_team_uri=args.red_team_uri,
        red_team_model=args.red_team_model,
        detector_uri=args.detector_uri,
        detector_model=args.detector_model,
        nim_api_key=args.nim_api_key,
        agent_uri=args.agent_uri,
        agent_config_file=args.agent_config_file,
        config_out=args.config_out,
        generations=args.generations,
        max_calls_per_conversation=args.max_calls_per_conversation,
        max_attempts_per_tool=args.max_attempts_per_tool,
    )
    print(report)


if __name__ == "__main__":
    main()

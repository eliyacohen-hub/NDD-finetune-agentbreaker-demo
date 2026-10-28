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
    generations: int = 1,
    red_team_max_tokens: int = 4096,
    detector_max_tokens: int = 1024,
) -> dict:
    """Return a Garak run-config dict wiring red-team + detector + target agent.

    Both the red-team attacker and the detector/judge are OpenAI-compatible
    `nim.NVOpenAIChat` generators, so a local vLLM endpoint and a hosted
    build.nvidia.com endpoint use the same plugin, differing only by `uri`.
    """
    model_block = {
        "uri": red_team_uri,
        "max_tokens": red_team_max_tokens,
        "suppressed_params": copy.copy(_SUPPRESSED),
    }
    detector_block = {
        "uri": detector_uri,
        "max_tokens": detector_max_tokens,
        "suppressed_params": copy.copy(_SUPPRESSED),
    }
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
                        "agent_config_file": "agent_breaker/agent.yaml",
                        "red_team_model_type": "nim.NVOpenAIChat",
                        "red_team_model_name": red_team_model,
                        "red_team_model_config": model_block,
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

    cmd = [
        sys.executable,
        "-m",
        "garak",
        "--config",
        str(Path(config_path).resolve()),
        "--report_prefix",
        report_prefix,
    ]
    print(f"[run_scan] launching: {' '.join(cmd)} (cwd={garak_dir})", flush=True)
    subprocess.run(cmd, cwd=str(garak_dir), env=env, check=True)

    # Default report location: ~/.local/share/garak/garak_runs/<prefix>.report.jsonl
    runs_dir = Path.home() / ".local" / "share" / "garak" / "garak_runs"
    report = runs_dir / f"{report_prefix}.report.jsonl"
    if not report.exists():
        # Fall back to a search in case the data dir is customized.
        matches = sorted(runs_dir.glob(f"{report_prefix}*.report.jsonl"))
        if matches:
            report = matches[-1]
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
) -> Path:
    """Convenience: build config, render it, run garak, return the report path."""
    cfg = build_scan_config(
        red_team_uri=red_team_uri,
        red_team_model=red_team_model,
        detector_uri=detector_uri,
        detector_model=detector_model,
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
    )
    print(report)


if __name__ == "__main__":
    main()

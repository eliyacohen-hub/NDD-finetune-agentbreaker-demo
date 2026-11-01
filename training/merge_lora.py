"""Merge a LoRA adapter into its base model for serving.

NeMo AutoModel trains a LoRA adapter; to serve it with vLLM as a plain model you
can either keep the adapter separate (`vllm serve ... --enable-lora`) or merge it
into the base weights. This script does the merge and, crucially for Nemotron Nano,
copies the base model's auxiliary hub files (custom tokenizer / modeling / chat
template) into the merged output so it loads standalone.

    python training/merge_lora.py \
        --base-model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
        --adapter runs/attacker_lora/checkpoints \
        --output  runs/attacker_merged \
        --trust-remote-code --dtype bfloat16

SECURITY: --trust-remote-code is required (Nemotron ships custom Python on the Hub).
Only merge checkpoints you trained from the official NVIDIA base checkpoint.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Auxiliary hub files that must travel with the merged model.
_AUX_PREFIXES = ("tokenization_", "configuration_", "modeling_", "nano_v3")
_AUX_SUFFIXES = (".jinja",)
_AUX_NAMES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def _copy_aux_files(base_path: Path, output: Path) -> list[str]:
    """Copy hub Python / tokenizer / template files from base into the merged dir."""
    copied = []
    for f in base_path.glob("*"):
        if not f.is_file():
            continue
        name = f.name
        keep = (
            name.startswith(_AUX_PREFIXES)
            or name.endswith(_AUX_SUFFIXES)
            or name in _AUX_NAMES
            or (name.endswith(".py") and "parser" in name)
        )
        if keep and not (output / name).exists():
            shutil.copy2(f, output / name)
            copied.append(name)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model.")
    parser.add_argument("--base-model", required=True, help="Base model HF id or local path")
    parser.add_argument("--adapter", required=True, type=Path, help="Trained LoRA adapter / checkpoint dir")
    parser.add_argument("--output", required=True, type=Path, help="Output dir for the merged model")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Required for Nemotron (executes Python from the Hub).",
    )
    args = parser.parse_args()

    if not args.trust_remote_code:
        sys.exit(
            "ERROR: refusing to load without --trust-remote-code "
            "(Nemotron requires hub Python; supply-chain risk)."
        )

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    print(f"Applying adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, str(args.adapter))
    print("Merging ...")
    model = model.merge_and_unload()

    generation_config = getattr(model, "generation_config", None)
    if (
        generation_config is not None
        and not generation_config.do_sample
        and generation_config.top_p != 1.0
    ):
        print(
            "Resetting inactive top_p to 1.0 so the greedy generation config "
            "passes Transformers validation."
        )
        generation_config.top_p = 1.0

    print(f"Saving merged model -> {args.output}")
    model.save_pretrained(str(args.output), safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True).save_pretrained(
        str(args.output)
    )

    # Copy the base model's auxiliary hub files so the merged model loads standalone.
    base_path = Path(args.base_model)
    if not base_path.exists():
        base_path = Path(
            snapshot_download(args.base_model, allow_patterns=["*.py", "*.json", "*.jinja"])
        )
    copied = _copy_aux_files(base_path, args.output)
    print(f"Copied {len(copied)} auxiliary files: {copied}")
    print("Done.")


if __name__ == "__main__":
    main()

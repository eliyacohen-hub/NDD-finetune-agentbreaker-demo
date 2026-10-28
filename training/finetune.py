"""Fine-tuning entry point for the NeMo Data Designer -> Nemotron demo.

Thin wrapper around NeMo AutoModel's LLM fine-tuning recipe. Launch it with
torchrun so FSDP2 initializes the (single- or multi-GPU) process group:

    torchrun --nproc-per-node=1 training/finetune.py --config training/automodel_lora.yaml

It loads the YAML recipe, builds the recipe object, and runs the
train/validation loop. Mirrors the Agent Breaker reference entrypoint.
"""

from __future__ import annotations

from pathlib import Path

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import (
    TrainFinetuneRecipeForNextTokenPrediction,
)

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "automodel_lora.yaml"


def main(default_config_path: str | Path | None = None) -> None:
    """Load the config, set up the recipe, and run the training loop."""
    path = str(default_config_path or _DEFAULT_CONFIG)
    try:
        cfg = parse_args_and_load_config(path)
        recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
        recipe.setup()
        recipe.run_train_validation_loop()
    except FileNotFoundError as e:
        raise RuntimeError(f"Configuration file not found: {path}") from e
    except Exception as e:  # noqa: BLE001 - surface the real training error
        raise RuntimeError(f"Fine-tuning failed: {e}") from e


if __name__ == "__main__":
    main()

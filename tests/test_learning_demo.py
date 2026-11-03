from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from garak_eval.compare import compare_run_matrix, hit_prompt_text
from notebook_utils import (
    build_merge_command,
    build_training_command,
    extract_judge_integer,
    prepare_training_recipe,
    select_top_profiles,
    serve_agent,
)


def write_jsonl(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"prompt {index}"},
                            {"role": "assistant", "content": f"answer {index}"},
                        ]
                    }
                )
                + "\n"
            )


def write_recipe(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "step_scheduler": {
                    "global_batch_size": 8,
                    "local_batch_size": 1,
                    "max_steps": 1,
                    "val_every_steps": 1,
                    "ckpt_every_steps": 1,
                },
                "rng": {"seed": 0},
                "model": {"pretrained_model_name_or_path": "old"},
                "peft": {"dim": 4, "alpha": 8},
                "dataset": {"path_or_dataset_id": "old-train"},
                "validation_dataset": {"path_or_dataset_id": "old-val"},
                "optimizer": {
                    "_target_": "torch.optim.AdamW",
                    "lr": 1e-4,
                    "weight_decay": 0.0,
                },
                "lr_scheduler": {
                    "lr_decay_style": "cosine",
                    "min_lr": 0.0,
                },
                "checkpoint": {"checkpoint_dir": "old-checkpoints"},
            }
        ),
        encoding="utf-8",
    )


def profile_candidate(slot: str, score: int, *, purpose: str | None = None) -> dict:
    purpose = purpose or f"Purpose for {slot}"
    profile_core = {
        "agent_purpose": purpose,
        "system_prompt": f"System prompt for {purpose}",
    }
    agent_profile = {
        "agent_purpose": purpose,
        "tools": [{"name": f"tool_{slot}", "description": "Demo tool"}],
        "system_prompt": profile_core["system_prompt"],
    }
    return {
        "profile_slot": slot,
        "domain": "demo",
        "num_tools": 1,
        "security_level": "permissive",
        "profile_core": profile_core,
        "security_instruction": "",
        "agent_profile": agent_profile,
        "profile_quality": {
            "overall_quality": {
                "score": score,
                "reasoning": f"Quality {score}",
            }
        },
    }


class ProfileSelectionTests(unittest.TestCase):
    def test_extracts_integer_score_and_rejects_bad_values(self) -> None:
        self.assertEqual(
            extract_judge_integer(
                {"overall_quality": {"score": 5}},
                score_name="overall_quality",
            ),
            5,
        )
        with self.assertRaises(TypeError):
            extract_judge_integer(
                {"overall_quality": {"score": True}},
                score_name="overall_quality",
            )
        with self.assertRaises(ValueError):
            extract_judge_integer(
                {"overall_quality": {"score": 6}},
                score_name="overall_quality",
            )

    def test_selects_two_profiles_independent_of_input_order(self) -> None:
        frame = pd.DataFrame(
            [
                profile_candidate("profile-c", 3),
                profile_candidate("profile-a", 5),
                profile_candidate("profile-e", 2),
                profile_candidate("profile-b", 5),
                profile_candidate("profile-d", 4),
            ]
        )
        ranked, selected = select_top_profiles(
            frame,
            candidate_count=5,
            selected_count=2,
        )
        _, reversed_selected = select_top_profiles(
            frame.iloc[::-1].reset_index(drop=True),
            candidate_count=5,
            selected_count=2,
        )
        self.assertEqual(len(ranked), 5)
        self.assertEqual(ranked["profile_selection_rank"].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(int(ranked["selected_for_analysis"].sum()), 2)
        self.assertEqual(
            selected["profile_slot"].tolist(),
            reversed_selected["profile_slot"].tolist(),
        )

    def test_exact_content_tie_falls_back_to_profile_slot(self) -> None:
        rows = [
            profile_candidate("profile-b", 5, purpose="Identical"),
            profile_candidate("profile-a", 5, purpose="Identical"),
            profile_candidate("profile-c", 4),
            profile_candidate("profile-d", 3),
            profile_candidate("profile-e", 2),
        ]
        rows[0]["agent_profile"]["tools"] = [{"name": "same", "description": "same"}]
        rows[1]["agent_profile"]["tools"] = [{"name": "same", "description": "same"}]
        ranked, selected = select_top_profiles(
            pd.DataFrame(rows),
            candidate_count=5,
            selected_count=2,
        )
        self.assertEqual(selected["profile_slot"].tolist(), ["profile-a", "profile-b"])
        self.assertTrue(
            selected["profile_quality_tiebreak"].nunique() == 1
        )
        self.assertEqual(ranked.iloc[0]["profile_selection_rank"], 1)


class TrainingPlanTests(unittest.TestCase):
    def test_generated_recipe_resolves_all_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.yaml"
            output = root / "run.yaml"
            write_recipe(template)
            bundled_train = root / "bundled-train.jsonl"
            bundled_val = root / "bundled-val.jsonl"
            generated_train = root / "generated-train.jsonl"
            generated_val = root / "generated-val.jsonl"
            for path, count in (
                (bundled_train, 8),
                (bundled_val, 2),
                (generated_train, 5),
                (generated_val, 2),
            ):
                write_jsonl(path, count)

            plan = prepare_training_recipe(
                template_path=template,
                output_path=output,
                checkpoint_dir=root / "checkpoints",
                dataset_mode="generated",
                bundled_train_path=bundled_train,
                bundled_validation_path=bundled_val,
                generated_train_path=generated_train,
                generated_validation_path=generated_val,
                base_model="nvidia/test-model",
                epochs=2,
                global_batch_size=8,
                local_batch_size=1,
                learning_rate=2e-5,
                optimizer="adamw",
                lr_scheduler="cosine",
                min_learning_rate=1e-6,
                weight_decay=0.01,
                lora_dim=16,
                lora_alpha=32,
                seed=42,
            )

            recipe = yaml.safe_load(output.read_text(encoding="utf-8"))
            self.assertEqual(plan["train_path"], generated_train)
            self.assertEqual(plan["global_batch_size"], 5)
            self.assertEqual(plan["steps_per_epoch"], 1)
            self.assertEqual(plan["max_steps"], 2)
            self.assertEqual(plan["lora_scaling"], 2)
            self.assertEqual(recipe["peft"], {"dim": 16, "alpha": 32})
            self.assertEqual(recipe["optimizer"]["lr"], 2e-5)

    def test_docker_training_command_is_one_gpu(self) -> None:
        command, backend = build_training_command(
            repo=Path("/repo"),
            recipe_path=Path("/repo/run.yaml"),
            gpu_id=3,
            backend="docker",
            image="image:test",
        )
        self.assertEqual(backend, "docker")
        self.assertIn("device=3", command)
        self.assertIn("--nproc-per-node=1", command)
        self.assertIn(
            "finetune-demo-hf-cache:/root/.cache/huggingface",
            command,
        )

    def test_docker_merge_reuses_hugging_face_cache(self) -> None:
        command, backend = build_merge_command(
            repo=Path("/repo"),
            adapter_path=Path("/repo/adapter"),
            output_path=Path("/repo/merged"),
            base_model="nvidia/test-model",
            gpu_id=0,
            backend="docker",
        )
        self.assertEqual(backend, "docker")
        self.assertIn(
            "finetune-demo-hf-cache:/root/.cache/huggingface",
            command,
        )


class ComparisonTests(unittest.TestCase):
    @staticmethod
    def _report(path: Path, total: int, hits: int) -> None:
        path.write_text(
            json.dumps(
                {
                    "entry_type": "eval",
                    "detector": "agent_breaker.AgentBreakerResult",
                    "fails": hits,
                    "passed": total - hits,
                    "total_evaluated": total,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_matrix_normalizes_to_configured_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.report.jsonl"
            tuned = root / "tuned.report.jsonl"
            self._report(base, total=2, hits=1)
            self._report(tuned, total=3, hits=2)
            rows = compare_run_matrix(
                {
                    ("base", "research"): base,
                    ("finetuned", "research"): tuned,
                },
                expected_attempts_per_run=3,
            )
            self.assertEqual([row["total"] for row in rows], [3, 3])
            self.assertEqual([row["evaluated"] for row in rows], [2, 3])
            self.assertEqual(
                [row["invalid_or_skipped"] for row in rows], [1, 0]
            )
            self.assertAlmostEqual(rows[0]["asr"], 1 / 3)

    def test_raw_matrix_rejects_unmatched_denominators(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.report.jsonl"
            tuned = root / "tuned.report.jsonl"
            self._report(base, total=2, hits=1)
            self._report(tuned, total=3, hits=2)
            with self.assertRaisesRegex(ValueError, "not matched"):
                compare_run_matrix(
                    {
                        ("base", "research"): base,
                        ("finetuned", "research"): tuned,
                    }
                )

    def test_nested_hit_prompt_is_rendered(self) -> None:
        hit = {"prompt": {"turns": [{"content": "Use the sensitive tool."}]}}
        self.assertEqual(hit_prompt_text(hit), "Use the sensitive tool.")


class VictimIsolationTests(unittest.TestCase):
    def test_fresh_outbox_path_is_passed_to_each_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            with patch("notebook_utils.subprocess.Popen") as popen:
                serve_agent(Path("/repo"), "research", state_dir=state_dir)
                first = Path(popen.call_args.kwargs["env"]["AGENT_OUTBOX_PATH"])
                serve_agent(Path("/repo"), "research", state_dir=state_dir)
                second = Path(popen.call_args.kwargs["env"]["AGENT_OUTBOX_PATH"])
            self.assertEqual(first.parent, state_dir)
            self.assertEqual(second.parent, state_dir)
            self.assertNotEqual(first, second)


class NotebookContractTests(unittest.TestCase):
    def test_main_notebook_scores_five_and_selects_two(self) -> None:
        notebook = json.loads(
            Path("ndd_nemotron_garak.ipynb").read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("N_PROFILE_CANDIDATES: Final[int] = 5", source)
        self.assertIn("N_SELECTED_PROFILES: Final[int] = 2", source)
        self.assertIn('name="profile_quality"', source)
        self.assertIn("scores=[", source)
        self.assertIn(
            "analysis_cb = analysis_builder(selected_profile_raw_df)",
            source,
        )
        self.assertIn(
            "summary_rows: list[dict[str, Any]] = list(all_summary_rows)",
            source,
        )

    def test_exercise_notebook_is_live_main_notebook_with_todos(self) -> None:
        exercise = json.loads(
            Path("ndd_nemotron_garak_workshop.ipynb").read_text(
                encoding="utf-8"
            )
        )
        main = json.loads(
            Path("ndd_nemotron_garak.ipynb").read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in exercise["cells"]
        )
        self.assertEqual(len(exercise["cells"]), len(main["cells"]))
        self.assertIn("Exercise 2.2: add the profile-quality NDD judge", source)
        self.assertIn("Exercise 2.2: add the profile_core NDD column", source)
        self.assertIn("Exercise 2.3: implement the tool-analysis output schemas", source)
        self.assertIn("Exercise 2.4: implement the second adaptive loop", source)
        self.assertIn("Exercise 3: build SFT records", source)
        self.assertIn("TRAIN_DATASET_MODE: Final[str] = None", source)
        self.assertNotIn("Exercise 5.", source)
        self.assertGreaterEqual(source.count("<details><summary>Hint"), 6)
        self.assertNotIn("REFERENCE FALLBACK", source)
        self.assertNotIn("FIXTURES", source)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

# base.py may re-exec the interpreter when PYTHONHASHSEED is absent. The test
# suite does not depend on hash iteration order, so provide the expected value
# before importing the shared training module.
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = ROOT / "algorithms"
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import base  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402


PLOT_MODULE_PATH = ROOT / "tools" / "paper_draw.py"
PLOT_SPEC = importlib.util.spec_from_file_location("paper_draw_test_module", PLOT_MODULE_PATH)
if PLOT_SPEC is None or PLOT_SPEC.loader is None:
    raise RuntimeError(f"无法加载绘图模块：{PLOT_MODULE_PATH}")
paper_draw = importlib.util.module_from_spec(PLOT_SPEC)
sys.modules[PLOT_SPEC.name] = paper_draw
PLOT_SPEC.loader.exec_module(paper_draw)


class ExperimentOutputRegressionTests(unittest.TestCase):
    def make_round_result(self) -> base.RoundResult:
        return base.RoundResult(
            round_id=1,
            selected_clients=[0, 2],
            train_metrics={
                "avg_train_loss": 1.25,
                "avg_train_acc": 42.0,
            },
            test_metrics={
                "test_loss": 1.5,
                "test_acc": 45.0,
            },
            best_acc=45.0,
            aggregation_info={
                "non_expert": {"method": "uniform"},
                "expert": {"method": "uniform"},
            },
            client_diagnostics={
                0: {
                    "num_samples": 10,
                    "metrics": {"train_loss": 1.2},
                    "expert_usage": None,
                }
            },
        )

    def make_server_result(self) -> base.ServerTrainResult:
        train_state = base.TrainState(
            round_id=1,
            best_acc=45.0,
            best_round=1,
            extra={"method_state": "ok"},
        )
        return base.ServerTrainResult(
            round_results=[self.make_round_result()],
            train_state=train_state,
        )

    def test_round_result_has_separated_metric_groups(self) -> None:
        result = self.make_round_result().to_dict()

        self.assertEqual(
            set(result),
            {
                "round_id",
                "selected_clients",
                "train_metrics",
                "test_metrics",
                "best_acc",
                "aggregation_info",
                "client_diagnostics",
            },
        )
        self.assertNotIn("client_metrics", result)
        self.assertIn("avg_train_loss", result["train_metrics"])
        self.assertIn("test_acc", result["test_metrics"])
        self.assertNotIn("avg_train_loss", result["aggregation_info"])
        self.assertNotIn("client_diagnostics", result["aggregation_info"])

    def test_train_state_and_server_result_do_not_duplicate_best_metrics(self) -> None:
        train_state = base.TrainState(
            round_id=3,
            best_acc=60.0,
            best_round=2,
            extra={"counter": 1},
        )
        state_without_best = train_state.to_dict(include_best_metrics=False)

        self.assertEqual(state_without_best["round_id"], 3)
        self.assertEqual(state_without_best["extra"], {"counter": 1})
        self.assertNotIn("best_acc", state_without_best)
        self.assertNotIn("best_round", state_without_best)

        result = self.make_server_result().to_dict()
        self.assertEqual(result["best_acc"], 45.0)
        self.assertEqual(result["best_round"], 1)
        self.assertNotIn("best_acc", result["train_state"])
        self.assertNotIn("best_round", result["train_state"])

    def test_json_safe_supports_numpy_tensor_and_rejects_non_finite_values(self) -> None:
        safe = base.make_json_safe(
            {
                "numpy_scalar": np.float32(1.25),
                "numpy_array": np.asarray([1, 2], dtype=np.int64),
                "tensor": torch.tensor([3.0, 4.0]),
                "tuple": (5, 6),
            }
        )
        decoded = json.loads(json.dumps(safe, allow_nan=False))

        self.assertAlmostEqual(decoded["numpy_scalar"], 1.25)
        self.assertEqual(decoded["numpy_array"], [1, 2])
        self.assertEqual(decoded["tensor"], [3.0, 4.0])
        self.assertEqual(decoded["tuple"], [5, 6])

        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    base.make_json_safe({"metrics": {"value": invalid}})

    def test_summary_json_and_results_csv_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            base.save_train_outputs(
                train_result=self.make_server_result(),
                output_dir=output_dir,
                save_csv=True,
            )

            summary_path = output_dir / "summary.json"
            csv_path = output_dir / "results.csv"
            self.assertTrue(summary_path.is_file())
            self.assertTrue(csv_path.is_file())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["best_acc"], 45.0)
            self.assertEqual(len(summary["round_results"]), 1)
            self.assertEqual(
                summary["round_results"][0]["test_metrics"]["test_acc"],
                45.0,
            )

            with csv_path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0]["selected_clients"]), [0, 2])
            self.assertAlmostEqual(float(rows[0]["test_acc"]), 45.0)
            self.assertAlmostEqual(float(rows[0]["test_loss"]), 1.5)

    def test_plot_metadata_supports_new_and_legacy_config_formats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            new_run = root / "rfedags_orthogonal_v2"
            new_run.mkdir()
            (new_run / "config_used.yaml").write_text(
                yaml.safe_dump(
                    {
                        "data": {"dataset": "cifar10"},
                        "model": {"backbone": "resnet_cifar"},
                        "aggregation": {"expert": "rfedags_orthogonal"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            new_metadata = paper_draw.load_run_metadata(new_run / "results.csv")
            self.assertEqual(
                new_metadata,
                ("cifar10", "resnet_cifar", "rfedags_orthogonal_v2"),
            )

            old_run = root / "legacy_run_name"
            old_run.mkdir()
            (old_run / "config_used.yaml").write_text(
                yaml.safe_dump(
                    {
                        "dataset": "cifar100",
                        "model_cfg": {"backbone": "resnet_cifar"},
                        "agg": {
                            "expert": {"method": "legacy_method_name"}
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            old_metadata = paper_draw.load_run_metadata(old_run / "results.csv")
            self.assertEqual(
                old_metadata,
                ("cifar100", "resnet_cifar", "legacy_run_name"),
            )

    def test_plot_keeps_multiple_runs_of_the_same_method(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "experiments"
            output_dir = root / "plots"

            for run_name, values in (
                ("uniform", [10.0, 20.0, 30.0]),
                ("uniform_v2", [12.0, 22.0, 32.0]),
            ):
                run_dir = input_dir / run_name
                run_dir.mkdir(parents=True)
                (run_dir / "config_used.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "data": {"dataset": "cifar10"},
                            "model": {"backbone": "resnet_cifar"},
                            "aggregation": {"expert": "uniform"},
                        },
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                with (run_dir / "results.csv").open(
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=["round_id", "test_acc"],
                    )
                    writer.writeheader()
                    for round_id, accuracy in enumerate(values, start=1):
                        writer.writerow(
                            {"round_id": round_id, "test_acc": accuracy}
                        )

            captured = io.StringIO()
            argv = [
                "paper_draw.py",
                "--input-dir",
                str(input_dir),
                "--window",
                "1",
                "--output-dir",
                str(output_dir),
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(captured):
                exit_code = paper_draw.main()

            output = captured.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertEqual(output.count("rounds="), 2)
            self.assertIn("Uniform (uniform)", output)
            self.assertIn("Uniform (uniform_v2)", output)
            self.assertTrue((output_dir / "cifar10_resnet_cifar.png").is_file())
            self.assertTrue((output_dir / "cifar10_resnet_cifar.pdf").is_file())

    def test_run_overwrite_cleans_only_framework_owned_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "manual_run"
            run_dir.mkdir()

            for filename in base.RUN_OUTPUT_FILENAMES:
                (run_dir / filename).write_text("stale", encoding="utf-8")
            user_file = run_dir / "user_notes.txt"
            user_file.write_text("keep", encoding="utf-8")

            # The training entrypoint passes ConfigNode, not a plain dict.
            resolved = base.ensure_run_dir(
                base.ConfigNode(
                    {
                        "run_dir": str(run_dir),
                        "run": {"overwrite": True},
                    }
                )
            )

            self.assertEqual(resolved, run_dir)
            for filename in base.RUN_OUTPUT_FILENAMES:
                self.assertFalse((run_dir / filename).exists())
            self.assertEqual(user_file.read_text(encoding="utf-8"), "keep")

            # Without overwrite, ensure_run_dir must preserve existing outputs.
            protected_summary = run_dir / "summary.json"
            protected_summary.write_text("keep summary", encoding="utf-8")
            base.ensure_run_dir(
                base.ConfigNode(
                    {
                        "run_dir": str(run_dir),
                        "run": {"overwrite": False},
                    }
                )
            )
            self.assertEqual(
                protected_summary.read_text(encoding="utf-8"),
                "keep summary",
            )

    def test_training_log_is_truncated_for_each_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "train.log"
            log_path.write_text("stale log\n", encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                with base.tee_output_to_file(log_path):
                    print("fresh log")

            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("fresh log", log_text)
            self.assertNotIn("stale log", log_text)

    def test_run_name_collision_policy_and_boolean_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "outputs"
            output_dir.mkdir()

            def config(
                run_name: str,
                unique_name: object,
                overwrite: object,
            ) -> dict:
                return {
                    "run_name": run_name,
                    "output_dir": str(output_dir),
                    "run": {
                        "unique_name": unique_name,
                        "overwrite": overwrite,
                    },
                    "agg": {"expert": {"method": "uniform"}},
                }

            (output_dir / "manual").mkdir()
            unique_cfg = base._finalize_run_info(
                config("manual", True, False)
            )
            self.assertEqual(unique_cfg["run_name"], "manual_v2")

            with self.assertRaises(base.ConfigError):
                base._finalize_run_info(
                    config("manual", False, False)
                )

            overwrite_cfg = base._finalize_run_info(
                config("manual", False, True)
            )
            self.assertEqual(
                Path(overwrite_cfg["run_dir"]),
                output_dir / "manual",
            )

            with self.assertRaises(base.ConfigError):
                base._finalize_run_info(
                    config("another", "false", False)
                )

            (output_dir / "uniform").mkdir()
            auto_cfg = base._finalize_run_info(
                {
                    "run_name": "auto",
                    "output_dir": str(output_dir),
                    "run": {"unique_name": False, "overwrite": True},
                    "agg": {"expert": {"method": "uniform"}},
                }
            )
            self.assertEqual(auto_cfg["run_name"], "uniform_v2")


if __name__ == "__main__":
    unittest.main()

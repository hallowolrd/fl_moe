from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = ROOT / "algorithms"
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import base  # noqa: E402
import yaml  # noqa: E402


class ConfigRecordTests(unittest.TestCase):
    def test_default_record_contains_effective_experiment_configuration(self) -> None:
        cfg = base.load_embedded_config()
        record = base._build_config_record(
            cfg,
            model_builder=base.build_switch_vit_model,
            resolved_device="cuda:0",
        )

        self.assertEqual(
            record["data"]["partition"],
            {
                "method": "dirichlet",
                "alpha": 0.1,
                "sample_balance": "unbalanced",
                "min_samples_per_client": 1,
                "max_retries": 100,
            },
        )
        self.assertTrue(record["data"]["augmentation"])
        self.assertTrue(record["data"]["download"])
        self.assertEqual(
            record["data"]["loader"],
            {
                "client_train_drop_last": False,
                "pin_memory": True,
                "persistent_workers": True,
            },
        )

        self.assertEqual(record["model"]["architecture"], "SwitchViT")
        self.assertEqual(record["model"]["builder"], "base.build_switch_vit_model")
        self.assertEqual(record["model"]["num_experts"], 4)
        self.assertEqual(record["model"]["num_classes"], 10)
        self.assertEqual(record["model"]["input_shape"], [3, 32, 32])
        self.assertEqual(
            record["model"]["config"]["moe_block_indices"],
            [2, 4, 6],
        )

        self.assertEqual(record["training"]["communication_rounds"], 50)
        self.assertEqual(record["training"]["local"]["epochs"], 5)
        self.assertEqual(record["training"]["local"]["batch_size"], 64)
        self.assertEqual(
            record["training"]["loss"],
            {"name": "cross_entropy", "label_smoothing": 0.0},
        )
        self.assertIsNone(record["training"]["gradient_clip"])
        self.assertEqual(
            record["training"]["optimizer"],
            {
                "name": "sgd",
                "learning_rate": 0.01,
                "weight_decay": 0.0005,
                "momentum": 0.9,
                "nesterov": False,
            },
        )

        self.assertEqual(record["reproducibility"]["seed"], cfg.seed)
        self.assertTrue(record["reproducibility"]["deterministic"])
        self.assertEqual(
            record["runtime"],
            {"device": "auto", "resolved_device": "cuda:0"},
        )
        self.assertEqual(
            record["server_evidence"],
            {"size": 0, "batch_size": 256, "class_balanced": True},
        )
        self.assertEqual(
            record["logging"],
            {
                "save_config": True,
                "save_results_csv": True,
                "progress_bar": True,
            },
        )
        self.assertEqual(
            record["output"],
            {
                "root": "outputs",
                "run_name": cfg.run_name,
                "auto_generated": True,
                "unique_name": True,
                "effective_unique_name": True,
                "overwrite": False,
            },
        )

    def test_explicit_false_and_optimizer_overrides_are_recorded(self) -> None:
        cfg = base.load_embedded_config(
            method_overrides={
                # This legacy CLI label does not select the actual model.
                "model": "resnet_cifar",
                "data_augmentation": False,
                "download_data": False,
                "drop_last": True,
                "pin_memory": False,
                "num_workers": 0,
                "min_samples_per_client": 3,
                "partition_max_retries": 17,
                "label_smooth": 0.15,
                "grad_clip": 2.5,
                "deterministic": False,
                "optimizer": {
                    "type": "adamw",
                    "lr": 0.001,
                    "weight_decay": 0.02,
                    "betas": [0.8, 0.95],
                    "eps": 1e-7,
                    # Nested optimizer setting takes precedence over top-level.
                    "grad_clip": 4.0,
                },
                "server_evidence": {
                    "size": 0,
                    "batch_size": 128,
                    "class_balanced": False,
                },
                "logging": {
                    "save_config": False,
                    "save_results_csv": False,
                    "progress_bar": False,
                },
                "run": {
                    "unique_name": False,
                    "overwrite": True,
                },
                "method_config": {"plugin_flag": False},
            }
        )
        record = base._build_config_record(
            cfg,
            model_builder=base.build_switch_vit_model,
            resolved_device="cpu",
        )

        self.assertEqual(record["model"]["architecture"], "SwitchViT")
        self.assertEqual(record["model"]["legacy_label"], "resnet_cifar")
        self.assertFalse(record["data"]["augmentation"])
        self.assertFalse(record["data"]["download"])
        self.assertEqual(
            record["data"]["loader"],
            {
                "client_train_drop_last": True,
                "pin_memory": False,
                "persistent_workers": False,
            },
        )
        self.assertEqual(
            record["data"]["partition"]["min_samples_per_client"],
            3,
        )
        self.assertEqual(record["data"]["partition"]["max_retries"], 17)

        self.assertEqual(
            record["training"]["loss"]["label_smoothing"],
            0.15,
        )
        self.assertEqual(record["training"]["gradient_clip"], 4.0)
        self.assertEqual(
            record["training"]["optimizer"],
            {
                "name": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.02,
                "betas": [0.8, 0.95],
                "eps": 1e-7,
            },
        )
        self.assertFalse(record["reproducibility"]["deterministic"])
        self.assertEqual(
            record["server_evidence"],
            {"size": 0, "batch_size": 128, "class_balanced": False},
        )
        self.assertEqual(
            record["logging"],
            {
                "save_config": False,
                "save_results_csv": False,
                "progress_bar": False,
            },
        )
        self.assertTrue(record["output"]["auto_generated"])
        self.assertFalse(record["output"]["unique_name"])
        self.assertTrue(record["output"]["effective_unique_name"])
        self.assertTrue(record["output"]["overwrite"])
        self.assertEqual(
            record["aggregation"]["method_config"],
            {"plugin_flag": False},
        )

        method_config = record["aggregation"]["method_config"]
        for framework_key in (
            "model",
            "data_augmentation",
            "download_data",
            "drop_last",
            "pin_memory",
            "label_smooth",
            "grad_clip",
            "min_samples_per_client",
            "partition_max_retries",
        ):
            self.assertNotIn(framework_key, method_config)

    def test_save_config_yaml_round_trip_keeps_false_and_null_values(self) -> None:
        cfg = base.load_embedded_config(
            method_overrides={
                "data_augmentation": False,
                "grad_clip": None,
                "deterministic": False,
                "logging": {
                    "save_config": True,
                    "save_results_csv": False,
                    "progress_bar": False,
                },
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "config_used.yaml"
            base.save_config(
                cfg,
                output_path,
                model_builder=base.build_switch_vit_model,
                resolved_device="cpu",
            )

            loaded = yaml.safe_load(output_path.read_text(encoding="utf-8"))

        self.assertFalse(loaded["data"]["augmentation"])
        self.assertFalse(loaded["reproducibility"]["deterministic"])
        self.assertIsNone(loaded["training"]["gradient_clip"])
        self.assertFalse(loaded["logging"]["save_results_csv"])
        self.assertFalse(loaded["logging"]["progress_bar"])
        self.assertEqual(loaded["runtime"]["resolved_device"], "cpu")


if __name__ == "__main__":
    unittest.main()

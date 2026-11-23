from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("PYTHONHASHSEED", "0")

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = ROOT / "algorithms"
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import base  # noqa: E402


class BooleanConfigValidationTests(unittest.TestCase):
    BOOLEAN_FIELDS = (
        "data_augmentation",
        "download_data",
        "drop_last",
        "deterministic",
        "model_cfg.token_dropping",
        "model_cfg.router_statistics",
        "optimizer.nesterov",
        "server_evidence.class_balanced",
        "logging.save_config",
        "logging.save_results_csv",
        "logging.progress_bar",
        "run.unique_name",
        "run.overwrite",
    )

    @staticmethod
    def _nested_override(path: str, value: object) -> dict:
        parts = path.split(".")
        result = current = {}
        for part in parts[:-1]:
            child = {}
            current[part] = child
            current = child
        current[parts[-1]] = value
        return result

    def test_all_public_boolean_fields_accept_only_python_bool(self) -> None:
        invalid_values = (1, 0, "true", "false", "yes", "no", None, [], {})
        for field in self.BOOLEAN_FIELDS:
            for value in invalid_values:
                with self.subTest(field=field, value=value):
                    overrides = self._nested_override(field, value)
                    with self.assertRaisesRegex(
                        base.ConfigError,
                        field.replace(".", r"\."),
                    ):
                        base.load_embedded_config(config_overrides=overrides)

    def test_all_public_boolean_fields_accept_true_and_false(self) -> None:
        for field in self.BOOLEAN_FIELDS:
            for value in (True, False):
                with self.subTest(field=field, value=value):
                    cfg = base.load_embedded_config(
                        config_overrides=self._nested_override(field, value)
                    )
                    self.assertIs(
                        base._cfg_get_bool(cfg, field, not value),
                        value,
                    )

    def test_pin_memory_accepts_none_or_bool_only(self) -> None:
        for value in (None, True, False):
            with self.subTest(value=value):
                cfg = base.load_embedded_config(
                    config_overrides={"pin_memory": value}
                )
                self.assertIs(cfg.pin_memory, value)

        for value in (1, 0, "false", "true", [], {}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(base.ConfigError, "pin_memory"):
                    base.load_embedded_config(
                        config_overrides={"pin_memory": value}
                    )

    def test_quoted_boolean_string_in_yaml_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid.yaml"
            config_path.write_text(
                'data_augmentation: "false"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                base.ConfigError,
                "data_augmentation",
            ):
                base.load_config(config_path)

    def test_boolean_reader_does_not_coerce_invalid_values(self) -> None:
        self.assertFalse(base._cfg_get_bool({"enabled": False}, "enabled", True))
        self.assertTrue(base._cfg_get_bool({"enabled": True}, "enabled", False))

        for value in (1, 0, "false", "true", [], {}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(base.ConfigError, "enabled"):
                    base._cfg_get_bool({"enabled": value}, "enabled", False)

    def test_method_config_boolean_values_remain_plugin_owned(self) -> None:
        cfg = base.load_embedded_config(
            config_overrides={
                "method_config": {
                    "plugin_flag": "false",
                    "plugin_switch": 1,
                }
            }
        )
        self.assertEqual(
            cfg.method_config.to_dict(),
            {"plugin_flag": "false", "plugin_switch": 1},
        )

    def test_config_record_does_not_silently_coerce_boolean_values(self) -> None:
        with self.assertRaisesRegex(base.ConfigError, "data_augmentation"):
            base._build_config_record({"data_augmentation": "false"})


class SeedValidationTests(unittest.TestCase):
    def test_seed_validator_accepts_supported_boundaries(self) -> None:
        for seed in (0, 1, 42, 2**32 - 1):
            with self.subTest(seed=seed):
                self.assertEqual(base._validate_seed_value(seed), seed)

    def test_seed_validator_rejects_invalid_types_and_ranges(self) -> None:
        for seed in (True, False, 1.5, "42", None):
            with self.subTest(seed=seed):
                with self.assertRaises(TypeError):
                    base._validate_seed_value(seed)

        for seed in (-1, 2**32):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    base._validate_seed_value(seed)

    def test_startup_seed_parser_keeps_cli_decimal_semantics(self) -> None:
        self.assertEqual(base._parse_startup_seed_value("42"), 42)
        self.assertEqual(base._parse_startup_seed_value("010"), 10)

        with self.assertRaises(ValueError):
            base._parse_startup_seed_value("0x10")

    def test_startup_seed_parser_keeps_yaml_scalar_types(self) -> None:
        self.assertEqual(
            base._parse_startup_seed_value("0x10", yaml_scalar=True),
            16,
        )
        with self.assertRaises(TypeError):
            base._parse_startup_seed_value('"123"', yaml_scalar=True)

    def test_cli_seed_lookup_uses_last_repeated_option(self) -> None:
        with patch.object(
            base.sys,
            "argv",
            ["train.py", "--seed=11", "--seed", "22"],
        ):
            self.assertEqual(base._get_cli_arg_value("--seed"), "22")

        with patch.object(
            base.sys,
            "argv",
            ["train.py", "--seed", "22", "--seed=33"],
        ):
            self.assertEqual(base._get_cli_arg_value("--seed"), "33")


class _SyntheticLabelDataset:
    def __init__(
        self,
        targets: list[object],
        *,
        length: int | None = None,
        classes: list[str] | None = None,
        class_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.targets = targets
        self._length = len(targets) if length is None else length
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return self._length


class DatasetClassValidationTests(unittest.TestCase):
    def _validate(
        self,
        dataset: _SyntheticLabelDataset,
        *,
        split_name: str = "train",
        require_all_classes: bool = True,
    ) -> set[int]:
        return base._validate_dataset_split_classes(
            dataset=dataset,
            dataset_name="synthetic",
            split_name=split_name,
            expected_num_classes=3,
            require_all_classes=require_all_classes,
        )

    def test_train_and_evidence_splits_require_all_registered_classes(self) -> None:
        for split_name in ("train", "train_evidence"):
            with self.subTest(split_name=split_name):
                with self.assertRaisesRegex(ValueError, "missing_class_ids"):
                    self._validate(
                        _SyntheticLabelDataset([0, 1, 1]),
                        split_name=split_name,
                        require_all_classes=True,
                    )

    def test_test_split_may_omit_classes_but_returns_observed_labels(self) -> None:
        observed = self._validate(
            _SyntheticLabelDataset([0, 2, 2]),
            split_name="test",
            require_all_classes=False,
        )
        self.assertEqual(observed, {0, 2})

    def test_dataset_labels_must_stay_inside_registered_class_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid_range"):
            self._validate(_SyntheticLabelDataset([0, 1, 3]))

        with self.assertRaisesRegex(ValueError, "valid_range"):
            self._validate(_SyntheticLabelDataset([-1, 0, 1]))

    def test_dataset_sample_count_must_match_target_count(self) -> None:
        dataset = _SyntheticLabelDataset([0, 1], length=3)
        with self.assertRaisesRegex(ValueError, "样本数与标签数不一致"):
            self._validate(dataset, require_all_classes=False)

    def test_dataset_labels_must_be_strict_integers(self) -> None:
        for label in (True, 1.0, "1"):
            with self.subTest(label=label):
                with self.assertRaises(TypeError):
                    self._validate(
                        _SyntheticLabelDataset([0, label, 2]),
                        require_all_classes=False,
                    )

    def test_dataset_class_metadata_must_match_registered_class_space(self) -> None:
        valid_dataset = _SyntheticLabelDataset(
            [0, 1, 2],
            classes=["a", "b", "c"],
            class_to_idx={"a": 0, "b": 1, "c": 2},
        )
        self.assertEqual(self._validate(valid_dataset), {0, 1, 2})

        invalid_mapping = _SyntheticLabelDataset(
            [0, 1, 2],
            class_to_idx={"a": 0, "b": 1, "c": 3},
        )
        with self.assertRaisesRegex(ValueError, "class_to_idx"):
            self._validate(invalid_mapping)

        invalid_classes = _SyntheticLabelDataset(
            [0, 1, 2],
            classes=["a", "b"],
        )
        with self.assertRaisesRegex(ValueError, "声明的类别数量"):
            self._validate(invalid_classes)


class PublicHyperparameterValidationTests(unittest.TestCase):
    @staticmethod
    def _load(overrides: dict) -> base.ConfigNode:
        return base.load_embedded_config(config_overrides=overrides)

    @staticmethod
    def _nested_override(path: str, value: object) -> dict:
        return BooleanConfigValidationTests._nested_override(path, value)

    def test_partition_integer_parameters_require_positive_integers(self) -> None:
        for field in ("min_samples_per_client", "partition_max_retries"):
            for value in (0, -1, True, 1.0, "1"):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(base.ConfigError, field):
                        self._load({field: value})

        self._load(
            {
                "min_samples_per_client": 1,
                "partition_max_retries": 1,
            }
        )

    def test_label_smoothing_accepts_closed_unit_interval(self) -> None:
        for value in (0.0, 0.5, 1.0):
            with self.subTest(value=value):
                self._load({"label_smooth": value})

    def test_label_smoothing_rejects_out_of_range_and_non_finite_values(self) -> None:
        for value in (-0.01, 1.01, float("nan"), float("inf"), float("-inf"), True, "invalid"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(base.ConfigError, "label_smooth"):
                    self._load({"label_smooth": value})

    def test_top_level_and_optimizer_gradient_clip_validation(self) -> None:
        for value in (None, 0.0, 1.5):
            with self.subTest(location="top-level", value=value):
                self._load({"grad_clip": value})
            with self.subTest(location="optimizer", value=value):
                self._load({"optimizer": {"grad_clip": value}})

        invalid_values = (-1.0, float("nan"), float("inf"), float("-inf"), True, "invalid")
        for value in invalid_values:
            with self.subTest(location="top-level", value=value):
                with self.assertRaises(base.ConfigError):
                    self._load({"grad_clip": value})
            with self.subTest(location="optimizer", value=value):
                with self.assertRaises(base.ConfigError):
                    self._load({"optimizer": {"grad_clip": value}})

    def test_optimizer_momentum_must_be_finite_and_non_negative(self) -> None:
        for value in (0.0, 0.9):
            with self.subTest(value=value):
                self._load({"optimizer": {"momentum": value}})

        for value in (-0.01, float("nan"), float("inf"), float("-inf"), True, "invalid"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(base.ConfigError, "optimizer.momentum"):
                    self._load({"optimizer": {"momentum": value}})

    def test_optimizer_betas_must_be_two_finite_values_in_zero_one(self) -> None:
        self._load({"optimizer": {"betas": [0.0, 0.999999]}})

        invalid_values = (
            [],
            [0.9],
            [0.9, 1.0],
            [-0.1, 0.9],
            [float("nan"), 0.9],
            [0.9, float("inf")],
            [True, 0.9],
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(base.ConfigError, "optimizer.betas"):
                    self._load({"optimizer": {"betas": value}})

    def test_optimizer_eps_must_be_finite_and_positive(self) -> None:
        self._load({"optimizer": {"eps": 1e-12}})

        for value in (0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, "invalid"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(base.ConfigError, "optimizer.eps"):
                    self._load({"optimizer": {"eps": value}})

    def test_optimizer_nesterov_requires_bool_and_positive_sgd_momentum(self) -> None:
        for value in (1, 0, "true", "false", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(base.ConfigError, "optimizer.nesterov"):
                    self._load({"optimizer": {"nesterov": value}})

        with self.assertRaisesRegex(base.ConfigError, "momentum 必须大于 0"):
            self._load(
                {
                    "optimizer": {
                        "type": "sgd",
                        "momentum": 0.0,
                        "nesterov": True,
                    }
                }
            )

        self._load(
            {
                "optimizer": {
                    "type": "sgd",
                    "momentum": 0.9,
                    "nesterov": True,
                }
            }
        )

    def test_combined_public_hyperparameter_boundaries_are_accepted(self) -> None:
        self._load(
            {
                "min_samples_per_client": 1,
                "partition_max_retries": 1,
                "label_smooth": 1.0,
                "grad_clip": 0.0,
                "optimizer": {
                    "momentum": 0.0,
                    "betas": [0.0, 0.999999],
                    "eps": 1e-12,
                    "nesterov": False,
                },
            }
        )


if __name__ == "__main__":
    unittest.main()

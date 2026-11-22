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


if __name__ == "__main__":
    unittest.main()

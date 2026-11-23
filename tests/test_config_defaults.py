from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("PYTHONHASHSEED", "0")

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = ROOT / "algorithms"
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import base  # noqa: E402


class DirichletDefaultConfigTests(unittest.TestCase):
    def test_default_dirichlet_configuration_without_cli_overrides(self) -> None:
        # 模拟不传入任何命令行参数的默认启动方式。
        with patch.object(sys, "argv", ["uniform.py"]):
            args = base.parse_args()

        cli_overrides = base.build_common_cli_overrides(args)
        self.assertNotIn("partition", cli_overrides)

        cfg = base.load_embedded_config(config_overrides=cli_overrides)

        self.assertEqual(cfg.partition.method, "dirichlet")
        self.assertEqual(cfg.partition.dirichlet.alpha, 0.1)
        self.assertEqual(
            cfg.partition.dirichlet.sample_balance,
            "unbalanced",
        )


class _FakeCIFAR10Dataset:
    targets = list(range(10))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return index, self.targets[index]


class PublicDefaultSourceTests(unittest.TestCase):
    def test_dataset_builder_reads_defaults_from_default_config(self) -> None:
        fake_dataset = _FakeCIFAR10Dataset()

        with patch.dict(
            base.DEFAULT_CONFIG,
            {
                "data_augmentation": False,
                "download_data": False,
            },
        ):
            with (
                patch.object(
                    base,
                    "build_train_transform",
                    side_effect=lambda **kwargs: kwargs["use_augmentation"],
                ) as transform_builder,
                patch.object(base, "build_test_transform", return_value=None),
                patch.object(
                    base.datasets,
                    "CIFAR10",
                    return_value=fake_dataset,
                ) as dataset_builder,
            ):
                base.build_datasets(
                    SimpleNamespace(
                        dataset="cifar10",
                        data_root="./datasets",
                    )
                )

        self.assertEqual(
            [
                call.kwargs["use_augmentation"]
                for call in transform_builder.call_args_list
            ],
            [False, False],
        )
        self.assertEqual(dataset_builder.call_count, 3)
        self.assertTrue(
            all(
                call.kwargs["download"] is False
                for call in dataset_builder.call_args_list
            )
        )

    def test_client_train_loader_reads_drop_last_default_from_default_config(self) -> None:
        dataset = _FakeCIFAR10Dataset()

        with patch.dict(base.DEFAULT_CONFIG, {"drop_last": True}):
            loaders = base.build_client_train_loaders(
                cfg=SimpleNamespace(),
                client_datasets=[dataset],
                batch_size=4,
                num_workers=0,
                pin_memory=False,
                seed=17,
            )

        self.assertEqual(len(loaders), 1)
        self.assertTrue(loaders[0].drop_last)


if __name__ == "__main__":
    unittest.main()

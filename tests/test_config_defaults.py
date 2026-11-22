from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()

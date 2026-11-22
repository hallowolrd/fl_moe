from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTHONHASHSEED", "0")

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = ROOT / "algorithms"
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import torch  # noqa: E402
import base  # noqa: E402


class AggregationTensorShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.global_state = {
            "weight": torch.zeros(2, 3),
            "bias": torch.zeros(2),
        }

    def test_subtract_state_dict_rejects_broadcastable_shape_mismatch(self) -> None:
        local_state = {
            "weight": torch.ones(1, 3),
            "bias": torch.ones(2),
        }

        with self.assertRaisesRegex(
            ValueError,
            r"参数 weight.*expected=\(2, 3\).*actual=\(1, 3\)",
        ):
            base.subtract_state_dict(
                local_state=local_state,
                global_state=self.global_state,
            )

    def test_weighted_delta_rejects_broadcastable_shape_mismatch(self) -> None:
        updates = [
            SimpleNamespace(
                client_id=7,
                model_delta={
                    "weight": torch.ones(1, 3),
                    "bias": torch.ones(2),
                },
            )
        ]

        with self.assertRaisesRegex(
            ValueError,
            r"客户端 7.*weight.*expected=\(2, 3\).*actual=\(1, 3\)",
        ):
            base.apply_weighted_delta(
                global_state=self.global_state,
                client_updates=updates,
                weights={7: 1.0},
            )

    def test_weighted_delta_rejects_wrong_base_state_shape(self) -> None:
        updates = [
            SimpleNamespace(
                client_id=0,
                model_delta={
                    "weight": torch.ones(2, 3),
                    "bias": torch.ones(2),
                },
            )
        ]
        base_state = {
            "weight": torch.zeros(1, 3),
            "bias": torch.zeros(2),
        }

        with self.assertRaisesRegex(
            ValueError,
            r"base_state.*weight.*expected=\(2, 3\).*actual=\(1, 3\)",
        ):
            base.apply_weighted_delta(
                global_state=self.global_state,
                client_updates=updates,
                weights={0: 1.0},
                base_state=base_state,
            )

    def test_state_dict_shape_validator_rejects_wrong_aggregator_output(self) -> None:
        candidate_state = {
            "weight": torch.zeros(2, 1),
            "bias": torch.zeros(2),
        }

        with self.assertRaisesRegex(
            ValueError,
            r"expert 聚合结果.*weight.*expected=\(2, 3\).*actual=\(2, 1\)",
        ):
            base.validate_state_dict_shapes(
                reference_state=self.global_state,
                candidate_state=candidate_state,
                context="expert 聚合结果",
            )

    def test_matching_shapes_are_aggregated_normally(self) -> None:
        updates = [
            SimpleNamespace(
                client_id=0,
                model_delta={
                    "weight": torch.ones(2, 3),
                    "bias": torch.full((2,), 2.0),
                },
            )
        ]

        result = base.apply_weighted_delta(
            global_state=self.global_state,
            client_updates=updates,
            weights={0: 1.0},
        )

        torch.testing.assert_close(result["weight"], torch.ones(2, 3))
        torch.testing.assert_close(result["bias"], torch.full((2,), 2.0))


if __name__ == "__main__":
    unittest.main()

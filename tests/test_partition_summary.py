from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Match the shared test-suite import contract before importing base.py.
os.environ.setdefault("PYTHONHASHSEED", "0")

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = ROOT / "algorithms"
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import base  # noqa: E402


class PartitionSummaryTests(unittest.TestCase):
    @staticmethod
    def make_dirichlet_partition() -> base.PartitionResult:
        return base.PartitionResult(
            client_indices=[
                [0, 2, 4],
                [1, 3],
            ],
            client_sample_counts={
                0: 3,
                1: 2,
            },
            client_class_counts={
                0: {0: 2, 1: 1, 2: 0},
                1: {0: 0, 1: 1, 2: 1},
            },
            num_classes=3,
            partition_method="dirichlet",
            sample_balance="unbalanced",
            dirichlet_alpha=0.1,
        )

    def test_partition_summary_contains_complete_client_counts(self) -> None:
        partition = self.make_dirichlet_partition()

        summary = base.partition_summary_to_dict(partition)

        self.assertEqual(
            set(summary),
            {
                "partition_method",
                "sample_balance",
                "dirichlet_alpha",
                "num_classes",
                "num_clients",
                "client_sample_counts",
                "client_sample_count_min",
                "client_sample_count_max",
                "client_sample_count_spread",
                "client_class_counts",
            },
        )
        self.assertEqual(summary["partition_method"], "dirichlet")
        self.assertEqual(summary["sample_balance"], "unbalanced")
        self.assertEqual(summary["dirichlet_alpha"], 0.1)
        self.assertEqual(summary["num_classes"], 3)
        self.assertEqual(summary["num_clients"], 2)
        self.assertEqual(summary["client_sample_counts"], {0: 3, 1: 2})
        self.assertEqual(summary["client_sample_count_min"], 2)
        self.assertEqual(summary["client_sample_count_max"], 3)
        self.assertEqual(summary["client_sample_count_spread"], 1)
        self.assertEqual(
            summary["client_class_counts"],
            {
                0: {0: 2, 1: 1, 2: 0},
                1: {0: 0, 1: 1, 2: 1},
            },
        )
        self.assertNotIn("client_indices", summary)

        total_samples = sum(summary["client_sample_counts"].values())
        self.assertEqual(total_samples, 5)
        self.assertEqual(
            total_samples,
            sum(len(indices) for indices in partition.client_indices),
        )
        for client_id, indices in enumerate(partition.client_indices):
            self.assertEqual(summary["client_sample_counts"][client_id], len(indices))
        for client_id, sample_count in summary["client_sample_counts"].items():
            self.assertEqual(
                sum(summary["client_class_counts"][client_id].values()),
                sample_count,
            )

    def test_partition_summary_json_round_trip(self) -> None:
        partition = self.make_dirichlet_partition()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "partition_summary.json"
            base.save_partition_summary(partition, output_path)

            self.assertTrue(output_path.is_file())
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["partition_method"], "dirichlet")
        self.assertEqual(saved["sample_balance"], "unbalanced")
        self.assertEqual(saved["dirichlet_alpha"], 0.1)
        self.assertEqual(saved["num_classes"], 3)
        self.assertEqual(saved["num_clients"], 2)
        self.assertEqual(
            saved["client_sample_counts"],
            {"0": 3, "1": 2},
        )
        self.assertEqual(saved["client_sample_count_min"], 2)
        self.assertEqual(saved["client_sample_count_max"], 3)
        self.assertEqual(saved["client_sample_count_spread"], 1)
        self.assertEqual(
            saved["client_class_counts"],
            {
                "0": {"0": 2, "1": 1, "2": 0},
                "1": {"0": 0, "1": 1, "2": 1},
            },
        )
        self.assertNotIn("client_indices", saved)

    def test_iid_partition_records_non_dirichlet_metadata(self) -> None:
        partition = base.PartitionResult(
            client_indices=[[0, 2], [1, 3]],
            client_sample_counts={0: 2, 1: 2},
            client_class_counts={
                0: {0: 1, 1: 1},
                1: {0: 1, 1: 1},
            },
            num_classes=2,
            partition_method="iid",
            sample_balance="none",
            dirichlet_alpha=None,
        )

        summary = base.partition_summary_to_dict(partition)

        self.assertEqual(summary["partition_method"], "iid")
        self.assertEqual(summary["sample_balance"], "none")
        self.assertIsNone(summary["dirichlet_alpha"])
        self.assertEqual(summary["client_sample_count_min"], 2)
        self.assertEqual(summary["client_sample_count_max"], 2)
        self.assertEqual(summary["client_sample_count_spread"], 0)

    def test_empty_client_keeps_zero_counts_for_every_class(self) -> None:
        partition = base.PartitionResult(
            client_indices=[[0], [1], []],
            client_sample_counts={0: 1, 1: 1, 2: 0},
            client_class_counts={
                0: {0: 1, 1: 0},
                1: {0: 0, 1: 1},
                2: {0: 0, 1: 0},
            },
            num_classes=2,
            partition_method="iid",
            sample_balance="none",
            dirichlet_alpha=None,
        )

        summary = base.partition_summary_to_dict(partition)

        self.assertEqual(summary["client_sample_counts"][2], 0)
        self.assertEqual(
            summary["client_class_counts"][2],
            {0: 0, 1: 0},
        )
        self.assertEqual(summary["client_sample_count_min"], 0)


if __name__ == "__main__":
    unittest.main()

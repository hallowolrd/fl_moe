from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTHONHASHSEED", "0")

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = ROOT / "algorithms"
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import base  # noqa: E402


class ClientLossMetricPipelineTests(unittest.TestCase):
    def make_updates(self):
        return [
            base.ClientUpdate(
                client_id=0,
                round_id=1,
                num_samples=100,
                model_delta={},
                metrics={
                    "train_loss": 1.1,
                    "train_task_loss": 1.0,
                    "train_aux_loss": 0.1,
                    "train_acc": 80.0,
                    "num_processed_samples": 10.0,
                },
            ),
            base.ClientUpdate(
                client_id=1,
                round_id=1,
                num_samples=300,
                model_delta={},
                metrics={
                    "train_loss": 3.3,
                    "train_task_loss": 3.0,
                    "train_aux_loss": 0.3,
                    "train_acc": 60.0,
                    "num_processed_samples": 30.0,
                },
            ),
        ]

    def make_round_result(self) -> base.RoundResult:
        server = object.__new__(base.FLServer)
        server.train_state = base.TrainState(
            round_id=1,
            best_acc=75.0,
            best_round=1,
        )
        server.method_client_diagnostics_builder = None

        updates = self.make_updates()
        result = server.build_round_result(
            round_id=1,
            selected_clients=[
                SimpleNamespace(client_id=0),
                SimpleNamespace(client_id=1),
            ],
            client_updates=updates,
            eval_result=base.EvalResult(
                loss=0.8,
                acc=75.0,
                correct=30,
                total=40,
                extra={},
            ),
            aggregation_info={},
        )
        return result

    def test_round_metrics_are_weighted_by_processed_samples(self) -> None:
        result = self.make_round_result()

        self.assertAlmostEqual(result.train_metrics["avg_train_loss"], 2.75)
        self.assertAlmostEqual(result.train_metrics["avg_train_task_loss"], 2.5)
        self.assertAlmostEqual(result.train_metrics["avg_train_aux_loss"], 0.25)

        # task loss + auxiliary loss must equal the aggregated total loss.
        self.assertAlmostEqual(
            result.train_metrics["avg_train_loss"],
            result.train_metrics["avg_train_task_loss"]
            + result.train_metrics["avg_train_aux_loss"],
        )

        self.assertEqual(
            result.client_diagnostics[0]["metrics"]["train_task_loss"],
            1.0,
        )
        self.assertEqual(
            result.client_diagnostics[1]["metrics"]["train_aux_loss"],
            0.3,
        )

    def test_round_and_client_logs_include_loss_components(self) -> None:
        result = self.make_round_result()
        server = object.__new__(base.FLServer)

        console = io.StringIO()
        with redirect_stdout(console):
            server.print_round_summary(result)

        console_text = console.getvalue()
        self.assertIn("task_loss=2.5000", console_text)
        self.assertIn("aux_loss=0.2500", console_text)

        messages = []
        server._write_log_only = messages.append
        server.print_file_round_detail(result)

        joined = "\n".join(messages)
        self.assertIn("task_loss=2.5000", joined)
        self.assertIn("aux_loss=0.2500", joined)
        self.assertIn("[Client][0]", joined)
        self.assertIn("task_loss=1.0000", joined)
        self.assertIn("aux_loss=0.1000", joined)
        self.assertIn("[Client][1]", joined)
        self.assertIn("task_loss=3.0000", joined)
        self.assertIn("aux_loss=0.3000", joined)

    def test_results_csv_contains_round_task_and_aux_losses(self) -> None:
        result = self.make_round_result()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.csv"
            base.save_round_results_csv(
                round_results=[result],
                output_path=output_path,
            )

            with output_path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["avg_train_loss"]), 2.75)
        self.assertAlmostEqual(float(rows[0]["avg_train_task_loss"]), 2.5)
        self.assertAlmostEqual(float(rows[0]["avg_train_aux_loss"]), 0.25)


if __name__ == "__main__":
    unittest.main()

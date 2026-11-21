from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = ROOT / "algorithms"
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import base  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


class _ObjectOutput:
    def __init__(self, logits, aux_loss):
        self.logits = logits
        self.aux_loss = aux_loss
        self.aux_loss_by_layer = {"layer_1": aux_loss}
        self.moe_stats = {"layer_1": {"tokens_per_expert": [2, 1]}}


class _AuxLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(2, 2)
        self.aux_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, inputs):
        logits = self.classifier(inputs)
        return base.ModelOutput(
            logits=logits,
            aux_loss=self.aux_scale.square(),
            aux_loss_by_layer={"layer_1": self.aux_scale.square()},
            moe_stats={"layer_1": {"tokens_per_expert": [2, 1]}},
        )


class _ExpertAuxLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2, bias=False)
        self.experts = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.aux_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, inputs):
        features = self.backbone(inputs)
        logits = self.experts[0](features)
        aux_loss = self.aux_scale.square()
        return base.ModelOutput(
            logits=logits,
            aux_loss=aux_loss,
            aux_loss_by_layer={"layer_0": aux_loss},
            moe_stats={"layer_0": {"tokens_per_expert": [1, 1]}},
        )


class _SimpleExpertStrategy(base.ExpertClientTrainingStrategy):
    def step(self, context: base.ExpertClientBatchContext) -> None:
        with torch.no_grad():
            for parameter in context.train_context.expert_parameters.values():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-0.01)


class ModelOutputAndAuxLossTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.logits = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            requires_grad=True,
        )
        self.targets = torch.tensor([0, 1])
        self.criterion = nn.CrossEntropyLoss()

    def test_extract_model_output_supports_legacy_and_new_formats(self) -> None:
        aux_loss = torch.tensor(0.3, requires_grad=True)

        tensor_output = base.extract_model_output(self.logits)
        self.assertTrue(torch.equal(tensor_output.logits, self.logits))
        self.assertIsNone(tensor_output.aux_loss)

        tuple_output = base.extract_model_output((self.logits, "legacy-extra"))
        self.assertTrue(torch.equal(tuple_output.logits, self.logits))
        self.assertIsNone(tuple_output.aux_loss)

        dict_output = base.extract_model_output(
            {
                "logits": self.logits,
                "aux_loss": aux_loss,
                "aux_loss_by_layer": {"layer_1": aux_loss},
                "moe_stats": {"layer_1": {"tokens_per_expert": [2, 1]}},
            }
        )
        self.assertIs(dict_output.aux_loss, aux_loss)
        self.assertIn("layer_1", dict_output.aux_loss_by_layer)
        self.assertIn("layer_1", dict_output.moe_stats)

        object_output = base.extract_model_output(
            _ObjectOutput(self.logits, aux_loss)
        )
        self.assertIs(object_output.aux_loss, aux_loss)
        self.assertIn("layer_1", object_output.aux_loss_by_layer)

    def test_compute_model_loss_adds_aux_loss_and_preserves_gradient(self) -> None:
        aux_parameter = nn.Parameter(torch.tensor(2.0))
        aux_loss = 0.5 * aux_parameter.square()

        result = base.compute_model_loss(
            outputs=base.ModelOutput(
                logits=self.logits,
                aux_loss=aux_loss,
            ),
            targets=self.targets,
            criterion=self.criterion,
        )

        expected_task_loss = self.criterion(self.logits, self.targets)
        self.assertAlmostEqual(
            result.total_loss.item(),
            (expected_task_loss + aux_loss).item(),
            places=7,
        )
        self.assertAlmostEqual(result.task_loss.item(), expected_task_loss.item(), places=7)
        self.assertAlmostEqual(result.aux_loss.item(), aux_loss.item(), places=7)

        result.total_loss.backward()
        self.assertAlmostEqual(aux_parameter.grad.item(), 2.0, places=7)

    def test_missing_aux_loss_is_zero_and_keeps_classification_loss(self) -> None:
        result = base.compute_model_loss(
            outputs=self.logits,
            targets=self.targets,
            criterion=self.criterion,
        )

        expected = self.criterion(self.logits, self.targets)
        self.assertEqual(result.aux_loss.item(), 0.0)
        self.assertAlmostEqual(result.total_loss.item(), expected.item(), places=7)
        self.assertAlmostEqual(result.task_loss.item(), expected.item(), places=7)

    def test_invalid_aux_loss_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "必须是标量"):
            base.compute_model_loss(
                outputs=base.ModelOutput(
                    logits=self.logits,
                    aux_loss=torch.tensor([0.1, 0.2]),
                ),
                targets=self.targets,
                criterion=self.criterion,
            )

    def test_ordinary_training_path_uses_aux_loss_and_reports_components(self) -> None:
        inputs = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]
        )
        targets = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(
            TensorDataset(inputs, targets),
            batch_size=2,
            shuffle=False,
        )
        model = _AuxLossModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        stats = base.train_local_model(
            model=model,
            train_loader=loader,
            criterion=self.criterion,
            optimizer=optimizer,
            device=torch.device("cpu"),
            local_epochs=1,
        )

        self.assertGreater(stats.avg_aux_loss, 0.0)
        self.assertGreater(stats.avg_aux_loss_by_layer["layer_1"], 0.0)
        self.assertEqual(
            stats.moe_stats["layer_1"]["tokens_per_expert"],
            [4, 2],
        )
        self.assertAlmostEqual(
            stats.avg_loss,
            stats.avg_task_loss + stats.avg_aux_loss,
            places=6,
        )
        metrics = stats.to_metrics()
        self.assertIn("train_task_loss", metrics)
        self.assertIn("train_aux_loss", metrics)

    def test_expert_strategy_training_path_uses_same_aux_loss(self) -> None:
        inputs = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]
        )
        targets = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(
            TensorDataset(inputs, targets),
            batch_size=2,
            shuffle=False,
        )
        model = _ExpertAuxLossModel()
        named_parameters = dict(model.named_parameters())
        optimization_groups = base.FederatedOptimizationGroups(
            non_expert_state_names=["backbone.weight", "aux_scale"],
            expert_state_names=["experts.0.weight"],
            non_expert_parameters=["backbone.weight", "aux_scale"],
            expert_parameters=["experts.0.weight"],
        )
        optimizer = torch.optim.SGD(
            [
                named_parameters["backbone.weight"],
                named_parameters["aux_scale"],
            ],
            lr=0.01,
        )

        result = base.train_local_model_with_expert_strategy(
            model=model,
            train_loader=loader,
            criterion=self.criterion,
            optimizer=optimizer,
            device=torch.device("cpu"),
            local_epochs=1,
            grad_clip=None,
            client_id=0,
            round_id=1,
            cfg={},
            global_state=model.state_dict(),
            optimization_groups=optimization_groups,
            strategy=_SimpleExpertStrategy(),
        )

        self.assertGreater(result.stats.avg_aux_loss, 0.0)
        self.assertGreater(result.stats.avg_aux_loss_by_layer["layer_0"], 0.0)
        self.assertEqual(
            result.stats.moe_stats["layer_0"]["tokens_per_expert"],
            [2, 2],
        )
        self.assertAlmostEqual(
            result.stats.avg_loss,
            result.stats.avg_task_loss + result.stats.avg_aux_loss,
            places=6,
        )

    def test_flclient_uploads_layer_losses_and_moe_stats(self) -> None:
        inputs = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]
        )
        targets = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(
            TensorDataset(inputs, targets),
            batch_size=2,
            shuffle=False,
        )
        client = base.FLClient(
            client_id=3,
            train_loader=loader,
            cfg={"local_epochs": 1, "local_steps": None},
            device="cpu",
        )
        update = client.train(
            global_model=_AuxLossModel(),
            round_id=1,
        )

        self.assertIn("layer_1", update.avg_aux_loss_by_layer)
        self.assertEqual(
            update.moe_stats["layer_1"]["tokens_per_expert"],
            [4, 2],
        )
        summary = update.summary()
        self.assertIn("layer_1", summary["avg_aux_loss_by_layer_keys"])
        self.assertIn("layer_1", summary["moe_stats_keys"])

    def test_evaluation_uses_classification_loss_only(self) -> None:
        inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = torch.tensor([0, 1])
        loader = DataLoader(
            TensorDataset(inputs, targets),
            batch_size=2,
            shuffle=False,
        )
        model = _AuxLossModel()
        model.eval()
        with torch.no_grad():
            expected_task_loss = self.criterion(model(inputs).logits, targets)

        result = base.evaluate(
            model=model,
            data_loader=loader,
            device="cpu",
            criterion=self.criterion,
        )
        self.assertAlmostEqual(result.loss, expected_task_loss.item(), places=7)


if __name__ == "__main__":
    unittest.main()

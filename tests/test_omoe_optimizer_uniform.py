from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = ROOT / "algorithms"
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import base  # noqa: E402
import omoe_optimizer_uniform as algorithm  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


class TinySwitchModel(nn.Module):
    def __init__(self, hidden_dim: int = 4, expert_hidden_dim: int = 6) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.moe = base.SwitchMoE(
            hidden_dim=hidden_dim,
            expert_hidden_dim=expert_hidden_dim,
            num_experts=3,
            dropout=0.0,
            collect_router_statistics=False,
            aux_loss_coef=0.0,
            capacity_factor=0.0,
            token_dropping=False,
        )
        self.classifier = nn.Linear(hidden_dim, 2)

        # Deterministically route every token to expert 0. This lets the test
        # verify that OWM observes the actual dispatched expert inputs.
        with torch.no_grad():
            self.moe.router.projection.weight.zero_()
            self.moe.router.projection.bias.copy_(
                torch.tensor([1.0, 0.0, -1.0])
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        features = self.projection(inputs)
        moe_outputs, _, _ = self.moe(features)
        pooled = moe_outputs.mean(dim=1)
        return self.classifier(pooled)


def build_cfg(
    *,
    skip_epochs: int = 2,
    rounds: int = 4,
):
    return SimpleNamespace(
        optimizer={
            "type": "sgd",
            "lr": 0.05,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "nesterov": False,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "grad_clip": None,
        },
        method_config={
            "skip_epochs": skip_epochs,
            "alpha0": 1.0,
            "alpha_decay": 0.99,
        },
        rounds=rounds,
        num_clients=1,
        frac=1.0,
        num_experts=3,
    )


def build_context(
    model: nn.Module,
    cfg,
    loader: DataLoader,
    round_id: int,
) -> base.ExpertClientTrainContext:
    groups = base.build_federated_optimization_groups(
        model=model,
        expected_num_experts=3,
        strict=True,
    )
    non_expert_optimizer = base.build_optimizer(
        model=model,
        cfg=cfg,
        param_names=groups.non_expert_parameters,
    )
    return base.ExpertClientTrainContext(
        client_id=0,
        round_id=round_id,
        cfg=cfg,
        device=torch.device("cpu"),
        model=model,
        global_state=base.clone_state_dict(model.state_dict()),
        train_loader=loader,
        criterion=nn.CrossEntropyLoss(),
        non_expert_optimizer=non_expert_optimizer,
        optimization_groups=groups,
        local_epochs=1,
        local_steps=len(loader),
        non_expert_grad_clip=None,
    )


def run_local_round(
    strategy: algorithm.OMoEOptimizerClientStrategy,
    context: base.ExpertClientTrainContext,
    loader: DataLoader,
) -> dict:
    strategy.on_train_start(context)
    for batch_index, (inputs, targets) in enumerate(loader):
        context.model.zero_grad(set_to_none=True)
        logits = context.model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        strategy.step(
            base.ExpertClientBatchContext(
                train_context=context,
                epoch=0,
                batch_index=batch_index,
                local_step=batch_index + 1,
                images=inputs,
                targets=targets,
                logits=logits,
                loss=loss,
            )
        )
        context.non_expert_optimizer.step()

    return dict(strategy.on_train_end(context))


class OMoEOptimizerUniformTests(unittest.TestCase):
    def test_owm_projector_update_matches_rls_formula(self) -> None:
        projector = torch.eye(3)
        vector = torch.tensor([1.0, 0.0, 0.0])
        alpha = 1.0

        algorithm.update_owm_projector(projector, vector, alpha)

        expected = torch.diag(torch.tensor([0.5, 1.0, 1.0]))
        torch.testing.assert_close(projector, expected)
        torch.testing.assert_close(projector, projector.T)

    def test_other_projector_average_uses_paper_equation_scaling(self) -> None:
        projectors = {
            0: torch.eye(2),
            1: 2.0 * torch.eye(2),
            2: 3.0 * torch.eye(2),
        }

        result = algorithm.average_other_projectors(
            projectors,
            current_expert_id=0,
        )

        # ECAI 2024 Eq. (8) prints 1/M, with the current expert excluded.
        torch.testing.assert_close(result, (5.0 / 3.0) * torch.eye(2))

    def test_linear_delta_is_projected_on_input_feature_axis(self) -> None:
        delta = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        projector = torch.diag(torch.tensor([1.0, 0.5, 0.0]))

        result = algorithm.project_linear_delta(delta, projector)

        expected = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [4.0, 2.5, 0.0],
            ]
        )
        torch.testing.assert_close(result, expected)

    def test_method_config_validation(self) -> None:
        algorithm.validate_method_config(
            {
                "method_config": {
                    "skip_epochs": 5,
                    "alpha0": 1.0,
                    "alpha_decay": 0.99,
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "skip_epochs"):
            algorithm.validate_method_config(
                {
                    "method_config": {
                        "skip_epochs": 0,
                        "alpha0": 1.0,
                        "alpha_decay": 0.99,
                    }
                }
            )
        with self.assertRaisesRegex(ValueError, "alpha_decay"):
            algorithm.validate_method_config(
                {
                    "method_config": {
                        "skip_epochs": 5,
                        "alpha0": 1.0,
                        "alpha_decay": 1.1,
                    }
                }
            )

    def test_uniform_aggregator_assigns_equal_client_weights(self) -> None:
        aggregator = algorithm.build_expert_aggregator(build_cfg())
        updates = [
            SimpleNamespace(client_id=0, num_samples=100),
            SimpleNamespace(client_id=1, num_samples=500),
            SimpleNamespace(client_id=2, num_samples=900),
        ]

        weights = aggregator.compute_weights(updates)

        self.assertEqual(aggregator.method_name, algorithm.ALGORITHM_NAME)
        self.assertEqual(weights, {0: 1.0 / 3.0, 1: 1.0 / 3.0, 2: 1.0 / 3.0})

    def test_client_strategy_accumulates_projectors_and_alternates_across_rounds(
        self,
    ) -> None:
        torch.manual_seed(31)
        cfg = build_cfg(skip_epochs=2, rounds=4)
        inputs = torch.randn(8, 4)
        targets = torch.randint(0, 2, (8,))
        loader = DataLoader(
            TensorDataset(inputs, targets),
            batch_size=2,
            shuffle=False,
        )

        strategy = algorithm.OMoEOptimizerClientStrategy(cfg, client_id=0)
        first_model = TinySwitchModel()
        first_context = build_context(first_model, cfg, loader, round_id=1)

        first_payload = run_local_round(
            strategy,
            first_context,
            loader,
        )
        first_diagnostics = first_payload["omoe_optimizer"]

        self.assertEqual(first_diagnostics["r_steps"], len(loader))
        self.assertEqual(first_diagnostics["o_steps"], 0)
        self.assertGreater(first_diagnostics["projector_updates"], 0)
        self.assertEqual(first_diagnostics["num_projectors"], 6)

        projector_key = ("moe", 0, "fc1")
        projector_after_r = strategy._projectors[projector_key].clone()
        identity = torch.eye(projector_after_r.shape[0])
        self.assertFalse(torch.allclose(projector_after_r, identity))

        # Simulate the next FL round: the local model is recreated, but the
        # client strategy and its OWM projectors remain alive.
        second_model = copy.deepcopy(first_model)
        second_context = build_context(second_model, cfg, loader, round_id=2)
        non_expert_before = {
            name: dict(second_model.named_parameters())[name].detach().clone()
            for name in second_context.optimization_groups.non_expert_parameters
        }
        expert_before = {
            name: dict(second_model.named_parameters())[name].detach().clone()
            for name in second_context.optimization_groups.expert_parameter_names
        }

        second_payload = run_local_round(
            strategy,
            second_context,
            loader,
        )
        second_diagnostics = second_payload["omoe_optimizer"]

        self.assertEqual(second_diagnostics["r_steps"], 0)
        self.assertEqual(second_diagnostics["o_steps"], len(loader))
        torch.testing.assert_close(
            strategy._projectors[projector_key],
            projector_after_r,
        )

        second_parameters = dict(second_model.named_parameters())
        for name, before in non_expert_before.items():
            torch.testing.assert_close(second_parameters[name], before)

        self.assertTrue(
            any(
                not torch.allclose(second_parameters[name], before)
                for name, before in expert_before.items()
            )
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = ROOT / "algorithms"
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import base  # noqa: E402
import rfedags_grassmann as algorithm  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


class TinyExpert(nn.Module):
    def __init__(self, hidden_dim: int = 4, expert_hidden_dim: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, expert_hidden_dim)
        self.fc2 = nn.Linear(expert_hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(inputs)))


class TinySwitchMoE(nn.Module):
    def __init__(self, num_experts: int = 3) -> None:
        super().__init__()
        self.experts = nn.ModuleDict(
            {
                f"expert_{expert_id}": TinyExpert()
                for expert_id in range(num_experts)
            }
        )


class TinySwitchModel(nn.Module):
    def __init__(self, num_experts: int = 3) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.moe = TinySwitchMoE(num_experts=num_experts)
        self.classifier = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor):
        features = self.backbone(inputs)
        expert_outputs = [
            expert(features)
            for expert in self.moe.experts.values()
        ]
        merged = torch.stack(expert_outputs, dim=0).mean(dim=0)
        return base.ModelOutput(logits=self.classifier(merged))


def build_cfg(
    *,
    local_steps: int | None = 2,
    frac: float = 1.0,
    method_config: dict | None = None,
    optimizer_overrides: dict | None = None,
):
    optimizer = {
        "type": "sgd",
        "lr": 0.05,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "nesterov": False,
        "grad_clip": None,
    }
    if optimizer_overrides:
        optimizer.update(optimizer_overrides)
    configured_method = dict(method_config or {})
    return base.load_embedded_config(
        algorithm.EMBEDDED_METHOD_CONFIG,
        method_defaults=algorithm.METHOD_CONFIG_DEFAULTS,
        method_validator=algorithm.validate_method_config,
        config_overrides={
            "dataset": "cifar10",
            "num_clients": 2,
            "frac": frac,
            "local_steps": local_steps,
            "num_experts": 3,
            "optimizer": optimizer,
            "method_config": configured_method,
        },
    )


def build_model_and_groups(num_experts: int = 3):
    model = TinySwitchModel(num_experts=num_experts)
    param_groups = base.build_param_groups(
        model=model,
        expected_num_experts=num_experts,
        strict=True,
    )
    optimization_groups = base.build_federated_optimization_groups(
        model=model,
        param_groups=param_groups,
        strict=True,
    )
    return model, param_groups, optimization_groups


def build_aggregator(cfg, model, optimization_groups):
    context = base.MethodContext(
        cfg=cfg,
        device=torch.device("cpu"),
        dataset_name="cifar10",
        global_model=model,
        optimization_groups=optimization_groups,
    )
    aggregator = algorithm.build_expert_aggregator(cfg)
    aggregator.set_method_context(context)
    aggregator.initialize_model(model)
    return aggregator


class RFedAGSGrassmannTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.cfg = build_cfg()

    def test_qr_retraction_and_tangent_projection(self) -> None:
        matrix = torch.randn(12, 4)
        basis = algorithm.qr_retraction(matrix)
        error = algorithm.orthogonality_error(basis)
        self.assertLess(float(error), 1e-5)

        ambient = torch.randn_like(basis)
        tangent = algorithm.stiefel_tangent_projection(basis, ambient)
        tangent_condition = basis.T @ tangent + tangent.T @ basis
        torch.testing.assert_close(
            tangent_condition,
            torch.zeros_like(tangent_condition),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_grassmann_distance_is_invariant_to_basis_rotation(self) -> None:
        basis = algorithm.qr_retraction(torch.randn(12, 4))
        rotation = algorithm.qr_retraction(torch.randn(4, 4))
        rotated_basis = basis @ rotation

        overlap = algorithm.grassmann_overlap(basis, rotated_basis)
        distance = algorithm.grassmann_projection_distance(
            basis,
            rotated_basis,
        )
        self.assertAlmostEqual(float(overlap), 4.0, places=4)
        self.assertLess(float(distance), 1e-3)

    def test_overlap_regularizer_distinguishes_identical_and_orthogonal_subspaces(self) -> None:
        basis_a = torch.eye(8)[:, :2].clone().requires_grad_(True)
        basis_identical = basis_a.detach().clone().requires_grad_(True)
        basis_orthogonal = torch.eye(8)[:, 2:4].clone().requires_grad_(True)

        identical_loss, identical_metrics = algorithm.compute_subspace_overlap_loss(
            {"layer": (basis_a, basis_identical)},
            rho0=0.25,
        )
        orthogonal_loss, orthogonal_metrics = algorithm.compute_subspace_overlap_loss(
            {"layer": (basis_a, basis_orthogonal)},
            rho0=0.25,
        )

        self.assertGreater(float(identical_loss.detach()), 0.0)
        self.assertEqual(float(orthogonal_loss.detach()), 0.0)
        self.assertAlmostEqual(identical_metrics["mean_normalized_overlap"], 1.0)
        self.assertAlmostEqual(orthogonal_metrics["mean_normalized_overlap"], 0.0)
        identical_loss.backward()
        self.assertIsNotNone(basis_a.grad)
        self.assertTrue(torch.isfinite(basis_a.grad).all())

    def test_overlap_regularizer_is_rank_normalized(self) -> None:
        eye4 = torch.eye(4)
        rank_one_a = eye4[:, :1]
        rank_one_b = (rank_one_a + eye4[:, 1:2]) / (2.0 ** 0.5)

        eye8 = torch.eye(8)
        rank_two_a = eye8[:, :2]
        rank_two_b = (rank_two_a + eye8[:, 2:4]) / (2.0 ** 0.5)

        rank_one_loss, rank_one_metrics = algorithm.compute_subspace_overlap_loss(
            {"layer": (rank_one_a, rank_one_b)},
            rho0=0.25,
        )
        rank_two_loss, rank_two_metrics = algorithm.compute_subspace_overlap_loss(
            {"layer": (rank_two_a, rank_two_b)},
            rho0=0.25,
        )

        torch.testing.assert_close(rank_one_loss, torch.tensor(0.25))
        torch.testing.assert_close(rank_two_loss, torch.tensor(0.25))
        self.assertAlmostEqual(
            rank_one_metrics["mean_normalized_overlap"],
            rank_two_metrics["mean_normalized_overlap"],
        )

    def test_overlap_regularizer_is_layer_balanced_and_reports_active_ratio(self) -> None:
        eye = torch.eye(8)
        basis_a = eye[:, :2]
        basis_b = (basis_a + eye[:, 2:4]) / (2.0 ** 0.5)
        basis_c = eye[:, 4:6]

        loss, metrics = algorithm.compute_subspace_overlap_loss(
            {
                "one_pair_layer": (basis_a, basis_b),
                "three_pair_layer": (basis_a, basis_b, basis_c),
            },
            rho0=0.25,
        )

        # Layer losses are 0.25 and (0.25 + 0 + 0) / 3.
        # The final loss averages the two layer means.
        torch.testing.assert_close(loss, torch.tensor(1.0 / 6.0))
        self.assertAlmostEqual(metrics["active_overlap_ratio"], 0.5)
        self.assertEqual(metrics["num_active_pairs"], 2)
        self.assertEqual(metrics["num_pairs"], 4)
        self.assertAlmostEqual(metrics["mean_normalized_overlap"], 1.0 / 3.0)

    def test_default_overlap_regularization_hyperparameters(self) -> None:
        self.assertAlmostEqual(self.cfg.method_config.lambda_div, 5.0)
        self.assertAlmostEqual(self.cfg.method_config.rho0, 0.24)
        self.assertAlmostEqual(self.cfg.method_config.server_lr, 1.0)

    def test_method_and_runtime_config_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "rho0"):
            algorithm.validate_method_config({"rho0": 1.0})
        with self.assertRaisesRegex(ValueError, "lambda_div"):
            algorithm.validate_method_config({"lambda_div": -0.1})
        with self.assertRaisesRegex(ValueError, "server_lr"):
            algorithm.validate_method_config({"server_lr": 0.0})
        with self.assertRaisesRegex(ValueError, "rho0"):
            build_cfg(method_config={"rho0": 1.0})

        with self.assertRaisesRegex(ValueError, "local_steps"):
            algorithm._validate_runtime_config(build_cfg(local_steps=None))
        with self.assertRaisesRegex(ValueError, "frac=1.0"):
            algorithm._validate_runtime_config(build_cfg(frac=0.5))
        with self.assertRaisesRegex(ValueError, "momentum=0"):
            algorithm._validate_runtime_config(
                build_cfg(optimizer_overrides={"momentum": 0.9})
            )
        with self.assertRaisesRegex(ValueError, "weight_decay=0"):
            algorithm._validate_runtime_config(
                build_cfg(optimizer_overrides={"weight_decay": 1e-4})
            )
        with self.assertRaisesRegex(ValueError, "grad_clip"):
            algorithm._validate_runtime_config(
                build_cfg(optimizer_overrides={"grad_clip": 1.0})
            )

    def test_model_initialization_projects_all_expert_fc1_weights(self) -> None:
        model, _, optimization_groups = build_model_and_groups()
        aggregator = build_aggregator(self.cfg, model, optimization_groups)
        parameter_map = dict(model.named_parameters())

        self.assertEqual(len(aggregator._layout.stiefel_parameter_names), 3)
        for name in aggregator._layout.stiefel_parameter_names:
            self.assertLess(
                float(algorithm.orthogonality_error(parameter_map[name].detach())),
                algorithm.ORTHOGONALITY_TOLERANCE,
            )

        self.assertEqual(
            set(aggregator._layout.expert_parameter_names),
            set(optimization_groups.expert_parameters),
        )

    def test_client_strategy_updates_product_manifold_and_returns_streams(self) -> None:
        cfg = build_cfg(local_steps=1)
        model, _, optimization_groups = build_model_and_groups()
        aggregator = build_aggregator(cfg, model, optimization_groups)
        del aggregator

        parameter_map = dict(model.named_parameters())
        global_state = base.clone_state_dict(model.state_dict())
        non_expert_parameters = [
            parameter
            for name, parameter in parameter_map.items()
            if name not in set(optimization_groups.expert_parameters)
        ]
        non_expert_optimizer = torch.optim.SGD(
            non_expert_parameters,
            lr=cfg.optimizer.lr,
            momentum=0.0,
            weight_decay=0.0,
        )
        loader = DataLoader(
            TensorDataset(torch.randn(2, 4), torch.tensor([0, 1])),
            batch_size=2,
        )
        context = base.ExpertClientTrainContext(
            client_id=0,
            round_id=1,
            cfg=cfg,
            device=torch.device("cpu"),
            model=model,
            global_state=global_state,
            train_loader=loader,
            criterion=nn.CrossEntropyLoss(),
            non_expert_optimizer=non_expert_optimizer,
            optimization_groups=optimization_groups,
            local_epochs=1,
            local_steps=1,
            non_expert_grad_clip=None,
        )
        strategy = algorithm.RFedAGSGrassmannClientStrategy(cfg, client_id=0)
        strategy.on_train_start(context)

        before_euclidean = {
            name: parameter.detach().clone()
            for name, parameter in context.expert_parameters.items()
            if name not in set(strategy._layout.stiefel_parameter_names)
        }
        for parameter in context.expert_parameters.values():
            parameter.grad = torch.randn_like(parameter) * 0.02

        batch_context = base.ExpertClientBatchContext(
            train_context=context,
            epoch=0,
            batch_index=0,
            local_step=1,
            images=torch.randn(2, 4),
            targets=torch.tensor([0, 1]),
            logits=torch.randn(2, 2),
            loss=torch.tensor(1.0),
        )
        strategy.step(batch_context)
        payload = strategy.on_train_end(context)

        self.assertEqual(payload["algorithm"], algorithm.ALGORITHM_NAME)
        self.assertEqual(payload["num_local_steps"], 1)
        self.assertEqual(
            set(payload["gradient_streams"]),
            set(optimization_groups.expert_parameters),
        )
        for name in strategy._layout.stiefel_parameter_names:
            self.assertLess(
                float(algorithm.orthogonality_error(parameter_map[name].detach())),
                algorithm.ORTHOGONALITY_TOLERANCE,
            )
            anchor = global_state[name]
            stream = payload["gradient_streams"][name]
            tangent_condition = anchor.T @ stream + stream.T @ anchor
            torch.testing.assert_close(
                tangent_condition,
                torch.zeros_like(tangent_condition),
                atol=2e-5,
                rtol=2e-5,
            )
        for name, before in before_euclidean.items():
            self.assertFalse(torch.equal(parameter_map[name].detach(), before))
        self.assertLess(
            payload["diagnostics"]["max_orthogonality_error"],
            algorithm.ORTHOGONALITY_TOLERANCE,
        )
        self.assertGreaterEqual(
            payload["diagnostics"]["avg_active_overlap_ratio"],
            0.0,
        )
        self.assertLessEqual(
            payload["diagnostics"]["avg_active_overlap_ratio"],
            1.0,
        )
        self.assertEqual(payload["diagnostics"]["final_num_pairs"], 3)

    def test_base_local_training_hook_emits_rfedags_gradient_stream(self) -> None:
        cfg = build_cfg(local_steps=2)
        model, _, optimization_groups = build_model_and_groups()
        aggregator = build_aggregator(cfg, model, optimization_groups)
        del aggregator

        global_state = base.clone_state_dict(model.state_dict())
        non_expert_optimizer = base.build_optimizer(
            model=model,
            cfg=cfg,
            param_names=optimization_groups.non_expert_parameters,
        )
        loader = DataLoader(
            TensorDataset(torch.randn(4, 4), torch.tensor([0, 1, 0, 1])),
            batch_size=2,
            shuffle=False,
        )
        strategy = algorithm.build_expert_client_training_strategy(
            cfg,
            client_id=0,
        )
        train_result = base.train_local_model_with_expert_strategy(
            model=model,
            train_loader=loader,
            criterion=base.build_criterion(cfg),
            optimizer=non_expert_optimizer,
            device=torch.device("cpu"),
            local_epochs=1,
            local_steps=2,
            grad_clip=None,
            client_id=0,
            round_id=1,
            cfg=cfg,
            global_state=global_state,
            optimization_groups=optimization_groups,
            strategy=strategy,
        )

        payload = train_result.algorithm_payload
        self.assertEqual(train_result.stats.num_batches, 2)
        self.assertEqual(payload["num_local_steps"], 2)
        self.assertEqual(
            set(payload["gradient_streams"]),
            set(optimization_groups.expert_parameters),
        )
        self.assertTrue(
            all(
                torch.isfinite(stream).all()
                for stream in payload["gradient_streams"].values()
            )
        )

    def test_server_aggregates_gradient_streams_and_retracts_stiefel_weights(self) -> None:
        model, param_groups, optimization_groups = build_model_and_groups()
        aggregator = build_aggregator(self.cfg, model, optimization_groups)
        global_state = base.clone_state_dict(model.state_dict())
        stiefel_names = set(aggregator._layout.stiefel_parameter_names)

        client_updates = []
        client_stream_values = []
        for client_id, scale in ((0, 0.01), (1, 0.03)):
            streams = {}
            for name in optimization_groups.expert_parameters:
                if name in stiefel_names:
                    ambient = torch.full_like(global_state[name], scale)
                    streams[name] = algorithm.stiefel_tangent_projection(
                        global_state[name],
                        ambient,
                    )
                else:
                    streams[name] = torch.full_like(global_state[name], scale)
            client_stream_values.append(streams)
            client_updates.append(
                base.ClientUpdate(
                    client_id=client_id,
                    round_id=1,
                    num_samples=10 + client_id,
                    model_delta={
                        "backbone.weight": torch.zeros_like(model.backbone.weight)
                    },
                    algorithm_payload={
                        "algorithm": algorithm.ALGORITHM_NAME,
                        "version": algorithm.PAYLOAD_VERSION,
                        "num_local_steps": self.cfg.local_steps,
                        "gradient_streams": streams,
                        "diagnostics": {
                            "avg_diversity_loss": 0.2 + client_id * 0.1,
                            "avg_normalized_overlap": 0.3 + client_id * 0.1,
                            "avg_active_overlap_ratio": 0.5 + client_id * 0.1,
                            "max_normalized_overlap": 0.4 + client_id * 0.1,
                            "avg_projection_distance": 1.0,
                            "max_orthogonality_error": 1e-6,
                        },
                    },
                )
            )

        result = aggregator.aggregate(
            global_state=global_state,
            client_updates=client_updates,
            param_names=param_groups.expert,
            base_state=global_state,
            strict=True,
        )
        self.assertEqual(result.weights, {0: 0.5, 1: 0.5})

        for name in optimization_groups.expert_parameters:
            average_stream = 0.5 * (
                client_stream_values[0][name] + client_stream_values[1][name]
            )
            if name in stiefel_names:
                average_stream = algorithm.stiefel_tangent_projection(
                    global_state[name],
                    average_stream,
                )
                expected = algorithm.qr_retraction(
                    global_state[name]
                    - self.cfg.method_config.server_lr * average_stream
                )
                self.assertLess(
                    float(algorithm.orthogonality_error(result.new_state_dict[name])),
                    algorithm.ORTHOGONALITY_TOLERANCE,
                )
            else:
                expected = global_state[name] - (
                    self.cfg.method_config.server_lr * average_stream
                )
            torch.testing.assert_close(
                result.new_state_dict[name],
                expected,
                atol=1e-6,
                rtol=1e-6,
            )

        self.assertEqual(
            result.diagnostics["aggregation"],
            "rfedags_gradient_stream",
        )
        self.assertAlmostEqual(
            result.diagnostics["mean_client_diversity_loss"],
            0.25,
        )
        self.assertAlmostEqual(
            result.diagnostics["mean_client_active_overlap_ratio"],
            0.55,
        )

    def test_aggregator_rejects_incomplete_gradient_stream_payload(self) -> None:
        model, param_groups, optimization_groups = build_model_and_groups()
        aggregator = build_aggregator(self.cfg, model, optimization_groups)
        global_state = base.clone_state_dict(model.state_dict())
        payload = {
            "algorithm": algorithm.ALGORITHM_NAME,
            "version": algorithm.PAYLOAD_VERSION,
            "num_local_steps": self.cfg.local_steps,
            "gradient_streams": {},
            "diagnostics": {},
        }
        update = base.ClientUpdate(
            client_id=0,
            round_id=1,
            num_samples=10,
            model_delta={"backbone.weight": torch.zeros_like(model.backbone.weight)},
            algorithm_payload=payload,
        )
        with self.assertRaisesRegex(ValueError, "梯度流参数集合不匹配"):
            aggregator.aggregate(
                global_state=global_state,
                client_updates=[update],
                param_names=param_groups.expert,
                base_state=global_state,
                strict=True,
            )


if __name__ == "__main__":
    unittest.main()

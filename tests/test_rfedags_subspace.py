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
import rfedags_subspace as algorithm  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


class RFedAGSSubspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.cfg = self._build_cfg(local_steps=4)

    @staticmethod
    def _build_cfg(local_steps: int | None = 4):
        return base.load_embedded_config(
            algorithm.EMBEDDED_METHOD_CONFIG,
            method_defaults=algorithm.METHOD_CONFIG_DEFAULTS,
            method_validator=algorithm.validate_method_config,
            config_overrides={
                "dataset": "cifar10",
                "num_clients": 2,
                "frac": 1.0,
                "local_steps": local_steps,
                "num_experts": 4,
                "model_cfg": {
                    "transformer_layers": 2,
                    "hidden_dim": 16,
                    "attention_heads": 4,
                    "patch_size": 4,
                    "expert_hidden_dim": 32,
                    "moe_block_indices": [1, 2],
                },
                "method_config": {
                    "subspace_rank": 8,
                },
            },
        )

    def test_fixed_local_steps_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "local_steps"):
            self._build_cfg(local_steps=None)

        self.assertEqual(int(self.cfg.local_steps), 4)

    def test_low_rank_stiefel_linear_matches_factorized_weight(self) -> None:
        original = nn.Linear(16, 32)
        layer = algorithm.LowRankStiefelLinear.from_linear(original, rank=8)
        inputs = torch.randn(5, 16)

        expected = nn.functional.linear(
            inputs,
            layer.A @ layer.P.transpose(0, 1),
            layer.bias,
        )
        actual = layer(inputs)

        self.assertEqual(tuple(layer.A.shape), (32, 8))
        self.assertEqual(tuple(layer.P.shape), (16, 8))
        self.assertTrue(
            torch.allclose(layer.P.transpose(0, 1) @ layer.P, torch.eye(8), atol=1e-5)
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_stiefel_projection_and_qr_retraction(self) -> None:
        point = torch.linalg.qr(torch.randn(16, 8), mode="reduced").Q
        ambient = torch.randn_like(point)

        tangent = algorithm.stiefel_tangent_projection(point, ambient)
        tangent_error = (
            point.transpose(0, 1) @ tangent
            + tangent.transpose(0, 1) @ point
        )
        self.assertLess(float(torch.linalg.norm(tangent_error)), 1e-5)

        updated = algorithm.qr_retraction(point - 0.01 * tangent)
        self.assertTrue(
            torch.allclose(
                updated.transpose(0, 1) @ updated,
                torch.eye(8),
                atol=1e-5,
            )
        )

    def test_model_builder_replaces_each_expert_first_layer(self) -> None:
        model = algorithm.build_rfedags_subspace_model(self.cfg)
        stiefel_layers = [
            module
            for module in model.modules()
            if isinstance(module, algorithm.LowRankStiefelLinear)
        ]

        self.assertEqual(len(stiefel_layers), 2 * 4)
        self.assertEqual(
            len(model._rfedags_subspace_layer_records),
            2,
        )
        self.assertEqual(
            set(model._rfedags_subspace_thresholds),
            set(model._rfedags_subspace_layer_records),
        )

        for layer in stiefel_layers:
            self.assertTrue(
                torch.allclose(
                    layer.P.transpose(0, 1) @ layer.P,
                    torch.eye(8),
                    atol=1e-5,
                )
            )

        output = base.extract_model_output(
            model(torch.randn(2, 3, 32, 32))
        )
        self.assertEqual(tuple(output.logits.shape), (2, 10))

    def test_diversity_loss_is_finite_and_differentiable(self) -> None:
        model = algorithm.build_rfedags_subspace_model(self.cfg)
        loss, overlap_by_layer = algorithm.compute_subspace_diversity_loss(model)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss.item()), 0.0)
        self.assertEqual(set(overlap_by_layer), set(model._rfedags_subspace_layer_records))

        loss.backward()
        p_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.endswith(".P")
        ]
        self.assertTrue(p_parameters)
        self.assertTrue(all(parameter.grad is not None for parameter in p_parameters))

    def test_client_strategy_executes_exact_fixed_step_count(self) -> None:
        model = algorithm.build_rfedags_subspace_model(self.cfg)
        global_state = base.clone_state_dict(model.state_dict())
        groups = base.build_federated_optimization_groups(
            model=model,
            expected_num_experts=4,
            strict=True,
        )
        non_expert_optimizer = base.build_optimizer(
            model=model,
            cfg=self.cfg,
            param_names=groups.non_expert_parameters,
        )

        dataset = TensorDataset(
            torch.randn(5, 3, 32, 32),
            torch.randint(0, 10, (5,)),
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False)
        strategy = algorithm.RFedAGSSubspaceClientStrategy(
            cfg=self.cfg,
            client_id=0,
        )

        result = base.train_local_model_with_expert_strategy(
            model=model,
            train_loader=loader,
            criterion=base.build_criterion(self.cfg),
            optimizer=non_expert_optimizer,
            device=torch.device("cpu"),
            local_epochs=5,
            grad_clip=base._get_grad_clip(self.cfg),
            client_id=0,
            round_id=1,
            cfg=self.cfg,
            global_state=global_state,
            optimization_groups=groups,
            strategy=strategy,
            local_steps=4,
        )

        self.assertEqual(result.stats.num_batches, 4)
        payload = result.algorithm_payload[algorithm.PAYLOAD_KEY]
        self.assertEqual(payload["local_steps"], 4)
        self.assertEqual(set(payload["gradient_stream"]), {
            item["p_name"]
            for records in model._rfedags_subspace_layer_records.values()
            for item in records
        })

        for name, parameter in model.named_parameters():
            if name.endswith(".P"):
                self.assertTrue(
                    torch.allclose(
                        parameter.transpose(0, 1) @ parameter,
                        torch.eye(parameter.shape[1]),
                        atol=1e-5,
                    )
                )

    def test_server_aggregator_uses_stream_for_p_and_delta_for_other_expert_params(self) -> None:
        model = algorithm.build_rfedags_subspace_model(self.cfg)
        state = base.clone_state_dict(model.state_dict())
        param_groups = base.build_param_groups(
            model=model,
            expected_num_experts=4,
            strict=True,
        )
        optimization_groups = base.build_federated_optimization_groups(
            model=model,
            param_groups=param_groups,
            expected_num_experts=4,
            strict=True,
        )
        method_context = base.MethodContext(
            cfg=self.cfg,
            device=torch.device("cpu"),
            dataset_name="cifar10",
            model_builder=algorithm.build_rfedags_subspace_model,
            global_model=model,
            param_groups=param_groups,
            optimization_groups=optimization_groups,
        )
        aggregator = algorithm.RFedAGSSubspaceExpertAggregator(self.cfg)
        aggregator.set_method_context(method_context)

        p_names = {
            item["p_name"]
            for records in model._rfedags_subspace_layer_records.values()
            for item in records
        }
        euclidean_name = next(
            name
            for name in optimization_groups.expert_parameter_names
            if name.endswith(".A")
        )
        client_updates = []
        for client_id, delta_value in ((0, 0.1), (1, 0.3)):
            model_delta = {
                name: torch.zeros_like(value)
                for name, value in state.items()
                if value.is_floating_point() or value.is_complex()
            }
            model_delta[euclidean_name] = torch.full_like(
                state[euclidean_name],
                delta_value,
            )
            payload = {
                algorithm.PAYLOAD_KEY: {
                    "version": algorithm.PAYLOAD_VERSION,
                    "local_steps": 4,
                    "gradient_stream": {
                        name: torch.zeros_like(state[name])
                        for name in p_names
                    },
                    "expert_buffers": {},
                    "mean_diversity_loss": 0.0,
                    "mean_overlap_by_layer": {
                        layer_name: 0.5
                        for layer_name in model._rfedags_subspace_layer_records
                    },
                }
            }
            client_updates.append(
                base.ClientUpdate(
                    client_id=client_id,
                    round_id=1,
                    num_samples=10,
                    model_delta=model_delta,
                    metrics={"num_batches": 4},
                    algorithm_payload=payload,
                )
            )

        result = aggregator.aggregate(
            global_state=state,
            client_updates=client_updates,
            param_names=param_groups.expert,
            base_state=state,
            strict=True,
        )

        expected = state[euclidean_name] + 0.2
        self.assertTrue(
            torch.allclose(result.new_state_dict[euclidean_name], expected, atol=1e-6)
        )
        for name in p_names:
            self.assertTrue(
                torch.allclose(result.new_state_dict[name], state[name], atol=1e-5)
            )
            p = result.new_state_dict[name]
            self.assertTrue(
                torch.allclose(
                    p.transpose(0, 1) @ p,
                    torch.eye(p.shape[1]),
                    atol=1e-5,
                )
            )
        self.assertEqual(result.diagnostics["local_steps"], 4)
        self.assertEqual(result.weights, {0: 0.5, 1: 0.5})


if __name__ == "__main__":
    unittest.main()

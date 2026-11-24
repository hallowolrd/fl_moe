from __future__ import annotations

import math
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
import rfedags_server_diversity as algorithm  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


class RFedAGSServerDiversityTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(29)
        self.cfg = self._build_cfg(local_steps=4)

    @staticmethod
    def _build_cfg(
        local_steps: int | None = 4,
        frac: float = 1.0,
        server_diversity_lr: float | None = 0.5,
        tau_margin: float = 0.05,
        threshold_cap: float = 0.95,
    ):
        method_config = {
            "subspace_rank": 8,
            "tau_margin": tau_margin,
            "subspace_overlap_threshold_cap": threshold_cap,
        }
        if server_diversity_lr is not None:
            method_config["server_diversity_lr"] = server_diversity_lr

        return base.load_embedded_config(
            algorithm.EMBEDDED_METHOD_CONFIG,
            method_defaults=algorithm.METHOD_CONFIG_DEFAULTS,
            method_validator=algorithm.validate_method_config,
            config_overrides={
                "dataset": "cifar10",
                "num_clients": 2,
                "frac": frac,
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
                "method_config": method_config,
            },
        )

    def test_scheme_two_is_independent_from_scheme_one(self) -> None:
        self.assertEqual(algorithm.PAYLOAD_KEY, algorithm.ALGORITHM_NAME)
        self.assertFalse(hasattr(algorithm, "subspace"))
        source = Path(algorithm.__file__).read_text(encoding="utf-8")
        self.assertNotIn("rfedags_subspace", source)
        self.assertNotIn("subspace.", source)
        self.assertIs(
            algorithm.RFedAGSServerDiversityClientStrategy.__bases__[0],
            base.ExpertClientTrainingStrategy,
        )
        self.assertIs(
            algorithm.RFedAGSServerDiversityExpertAggregator.__bases__[0],
            base.Aggregator,
        )
        self.assertFalse(
            any("rfedags_subspace" in cls.__module__
                for cls in algorithm.RFedAGSServerDiversityClientStrategy.__mro__)
        )
        self.assertFalse(
            any("rfedags_subspace" in cls.__module__
                for cls in algorithm.RFedAGSServerDiversityExpertAggregator.__mro__)
        )

    def test_fixed_local_steps_and_full_participation_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "local_steps"):
            self._build_cfg(local_steps=None)

        with self.assertRaisesRegex(ValueError, "frac=1.0"):
            self._build_cfg(frac=0.5)

        self.assertEqual(int(self.cfg.local_steps), 4)
        self.assertEqual(
            float(self.cfg.method_config.server_diversity_lr),
            0.5,
        )

    def _build_server_aggregator(self, cfg, model):
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
            cfg=cfg,
            device=torch.device("cpu"),
            dataset_name="cifar10",
            model_builder=algorithm.build_rfedags_server_diversity_model,
            global_model=model,
            param_groups=param_groups,
            optimization_groups=optimization_groups,
        )
        aggregator = algorithm.RFedAGSServerDiversityExpertAggregator(cfg)
        aggregator.set_method_context(method_context)
        aggregator.initialize_model(model)
        return param_groups, optimization_groups, aggregator

    @staticmethod
    def _set_overlapping_subspaces(model, angle: float) -> None:
        records = model._rfedags_server_diversity_layer_records
        parameter_map = dict(model.named_parameters())
        basis = torch.eye(16)[:, :8]
        rotated = basis.clone()
        rotated[:, 0] = (
            torch.cos(torch.tensor(angle)) * basis[:, 0]
            + torch.sin(torch.tensor(angle)) * torch.eye(16)[:, 8]
        )
        with torch.no_grad():
            for layer_records in records.values():
                for expert_index, item in enumerate(layer_records):
                    parameter_map[item["p_name"]].copy_(
                        basis if expert_index < 3 else rotated
                    )

    def test_model_builder_uses_shared_stiefel_parameterization(self) -> None:
        model = algorithm.build_rfedags_server_diversity_model(self.cfg)
        stiefel_layers = [
            module
            for module in model.modules()
            if isinstance(module, algorithm.ServerDiversityLowRankStiefelLinear)
        ]

        self.assertEqual(len(stiefel_layers), 2 * 4)
        self.assertEqual(
            len(model._rfedags_server_diversity_layer_records),
            2,
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

    def test_server_thresholds_are_capped_strictly_below_one(self) -> None:
        cfg = self._build_cfg(tau_margin=0.99, threshold_cap=0.95)
        model = algorithm.build_rfedags_server_diversity_model(cfg)

        self.assertTrue(model._rfedags_server_diversity_thresholds)
        self.assertTrue(
            all(
                0.0 <= threshold <= 0.95
                for threshold in model._rfedags_server_diversity_thresholds.values()
            )
        )
        self.assertTrue(
            all(
                threshold < 1.0
                for threshold in model._rfedags_server_diversity_thresholds.values()
            )
        )
        self.assertTrue(
            all(
                abs(threshold - 0.95) < 1e-12
                for threshold in model._rfedags_server_diversity_thresholds.values()
            )
        )

    def test_threshold_cap_must_be_strictly_between_zero_and_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold_cap"):
            self._build_cfg(threshold_cap=1.0)
        with self.assertRaisesRegex(ValueError, "threshold_cap"):
            self._build_cfg(threshold_cap=0.0)

    def test_default_server_diversity_lr_is_used(self) -> None:
        cfg = self._build_cfg(server_diversity_lr=None)
        self.assertAlmostEqual(
            float(cfg.method_config.server_diversity_lr),
            algorithm.SERVER_DIVERSITY_LR,
        )

        model = algorithm.build_rfedags_server_diversity_model(cfg)
        self._set_overlapping_subspaces(model, angle=0.4)
        _, _, aggregator = self._build_server_aggregator(cfg, model)
        self.assertAlmostEqual(
            aggregator.diversity_lr,
            algorithm.SERVER_DIVERSITY_LR,
        )

        aggregator.after_aggregate(model)
        diagnostics = aggregator.build_post_aggregate_diagnostics(model)
        self.assertTrue(diagnostics["server_diversity_applied"])
        self.assertIsNone(diagnostics["server_diversity_skip_reason"])
        self.assertGreater(
            diagnostics["server_diversity_parameter_step_norm"],
            0.0,
        )

    def test_exactly_identical_subspaces_report_zero_tangent_gradient(self) -> None:
        model = algorithm.build_rfedags_server_diversity_model(self.cfg)
        self._set_overlapping_subspaces(model, angle=0.0)
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name.endswith(".P")
        }
        _, _, aggregator = self._build_server_aggregator(self.cfg, model)

        aggregator.after_aggregate(model)
        diagnostics = aggregator.build_post_aggregate_diagnostics(model)

        self.assertGreater(diagnostics["server_diversity_loss_before"], 0.0)
        self.assertLess(
            diagnostics["server_diversity_gradient_norm"],
            1e-6,
        )
        self.assertFalse(diagnostics["server_diversity_applied"])
        self.assertEqual(
            diagnostics["server_diversity_skip_reason"],
            "zero_riemannian_gradient",
        )
        self.assertEqual(
            diagnostics["server_diversity_parameter_step_norm"],
            0.0,
        )
        for name, parameter in model.named_parameters():
            if name in before:
                self.assertTrue(torch.equal(parameter, before[name]))

    def test_client_training_does_not_compute_diversity_loss(self) -> None:
        model = algorithm.build_rfedags_server_diversity_model(self.cfg)
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
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            drop_last=False,
        )
        strategy = algorithm.RFedAGSServerDiversityClientStrategy(
            cfg=self.cfg,
            client_id=0,
        )

        # If the client path tries to evaluate the diversity objective, fail
        # immediately. Scheme two computes that objective only on the server.
        with patch.object(
            algorithm,
            "compute_subspace_diversity_loss",
            side_effect=AssertionError(
                "client must not compute the diversity loss"
            ),
        ):
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
        self.assertNotIn("mean_diversity_loss", payload)
        self.assertNotIn("mean_overlap_by_layer", payload)
        self.assertTrue(payload["gradient_stream"])

        for name, parameter in model.named_parameters():
            if name.endswith(".P"):
                self.assertTrue(
                    torch.allclose(
                        parameter.transpose(0, 1) @ parameter,
                        torch.eye(parameter.shape[1]),
                        atol=1e-5,
                    )
                )

    def test_server_correction_runs_after_rfedags_aggregation(self) -> None:
        model = algorithm.build_rfedags_server_diversity_model(self.cfg)
        records = model._rfedags_server_diversity_layer_records
        self._set_overlapping_subspaces(model, angle=0.1)

        state = base.clone_state_dict(model.state_dict())
        param_groups, optimization_groups, aggregator = (
            self._build_server_aggregator(self.cfg, model)
        )

        p_names = {
            item["p_name"]
            for layer_records in records.values()
            for item in layer_records
        }
        expert_parameter_names = set(optimization_groups.expert_parameter_names)
        euclidean_name = next(
            name for name in expert_parameter_names if name.endswith(".A")
        )
        expert_buffer_names = (
            set(optimization_groups.expert_state_names)
            - set(dict(model.named_parameters()))
        )
        parameter_map = dict(model.named_parameters())
        stream_by_name = {}
        for name in p_names:
            parameter = parameter_map[name].detach()
            ambient_direction = torch.randn_like(parameter)
            tangent_direction = algorithm.stiefel_tangent_projection(
                parameter,
                ambient_direction,
            )
            self.assertGreater(
                float(torch.linalg.norm(tangent_direction).item()),
                0.0,
            )
            stream_by_name[name] = 1e-3 * tangent_direction

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
                        name: stream_by_name[name]
                        * (0.5 if client_id == 0 else 1.5)
                        for name in p_names
                    },
                    "expert_buffers": {
                        name: state[name].clone()
                        for name in expert_buffer_names
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

        # aggregate() performs only stage one: RFedAGS. The server correction
        # is invoked by base.py afterward through after_aggregate().
        result = aggregator.aggregate(
            global_state=state,
            client_updates=client_updates,
            param_names=param_groups.expert,
            base_state=state,
            strict=True,
        )
        for name in p_names:
            self.assertFalse(
                torch.allclose(
                    result.new_state_dict[name],
                    state[name],
                    atol=1e-7,
                ),
                msg=f"RFedAGS 第一阶段没有应用非零梯度流：{name}",
            )
        self.assertGreater(
            result.diagnostics["mean_aggregated_stream_norm"],
            0.0,
        )

        self.assertEqual(result.diagnostics["aggregation_method"], algorithm.ALGORITHM_NAME)
        self.assertNotIn("mean_diversity_loss", result.diagnostics)
        self.assertNotIn("lambda_div", result.diagnostics)

        model.load_state_dict(result.new_state_dict, strict=True)

        # Independently construct the expected server update using the
        # Stiefel tangent projection. This fails if the implementation
        # accidentally applies the ambient Euclidean gradient directly.
        ordered_p_names = [
            item["p_name"]
            for layer_records in records.values()
            for item in layer_records
        ]
        parameter_map = dict(model.named_parameters())
        p_parameters = [parameter_map[name] for name in ordered_p_names]
        with torch.enable_grad():
            expected_loss, _ = algorithm.compute_subspace_diversity_loss(
                model,
                thresholds=model._rfedags_server_diversity_thresholds,
            )
            euclidean_gradients = torch.autograd.grad(
                expected_loss,
                p_parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
        expected_riemannian_gradients = [
            algorithm.stiefel_tangent_projection(
                parameter.detach(),
                gradient.detach(),
            )
            for parameter, gradient in zip(p_parameters, euclidean_gradients)
        ]
        expected_gradient_norm = math.sqrt(
            sum(
                float(torch.sum(gradient.to(torch.float64).square()).item())
                for gradient in expected_riemannian_gradients
            )
        )
        expected_parameters = {
            name: algorithm.qr_retraction(
                parameter.detach()
                - aggregator.diversity_lr * gradient
            )
            for name, parameter, gradient in zip(
                ordered_p_names,
                p_parameters,
                expected_riemannian_gradients,
            )
        }

        aggregator.after_aggregate(model)
        diagnostics = aggregator.build_post_aggregate_diagnostics(model)

        actual_parameter_map = dict(model.named_parameters())
        for name, expected_parameter in expected_parameters.items():
            self.assertTrue(
                torch.allclose(
                    actual_parameter_map[name],
                    expected_parameter,
                    atol=1e-6,
                    rtol=1e-5,
                ),
                msg=f"服务器端参数 {name} 未按黎曼梯度方向更新。",
            )
        self.assertAlmostEqual(
            diagnostics["server_diversity_gradient_norm"],
            expected_gradient_norm,
            places=6,
        )

        self.assertTrue(diagnostics["server_diversity_applied"])
        self.assertIsNone(diagnostics["server_diversity_skip_reason"])
        self.assertLess(
            diagnostics["server_diversity_threshold_cap"],
            1.0,
        )
        self.assertGreater(
            diagnostics["server_diversity_loss_before"],
            diagnostics["server_diversity_loss_after"],
        )
        self.assertGreater(
            diagnostics["server_diversity_pairs_above_threshold_before"],
            0,
        )
        self.assertEqual(
            len(diagnostics["server_subspace_pair_overlap_before"]),
            2,
        )
        self.assertTrue(
            all(
                len(pair_values) == 6
                for pair_values in diagnostics[
                    "server_subspace_pair_overlap_before"
                ].values()
            )
        )
        self.assertEqual(
            set(diagnostics["server_diversity_thresholds_by_layer"]),
            set(records),
        )
        self.assertGreater(
            diagnostics["server_diversity_parameter_step_norm"],
            0.0,
        )
        self.assertLess(
            diagnostics["max_stiefel_orthogonality_error"],
            1e-4,
        )

        for name, parameter in model.named_parameters():
            if name.endswith(".P"):
                self.assertTrue(
                    torch.allclose(
                        parameter.transpose(0, 1) @ parameter,
                        torch.eye(parameter.shape[1]),
                        atol=1e-5,
                    )
                )


if __name__ == "__main__":
    unittest.main()

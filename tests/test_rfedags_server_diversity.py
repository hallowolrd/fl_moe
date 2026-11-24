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
import rfedags_server_diversity as algorithm  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


class RFedAGSServerDiversityTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(29)
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
                    "server_diversity_lr": 0.5,
                },
            },
        )

    def test_fixed_local_steps_and_full_participation_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "local_steps"):
            self._build_cfg(local_steps=None)

        invalid_cfg = self._build_cfg()
        invalid_cfg.frac = 0.5
        with self.assertRaisesRegex(ValueError, "frac=1.0"):
            algorithm.validate_method_config(invalid_cfg)

        self.assertEqual(int(self.cfg.local_steps), 4)
        self.assertEqual(
            float(self.cfg.method_config.server_diversity_lr),
            0.5,
        )

    def test_model_builder_uses_shared_stiefel_parameterization(self) -> None:
        model = algorithm.build_rfedags_server_diversity_model(self.cfg)
        stiefel_layers = [
            module
            for module in model.modules()
            if isinstance(module, algorithm.subspace.LowRankStiefelLinear)
        ]

        self.assertEqual(len(stiefel_layers), 2 * 4)
        self.assertEqual(
            len(model._rfedags_subspace_layer_records),
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
            algorithm.subspace,
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
        self.assertEqual(payload["mean_diversity_loss"], 0.0)
        self.assertEqual(payload["mean_overlap_by_layer"], {})
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
        records = model._rfedags_subspace_layer_records
        parameter_map = dict(model.named_parameters())
        basis = torch.eye(16)[:, :8]
        rotated = basis.clone()
        angle = 0.1
        rotated[:, 0] = (
            torch.cos(torch.tensor(angle)) * basis[:, 0]
            + torch.sin(torch.tensor(angle)) * torch.eye(16)[:, 8]
        )

        # Make each layer's experts highly overlapping while keeping every P
        # on the Stiefel manifold. The thresholds remain those calibrated at
        # model initialization.
        with torch.no_grad():
            for layer_records in records.values():
                for expert_index, item in enumerate(layer_records):
                    parameter = parameter_map[item["p_name"]]
                    parameter.copy_(basis if expert_index < 3 else rotated)

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
            model_builder=algorithm.build_rfedags_server_diversity_model,
            global_model=model,
            param_groups=param_groups,
            optimization_groups=optimization_groups,
        )
        aggregator = algorithm.RFedAGSServerDiversityExpertAggregator(self.cfg)
        aggregator.set_method_context(method_context)
        aggregator.initialize_model(model)

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
                    "expert_buffers": {
                        name: state[name].clone()
                        for name in expert_buffer_names
                    },
                    "mean_diversity_loss": 0.0,
                    "mean_overlap_by_layer": {},
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
            self.assertTrue(
                torch.allclose(
                    result.new_state_dict[name],
                    state[name],
                    atol=1e-5,
                )
            )

        self.assertNotIn("mean_diversity_loss", result.diagnostics)
        self.assertNotIn("lambda_div", result.diagnostics)

        model.load_state_dict(result.new_state_dict, strict=True)
        aggregator.after_aggregate(model)
        diagnostics = aggregator.build_post_aggregate_diagnostics(model)

        self.assertTrue(diagnostics["server_diversity_applied"])
        self.assertGreater(
            diagnostics["server_diversity_loss_before"],
            diagnostics["server_diversity_loss_after"],
        )
        self.assertGreater(
            diagnostics["server_diversity_pairs_above_threshold_before"],
            0,
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

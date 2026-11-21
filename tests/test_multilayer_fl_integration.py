from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONHASHSEED", "0")

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = ROOT / "algorithms"
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import base  # noqa: E402
import fedavg_orthogonal  # noqa: E402
import rfedags_orthogonal  # noqa: E402


MOE_LAYER_NAMES = ("blocks.0.moe", "blocks.1.moe")
STIEFEL_BLOCK_NAMES = ("block_0_fc1", "block_1_fc1")


class TinyExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Two experts per layer give a 4x4 stacked Stiefel coordinate:
        # cat([W_0, W_1], dim=0).T, with n=p=4.
        self.fc1 = nn.Linear(4, 2)
        self.fc2 = nn.Linear(2, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(inputs)))


class TinyMoE(nn.Module):
    def __init__(self, num_experts: int = 2) -> None:
        super().__init__()
        self.router = nn.Linear(4, num_experts)
        self.experts = nn.ModuleList(
            [TinyExpert() for _ in range(num_experts)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        routing_weights = torch.softmax(self.router(inputs), dim=-1)
        expert_outputs = torch.stack(
            [expert(inputs) for expert in self.experts],
            dim=1,
        )
        return torch.sum(
            expert_outputs * routing_weights.unsqueeze(-1),
            dim=1,
        )


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moe = TinyMoE(num_experts=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.moe(inputs)


class TinyTwoLayerMoEClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])
        self.classifier = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> base.ModelOutput:
        features = torch.tanh(self.backbone(inputs))
        for block in self.blocks:
            features = block(features)
        return base.ModelOutput(logits=self.classifier(features))


def make_cfg() -> dict[str, Any]:
    return {
        "num_clients": 2,
        "num_experts": 2,
        "frac": 1.0,
        "local_epochs": 1,
        "local_steps": 1,
        "optimizer": {
            "type": "sgd",
            "lr": 1e-3,
            "momentum": 0.0,
            "nesterov": False,
            "weight_decay": 0.0,
        },
        "orthogonal": {
            "stiefel_blocks": {
                "block_0_fc1": {
                    "parameter_patterns": [
                        "blocks.0.moe.experts.*.fc1.weight"
                    ]
                },
                "block_1_fc1": {
                    "parameter_patterns": [
                        "blocks.1.moe.experts.*.fc1.weight"
                    ]
                },
            }
        },
    }


def make_loader(seed: int, num_samples: int = 8) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(num_samples, 4, generator=generator)
    labels = torch.randint(0, 2, (num_samples,), generator=generator)
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=4,
        shuffle=False,
    )


class MultiLayerClientServerIntegrationTests(unittest.TestCase):
    def _run_one_round(
        self,
        algorithm_module: Any,
        *,
        expects_payload: bool,
    ) -> None:
        torch.manual_seed(2026)
        cfg = make_cfg()
        global_model = TinyTwoLayerMoEClassifier()
        client_loaders = [make_loader(11), make_loader(22)]
        test_loader = make_loader(33, num_samples=4)

        method_context = base.MethodContext(
            cfg=cfg,
            device=torch.device("cpu"),
            dataset_name="tiny_multilayer_moe",
        )
        server = base.FLServer(
            cfg=cfg,
            client_loaders=client_loaders,
            test_loader=test_loader,
            device="cpu",
            expert_aggregator_builder=algorithm_module.build_expert_aggregator,
            method_context=method_context,
            global_model=global_model,
            expert_client_training_strategy_builder=(
                algorithm_module.build_expert_client_training_strategy
            ),
        )

        self.assertEqual(
            set(server.param_groups.expert_by_moe_layer),
            set(MOE_LAYER_NAMES),
        )
        self.assertEqual(
            set(server.optimization_groups.expert_parameters_by_moe_layer),
            set(MOE_LAYER_NAMES),
        )

        # FLServer injects the same hierarchical grouping into the expert
        # aggregation plugin through MethodContext.
        plugin_groups = server.aggregators.expert.method_context.optimization_groups
        self.assertIsNotNone(plugin_groups)
        self.assertEqual(
            set(plugin_groups.expert_parameters_by_moe_layer),
            set(MOE_LAYER_NAMES),
        )
        for layer_name in MOE_LAYER_NAMES:
            self.assertEqual(
                set(plugin_groups.expert_parameters_by_layer_and_id[layer_name]),
                {0, 1},
            )

        updates = base.train_selected_clients(
            clients=server.clients,
            global_model=server.global_model,
            round_id=1,
        )
        self.assertEqual(len(updates), 2)

        expected_non_expert_names = set(server.param_groups.non_expert)
        expected_expert_names = set(server.param_groups.expert)
        expected_all_names = set(server.param_groups.all)

        for update in updates:
            if expects_payload:
                # RFedAGS sends non-expert deltas through model_delta and all
                # expert updates through its algorithm-specific payload.
                self.assertEqual(
                    set(update.model_delta),
                    expected_non_expert_names,
                )
                payload = update.algorithm_payload["rfedags"]
                self.assertEqual(
                    set(payload["stiefel_stream"]),
                    set(STIEFEL_BLOCK_NAMES),
                )
                self.assertTrue(payload["euclidean_stream"])
                self.assertEqual(
                    set(payload["euclidean_stream"])
                    | {
                        name
                        for layer_name in MOE_LAYER_NAMES
                        for names in (
                            plugin_groups.expert_parameters_by_layer_and_id[
                                layer_name
                            ].values()
                        )
                        for name in names
                        if name.endswith("fc1.weight")
                    },
                    expected_expert_names,
                )
            else:
                # FedAvg-Orthogonal uploads the complete model delta,
                # including experts from both MoE layers.
                self.assertEqual(set(update.model_delta), expected_all_names)
                self.assertEqual(update.algorithm_payload, {})

        aggregation_info = server.aggregate_client_updates(updates)
        expert_summary = aggregation_info["expert"]
        diagnostics = expert_summary["diagnostics"]

        self.assertEqual(
            set(diagnostics["orthogonality"]["blocks"]),
            set(STIEFEL_BLOCK_NAMES),
        )
        for block_info in diagnostics["orthogonality"]["blocks"].values():
            self.assertLess(block_info["orthogonality_error"], 1e-4)

        if expects_payload:
            tangent_blocks = diagnostics["manifold_diagnostics"][
                "stiefel_tangent"
            ]["blocks"]
            self.assertEqual(set(tangent_blocks), set(STIEFEL_BLOCK_NAMES))

        # The aggregation round must leave a finite, correctly structured
        # global state after both expert layers have been updated.
        global_state = server.global_model.state_dict()
        self.assertEqual(set(global_state), expected_all_names)
        for name in expected_expert_names:
            self.assertTrue(bool(torch.isfinite(global_state[name]).all()))

    def test_fedavg_orthogonal_client_server_support_two_moe_layers(self) -> None:
        self._run_one_round(
            fedavg_orthogonal,
            expects_payload=False,
        )

    def test_rfedags_client_server_support_two_moe_layers(self) -> None:
        self._run_one_round(
            rfedags_orthogonal,
            expects_payload=True,
        )


if __name__ == "__main__":
    unittest.main()

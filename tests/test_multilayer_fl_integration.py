from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONHASHSEED", "0")

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = ROOT / "algorithms"
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import base  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
import fedavg_orthogonal  # noqa: E402
import rfedags_orthogonal  # noqa: E402


MOE_LAYER_NAMES = ("blocks.0.moe", "blocks.1.moe")
STIEFEL_BLOCK_NAMES = ("block_0_fc1", "block_1_fc1")
SWITCH_MOE_LAYER_NAMES = ("blocks.0.mlp", "blocks.1.mlp")
SWITCH_STIEFEL_BLOCK_NAMES = ("switch_block_0_fc1", "switch_block_1_fc1")


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


class TinySwitchMoE(nn.Module):
    """A compact Switch-style MoE whose experts use the ModuleDict naming layout."""

    def __init__(self, num_experts: int = 2) -> None:
        super().__init__()
        self.router = nn.Linear(4, num_experts)
        self.experts = nn.ModuleDict({
            f"expert_{expert_id}": TinyExpert()
            for expert_id in range(num_experts)
        })

    def forward(self, inputs: torch.Tensor):
        router_probabilities = torch.softmax(self.router(inputs), dim=-1)
        expert_outputs = torch.stack(
            [expert(inputs) for expert in self.experts.values()],
            dim=1,
        )
        outputs = torch.sum(
            expert_outputs * router_probabilities.unsqueeze(-1),
            dim=1,
        )
        selected_experts = router_probabilities.argmax(dim=-1)
        tokens_per_expert = torch.bincount(
            selected_experts,
            minlength=len(self.experts),
        )
        router_prob_sum = router_probabilities.sum(dim=0)
        batch_size = int(inputs.shape[0])
        token_fraction = tokens_per_expert.to(router_probabilities.dtype) / batch_size
        probability_fraction = router_prob_sum / batch_size
        auxiliary_loss = len(self.experts) * torch.sum(
            token_fraction * probability_fraction
        )
        stats = {
            "tokens_per_expert": tokens_per_expert,
            "router_prob_sum": router_prob_sum,
            "num_tokens": batch_size,
        }
        return outputs, auxiliary_loss, stats


class TinySwitchBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = TinySwitchMoE(num_experts=2)

    def forward(self, inputs: torch.Tensor):
        return self.mlp(inputs)


class TinySwitchTwoLayerMoEClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([TinySwitchBlock(), TinySwitchBlock()])
        self.classifier = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> base.ModelOutput:
        features = torch.tanh(self.backbone(inputs))
        aux_loss_by_layer = {}
        moe_stats = {}
        for layer_index, block in enumerate(self.blocks):
            layer_output, layer_aux_loss, layer_stats = block(features)
            features = features + layer_output
            layer_name = f"blocks.{layer_index}.mlp"
            aux_loss_by_layer[layer_name] = layer_aux_loss
            moe_stats[layer_name] = layer_stats
        return base.ModelOutput(
            logits=self.classifier(features),
            aux_loss=sum(aux_loss_by_layer.values()),
            aux_loss_by_layer=aux_loss_by_layer,
            moe_stats=moe_stats,
        )


def make_switch_cfg() -> dict[str, Any]:
    cfg = make_cfg()
    cfg["orthogonal"]["stiefel_blocks"] = {
        f"switch_block_{layer_index}_fc1": {
            "parameter_patterns": [
                f"blocks.{layer_index}.mlp.experts.expert_*.fc1.weight"
            ]
        }
        for layer_index in range(2)
    }
    return cfg


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
        cfg_override: dict[str, Any] | None = None,
        global_model_override: nn.Module | None = None,
        expected_moe_layer_names: tuple[str, ...] = MOE_LAYER_NAMES,
        expected_stiefel_block_names: tuple[str, ...] = STIEFEL_BLOCK_NAMES,
        expects_runtime_stats: bool = False,
    ) -> None:
        torch.manual_seed(2026)
        cfg = make_cfg() if cfg_override is None else cfg_override
        global_model = (
            TinyTwoLayerMoEClassifier()
            if global_model_override is None
            else global_model_override
        )
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
            set(expected_moe_layer_names),
        )
        self.assertEqual(
            set(server.optimization_groups.expert_parameters_by_moe_layer),
            set(expected_moe_layer_names),
        )

        # FLServer injects the same hierarchical grouping into the expert
        # aggregation plugin through MethodContext.
        plugin_groups = server.aggregators.expert.method_context.optimization_groups
        self.assertIsNotNone(plugin_groups)
        self.assertEqual(
            set(plugin_groups.expert_parameters_by_moe_layer),
            set(expected_moe_layer_names),
        )
        for layer_name in expected_moe_layer_names:
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
        if expects_runtime_stats:
            for update in updates:
                self.assertEqual(
                    set(update.avg_aux_loss_by_layer),
                    set(expected_moe_layer_names),
                )
                self.assertEqual(
                    set(update.moe_stats),
                    set(expected_moe_layer_names),
                )

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
                    set(expected_stiefel_block_names),
                )
                self.assertTrue(payload["euclidean_stream"])
                self.assertEqual(
                    set(payload["euclidean_stream"])
                    | {
                        name
                        for layer_name in expected_moe_layer_names
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
            set(expected_stiefel_block_names),
        )
        for block_info in diagnostics["orthogonality"]["blocks"].values():
            self.assertLess(block_info["orthogonality_error"], 1e-4)

        if expects_payload:
            tangent_blocks = diagnostics["manifold_diagnostics"][
                "stiefel_tangent"
            ]["blocks"]
            self.assertEqual(set(tangent_blocks), set(expected_stiefel_block_names))

        round_result = server.build_round_result(
            round_id=1,
            selected_clients=server.clients,
            client_updates=updates,
            eval_result=base.EvalResult(
                loss=0.5,
                acc=50.0,
                correct=2,
                total=4,
                extra={},
            ),
            aggregation_info=aggregation_info,
        )
        if expects_runtime_stats:
            self.assertEqual(
                set(round_result.avg_train_aux_loss_by_layer),
                set(expected_moe_layer_names),
            )
            self.assertEqual(
                set(round_result.moe_stats),
                set(expected_moe_layer_names),
            )
            for layer_name in expected_moe_layer_names:
                self.assertEqual(round_result.moe_stats[layer_name]["num_tokens"], 8)

            log_messages = []
            server._write_log_only = log_messages.append
            server.print_file_round_detail(round_result)
            joined_log = "\n".join(log_messages)
            self.assertIn("[AuxLossByLayer]", joined_log)
            self.assertIn("[MoEStats]", joined_log)
            self.assertIn("tokens_per_expert", joined_log)

            serialized = base.make_json_safe(round_result.to_dict())
            self.assertEqual(
                serialized["moe_stats"][expected_moe_layer_names[0]]["num_tokens"],
                8,
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                csv_path = Path(temp_dir) / "results.csv"
                base.save_round_results_csv([round_result], csv_path)
                with csv_path.open("r", encoding="utf-8", newline="") as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(
                    set(json.loads(row["avg_train_aux_loss_by_layer_json"])),
                    set(expected_moe_layer_names),
                )
                self.assertEqual(
                    json.loads(row["moe_stats_json"])[expected_moe_layer_names[0]]["num_tokens"],
                    8,
                )

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

    def test_fedavg_orthogonal_client_server_support_switch_module_dict(self) -> None:
        self._run_one_round(
            fedavg_orthogonal,
            expects_payload=False,
            cfg_override=make_switch_cfg(),
            global_model_override=TinySwitchTwoLayerMoEClassifier(),
            expected_moe_layer_names=SWITCH_MOE_LAYER_NAMES,
            expected_stiefel_block_names=SWITCH_STIEFEL_BLOCK_NAMES,
            expects_runtime_stats=True,
        )

    def test_rfedags_client_server_support_switch_module_dict(self) -> None:
        self._run_one_round(
            rfedags_orthogonal,
            expects_payload=True,
            cfg_override=make_switch_cfg(),
            global_model_override=TinySwitchTwoLayerMoEClassifier(),
            expected_moe_layer_names=SWITCH_MOE_LAYER_NAMES,
            expected_stiefel_block_names=SWITCH_STIEFEL_BLOCK_NAMES,
            expects_runtime_stats=True,
        )

    def test_build_server_uses_injected_model_builder(self) -> None:
        cfg = make_cfg()
        built_configs = []

        def model_builder(received_cfg):
            built_configs.append(received_cfg)
            return TinyTwoLayerMoEClassifier()

        method_context = base.MethodContext(
            cfg=cfg,
            device=torch.device("cpu"),
            dataset_name="tiny_multilayer_moe",
            model_builder=model_builder,
        )
        server = base.build_server(
            cfg=cfg,
            client_loaders=[make_loader(11), make_loader(22)],
            test_loader=make_loader(33, num_samples=4),
            device="cpu",
            expert_aggregator_builder=fedavg_orthogonal.build_expert_aggregator,
            method_context=method_context,
            model_builder=model_builder,
        )

        self.assertEqual(built_configs, [cfg])
        self.assertIsInstance(server.global_model, TinyTwoLayerMoEClassifier)
        self.assertIs(server.method_context.model_builder, model_builder)


if __name__ == "__main__":
    unittest.main()

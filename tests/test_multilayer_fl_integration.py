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
import uniform  # noqa: E402
import rfedags  # noqa: E402


MOE_LAYER_NAMES = ("blocks.0.moe", "blocks.1.moe")
SWITCH_MOE_LAYER_NAMES = ("blocks.0.mlp", "blocks.1.mlp")


class TinyExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Keep expert dimensions small for fast client/server integration tests.
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
    return make_cfg()


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
        expects_runtime_stats: bool = False,
    ) -> dict[str, torch.Tensor]:
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
            expert_client_training_strategy_builder=getattr(
                algorithm_module,
                "build_expert_client_training_strategy",
                None,
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
        expected_expert_parameter_names = set(
            server.optimization_groups.expert_parameter_names
        )
        expected_expert_buffer_names = expected_expert_names - set(
            dict(server.global_model.named_parameters())
        )
        expected_all_names = set(server.param_groups.all)

        for update in updates:
            if expects_payload:
                self.assertEqual(set(update.model_delta), expected_non_expert_names)
                payload = update.algorithm_payload["rfedags"]
                self.assertEqual(payload["version"], rfedags.PAYLOAD_VERSION)
                self.assertEqual(set(payload["expert_stream"]), expected_expert_parameter_names)
                self.assertEqual(set(payload["expert_buffers"]), expected_expert_buffer_names)
            else:
                self.assertEqual(set(update.model_delta), expected_all_names)
                self.assertEqual(update.algorithm_payload, {})

        aggregation_info = server.aggregate_client_updates(updates)
        diagnostics = aggregation_info["expert"]["diagnostics"]
        self.assertNotIn("manifold_diagnostics", diagnostics)
        if expects_payload:
            self.assertEqual(diagnostics["update_space"], "euclidean")
            self.assertEqual(diagnostics["local_steps"], 1)

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
        return {
            name: value.detach().clone()
            for name, value in global_state.items()
        }


    def test_after_aggregate_rejects_shape_dtype_and_non_finite_state(self) -> None:
        class MutatingAfterAggregateAggregator(base.Aggregator):
            def __init__(self, cfg: Any, mutation: str) -> None:
                super().__init__(cfg=cfg, param_group_name="expert")
                self.mutation = mutation

            @property
            def method_name(self) -> str:
                return "test_mutating_after_aggregate"

            def compute_weights(self, client_updates):
                return base.build_uniform_weights(client_updates)

            def after_aggregate(self, model: nn.Module) -> None:
                parameter = model.classifier.weight
                if self.mutation == "shape":
                    parameter.data = parameter.data[:1].clone()
                elif self.mutation == "dtype":
                    parameter.data = parameter.data.to(dtype=torch.float64)
                elif self.mutation == "nan":
                    parameter.data.fill_(float("nan"))
                elif self.mutation == "inf":
                    parameter.data.fill_(float("inf"))
                else:
                    raise AssertionError(f"Unknown mutation: {self.mutation}")

        cases = (
            ("shape", ValueError, "聚合后处理结果.*Tensor Shape 不匹配"),
            ("dtype", TypeError, "聚合后处理结果.*Tensor Dtype 不匹配"),
            ("nan", ValueError, "classifier.weight 中存在 NaN 或 Inf"),
            ("inf", ValueError, "classifier.weight 中存在 NaN 或 Inf"),
        )

        for mutation, exception_type, message_pattern in cases:
            with self.subTest(mutation=mutation):
                cfg = make_cfg()
                model = TinyTwoLayerMoEClassifier()
                server = base.FLServer(
                    cfg=cfg,
                    client_loaders=[make_loader(11), make_loader(22)],
                    test_loader=make_loader(33, num_samples=4),
                    device="cpu",
                    expert_aggregator_builder=(
                        lambda received_cfg, selected_mutation=mutation:
                        MutatingAfterAggregateAggregator(
                            received_cfg,
                            selected_mutation,
                        )
                    ),
                    global_model=model,
                )

                model_state = server.global_model.state_dict()
                model_delta = {
                    name: torch.zeros_like(tensor)
                    for name, tensor in model_state.items()
                    if torch.is_floating_point(tensor)
                }
                update = base.ClientUpdate(
                    client_id=0,
                    round_id=1,
                    num_samples=4,
                    model_delta=model_delta,
                    metrics={"num_batches": 1},
                )

                with self.assertRaisesRegex(exception_type, message_pattern):
                    server.aggregate_client_updates([update])

    def test_server_aggregators_receive_independent_state_snapshots(self) -> None:
        captured: dict[str, Any] = {}

        class MutatingExpertAggregator(base.Aggregator):
            @property
            def method_name(self) -> str:
                return "test_snapshot_isolation"

            def compute_weights(self, client_updates):
                return base.build_uniform_weights(client_updates)

            def aggregate(
                self,
                global_state,
                client_updates,
                param_names=None,
                base_state=None,
                strict=True,
            ):
                if base_state is None:
                    raise AssertionError("Expert aggregation must receive base_state.")

                captured["expert_global_ref"] = global_state
                captured["expert_base_ref"] = base_state
                captured["expert_global_before"] = base.clone_state_dict(global_state)
                captured["expert_base_before"] = base.clone_state_dict(base_state)
                result_state = base.clone_state_dict(base_state)

                # Deliberately mutate both inputs to emulate an unsafe custom plugin.
                for tensor in global_state.values():
                    tensor.add_(100)
                for tensor in base_state.values():
                    tensor.add_(200)

                return base.AggregationResult(
                    new_state_dict=result_state,
                    weights={0: 1.0},
                )

        cfg = make_cfg()
        server = base.FLServer(
            cfg=cfg,
            client_loaders=[make_loader(11), make_loader(22)],
            test_loader=make_loader(33, num_samples=4),
            device="cpu",
            expert_aggregator_builder=lambda received_cfg: MutatingExpertAggregator(
                cfg=received_cfg,
                param_group_name="expert",
            ),
            global_model=TinyTwoLayerMoEClassifier(),
        )
        initial_state = base.clone_state_dict(server.global_model.state_dict())

        def destructive_non_expert_aggregate(
            *,
            global_state,
            client_updates,
            param_names,
            base_state,
            strict,
        ):
            captured["non_expert_global_ref"] = global_state
            result_state = base.clone_state_dict(global_state)
            result = base.AggregationResult(
                new_state_dict=result_state,
                weights={0: 1.0},
            )
            captured["non_expert_result"] = result
            captured["non_expert_result_before_expert"] = base.clone_state_dict(
                result.new_state_dict
            )

            # The non-expert plugin mutates its received global-state argument.
            for tensor in global_state.values():
                tensor.add_(10)

            return result

        # Replace the fixed aggregator method only for this adversarial isolation test.
        server.aggregators.non_expert.aggregate = destructive_non_expert_aggregate

        update = base.ClientUpdate(
            client_id=0,
            round_id=1,
            num_samples=4,
            model_delta={},
            metrics={"num_batches": 1},
        )
        server.aggregate_client_updates([update])

        non_expert_input = captured["non_expert_global_ref"]
        expert_input = captured["expert_global_ref"]
        expert_base_input = captured["expert_base_ref"]
        non_expert_result = captured["non_expert_result"]
        non_expert_result_before = captured["non_expert_result_before_expert"]

        for name, original_tensor in initial_state.items():
            with self.subTest(parameter=name):
                torch.testing.assert_close(
                    captured["expert_global_before"][name],
                    original_tensor,
                )
                torch.testing.assert_close(
                    captured["expert_base_before"][name],
                    non_expert_result_before[name],
                )
                torch.testing.assert_close(
                    non_expert_result.new_state_dict[name],
                    non_expert_result_before[name],
                )
                torch.testing.assert_close(
                    server.global_model.state_dict()[name],
                    original_tensor,
                )

                self.assertNotEqual(
                    non_expert_input[name].data_ptr(),
                    expert_input[name].data_ptr(),
                )
                self.assertNotEqual(
                    non_expert_result.new_state_dict[name].data_ptr(),
                    expert_base_input[name].data_ptr(),
                )
                self.assertTrue(
                    torch.equal(
                        non_expert_input[name],
                        original_tensor + 10,
                    )
                )


    def test_uniform_client_server_support_two_moe_layers(self) -> None:
        self._run_one_round(
            uniform,
            expects_payload=False,
        )

    def test_rfedags_client_server_support_two_moe_layers(self) -> None:
        self._run_one_round(
            rfedags,
            expects_payload=True,
        )

    def test_uniform_client_server_support_switch_module_dict(self) -> None:
        self._run_one_round(
            uniform,
            expects_payload=False,
            cfg_override=make_switch_cfg(),
            global_model_override=TinySwitchTwoLayerMoEClassifier(),
            expected_moe_layer_names=SWITCH_MOE_LAYER_NAMES,
            expects_runtime_stats=True,
        )

    def test_rfedags_client_server_support_switch_module_dict(self) -> None:
        self._run_one_round(
            rfedags,
            expects_payload=True,
            cfg_override=make_switch_cfg(),
            global_model_override=TinySwitchTwoLayerMoEClassifier(),
            expected_moe_layer_names=SWITCH_MOE_LAYER_NAMES,
            expects_runtime_stats=True,
        )

    def test_euclidean_rfedags_matches_uniform_fedavg(self) -> None:
        uniform_state = self._run_one_round(uniform, expects_payload=False)
        rfedags_state = self._run_one_round(rfedags, expects_payload=True)
        self.assertEqual(set(uniform_state), set(rfedags_state))
        for name in uniform_state:
            torch.testing.assert_close(
                rfedags_state[name], uniform_state[name], rtol=1e-5, atol=1e-7
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
            expert_aggregator_builder=uniform.build_expert_aggregator,
            method_context=method_context,
            model_builder=model_builder,
        )

        self.assertEqual(built_configs, [cfg])
        self.assertIsInstance(server.global_model, TinyTwoLayerMoEClassifier)
        self.assertIs(server.method_context.model_builder, model_builder)


if __name__ == "__main__":
    unittest.main()

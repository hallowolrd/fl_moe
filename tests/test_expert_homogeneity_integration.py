from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = ROOT / "algorithms"
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import base
from tools import diagnose_expert_homogeneity as diagnostics


class SwitchViTExpertDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(101)
        self.model = base.SwitchViT(
            num_classes=4,
            input_shape=(3, 8, 8),
            transformer_layers=2,
            hidden_dim=8,
            attention_heads=2,
            patch_size=4,
            expert_hidden_dim=16,
            num_experts=3,
            moe_block_indices=(1, 2),
            dropout=0.25,
            router_statistics=False,
        )
        images = torch.randn(4, 3, 8, 8)
        labels = torch.zeros(4, dtype=torch.long)
        self.loader = DataLoader(
            TensorDataset(images, labels),
            batch_size=2,
            shuffle=False,
        )

    def test_discovers_each_switch_vit_moe_layer(self) -> None:
        layers = diagnostics._discover_moe_layers(self.model)
        self.assertEqual(
            set(layers),
            {
                "blocks.0.mlp",
                "blocks.1.mlp",
            },
        )
        for experts in layers.values():
            self.assertEqual(
                [name for name, _ in experts],
                ["expert_0", "expert_1", "expert_2"],
            )

    def test_collects_all_experts_on_identical_layer_inputs(self) -> None:
        captured_inputs: dict[str, list[torch.Tensor]] = {}
        handles = []

        for layer_name, expert_items in diagnostics._discover_moe_layers(
            self.model
        ).items():
            for expert_name, expert in expert_items:
                key = f"{layer_name}.{expert_name}"
                captured_inputs[key] = []

                def capture_fc1_input(
                    _module: torch.nn.Module,
                    inputs: tuple[torch.Tensor, ...],
                    *,
                    capture_key: str = key,
                ) -> None:
                    captured_inputs[capture_key].append(
                        inputs[0].detach().cpu().clone()
                    )

                handles.append(
                    expert.fc1.register_forward_pre_hook(capture_fc1_input)
                )

        try:
            report = diagnostics.diagnose_model_experts(
                model=self.model,
                data_loader=self.loader,
                device="cpu",
                max_batches=2,
                max_tokens_per_layer=13,
                subspace_k=2,
                expert_batch_size=32,
            )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(report["num_batches"], 2)
        self.assertEqual(report["num_images"], 4)
        self.assertEqual(set(report["layers"]), {
            "blocks.0.mlp",
            "blocks.1.mlp",
        })

        for layer_name in report["layers"]:
            expert_inputs = []
            for expert_id in range(3):
                key = f"{layer_name}.expert_{expert_id}"
                self.assertTrue(captured_inputs[key])
                # 最后一次 fc1 调用来自诊断阶段的完整专家独立前向，
                # 正常 SwitchMoE 稀疏路由产生的调用发生在它之前。
                expert_inputs.append(captured_inputs[key][-1])

            self.assertEqual(
                [tuple(value.shape) for value in expert_inputs],
                [(13, 8), (13, 8), (13, 8)],
            )
            self.assertTrue(torch.equal(expert_inputs[0], expert_inputs[1]))
            self.assertTrue(torch.equal(expert_inputs[0], expert_inputs[2]))

    def test_report_contains_all_pairwise_metrics(self) -> None:
        report = diagnostics.diagnose_model_experts(
            model=self.model,
            data_loader=self.loader,
            device="cpu",
            max_batches=1,
            max_tokens_per_layer=10,
            subspace_k=2,
            expert_batch_size=16,
        )

        expected_metrics = {
            "fc1_weight_cosine",
            "fc2_weight_cosine",
            "fc1_right_subspace_overlap",
            "fc2_left_subspace_overlap",
            "hidden_cka",
            "output_cka",
            "output_normalized_mse",
            "errors",
        }
        for layer_report in report["layers"].values():
            self.assertEqual(layer_report["num_experts"], 3)
            self.assertEqual(layer_report["num_tokens_used"], 10)
            self.assertFalse(
                layer_report["router_gate_applied_to_expert_outputs"]
            )
            self.assertEqual(len(layer_report["pairwise"]), 3)
            for pair_report in layer_report["pairwise"].values():
                self.assertEqual(set(pair_report), expected_metrics)
                for metric_name in expected_metrics - {"errors"}:
                    value = pair_report[metric_name]
                    if value is not None:
                        self.assertTrue(torch.isfinite(torch.tensor(value)))
                for metric_name in (
                    "fc1_right_subspace_overlap",
                    "fc2_left_subspace_overlap",
                    "hidden_cka",
                    "output_cka",
                ):
                    value = pair_report[metric_name]
                    if value is not None:
                        self.assertGreaterEqual(value, 0.0)
                        self.assertLessEqual(value, 1.0)

    def test_diagnostic_preserves_model_state_and_removes_hooks(self) -> None:
        self.model.train()
        # 保留一个子模块与父模型不同的 training 状态，检查精确恢复。
        self.model.blocks[0].mlp.experts["expert_0"].dropout.eval()

        training_states = {
            module: module.training
            for module in self.model.modules()
        }
        parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }
        moe_hook_counts = {
            name: len(module._forward_pre_hooks)
            for name, module in self.model.named_modules()
            if isinstance(module, base.SwitchMoE)
        }

        diagnostics.diagnose_model_experts(
            model=self.model,
            data_loader=self.loader,
            device="cpu",
            max_batches=1,
            max_tokens_per_layer=10,
            subspace_k=2,
            expert_batch_size=16,
        )

        for module, was_training in training_states.items():
            self.assertEqual(module.training, was_training)
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, parameters_before[name]))
        for name, module in self.model.named_modules():
            if isinstance(module, base.SwitchMoE):
                self.assertEqual(
                    len(module._forward_pre_hooks),
                    moe_hook_counts[name],
                )

    def test_empty_loader_raises_and_restores_model_state(self) -> None:
        self.model.train()
        training_states = {
            module: module.training
            for module in self.model.modules()
        }
        hook_counts = {
            name: len(module._forward_pre_hooks)
            for name, module in self.model.named_modules()
            if isinstance(module, base.SwitchMoE)
        }

        with self.assertRaisesRegex(ValueError, "没有提供任何 batch"):
            diagnostics.diagnose_model_experts(
                model=self.model,
                data_loader=[],
                device="cpu",
                max_batches=1,
                subspace_k=2,
            )

        for module, was_training in training_states.items():
            self.assertEqual(module.training, was_training)
        for name, module in self.model.named_modules():
            if isinstance(module, base.SwitchMoE):
                self.assertEqual(
                    len(module._forward_pre_hooks),
                    hook_counts[name],
                )

    def test_invalid_limits_are_rejected(self) -> None:
        for kwargs in (
            {"max_batches": 0},
            {"max_tokens_per_layer": 0},
            {"expert_batch_size": 0},
            {"subspace_k": 0},
            {"subspace_k": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    diagnostics.diagnose_model_experts(
                        model=self.model,
                        data_loader=self.loader,
                        **kwargs,
                    )

    def test_model_without_moe_layers_is_rejected(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with self.assertRaisesRegex(ValueError, "没有发现"):
            diagnostics.diagnose_model_experts(
                model=model,
                data_loader=self.loader,
            )


if __name__ == "__main__":
    unittest.main()

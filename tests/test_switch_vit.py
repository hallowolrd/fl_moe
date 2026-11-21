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


class SwitchViTTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.cfg = {
            "num_classes": 10,
            "num_experts": 4,
            "input_shape": (3, 32, 32),
            "model_cfg": {},
        }

    def test_default_architecture_and_parameter_count(self) -> None:
        model = base.build_switch_vit_model(self.cfg)

        self.assertIsInstance(model, base.SwitchViT)
        self.assertEqual(len(model.blocks), 6)
        self.assertEqual(model.moe_block_indices, (2, 4, 6))
        self.assertEqual(
            [index + 1 for index, block in enumerate(model.blocks) if block.use_moe],
            [2, 4, 6],
        )
        self.assertEqual(
            [index + 1 for index, block in enumerate(model.blocks) if not block.use_moe],
            [1, 3, 5],
        )

        for block_index in (1, 3, 5):
            moe = model.blocks[block_index].mlp
            self.assertIsInstance(moe, base.SwitchMoE)
            self.assertEqual(
                list(moe.experts.keys()),
                ["expert_0", "expert_1", "expert_2", "expert_3"],
            )

        self.assertEqual(model.pos_embed.shape[1], 65)
        parameter_count = base.count_parameters(model)
        self.assertEqual(parameter_count, 9_504_278)

    def test_forward_top1_dispatch_and_router_statistics(self) -> None:
        model = base.build_switch_vit_model(self.cfg)
        images = torch.randn(2, 3, 32, 32)

        output = model(images)

        self.assertIsInstance(output, base.ModelOutput)
        self.assertEqual(tuple(output.logits.shape), (2, 10))
        self.assertIsNone(output.aux_loss)
        self.assertEqual(output.aux_loss_by_layer, {})
        self.assertEqual(
            set(output.moe_stats.keys()),
            {
                "blocks.1.mlp",
                "blocks.3.mlp",
                "blocks.5.mlp",
            },
        )

        expected_tokens = 2 * 65
        for layer_stats in output.moe_stats.values():
            self.assertEqual(layer_stats["num_tokens"], expected_tokens)
            self.assertEqual(
                int(layer_stats["tokens_per_expert"].sum().item()),
                expected_tokens,
            )
            self.assertEqual(
                tuple(layer_stats["tokens_per_expert"].shape),
                (4,),
            )
            self.assertEqual(
                tuple(layer_stats["router_prob_sum"].shape),
                (4,),
            )

    def test_task_loss_backpropagates_through_router_and_experts(self) -> None:
        model = base.build_switch_vit_model(self.cfg)
        images = torch.randn(2, 3, 32, 32)
        targets = torch.tensor([1, 4])

        output = model(images)
        loss = torch.nn.functional.cross_entropy(output.logits, targets)
        loss.backward()

        moe = model.blocks[1].mlp
        self.assertIsNotNone(moe.router.projection.weight.grad)
        self.assertGreater(
            float(moe.router.projection.weight.grad.abs().sum().item()),
            0.0,
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in moe.experts["expert_0"].parameters()
            )
            or any(
                parameter.grad is not None
                for parameter in moe.experts["expert_1"].parameters()
            )
            or any(
                parameter.grad is not None
                for parameter in moe.experts["expert_2"].parameters()
            )
            or any(
                parameter.grad is not None
                for parameter in moe.experts["expert_3"].parameters()
            )
        )

    def test_module_dict_experts_are_compatible_with_federated_groups(self) -> None:
        model = base.build_switch_vit_model(self.cfg)
        groups = base.build_param_groups(
            model=model,
            expected_num_experts=4,
            strict=True,
        )

        self.assertEqual(set(groups.expert_by_moe_layer), {
            "blocks.1.mlp",
            "blocks.3.mlp",
            "blocks.5.mlp",
        })
        self.assertEqual(len(groups.expert_by_layer_and_id), 3)
        for layer_name, experts in groups.expert_by_layer_and_id.items():
            self.assertEqual(set(experts), {0, 1, 2, 3})
            self.assertTrue(all(experts[expert_id] for expert_id in range(4)))

    def test_auxiliary_loss_cannot_be_enabled_yet(self) -> None:
        cfg = {
            **self.cfg,
            "model_cfg": {"auxiliary_loss": True},
        }
        with self.assertRaisesRegex(ValueError, "Auxiliary Loss"):
            base.build_switch_vit_model(cfg)


if __name__ == "__main__":
    unittest.main()

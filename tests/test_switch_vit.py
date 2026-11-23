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

    def test_invalid_switch_parameters_are_rejected_during_config_loading(self) -> None:
        invalid_model_values = (
            ("transformer_layers", 0),
            ("hidden_dim", True),
            ("attention_heads", 3),
            ("patch_size", 7),
            ("expert_hidden_dim", 0),
            ("moe_block_indices", [2, 2]),
            ("dropout", 1.0),
            ("routing", "top2"),
            ("aux_loss_coef", -0.1),
            ("capacity_factor", float("nan")),
            ("capacity_factor", 0.5),
            ("token_dropping", 1),
            ("token_dropping", True),
            ("router_statistics", 1),
        )
        for field, value in invalid_model_values:
            with self.subTest(field=field, value=value):
                with self.assertRaises(base.ConfigError):
                    base.load_embedded_config(
                        config_overrides={"model_cfg": {field: value}}
                    )

        with self.assertRaisesRegex(base.ConfigError, "num_experts"):
            base.load_embedded_config(config_overrides={"num_experts": 0})

    def test_switch_vit_constructor_uses_shared_parameter_validation(self) -> None:
        common = {
            "num_classes": 10,
            "input_shape": (3, 32, 32),
        }
        with self.assertRaisesRegex(ValueError, "hidden_dim"):
            base.SwitchViT(**common, hidden_dim=255)
        with self.assertRaisesRegex(ValueError, "dropout"):
            base.SwitchViT(**common, dropout=1.0)
        with self.assertRaisesRegex(TypeError, "router_statistics"):
            base.SwitchViT(**common, router_statistics=1)
        with self.assertRaisesRegex(ValueError, "moe_block_indices"):
            base.SwitchViT(**common, moe_block_indices=(2, 2))

    def test_switch_vit_cli_options_map_to_model_cfg(self) -> None:
        argv = [
            "uniform.py",
            "--transformer-layers", "8",
            "--hidden-dim", "384",
            "--attention-heads", "6",
            "--patch-size", "4",
            "--expert-hidden-dim", "1536",
            "--moe-block-indices", "2", "4", "6",
            "--routing", "top1",
            "--dropout", "0.1",
            "--aux-loss-coef", "0.02",
            "--capacity-factor", "1.5",
            "--token-dropping", "False",
            "--router-statistics", "False",
            "--data-augmentation", "True",
        ]
        with patch.object(sys, "argv", argv):
            args = base.parse_args()

        overrides = base.build_common_cli_overrides(args)
        self.assertTrue(overrides["data_augmentation"])
        self.assertEqual(
            overrides["model_cfg"],
            {
                "transformer_layers": 8,
                "hidden_dim": 384,
                "attention_heads": 6,
                "patch_size": 4,
                "expert_hidden_dim": 1536,
                "moe_block_indices": [2, 4, 6],
                "routing": "top1",
                "dropout": 0.1,
                "aux_loss_coef": 0.02,
                "capacity_factor": 1.5,
                "token_dropping": False,
                "router_statistics": False,
            },
        )

    def test_switch_vit_cli_accepts_explicit_boolean_values(self) -> None:
        argv = [
            "uniform.py",
            "--aux-loss-coef", "0",
            "--capacity-factor", "0",
            "--token-dropping", "False",
            "--router-statistics", "True",
            "--data-augmentation", "False",
        ]
        with patch.object(sys, "argv", argv):
            args = base.parse_args()

        overrides = base.build_common_cli_overrides(args)
        self.assertEqual(
            overrides["model_cfg"],
            {
                "aux_loss_coef": 0.0,
                "capacity_factor": 0.0,
                "token_dropping": False,
                "router_statistics": True,
            },
        )
        self.assertFalse(overrides["data_augmentation"])

    def test_switch_vit_cli_rejects_invalid_boolean_values(self) -> None:
        argv = ["uniform.py", "--token-dropping", "sometimes"]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                base.parse_args()

    def test_switch_vit_cli_omitted_options_preserve_config_defaults(self) -> None:
        with patch.object(sys, "argv", ["uniform.py"]):
            args = base.parse_args()

        overrides = base.build_common_cli_overrides(args)
        self.assertNotIn("model_cfg", overrides)

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
        for block_index in (1, 3, 5):
            moe = model.blocks[block_index].mlp
            self.assertFalse(moe.auxiliary_loss)
            self.assertFalse(moe.expert_capacity)
            self.assertFalse(moe.token_dropping)
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

    def test_auxiliary_loss_can_be_enabled_and_is_added_to_task_loss(self) -> None:
        cfg = {
            **self.cfg,
            "model_cfg": {
                "aux_loss_coef": 0.01,
            },
        }
        model = base.build_switch_vit_model(cfg)
        images = torch.randn(2, 3, 32, 32)
        targets = torch.tensor([1, 4])

        output = model(images)

        self.assertIsNotNone(output.aux_loss)
        self.assertGreater(float(output.aux_loss.detach().item()), 0.0)
        self.assertEqual(
            set(output.aux_loss_by_layer),
            {"blocks.1.mlp", "blocks.3.mlp", "blocks.5.mlp"},
        )
        losses = base.compute_model_loss(
            output,
            targets,
            torch.nn.CrossEntropyLoss(),
        )
        self.assertTrue(torch.allclose(
            losses.total_loss,
            losses.task_loss + losses.aux_loss,
        ))
        losses.total_loss.backward()
        for block_index in (1, 3, 5):
            router_grad = model.blocks[block_index].mlp.router.projection.weight.grad
            self.assertIsNotNone(router_grad)
            self.assertGreater(float(router_grad.abs().sum().item()), 0.0)

    def _make_constant_router_moe(
        self,
        num_experts: int,
        capacity_factor: float,
        token_dropping: bool,
    ) -> base.SwitchMoE:
        moe = base.SwitchMoE(
            hidden_dim=4,
            expert_hidden_dim=8,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            token_dropping=token_dropping,
        )
        with torch.no_grad():
            moe.router.projection.weight.zero_()
            # Every token prefers expert_0, then expert_1, then expert_2, etc.
            moe.router.projection.bias.copy_(
                torch.arange(num_experts, 0, -1, dtype=torch.float32)
            )
        return moe

    def test_switch_vit_builder_rejects_invalid_expert_counts(self) -> None:
        for value in (0, -1, 2.5, True):
            with self.subTest(num_experts=value):
                cfg = {
                    **self.cfg,
                    "num_experts": value,
                }
                with self.assertRaisesRegex(base.ConfigError, "num_experts"):
                    base.build_switch_vit_model(cfg)

    def test_switch_moe_requires_positive_integer_expert_count(self) -> None:
        common = {
            "hidden_dim": 4,
            "expert_hidden_dim": 8,
        }
        for value in (0, -1, 2.5, True):
            with self.subTest(num_experts=value):
                with self.assertRaisesRegex(ValueError, "num_experts"):
                    base.SwitchMoE(**common, num_experts=value)

    def test_switch_moe_constructor_validates_boolean_options(self) -> None:
        common = {
            "hidden_dim": 4,
            "expert_hidden_dim": 8,
            "num_experts": 2,
        }
        for field in ("collect_router_statistics", "token_dropping"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(TypeError, field):
                    base.SwitchMoE(**common, **{field: 1})

    def test_switch_moe_constructor_validates_dimensions_and_dropout(self) -> None:
        common = {
            "hidden_dim": 4,
            "expert_hidden_dim": 8,
            "num_experts": 2,
        }
        with self.assertRaisesRegex(ValueError, "hidden_dim"):
            base.SwitchMoE(**{**common, "hidden_dim": 0})
        with self.assertRaisesRegex(ValueError, "expert_hidden_dim"):
            base.SwitchMoE(**{**common, "expert_hidden_dim": 0})
        with self.assertRaisesRegex(ValueError, "dropout"):
            base.SwitchMoE(**common, dropout=1.0)
        with self.assertRaisesRegex(TypeError, "dropout"):
            base.SwitchMoE(**common, dropout="0.1")

    def test_switch_moe_constructor_validates_numeric_and_capacity_options(self) -> None:
        common = {
            "hidden_dim": 4,
            "expert_hidden_dim": 8,
            "num_experts": 2,
        }
        for field in ("aux_loss_coef", "capacity_factor"):
            for value in ("0.1", True):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(TypeError, field):
                        base.SwitchMoE(**common, **{field: value})

            for value in (-0.1, float("inf"), float("nan")):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        base.SwitchMoE(**common, **{field: value})

        with self.assertRaisesRegex(ValueError, "capacity_factor"):
            base.SwitchMoE(
                **common,
                capacity_factor=0.0,
                token_dropping=True,
            )
        with self.assertRaisesRegex(ValueError, "capacity_factor"):
            base.SwitchMoE(
                **common,
                capacity_factor=0.5,
                token_dropping=False,
            )

    def test_switch_vit_builder_rejects_invalid_moe_boolean_options(self) -> None:
        for field in ("token_dropping", "router_statistics"):
            with self.subTest(field=field):
                cfg = {
                    **self.cfg,
                    "model_cfg": {field: 1},
                }
                with self.assertRaisesRegex(base.ConfigError, field):
                    base.build_switch_vit_model(cfg)

        with self.assertRaisesRegex(base.ConfigError, "routing"):
            base.build_switch_vit_model({
                **self.cfg,
                "model_cfg": {"routing": "top2"},
            })

    def test_capacity_reroutes_overflow_across_all_experts_without_dropping(self) -> None:
        moe = self._make_constant_router_moe(
            num_experts=3,
            capacity_factor=1.0,
            token_dropping=False,
        )
        inputs = torch.randn(1, 9, 4)

        outputs, stats, aux_loss = moe(inputs)

        self.assertEqual(tuple(outputs.shape), (1, 9, 4))
        self.assertIsNone(aux_loss)
        self.assertEqual(stats["tokens_per_expert"].tolist(), [9, 0, 0])
        self.assertEqual(
            stats["processed_tokens_per_expert"].tolist(),
            [3, 3, 3],
        )
        self.assertEqual(stats["rerouted_tokens"], 6)
        self.assertEqual(stats["dropped_tokens"], 0)
        self.assertEqual(
            stats["rerouted_tokens_per_expert"].tolist(),
            [0, 3, 3],
        )

    def test_switch_vit_capacity_config_reaches_every_moe_layer(self) -> None:
        cfg = {
            **self.cfg,
            "model_cfg": {
                "capacity_factor": 1.0,
                "token_dropping": False,
            },
        }
        model = base.build_switch_vit_model(cfg)
        images = torch.randn(1, 3, 32, 32)

        output = model(images)

        self.assertEqual(
            set(output.moe_stats),
            {"blocks.1.mlp", "blocks.3.mlp", "blocks.5.mlp"},
        )
        for block_index in (1, 3, 5):
            moe = model.blocks[block_index].mlp
            self.assertTrue(moe.expert_capacity)
            self.assertFalse(moe.token_dropping)
            layer_stats = output.moe_stats[f"blocks.{block_index}.mlp"]
            self.assertEqual(
                int(layer_stats["processed_tokens_per_expert"].sum().item()),
                65,
            )
            self.assertEqual(layer_stats["dropped_tokens"], 0)

    def test_token_dropping_only_occurs_after_all_experts_are_full(self) -> None:
        moe = self._make_constant_router_moe(
            num_experts=3,
            capacity_factor=0.5,
            token_dropping=True,
        )
        inputs = torch.randn(1, 8, 4)

        outputs, stats, _ = moe(inputs)

        self.assertEqual(tuple(outputs.shape), (1, 8, 4))
        self.assertEqual(
            stats["processed_tokens_per_expert"].tolist(),
            [2, 2, 2],
        )
        self.assertEqual(stats["rerouted_tokens"], 4)
        self.assertEqual(stats["dropped_tokens"], 2)
        self.assertEqual(int(stats["overflow_tokens_per_expert"].sum().item()), 6)

    def test_capacity_and_token_dropping_configuration_validation(self) -> None:
        with self.assertRaisesRegex(base.ConfigError, "capacity_factor"):
            base.build_switch_vit_model({
                **self.cfg,
                "model_cfg": {
                    "capacity_factor": 0.5,
                    "token_dropping": False,
                },
            })

        with self.assertRaisesRegex(base.ConfigError, "capacity_factor"):
            base.build_switch_vit_model({
                **self.cfg,
                "model_cfg": {
                    "capacity_factor": 0.0,
                    "token_dropping": True,
                },
            })


if __name__ == "__main__":
    unittest.main()

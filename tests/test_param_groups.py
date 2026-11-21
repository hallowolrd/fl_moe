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

import torch.nn as nn  # noqa: E402
import base  # noqa: E402


class TinyExpert(nn.Module):
    def __init__(self, hidden_dim: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.BatchNorm1d(hidden_dim)


class TinyMoE(nn.Module):
    def __init__(self, num_experts: int = 4) -> None:
        super().__init__()
        self.router = nn.Linear(4, num_experts)
        self.experts = nn.ModuleList(
            [TinyExpert() for _ in range(num_experts)]
        )


class TinyBlock(nn.Module):
    def __init__(self, num_experts: int = 4) -> None:
        super().__init__()
        self.moe = TinyMoE(num_experts=num_experts)


class TinyMultiLayerMoE(nn.Module):
    def __init__(self, expert_counts: tuple[int, ...] = (4, 4)) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.blocks = nn.ModuleList(
            [TinyBlock(num_experts=count) for count in expert_counts]
        )
        self.classifier = nn.Linear(4, 2)


class TinySingleLayerMoE(nn.Module):
    def __init__(self, num_experts: int = 4) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.moe_head = TinyMoE(num_experts=num_experts)
        self.classifier = nn.Linear(4, 2)


class TinyRootExperts(nn.Module):
    def __init__(self, num_experts: int = 4) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [TinyExpert() for _ in range(num_experts)]
        )


class MultiLayerParamGroupTests(unittest.TestCase):
    def test_single_layer_keeps_legacy_expert_groups(self) -> None:
        model = TinySingleLayerMoE(num_experts=4)

        groups = base.build_param_groups(
            model,
            expected_num_experts=4,
            strict=True,
        )

        self.assertEqual(set(groups.expert_by_id), {0, 1, 2, 3})
        self.assertEqual(set(groups.expert_by_moe_layer), {"moe_head"})
        self.assertEqual(
            set(groups.expert_by_layer_and_id["moe_head"]),
            {0, 1, 2, 3},
        )

        # The legacy expert_by_id view still aggregates the same expert id
        # across all MoE layers (one layer in this compatibility test).
        self.assertEqual(
            set(groups.expert_by_id[0]),
            set(groups.expert_by_layer_and_id["moe_head"][0]),
        )

    def test_multiple_moe_layers_are_separated(self) -> None:
        model = TinyMultiLayerMoE(expert_counts=(4, 4))

        groups = base.build_param_groups(
            model,
            expected_num_experts=4,
            strict=True,
        )

        self.assertEqual(
            set(groups.expert_by_moe_layer),
            {"blocks.0.moe", "blocks.1.moe"},
        )
        for layer_name in ("blocks.0.moe", "blocks.1.moe"):
            self.assertEqual(
                set(groups.expert_by_layer_and_id[layer_name]),
                {0, 1, 2, 3},
            )
            for expert_id in range(4):
                names = groups.expert_by_layer_and_id[layer_name][expert_id]
                self.assertTrue(names)
                self.assertTrue(
                    all(
                        name.startswith(
                            f"{layer_name}.experts.{expert_id}."
                        )
                        for name in names
                    )
                )

        # expert_by_id remains a cross-layer compatibility view.
        self.assertEqual(len(groups.expert_by_id[0]), 2 * len(
            groups.expert_by_layer_and_id["blocks.0.moe"][0]
        ))

    def test_expected_expert_count_is_checked_per_layer(self) -> None:
        # The global union still contains ids 0..3, but layer 0 is missing
        # expert 3. Per-layer validation must catch this inconsistency.
        model = TinyMultiLayerMoE(expert_counts=(3, 4))

        with self.assertRaisesRegex(ValueError, "MoE 层"):
            base.build_param_groups(
                model,
                expected_num_experts=4,
                strict=True,
            )

    def test_expert_and_non_expert_groups_partition_state_dict(self) -> None:
        model = TinyMultiLayerMoE(expert_counts=(4, 4))
        state_names = set(model.state_dict())

        groups = base.build_param_groups(
            model,
            expected_num_experts=4,
            strict=True,
        )

        self.assertEqual(set(groups.all), state_names)
        self.assertEqual(
            set(groups.non_expert) | set(groups.expert),
            state_names,
        )
        self.assertFalse(set(groups.non_expert) & set(groups.expert))
        self.assertIn("backbone.weight", groups.non_expert)
        self.assertIn("classifier.weight", groups.non_expert)
        self.assertIn("blocks.0.moe.router.weight", groups.non_expert)

    def test_federated_optimization_groups_expose_trainable_layer_groups(self) -> None:
        model = TinyMultiLayerMoE(expert_counts=(4, 4))
        frozen_name = "blocks.0.moe.experts.0.fc1.weight"
        model.blocks[0].moe.experts[0].fc1.weight.requires_grad_(False)

        param_groups = base.build_param_groups(
            model,
            expected_num_experts=4,
            strict=True,
        )
        optimization_groups = base.build_federated_optimization_groups(
            model,
            param_groups=param_groups,
            strict=True,
        )

        self.assertIn(
            frozen_name,
            optimization_groups.expert_state_names_by_layer_and_id[
                "blocks.0.moe"
            ][0],
        )
        self.assertNotIn(
            frozen_name,
            optimization_groups.expert_parameters_by_layer_and_id[
                "blocks.0.moe"
            ][0],
        )
        self.assertIn(frozen_name, optimization_groups.frozen_parameters)

        for layer_name in ("blocks.0.moe", "blocks.1.moe"):
            layer_parameters = set(
                optimization_groups.expert_parameters_by_moe_layer[layer_name]
            )
            layer_parameters_by_id = {
                name
                for names in (
                    optimization_groups.expert_parameters_by_layer_and_id[
                        layer_name
                    ].values()
                )
                for name in names
            }
            self.assertEqual(layer_parameters, layer_parameters_by_id)

        grouped_trainable_names = {
            name
            for layer_groups in (
                optimization_groups.expert_parameters_by_layer_and_id.values()
            )
            for names in layer_groups.values()
            for name in names
        }
        self.assertEqual(
            grouped_trainable_names,
            set(optimization_groups.expert_parameters),
        )

    def test_parameter_statistics_include_layer_and_expert_dimensions(self) -> None:
        model = TinyMultiLayerMoE(expert_counts=(4, 4))
        groups = base.build_param_groups(
            model,
            expected_num_experts=4,
            strict=True,
        )
        state_dict = model.state_dict()

        tensor_counts = base.count_tensors_by_group(groups)
        numel_counts = base.count_numel_by_group(
            state_dict,
            groups,
            only_floating=True,
        )
        summary = base.summarize_param_groups(state_dict, groups)

        self.assertEqual(
            set(tensor_counts["expert_by_moe_layer"]),
            {"blocks.0.moe", "blocks.1.moe"},
        )
        self.assertGreater(
            tensor_counts["expert_by_layer_and_id"]["blocks.1.moe"][3],
            0,
        )
        self.assertGreater(
            numel_counts["expert_by_layer_and_id"]["blocks.0.moe"][0],
            0,
        )
        self.assertEqual(
            summary["tensor_counts"]["expert_by_moe_layer"],
            tensor_counts["expert_by_moe_layer"],
        )

    def test_root_level_experts_have_explicit_root_layer_key(self) -> None:
        model = TinyRootExperts()

        groups = base.build_param_groups(
            model,
            expected_num_experts=4,
            strict=True,
        )

        self.assertEqual(set(groups.expert_by_moe_layer), {"<root>"})
        self.assertEqual(
            set(groups.expert_by_layer_and_id["<root>"]),
            {0, 1, 2, 3},
        )


if __name__ == "__main__":
    unittest.main()

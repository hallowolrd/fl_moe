from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from tools import diagnose_expert_homogeneity as metrics


class WeightCosineTests(unittest.TestCase):
    def test_identical_weights_have_cosine_one(self) -> None:
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertAlmostEqual(
            metrics.compute_weight_cosine(weight, weight.clone()),
            1.0,
            places=12,
        )

    def test_opposite_weights_have_cosine_minus_one(self) -> None:
        weight = torch.tensor([1.0, -2.0, 3.0])
        self.assertAlmostEqual(
            metrics.compute_weight_cosine(weight, -weight),
            -1.0,
            places=12,
        )

    def test_cosine_is_scale_invariant_for_positive_scaling(self) -> None:
        weight = torch.randn(4, 5, dtype=torch.float64)
        self.assertAlmostEqual(
            metrics.compute_weight_cosine(weight, 3.5 * weight),
            1.0,
            places=12,
        )

    def test_zero_norm_and_shape_mismatch_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "零范数"):
            metrics.compute_weight_cosine(torch.zeros(4), torch.ones(4))
        with self.assertRaisesRegex(ValueError, "Shape"):
            metrics.compute_weight_cosine(torch.ones(2, 3), torch.ones(3, 2))


class SubspaceOverlapTests(unittest.TestCase):
    def test_identical_top_k_subspaces_have_overlap_one(self) -> None:
        weight = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
        for side in ("left", "right"):
            self.assertAlmostEqual(
                metrics.compute_subspace_overlap(weight, weight, k=2, side=side),
                1.0,
                places=12,
            )

    def test_left_and_right_subspaces_are_distinguished(self) -> None:
        # 两个矩阵共享相同的右主子空间，但左主子空间正交。
        u_a = torch.eye(5)[:, :4]
        u_b = torch.eye(5)[:, [2, 3, 0, 1]]
        singular_values = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
        v = torch.eye(4)
        weight_a = u_a @ singular_values @ v.transpose(0, 1)
        weight_b = u_b @ singular_values @ v.transpose(0, 1)

        right_overlap = metrics.compute_subspace_overlap(
            weight_a, weight_b, k=2, side="right"
        )
        left_overlap = metrics.compute_subspace_overlap(
            weight_a, weight_b, k=2, side="left"
        )
        self.assertAlmostEqual(right_overlap, 1.0, places=12)
        self.assertAlmostEqual(left_overlap, 0.0, places=12)

    def test_disjoint_top_k_subspaces_have_overlap_zero(self) -> None:
        weight_a = torch.diag(torch.tensor([4.0, 3.0, 0.0, 0.0]))
        weight_b = torch.diag(torch.tensor([0.0, 0.0, 4.0, 3.0]))
        self.assertAlmostEqual(
            metrics.compute_subspace_overlap(
                weight_a, weight_b, k=2, side="right"
            ),
            0.0,
            places=12,
        )

    def test_cached_subspace_bases_match_direct_overlap(self) -> None:
        weight_a = torch.randn(12, 8)
        weight_b = torch.randn(12, 8)
        basis_a, singular_values_a, rank_a = metrics.extract_topk_subspace(
            weight_a, max_k=4, side="right"
        )
        basis_b, singular_values_b, rank_b = metrics.extract_topk_subspace(
            weight_b, max_k=4, side="right"
        )

        self.assertEqual(basis_a.shape, (8, 4))
        self.assertEqual(basis_b.shape, (8, 4))
        self.assertEqual(singular_values_a.numel(), 8)
        self.assertEqual(singular_values_b.numel(), 8)
        self.assertEqual(rank_a, 8)
        self.assertEqual(rank_b, 8)
        cached = metrics.compute_subspace_overlap_from_bases(
            basis_a, basis_b, k=3
        )
        direct = metrics.compute_subspace_overlap(
            weight_a, weight_b, k=3, side="right"
        )
        self.assertAlmostEqual(cached, direct, places=12)

    def test_k_larger_than_effective_rank_is_rejected(self) -> None:
        weight = torch.diag(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        with self.assertRaisesRegex(ValueError, "有效数值秩"):
            metrics.compute_subspace_overlap(
                weight, weight, k=2, side="right"
            )

    def test_invalid_k_side_and_tensor_shape_are_rejected(self) -> None:
        weight = torch.eye(4)
        with self.assertRaisesRegex(ValueError, "正整数"):
            metrics.compute_subspace_overlap(weight, weight, k=0, side="right")
        with self.assertRaisesRegex(ValueError, "side"):
            metrics.compute_subspace_overlap(weight, weight, k=2, side="middle")
        with self.assertRaisesRegex(ValueError, "Shape"):
            metrics.compute_subspace_overlap(
                torch.ones(4, 3), torch.ones(3, 4), k=2, side="right"
            )


class LinearCKATests(unittest.TestCase):
    def test_identical_representations_have_cka_one(self) -> None:
        features = torch.randn(40, 12)
        self.assertAlmostEqual(
            metrics.compute_linear_cka(features, features.clone()),
            1.0,
            places=12,
        )

    def test_cka_is_invariant_to_orthogonal_transform_and_translation(self) -> None:
        torch.manual_seed(13)
        features = torch.randn(64, 8)
        orthogonal, _ = torch.linalg.qr(torch.randn(8, 8))
        transformed = features @ orthogonal + torch.randn(8)
        self.assertAlmostEqual(
            metrics.compute_linear_cka(features, transformed),
            1.0,
            places=10,
        )

    def test_cka_allows_different_feature_dimensions(self) -> None:
        features_a = torch.randn(32, 8)
        features_b = torch.randn(32, 5)
        value = metrics.compute_linear_cka(features_a, features_b)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_constant_representation_is_rejected(self) -> None:
        constant = torch.ones(16, 4)
        variable = torch.randn(16, 4)
        with self.assertRaisesRegex(ValueError, "没有有效方差"):
            metrics.compute_linear_cka(constant, variable)

    def test_misaligned_sample_count_and_invalid_rank_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "样本数"):
            metrics.compute_linear_cka(torch.randn(10, 4), torch.randn(9, 4))
        with self.assertRaisesRegex(ValueError, "Shape"):
            metrics.compute_linear_cka(torch.randn(10), torch.randn(10))


class OutputDifferenceTests(unittest.TestCase):
    def test_identical_outputs_have_zero_normalized_mse(self) -> None:
        output = torch.randn(20, 6)
        self.assertAlmostEqual(
            metrics.compute_normalized_output_mse(output, output.clone()),
            0.0,
            places=12,
        )

    def test_normalized_mse_matches_expected_scale_difference(self) -> None:
        output = torch.randn(20, 6)
        # For Y_b=2Y_a, the normalized squared difference is 1 / 2.5 = 0.4.
        self.assertAlmostEqual(
            metrics.compute_normalized_output_mse(output, 2.0 * output),
            0.4,
            places=12,
        )

    def test_low_energy_and_shape_mismatch_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "总能量过低"):
            metrics.compute_normalized_output_mse(
                torch.zeros(4, 3), torch.zeros(4, 3)
            )
        with self.assertRaisesRegex(ValueError, "Shape"):
            metrics.compute_normalized_output_mse(
                torch.ones(4, 3), torch.ones(3, 4)
            )


class ExpertPermutationTests(unittest.TestCase):
    def test_hidden_neuron_permutation_preserves_expert_function(self) -> None:
        torch.manual_seed(29)
        inputs = torch.randn(48, 5)
        fc1_weight = torch.randn(7, 5)
        fc1_bias = torch.randn(7)
        fc2_weight = torch.randn(3, 7)
        fc2_bias = torch.randn(3)
        permutation = torch.randperm(7)

        hidden_a = F.gelu(F.linear(inputs, fc1_weight, fc1_bias))
        output_a = F.linear(hidden_a, fc2_weight, fc2_bias)

        permuted_fc1_weight = fc1_weight[permutation]
        permuted_fc1_bias = fc1_bias[permutation]
        permuted_fc2_weight = fc2_weight[:, permutation]
        hidden_b = F.gelu(
            F.linear(inputs, permuted_fc1_weight, permuted_fc1_bias)
        )
        output_b = F.linear(hidden_b, permuted_fc2_weight, fc2_bias)

        self.assertTrue(torch.allclose(output_a, output_b, atol=1e-6, rtol=1e-6))
        self.assertLess(
            metrics.compute_normalized_output_mse(output_a, output_b),
            1e-12,
        )
        self.assertAlmostEqual(
            metrics.compute_linear_cka(output_a, output_b),
            1.0,
            places=10,
        )


class InputImmutabilityTests(unittest.TestCase):
    def test_metric_functions_do_not_modify_inputs(self) -> None:
        weight_a = torch.randn(6, 4)
        weight_b = torch.randn(6, 4)
        features_a = torch.randn(20, 6)
        features_b = torch.randn(20, 6)
        original = [
            value.clone()
            for value in (weight_a, weight_b, features_a, features_b)
        ]

        metrics.compute_weight_cosine(weight_a, weight_b)
        metrics.compute_subspace_overlap(
            weight_a, weight_b, k=2, side="right"
        )
        metrics.compute_linear_cka(features_a, features_b)
        metrics.compute_normalized_output_mse(features_a, features_b)

        for actual, expected in zip(
            (weight_a, weight_b, features_a, features_b),
            original,
        ):
            self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()

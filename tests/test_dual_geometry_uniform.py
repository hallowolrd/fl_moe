"""Unit tests for the dual-geometry expert aggregation algorithm."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = ROOT / "algorithms"
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))

import base  # noqa: E402
import dual_geometry_uniform as algorithm  # noqa: E402
import uniform  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


class TinyExpert(nn.Module):
    def __init__(self, hidden_dim: int = 4, expert_hidden_dim: int = 6) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, expert_hidden_dim)
        self.fc2 = nn.Linear(expert_hidden_dim, hidden_dim)


class TinySwitchMoE(nn.Module):
    def __init__(self, num_experts: int = 2) -> None:
        super().__init__()
        self.experts = nn.ModuleDict(
            {
                f"expert_{expert_id}": TinyExpert()
                for expert_id in range(num_experts)
            }
        )


class TinySwitchModel(nn.Module):
    def __init__(self, num_experts: int = 2) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.moe = TinySwitchMoE(num_experts=num_experts)
        self.classifier = nn.Linear(4, 2)


def build_cfg(
    *,
    align_beta: float = 0.0,
    diversity_gamma: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_experts=2,
        method_config=SimpleNamespace(
            subspace_rank=2,
            align_beta=align_beta,
            diversity_gamma=diversity_gamma,
            overlap_threshold=0.5,
        ),
    )


def build_client_updates(
    cfg: SimpleNamespace,
):
    """Build deterministic client deltas and valid local subspace evidence."""
    torch.manual_seed(41)
    global_model = TinySwitchModel(num_experts=cfg.num_experts)
    global_state = base.clone_state_dict(global_model.state_dict())
    optimization_groups = base.build_federated_optimization_groups(
        model=global_model,
        expected_num_experts=cfg.num_experts,
        strict=True,
    )

    client_updates = []
    for client_id, perturbation_scale in ((0, 0.01), (1, 0.03)):
        local_model = copy.deepcopy(global_model)
        generator = torch.Generator(device="cpu").manual_seed(100 + client_id)

        with torch.no_grad():
            for parameter in local_model.parameters():
                perturbation = torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device=parameter.device,
                    generator=generator,
                )
                parameter.add_(perturbation_scale * perturbation)

        local_state = local_model.state_dict()
        model_delta = base.subtract_state_dict(
            local_state=local_state,
            global_state=global_state,
            strict=True,
        )
        evidence = algorithm.collect_dual_geometry_evidence(
            model=local_model,
            evidence_loader=None,
            criterion=None,
            device=torch.device("cpu"),
            cfg=cfg,
        )

        client_updates.append(
            base.ClientUpdate(
                client_id=client_id,
                round_id=1,
                num_samples=10 if client_id == 0 else 100,
                model_delta=model_delta,
                extra=dict(evidence),
            )
        )

    return global_model, global_state, optimization_groups, client_updates


class DualGeometryUniformTests(unittest.TestCase):
    def test_extracted_right_singular_subspace_is_orthonormal(self) -> None:
        torch.manual_seed(7)
        weight = torch.randn(6, 4)

        basis = algorithm.extract_right_singular_subspace(
            weight=weight,
            rank=2,
        )

        self.assertEqual(tuple(basis.shape), (4, 2))
        torch.testing.assert_close(
            basis.T @ basis,
            torch.eye(2),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_grassmann_distance_of_identical_subspaces_is_zero(self) -> None:
        basis = torch.eye(4, dtype=torch.float64)[:, :2]

        distance_squared = algorithm._normalized_grassmann_distance_squared(
            basis,
            basis.clone(),
        )

        self.assertAlmostEqual(distance_squared, 0.0, places=10)

    def test_orthogonal_subspaces_have_zero_normalized_overlap(self) -> None:
        basis_a = torch.eye(4)[:, :2]
        basis_b = torch.eye(4)[:, 2:]

        overlap = algorithm._normalized_subspace_overlap(basis_a, basis_b)

        self.assertAlmostEqual(overlap, 0.0, places=7)

    def test_identical_subspaces_have_unit_normalized_overlap(self) -> None:
        basis = torch.eye(4)[:, :2]

        overlap = algorithm._normalized_subspace_overlap(
            basis,
            basis.clone(),
        )

        self.assertAlmostEqual(overlap, 1.0, places=7)

    def test_zero_geometry_coefficients_match_uniform_expert_aggregation(
        self,
    ) -> None:
        cfg = build_cfg(align_beta=0.0, diversity_gamma=0.0)
        (
            global_model,
            global_state,
            optimization_groups,
            client_updates,
        ) = build_client_updates(cfg)

        dual_aggregator = algorithm.build_expert_aggregator(cfg)
        dual_aggregator.initialize_model(global_model)
        dual_result = dual_aggregator.aggregate(
            global_state=global_state,
            client_updates=client_updates,
            param_names=optimization_groups.expert_state_names,
        )

        uniform_aggregator = uniform.UniformExpertAggregator(
            cfg=cfg,
            param_group_name="expert",
        )
        uniform_result = uniform_aggregator.aggregate(
            global_state=global_state,
            client_updates=client_updates,
            param_names=optimization_groups.expert_state_names,
        )

        self.assertEqual(set(dual_result.weights), {0, 1})
        for client_id in (0, 1):
            self.assertAlmostEqual(
                dual_result.weights[client_id],
                0.5,
                places=7,
            )

        per_expert_weights = dual_result.diagnostics[
            "per_expert_client_weights"
        ]
        self.assertTrue(per_expert_weights)
        for layer_weights in per_expert_weights.values():
            for client_weights in layer_weights.values():
                self.assertAlmostEqual(
                    float(client_weights["0"]),
                    0.5,
                    places=7,
                )
                self.assertAlmostEqual(
                    float(client_weights["1"]),
                    0.5,
                    places=7,
                )

        self.assertEqual(
            set(dual_result.new_state_dict),
            set(uniform_result.new_state_dict),
        )
        for name, expected in uniform_result.new_state_dict.items():
            torch.testing.assert_close(
                dual_result.new_state_dict[name],
                expected,
                msg=f"参数 {name} 未退化为 Uniform 聚合结果。",
            )


if __name__ == "__main__":
    unittest.main()

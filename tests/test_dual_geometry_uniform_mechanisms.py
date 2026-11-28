"""Mechanism tests for dual-geometry expert aggregation weights."""

from __future__ import annotations

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
    align_beta: float,
    diversity_gamma: float,
    overlap_threshold: float = 0.5,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_experts=2,
        method_config=SimpleNamespace(
            subspace_rank=2,
            align_beta=align_beta,
            diversity_gamma=diversity_gamma,
            overlap_threshold=overlap_threshold,
        ),
    )


def coordinate_subspace(start: int) -> torch.Tensor:
    """Return a 4-by-2 orthonormal basis spanning coordinate axes."""
    return torch.eye(4, dtype=torch.float32)[:, start : start + 2].clone()


def install_global_subspace(
    model: nn.Module,
    parameter_name: str,
    basis: torch.Tensor,
) -> None:
    """Set fc1 so its leading right-singular subspace is exactly basis."""
    parameters = dict(model.named_parameters())
    weight = parameters[parameter_name]
    if weight.ndim != 2 or weight.shape[1] != basis.shape[0]:
        raise AssertionError(f"Unexpected fc1 shape for {parameter_name}: {weight.shape}")
    if weight.shape[0] < basis.shape[1]:
        raise AssertionError("fc1 output dimension must be at least the subspace rank.")

    controlled_weight = torch.zeros_like(weight)
    controlled_weight[: basis.shape[1], :] = basis.T
    with torch.no_grad():
        weight.copy_(controlled_weight)


def run_controlled_aggregation(
    *,
    align_beta: float,
    diversity_gamma: float,
    local_bases_by_client: dict[int, dict[int, torch.Tensor]],
):
    """Run the real expert aggregator with deterministic subspace evidence."""
    cfg = build_cfg(
        align_beta=align_beta,
        diversity_gamma=diversity_gamma,
    )
    global_model = TinySwitchModel(num_experts=cfg.num_experts)
    optimization_groups = base.build_federated_optimization_groups(
        model=global_model,
        expected_num_experts=cfg.num_experts,
        strict=True,
    )
    layout = algorithm._build_parameter_layout(
        optimization_groups,
        expected_num_experts=cfg.num_experts,
    )

    global_bases = {
        0: coordinate_subspace(0),
        1: coordinate_subspace(2),
    }
    layer_names = sorted(layout.fc1_names_by_layer_and_id)
    if len(layer_names) != 1:
        raise AssertionError(f"Expected one MoE layer, got {layer_names!r}")
    layer_name = layer_names[0]

    for expert_id, parameter_name in layout.fc1_names_by_layer_and_id[
        layer_name
    ].items():
        install_global_subspace(
            global_model,
            parameter_name,
            global_bases[expert_id],
        )

    global_state = base.clone_state_dict(global_model.state_dict())
    zero_delta = base.subtract_state_dict(
        local_state=global_state,
        global_state=global_state,
        strict=True,
    )

    client_updates = []
    for client_id in sorted(local_bases_by_client):
        local_subspaces = {
            layout.fc1_names_by_layer_and_id[layer_name][expert_id]: basis
            for expert_id, basis in local_bases_by_client[client_id].items()
        }
        client_updates.append(
            base.ClientUpdate(
                client_id=client_id,
                round_id=1,
                num_samples=1,
                model_delta=base.clone_state_dict(zero_delta),
                extra={
                    "dual_geometry": {
                        "version": algorithm.EVIDENCE_VERSION,
                        "subspaces": local_subspaces,
                    }
                },
            )
        )

    aggregator = algorithm.build_expert_aggregator(cfg)
    aggregator.initialize_model(global_model)
    result = aggregator.aggregate(
        global_state=global_state,
        client_updates=client_updates,
        param_names=optimization_groups.expert_state_names,
    )
    return result, layer_name


class DualGeometryMechanismTests(unittest.TestCase):
    def test_cross_client_alignment_favors_the_closer_expert_subspace(
        self,
    ) -> None:
        basis_a = coordinate_subspace(0)
        basis_b = coordinate_subspace(2)

        result, layer_name = run_controlled_aggregation(
            align_beta=1.0,
            diversity_gamma=0.0,
            local_bases_by_client={
                0: {0: basis_a, 1: basis_b},
                1: {0: basis_b, 1: basis_a},
            },
        )

        weights = result.diagnostics["per_expert_client_weights"][
            layer_name
        ]["0"]
        client_0_weight = float(weights["0"])
        client_1_weight = float(weights["1"])

        expected_client_0_weight = 1.0 / (
            1.0 + torch.exp(torch.tensor(-(torch.pi**2 / 4.0))).item()
        )
        self.assertGreater(client_0_weight, client_1_weight)
        self.assertAlmostEqual(
            client_0_weight,
            expected_client_0_weight,
            places=6,
        )
        self.assertAlmostEqual(client_0_weight + client_1_weight, 1.0, places=7)

    def test_within_client_diversity_favors_the_less_redundant_client(
        self,
    ) -> None:
        basis_a = coordinate_subspace(0)
        basis_b = coordinate_subspace(2)

        result, layer_name = run_controlled_aggregation(
            align_beta=0.0,
            diversity_gamma=4.0,
            local_bases_by_client={
                0: {0: basis_a, 1: basis_b},
                1: {0: basis_a, 1: basis_a},
            },
        )

        weights = result.diagnostics["per_expert_client_weights"][
            layer_name
        ]["0"]
        client_0_weight = float(weights["0"])
        client_1_weight = float(weights["1"])

        expected_client_0_weight = 1.0 / (1.0 + torch.exp(torch.tensor(-1.0)).item())
        self.assertGreater(client_0_weight, client_1_weight)
        self.assertAlmostEqual(
            client_0_weight,
            expected_client_0_weight,
            places=6,
        )
        self.assertAlmostEqual(client_0_weight + client_1_weight, 1.0, places=7)


if __name__ == "__main__":
    unittest.main()

"""Dual-geometry adaptive aggregation for Switch Transformer experts.

Each expert is represented by the dominant right-singular subspace of its
fc1 weight. The server combines cross-client alignment and within-client
expert diversity to compute a separate aggregation distribution for every
(MoE layer, expert) pair. Non-expert parameters remain under base.py's fixed
uniform aggregation policy.
"""

from __future__ import annotations

# Import base before torch so shared deterministic CUDA settings are applied
# before PyTorch initializes CUDA/cuBLAS.
import base

import argparse
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn


ALGORITHM_NAME = "dual_geometry_uniform"
EVIDENCE_VERSION = 1
BASIS_ORTHOGONALITY_TOLERANCE = 5e-3

METHOD_CONFIG_DEFAULTS = {
    "method_config": {
        "subspace_rank": 128,
        "align_beta": 1.0,
        "diversity_gamma": 2.0,
        "overlap_threshold": 0.5,
    },
}

EMBEDDED_METHOD_CONFIG = {
    "agg": {
        "non_expert": {"method": "uniform"},
        "expert": {"method": ALGORITHM_NAME},
    },
}


_MISSING = object()


def _read_nested_config(cfg: Any, path: str, default: Any = None) -> Any:
    """Read a dotted configuration path from ConfigNode or a mapping."""
    if hasattr(cfg, "get"):
        try:
            value = cfg.get(path, _MISSING)
        except TypeError:
            value = _MISSING
        if value is not _MISSING:
            return value

    current = cfg
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return default
    return current


def _finite_real(value: Any, name: str) -> float:
    """Validate and convert a finite real-valued configuration item."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是有限实数，当前值：{value!r}。")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限实数，当前值：{value!r}。")
    return result


def _read_method_config_value(
    config: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    method_config = config.get("method_config", config)
    if hasattr(method_config, "to_dict"):
        method_config = method_config.to_dict()
    if not isinstance(method_config, Mapping):
        raise ValueError("method_config 必须是 Mapping。")
    return method_config.get(key, default)


def validate_method_config(config: Mapping[str, Any]) -> None:
    """Validate the algorithm-specific configuration."""
    if not isinstance(config, Mapping):
        raise ValueError("配置必须是 Mapping。")

    defaults = METHOD_CONFIG_DEFAULTS["method_config"]
    rank = _read_method_config_value(
        config,
        "subspace_rank",
        defaults["subspace_rank"],
    )
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("method_config.subspace_rank 必须是正整数。")

    align_beta = _finite_real(
        _read_method_config_value(config, "align_beta", defaults["align_beta"]),
        "method_config.align_beta",
    )
    if align_beta < 0.0:
        raise ValueError("method_config.align_beta 必须大于或等于 0。")

    diversity_gamma = _finite_real(
        _read_method_config_value(
            config,
            "diversity_gamma",
            defaults["diversity_gamma"],
        ),
        "method_config.diversity_gamma",
    )
    if diversity_gamma < 0.0:
        raise ValueError("method_config.diversity_gamma 必须大于或等于 0。")

    overlap_threshold = _finite_real(
        _read_method_config_value(
            config,
            "overlap_threshold",
            defaults["overlap_threshold"],
        ),
        "method_config.overlap_threshold",
    )
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError(
            "method_config.overlap_threshold 必须满足 0 <= value <= 1。"
        )


def _read_runtime_settings(cfg: Any) -> Dict[str, Any]:
    """Resolve and validate the method settings from the full runtime config."""
    defaults = METHOD_CONFIG_DEFAULTS["method_config"]

    rank = _read_nested_config(
        cfg,
        "method_config.subspace_rank",
        defaults["subspace_rank"],
    )
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("method_config.subspace_rank 必须是正整数。")

    align_beta = _finite_real(
        _read_nested_config(
            cfg,
            "method_config.align_beta",
            defaults["align_beta"],
        ),
        "method_config.align_beta",
    )
    diversity_gamma = _finite_real(
        _read_nested_config(
            cfg,
            "method_config.diversity_gamma",
            defaults["diversity_gamma"],
        ),
        "method_config.diversity_gamma",
    )
    overlap_threshold = _finite_real(
        _read_nested_config(
            cfg,
            "method_config.overlap_threshold",
            defaults["overlap_threshold"],
        ),
        "method_config.overlap_threshold",
    )
    num_experts = _read_nested_config(cfg, "num_experts", 4)

    if align_beta < 0.0:
        raise ValueError("method_config.align_beta 必须大于或等于 0。")
    if diversity_gamma < 0.0:
        raise ValueError("method_config.diversity_gamma 必须大于或等于 0。")
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError(
            "method_config.overlap_threshold 必须满足 0 <= value <= 1。"
        )
    if (
        isinstance(num_experts, bool)
        or not isinstance(num_experts, int)
        or num_experts < 2
    ):
        raise ValueError("dual_geometry_uniform 要求 num_experts 至少为 2。")

    return {
        "subspace_rank": int(rank),
        "align_beta": align_beta,
        "diversity_gamma": diversity_gamma,
        "overlap_threshold": overlap_threshold,
        "num_experts": int(num_experts),
    }


@dataclass(frozen=True)
class ExpertParameterLayout:
    """State-dict names and fc1 weight names grouped by MoE layer and expert."""

    expert_state_names_by_layer_and_id: Mapping[
        str, Mapping[int, Tuple[str, ...]]
    ]
    fc1_names_by_layer_and_id: Mapping[str, Mapping[int, str]]
    expert_state_names: Tuple[str, ...]


def _build_parameter_layout(
    optimization_groups: Any,
    expected_num_experts: int,
) -> ExpertParameterLayout:
    """Resolve every expert's state entries and its fc1 weight name."""
    state_groups = getattr(
        optimization_groups,
        "expert_state_names_by_layer_and_id",
        None,
    )
    if not isinstance(state_groups, Mapping) or not state_groups:
        raise ValueError("没有找到逐层、逐专家的 expert state 分组。")

    expected_ids = list(range(expected_num_experts))
    normalized_state_groups: Dict[str, Dict[int, Tuple[str, ...]]] = {}
    fc1_names: Dict[str, Dict[int, str]] = {}
    all_names = set()

    for layer_name, experts in sorted(state_groups.items()):
        if not isinstance(experts, Mapping):
            raise TypeError(
                f"MoE 层 {layer_name!r} 的专家 state 分组必须是 Mapping。"
            )
        found_ids = sorted(int(expert_id) for expert_id in experts)
        if found_ids != expected_ids:
            raise ValueError(
                f"MoE 层 {layer_name!r} 的 expert id 不符合配置："
                f"found={found_ids}, expected={expected_ids}。"
            )

        layer_state_groups: Dict[int, Tuple[str, ...]] = {}
        layer_fc1_names: Dict[int, str] = {}
        for expert_id in expected_ids:
            names = tuple(str(name) for name in experts[expert_id])
            if not names:
                raise ValueError(
                    f"MoE 层 {layer_name!r} 的专家 {expert_id} 没有 state 条目。"
                )
            candidates = [
                name for name in names if name.endswith(".fc1.weight")
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"MoE 层 {layer_name!r} 的专家 {expert_id} 必须且只能有一个 "
                    f"fc1.weight，当前找到：{candidates}。"
                )
            layer_state_groups[expert_id] = names
            layer_fc1_names[expert_id] = candidates[0]
            all_names.update(names)

        normalized_state_groups[str(layer_name)] = layer_state_groups
        fc1_names[str(layer_name)] = layer_fc1_names

    expected_state_names = set(
        str(name)
        for name in getattr(optimization_groups, "expert_state_names", [])
    )
    if all_names != expected_state_names:
        missing = sorted(expected_state_names - all_names)
        extra = sorted(all_names - expected_state_names)
        raise ValueError(
            "逐层专家 state 分组与 expert_state_names 不一致；"
            f"missing={missing[:10]}, extra={extra[:10]}。"
        )

    return ExpertParameterLayout(
        expert_state_names_by_layer_and_id=normalized_state_groups,
        fc1_names_by_layer_and_id=fc1_names,
        expert_state_names=tuple(sorted(all_names)),
    )


def _validate_fc1_weight(
    weight: torch.Tensor,
    rank: int,
    parameter_name: str,
) -> None:
    """Validate the matrix used to derive a right-singular subspace."""
    if not torch.is_tensor(weight) or weight.ndim != 2:
        raise ValueError(
            f"{parameter_name} 必须是二维 Tensor，"
            f"当前 shape={getattr(weight, 'shape', None)}。"
        )
    if not weight.is_floating_point():
        raise TypeError(f"{parameter_name} 必须是浮点 Tensor。")
    if rank > min(weight.shape):
        raise ValueError(
            f"subspace_rank={rank} 超过 {parameter_name} 的可用秩上限 "
            f"{min(weight.shape)}，shape={tuple(weight.shape)}。"
        )
    if rank >= int(weight.shape[1]):
        raise ValueError(
            f"subspace_rank 必须小于 fc1.weight 的输入维度，"
            f"rank={rank}, input_dim={weight.shape[1]}。"
        )


def extract_right_singular_subspace(
    weight: torch.Tensor,
    rank: int,
    parameter_name: str = "fc1.weight",
) -> torch.Tensor:
    """Extract the top-r right-singular subspace and return a CPU float32 basis.

    For W in R^(d_ff x d_model), the returned basis is V[:, :rank] in
    R^(d_model x rank). The basis represents a Grassmann point; its choice of
    orthonormal coordinates is not treated as semantically meaningful.
    """
    _validate_fc1_weight(weight, rank, parameter_name)
    detached = weight.detach()
    if not bool(torch.isfinite(detached).all().item()):
        raise ValueError(f"{parameter_name} 包含 NaN 或 Inf。")

    work_dtype = (
        torch.float64 if detached.dtype == torch.float64 else torch.float32
    )
    with torch.no_grad():
        _, _, vh = torch.linalg.svd(
            detached.to(dtype=work_dtype),
            full_matrices=False,
        )
        basis = vh[:rank, :].transpose(0, 1).contiguous()

    return basis.to(device="cpu", dtype=torch.float32).clone()


def _validate_uploaded_basis(
    basis: Any,
    expected_shape: Tuple[int, int],
    client_id: int,
    parameter_name: str,
) -> torch.Tensor:
    """Validate a client-uploaded orthonormal basis before using it."""
    if not torch.is_tensor(basis):
        raise TypeError(
            f"客户端 {client_id} 上传的子空间 {parameter_name} 必须是 Tensor。"
        )
    if basis.ndim != 2 or tuple(basis.shape) != expected_shape:
        raise ValueError(
            f"客户端 {client_id} 的子空间 {parameter_name} shape 不匹配："
            f"expected={expected_shape}, actual={tuple(basis.shape)}。"
        )
    if not basis.is_floating_point():
        raise TypeError(
            f"客户端 {client_id} 的子空间 {parameter_name} 必须是浮点 Tensor。"
        )

    result = basis.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(
            f"客户端 {client_id} 的子空间 {parameter_name} 包含 NaN 或 Inf。"
        )

    rank = expected_shape[1]
    identity = torch.eye(rank, dtype=result.dtype)
    error = torch.linalg.matrix_norm(
        result.transpose(0, 1) @ result - identity,
        ord="fro",
    )
    if float(error.item()) > BASIS_ORTHOGONALITY_TOLERANCE:
        raise ValueError(
            f"客户端 {client_id} 上传的子空间基不满足正交性："
            f"name={parameter_name}, error={float(error.item()):.6e}。"
        )
    return result


def _normalized_grassmann_distance_squared(
    basis_a: torch.Tensor,
    basis_b: torch.Tensor,
) -> float:
    """Return d_G^2 / r using the principal angles between two subspaces."""
    cross_gram = basis_a.transpose(0, 1) @ basis_b
    singular_values = torch.linalg.svdvals(cross_gram).clamp(0.0, 1.0)
    principal_angles = torch.acos(singular_values)
    value = principal_angles.square().mean()
    return float(value.item())


def _normalized_subspace_overlap(
    basis_a: torch.Tensor,
    basis_b: torch.Tensor,
) -> float:
    """Return ||U^T V||_F^2 / r, a basis-invariant overlap in [0, 1]."""
    rank = int(basis_a.shape[1])
    overlap = (basis_a.transpose(0, 1) @ basis_b).square().sum() / rank
    return float(overlap.item())


def _softmax_client_weights(scores: Mapping[int, float]) -> Dict[int, float]:
    """Convert finite per-client log scores into normalized aggregation weights."""
    if not scores:
        raise ValueError("几何聚合分数不能为空。")
    if any(not math.isfinite(float(value)) for value in scores.values()):
        raise ValueError("几何聚合分数包含 NaN 或 Inf。")

    max_score = max(float(value) for value in scores.values())
    exponentials = {
        int(client_id): math.exp(float(score) - max_score)
        for client_id, score in scores.items()
    }
    return base.normalize_weights(exponentials)


def collect_dual_geometry_evidence(
    model: nn.Module,
    evidence_loader: Any,
    criterion: nn.Module,
    device: torch.device,
    cfg: Any,
) -> Mapping[str, Any]:
    """Collect each locally trained expert's fc1 right-singular subspace.

    The current method uses weight-space evidence only. The evidence loader,
    criterion, and device are accepted to match base.py's collector contract.
    """
    _ = evidence_loader, criterion, device
    settings = _read_runtime_settings(cfg)
    optimization_groups = base.build_federated_optimization_groups(
        model=model,
        expected_num_experts=settings["num_experts"],
        strict=True,
    )
    layout = _build_parameter_layout(
        optimization_groups,
        expected_num_experts=settings["num_experts"],
    )
    state = model.state_dict()

    subspaces: Dict[str, torch.Tensor] = {}
    for layer_name in sorted(layout.fc1_names_by_layer_and_id):
        for expert_id, parameter_name in sorted(
            layout.fc1_names_by_layer_and_id[layer_name].items()
        ):
            if parameter_name not in state:
                raise KeyError(
                    f"本地模型 state_dict 中不存在专家参数：{parameter_name}。"
                )
            subspaces[parameter_name] = extract_right_singular_subspace(
                weight=state[parameter_name],
                rank=settings["subspace_rank"],
                parameter_name=parameter_name,
            )

    return {
        "dual_geometry": {
            "version": EVIDENCE_VERSION,
            "subspaces": subspaces,
        }
    }


class DualGeometryUniformExpertAggregator(base.Aggregator):
    """Aggregate each expert using alignment and within-client diversity."""

    def __init__(
        self,
        cfg: Any,
        param_group_name: str = "expert",
    ) -> None:
        super().__init__(cfg=cfg, param_group_name=param_group_name)
        if param_group_name != "expert":
            raise ValueError(
                "DualGeometryUniformExpertAggregator 只能负责 expert 参数组。"
            )
        self.settings = _read_runtime_settings(cfg)
        self._layout: Optional[ExpertParameterLayout] = None

    @property
    def method_name(self) -> str:
        return ALGORITHM_NAME

    def compute_weights(
        self,
        client_updates: Sequence[base.ClientUpdate],
    ) -> Dict[int, float]:
        """Return a neutral uniform fallback for the base Aggregator contract.

        The actual aggregation path overrides aggregate() and computes a
        distinct client-weight distribution for every (layer, expert) pair.
        """
        return base.build_uniform_weights(client_updates)

    def _resolve_layout(
        self,
        model: Optional[nn.Module] = None,
    ) -> ExpertParameterLayout:
        if self._layout is not None:
            return self._layout

        if model is not None:
            optimization_groups = base.build_federated_optimization_groups(
                model=model,
                expected_num_experts=self.settings["num_experts"],
                strict=True,
            )
        elif (
            self.method_context is not None
            and self.method_context.optimization_groups is not None
        ):
            optimization_groups = self.method_context.optimization_groups
        elif (
            self.method_context is not None
            and self.method_context.global_model is not None
        ):
            optimization_groups = base.build_federated_optimization_groups(
                model=self.method_context.global_model,
                expected_num_experts=self.settings["num_experts"],
                strict=True,
            )
        else:
            raise RuntimeError(
                "无法解析专家参数布局；请先注入 MethodContext 或调用 initialize_model()。"
            )

        self._layout = _build_parameter_layout(
            optimization_groups,
            expected_num_experts=self.settings["num_experts"],
        )
        return self._layout

    def initialize_model(self, model: nn.Module) -> None:
        """Validate the model layout and subspace rank without modifying weights."""
        layout = self._resolve_layout(model)
        state = model.state_dict()
        for layer_name in sorted(layout.fc1_names_by_layer_and_id):
            for expert_id, parameter_name in sorted(
                layout.fc1_names_by_layer_and_id[layer_name].items()
            ):
                if parameter_name not in state:
                    raise KeyError(
                        f"全局模型 state_dict 中不存在专家参数：{parameter_name}。"
                    )
                _validate_fc1_weight(
                    state[parameter_name],
                    self.settings["subspace_rank"],
                    parameter_name,
                )

    def _collect_client_subspaces(
        self,
        client_updates: Sequence[base.ClientUpdate],
        layout: ExpertParameterLayout,
        global_state: Mapping[str, torch.Tensor],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Validate and materialize all client subspaces for the current round."""
        expected_names = {
            name
            for layer in layout.fc1_names_by_layer_and_id.values()
            for name in layer.values()
        }
        subspaces_by_client: Dict[int, Dict[str, torch.Tensor]] = {}

        for update in client_updates:
            client_id = int(update.client_id)
            extra = update.extra
            if not isinstance(extra, Mapping):
                raise TypeError(
                    f"客户端 {client_id} 的 extra 必须是 Mapping。"
                )
            evidence = extra.get("dual_geometry")
            if not isinstance(evidence, Mapping):
                raise ValueError(
                    f"客户端 {client_id} 缺少 dual_geometry 子空间证据。"
                )
            if evidence.get("version") != EVIDENCE_VERSION:
                raise ValueError(
                    f"客户端 {client_id} 的 dual_geometry evidence 版本不受支持："
                    f"{evidence.get('version')!r}。"
                )
            client_subspaces = evidence.get("subspaces")
            if not isinstance(client_subspaces, Mapping):
                raise TypeError(
                    f"客户端 {client_id} 的 dual_geometry.subspaces 必须是 Mapping。"
                )

            found_names = set(client_subspaces)
            if found_names != expected_names:
                missing = sorted(expected_names - found_names)
                extra_names = sorted(found_names - expected_names)
                raise ValueError(
                    f"客户端 {client_id} 上传的子空间参数集合不匹配；"
                    f"missing={missing[:10]}, extra={extra_names[:10]}。"
                )

            validated: Dict[str, torch.Tensor] = {}
            for parameter_name in sorted(expected_names):
                global_weight = global_state.get(parameter_name)
                if global_weight is None:
                    raise KeyError(
                        f"全局 state_dict 中不存在专家参数：{parameter_name}。"
                    )
                _validate_fc1_weight(
                    global_weight,
                    self.settings["subspace_rank"],
                    parameter_name,
                )
                expected_shape = (
                    int(global_weight.shape[1]),
                    self.settings["subspace_rank"],
                )
                validated[parameter_name] = _validate_uploaded_basis(
                    basis=client_subspaces[parameter_name],
                    expected_shape=expected_shape,
                    client_id=client_id,
                    parameter_name=parameter_name,
                )
            subspaces_by_client[client_id] = validated

        return subspaces_by_client

    def _compute_geometry_weights(
        self,
        client_updates: Sequence[base.ClientUpdate],
        layout: ExpertParameterLayout,
        global_state: Mapping[str, torch.Tensor],
        subspaces_by_client: Mapping[int, Mapping[str, torch.Tensor]],
    ) -> Tuple[
        Dict[Tuple[str, int], Dict[int, float]],
        Dict[str, Any],
        Dict[str, Any],
    ]:
        """Compute one normalized client-weight distribution per expert."""
        rank = self.settings["subspace_rank"]
        beta = self.settings["align_beta"]
        gamma = self.settings["diversity_gamma"]
        overlap_threshold = self.settings["overlap_threshold"]

        global_bases: Dict[str, torch.Tensor] = {}
        for layer_name in sorted(layout.fc1_names_by_layer_and_id):
            for expert_id, parameter_name in sorted(
                layout.fc1_names_by_layer_and_id[layer_name].items()
            ):
                global_bases[parameter_name] = extract_right_singular_subspace(
                    weight=global_state[parameter_name],
                    rank=rank,
                    parameter_name=parameter_name,
                )

        weights_by_expert: Dict[Tuple[str, int], Dict[int, float]] = {}
        weights_diagnostics: Dict[str, Any] = {}
        geometry_diagnostics: Dict[str, Any] = {}
        all_alignment_distances = []
        all_diversity_penalties = []
        all_internal_overlaps = []

        client_ids = [int(update.client_id) for update in client_updates]

        for layer_name in sorted(layout.fc1_names_by_layer_and_id):
            expert_ids = sorted(
                layout.fc1_names_by_layer_and_id[layer_name]
            )
            layer_weights: Dict[str, Any] = {}
            layer_geometry: Dict[str, Any] = {}

            for expert_id in expert_ids:
                parameter_name = layout.fc1_names_by_layer_and_id[
                    layer_name
                ][expert_id]
                scores: Dict[int, float] = {}
                distances: Dict[int, float] = {}
                penalties: Dict[int, float] = {}
                overlaps_for_expert: Dict[int, float] = {}

                for client_id in client_ids:
                    local_experts = {
                        other_id: subspaces_by_client[client_id][other_name]
                        for other_id, other_name in (
                            layout.fc1_names_by_layer_and_id[layer_name].items()
                        )
                    }
                    local_basis = local_experts[expert_id]
                    distance = _normalized_grassmann_distance_squared(
                        global_bases[parameter_name],
                        local_basis,
                    )

                    pair_overlaps = [
                        _normalized_subspace_overlap(
                            local_basis,
                            local_experts[other_id],
                        )
                        for other_id in expert_ids
                        if other_id != expert_id
                    ]
                    if not pair_overlaps:
                        raise ValueError(
                            f"MoE 层 {layer_name!r} 缺少用于多样性计算的其他专家。"
                        )
                    diversity_penalty = sum(
                        max(0.0, overlap - overlap_threshold) ** 2
                        for overlap in pair_overlaps
                    ) / len(pair_overlaps)
                    mean_overlap = sum(pair_overlaps) / len(pair_overlaps)

                    score = -beta * distance - gamma * diversity_penalty
                    scores[client_id] = score
                    distances[client_id] = distance
                    penalties[client_id] = diversity_penalty
                    overlaps_for_expert[client_id] = mean_overlap

                expert_weights = _softmax_client_weights(scores)
                weights_by_expert[(layer_name, expert_id)] = expert_weights

                mean_distance = sum(distances.values()) / len(distances)
                mean_penalty = sum(penalties.values()) / len(penalties)
                mean_overlap = (
                    sum(overlaps_for_expert.values())
                    / len(overlaps_for_expert)
                )
                mean_alignment_factor = sum(
                    math.exp(-beta * value) for value in distances.values()
                ) / len(distances)
                mean_diversity_factor = sum(
                    math.exp(-gamma * value) for value in penalties.values()
                ) / len(penalties)

                all_alignment_distances.extend(distances.values())
                all_diversity_penalties.extend(penalties.values())
                all_internal_overlaps.extend(overlaps_for_expert.values())

                layer_weights[str(expert_id)] = {
                    str(client_id): float(weight)
                    for client_id, weight in sorted(expert_weights.items())
                }
                layer_geometry[str(expert_id)] = {
                    "mean_normalized_grassmann_distance_squared": mean_distance,
                    "mean_internal_normalized_overlap": mean_overlap,
                    "mean_diversity_penalty": mean_penalty,
                    "mean_alignment_factor": mean_alignment_factor,
                    "mean_diversity_factor": mean_diversity_factor,
                }

            weights_diagnostics[layer_name] = layer_weights
            geometry_diagnostics[layer_name] = layer_geometry

        summary = {
            "mean_normalized_grassmann_distance_squared": (
                sum(all_alignment_distances) / len(all_alignment_distances)
                if all_alignment_distances
                else 0.0
            ),
            "mean_internal_normalized_overlap": (
                sum(all_internal_overlaps) / len(all_internal_overlaps)
                if all_internal_overlaps
                else 0.0
            ),
            "mean_diversity_penalty": (
                sum(all_diversity_penalties) / len(all_diversity_penalties)
                if all_diversity_penalties
                else 0.0
            ),
        }
        return weights_by_expert, weights_diagnostics, {
            "summary": summary,
            "by_layer_and_expert": geometry_diagnostics,
        }

    def aggregate(
        self,
        global_state: Mapping[str, torch.Tensor],
        client_updates: Sequence[base.ClientUpdate],
        param_names: Optional[Sequence[str]] = None,
        base_state: Optional[Mapping[str, torch.Tensor]] = None,
        strict: bool = True,
    ) -> base.AggregationResult:
        """Apply geometry-derived client weights independently to each expert."""
        self._validate_client_updates(client_updates)
        if not client_updates:
            raise ValueError("client_updates 不能为空。")

        round_ids = {int(update.round_id) for update in client_updates}
        if len(round_ids) != 1:
            raise ValueError(
                f"客户端更新轮数不一致：{sorted(round_ids)}。"
            )

        layout = self._resolve_layout()
        expected_names = set(layout.expert_state_names)
        selected_names = (
            expected_names if param_names is None else set(param_names)
        )
        if selected_names != expected_names:
            missing = sorted(expected_names - selected_names)
            extra = sorted(selected_names - expected_names)
            raise ValueError(
                "服务端传入的 expert param_names 与算法专家分组不一致；"
                f"missing={missing[:10]}, extra={extra[:10]}。"
            )

        for name in expected_names:
            if name not in global_state:
                raise KeyError(f"全局模型状态中不存在专家参数：{name}。")

        subspaces_by_client = self._collect_client_subspaces(
            client_updates=client_updates,
            layout=layout,
            global_state=global_state,
        )
        (
            weights_by_expert,
            weights_diagnostics,
            geometry_diagnostics,
        ) = self._compute_geometry_weights(
            client_updates=client_updates,
            layout=layout,
            global_state=global_state,
            subspaces_by_client=subspaces_by_client,
        )

        result_state = base.clone_state_dict(
            global_state if base_state is None else base_state
        )
        for layer_name in sorted(layout.expert_state_names_by_layer_and_id):
            for expert_id, names in sorted(
                layout.expert_state_names_by_layer_and_id[layer_name].items()
            ):
                result_state = base.apply_weighted_delta(
                    global_state=global_state,
                    client_updates=client_updates,
                    weights=weights_by_expert[(layer_name, expert_id)],
                    param_names=names,
                    base_state=result_state,
                    strict=strict,
                )

        base.validate_state_dict_shapes(
            reference_state=global_state,
            candidate_state=result_state,
            context="dual_geometry_uniform 专家聚合结果",
        )
        base.check_finite_state_dict(
            state_dict=result_state,
            param_names=layout.expert_state_names,
        )

        # AggregationResult.weights is a client-level compatibility summary.
        # The exact weights used by each expert are recorded separately below.
        num_expert_groups = len(weights_by_expert)
        mean_client_weights = {
            int(update.client_id): (
                sum(
                    weights[int(update.client_id)]
                    for weights in weights_by_expert.values()
                )
                / num_expert_groups
            )
            for update in client_updates
        }

        diagnostics = {
            "method": self.method_name,
            "param_group": self.param_group_name,
            "num_clients": len(client_updates),
            "num_state_entries": len(expected_names),
            "num_moe_layers": len(layout.fc1_names_by_layer_and_id),
            "num_experts_per_layer": self.settings["num_experts"],
            "subspace_rank": self.settings["subspace_rank"],
            "align_beta": self.settings["align_beta"],
            "diversity_gamma": self.settings["diversity_gamma"],
            "overlap_threshold": self.settings["overlap_threshold"],
            "aggregation": "per_expert_dual_geometry_weighted_delta",
            "client_weight_summary": (
                "mean of the per-expert client weights; exact values are in "
                "per_expert_client_weights"
            ),
            "per_expert_client_weights": weights_diagnostics,
            "geometry": geometry_diagnostics,
            **geometry_diagnostics["summary"],
        }

        return base.AggregationResult(
            new_state_dict=result_state,
            weights=mean_client_weights,
            diagnostics=diagnostics,
        )

    def build_post_aggregate_diagnostics(
        self,
        model: nn.Module,
    ) -> Dict[str, Any]:
        """Measure global expert diversity after each aggregation round."""
        layout = self._resolve_layout(model)
        state = model.state_dict()
        rank = self.settings["subspace_rank"]
        pair_overlaps = []
        pair_distances = []

        for layer_name in sorted(layout.fc1_names_by_layer_and_id):
            bases = {}
            for expert_id, parameter_name in sorted(
                layout.fc1_names_by_layer_and_id[layer_name].items()
            ):
                bases[expert_id] = extract_right_singular_subspace(
                    weight=state[parameter_name],
                    rank=rank,
                    parameter_name=parameter_name,
                )

            expert_ids = sorted(bases)
            for left_index, left_id in enumerate(expert_ids):
                for right_id in expert_ids[left_index + 1:]:
                    pair_overlaps.append(
                        _normalized_subspace_overlap(
                            bases[left_id],
                            bases[right_id],
                        )
                    )
                    distance_squared = _normalized_grassmann_distance_squared(
                        bases[left_id],
                        bases[right_id],
                    )
                    pair_distances.append(math.sqrt(distance_squared))

        if not pair_overlaps:
            return {}

        return {
            "post_mean_internal_normalized_overlap": (
                sum(pair_overlaps) / len(pair_overlaps)
            ),
            "post_max_internal_normalized_overlap": max(pair_overlaps),
            "post_mean_normalized_grassmann_distance": (
                sum(pair_distances) / len(pair_distances)
            ),
            "post_min_normalized_grassmann_distance": min(pair_distances),
            "post_num_expert_pairs": len(pair_overlaps),
        }


def build_expert_aggregator(cfg: Any) -> base.Aggregator:
    """Build the dual-geometry expert-only aggregator."""
    return DualGeometryUniformExpertAggregator(
        cfg=cfg,
        param_group_name="expert",
    )


def register_method_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Register dual-geometry-specific command-line arguments."""
    parser.add_argument("--subspace-rank", type=int, default=None)
    parser.add_argument("--align-beta", type=float, default=None)
    parser.add_argument("--diversity-gamma", type=float, default=None)
    parser.add_argument("--overlap-threshold", type=float, default=None)


def build_method_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert algorithm-specific CLI arguments into method_config overrides."""
    method_overrides: Dict[str, Any] = {}
    if args.subspace_rank is not None:
        method_overrides["subspace_rank"] = args.subspace_rank
    if args.align_beta is not None:
        method_overrides["align_beta"] = args.align_beta
    if args.diversity_gamma is not None:
        method_overrides["diversity_gamma"] = args.diversity_gamma
    if args.overlap_threshold is not None:
        method_overrides["overlap_threshold"] = args.overlap_threshold
    return {"method_config": method_overrides} if method_overrides else {}


def main(model_builder: Optional[base.ModelBuilder] = None) -> int:
    """Run the dual-geometry federated experiment."""
    return base.main(
        expert_aggregator_builder=build_expert_aggregator,
        embedded_method_config=EMBEDDED_METHOD_CONFIG,
        expert_method_name=ALGORITHM_NAME,
        method_config_defaults=METHOD_CONFIG_DEFAULTS,
        method_config_validator=validate_method_config,
        expert_evidence_collector=collect_dual_geometry_evidence,
        method_cli_argument_registrar=register_method_cli_arguments,
        method_cli_overrides_builder=build_method_cli_overrides,
        model_builder=model_builder,
    )


if __name__ == "__main__":
    raise SystemExit(main())

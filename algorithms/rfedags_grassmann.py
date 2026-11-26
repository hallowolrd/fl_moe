"""RFedAGS expert aggregation with Stiefel-Grassmann subspace regularization.

The first-layer weight of every Switch expert is used directly as its
orthonormal subspace basis. Client-side optimization follows a product
manifold: Stiefel Riemannian SGD for fc1.weight and Euclidean SGD for the
remaining expert parameters. The server aggregates the complete expert
parameter group's gradient streams with RFedAGS; non-expert parameters remain
under the framework's fixed uniform aggregation policy.
"""

from __future__ import annotations

# Import base before torch so the shared module can prepare deterministic
# CUDA environment variables before PyTorch initializes CUDA/cuBLAS.
import base

import argparse
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ALGORITHM_NAME = "rfedags_grassmann"
PAYLOAD_VERSION = 1
ORTHOGONALITY_TOLERANCE = 5e-4

METHOD_CONFIG_DEFAULTS = {
    "method_config": {
        # Scale-normalized Grassmann overlap regularization.
        "lambda_div": 5.0,
        "rho0": 0.24,
        # RFedAGS global step size (varpi).
        "server_lr": 1.0,
    },
}

EMBEDDED_METHOD_CONFIG = {
    "agg": {
        "non_expert": {"method": "uniform"},
        "expert": {"method": ALGORITHM_NAME},
    },
}


@dataclass(frozen=True)
class ExpertParameterLayout:
    """Names and per-layer grouping of the trainable expert parameters."""

    expert_parameter_names: Tuple[str, ...]
    stiefel_parameter_names: Tuple[str, ...]
    stiefel_names_by_layer: Mapping[str, Tuple[str, ...]]


def _read_nested_config(cfg: Any, path: str, default: Any = None) -> Any:
    """Read a dotted path from either ConfigNode or a nested mapping."""
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


_MISSING = object()


def _read_mapping_value(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    if hasattr(mapping, "get"):
        try:
            return mapping.get(key, default)
        except TypeError:
            pass
    if hasattr(mapping, key):
        return getattr(mapping, key)
    return default


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是有限实数，当前值：{value!r}。")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限实数，当前值：{value!r}。")
    return result


def validate_method_config(config: Mapping[str, Any]) -> None:
    """Validate RFedAGS-Grassmann settings from the complete resolved config.

    base.load_config() and base.load_embedded_config() pass the complete
    configuration mapping to method_validator. Direct callers may also pass
    the method_config mapping itself, which is supported for unit testing.
    """
    if not isinstance(config, Mapping):
        raise ValueError("配置必须是 Mapping。")
    method_config = config.get("method_config", config)
    if not isinstance(method_config, Mapping):
        raise ValueError("method_config 必须是 Mapping。")

    lambda_div = _finite_real(
        _read_mapping_value(
            method_config,
            "lambda_div",
            METHOD_CONFIG_DEFAULTS["method_config"]["lambda_div"],
        ),
        "method_config.lambda_div",
    )
    if lambda_div < 0.0:
        raise ValueError("method_config.lambda_div 必须大于或等于 0。")

    rho0 = _finite_real(
        _read_mapping_value(
            method_config,
            "rho0",
            METHOD_CONFIG_DEFAULTS["method_config"]["rho0"],
        ),
        "method_config.rho0",
    )
    if not 0.0 <= rho0 < 1.0:
        raise ValueError("method_config.rho0 必须满足 0 <= rho0 < 1。")

    server_lr = _finite_real(
        _read_mapping_value(
            method_config,
            "server_lr",
            METHOD_CONFIG_DEFAULTS["method_config"]["server_lr"],
        ),
        "method_config.server_lr",
    )
    if server_lr <= 0.0:
        raise ValueError("method_config.server_lr 必须大于 0。")

def _validate_runtime_config(cfg: Any) -> Dict[str, Any]:
    """Validate the RFedAGS assumptions and return resolved scalar settings."""
    optimizer_type = str(
        _read_nested_config(cfg, "optimizer.type", "sgd")
    ).strip().lower()
    if optimizer_type != "sgd":
        raise ValueError("RFedAGS-Grassmann 要求 optimizer.type='sgd'。")

    momentum = _finite_real(
        _read_nested_config(cfg, "optimizer.momentum", 0.0),
        "optimizer.momentum",
    )
    if momentum != 0.0:
        raise ValueError("RFedAGS-Grassmann 要求 optimizer.momentum=0。")

    weight_decay = _finite_real(
        _read_nested_config(cfg, "optimizer.weight_decay", 0.0),
        "optimizer.weight_decay",
    )
    if weight_decay != 0.0:
        raise ValueError("RFedAGS-Grassmann 要求 optimizer.weight_decay=0。")

    nesterov = _read_nested_config(cfg, "optimizer.nesterov", False)
    if not isinstance(nesterov, bool) or nesterov:
        raise ValueError("RFedAGS-Grassmann 要求 optimizer.nesterov=False。")

    grad_clip = _read_nested_config(cfg, "optimizer.grad_clip", None)
    if grad_clip is not None and _finite_real(
        grad_clip,
        "optimizer.grad_clip",
    ) > 0.0:
        raise ValueError(
            "RFedAGS-Grassmann 首版不启用梯度裁剪；"
            "请将 optimizer.grad_clip 设置为 null 或 0。"
        )

    local_lr = _finite_real(
        _read_nested_config(cfg, "optimizer.lr", 0.01),
        "optimizer.lr",
    )
    if local_lr <= 0.0:
        raise ValueError("optimizer.lr 必须大于 0。")

    local_steps = _read_nested_config(cfg, "local_steps", None)
    if (
        isinstance(local_steps, bool)
        or not isinstance(local_steps, int)
        or local_steps <= 0
    ):
        raise ValueError(
            "RFedAGS-Grassmann 要求设置正整数 local_steps，"
            "以保证所有客户端执行相同数量的本地更新。"
        )

    frac = _finite_real(
        _read_nested_config(cfg, "frac", 1.0),
        "frac",
    )
    if frac != 1.0:
        raise ValueError(
            "RFedAGS-Grassmann 首版要求全客户端参与，必须设置 frac=1.0。"
        )

    num_experts = _read_nested_config(cfg, "num_experts", 2)
    if (
        isinstance(num_experts, bool)
        or not isinstance(num_experts, int)
        or num_experts < 2
    ):
        raise ValueError("RFedAGS-Grassmann 要求 num_experts 至少为 2。")

    method_config = _read_nested_config(cfg, "method_config", {})
    if hasattr(method_config, "to_dict"):
        method_config = method_config.to_dict()
    if not isinstance(method_config, Mapping):
        raise ValueError("method_config 必须是 Mapping。")
    validate_method_config(method_config)

    lambda_div = _finite_real(
        _read_mapping_value(
            method_config,
            "lambda_div",
            METHOD_CONFIG_DEFAULTS["method_config"]["lambda_div"],
        ),
        "method_config.lambda_div",
    )
    rho0 = _finite_real(
        _read_mapping_value(
            method_config,
            "rho0",
            METHOD_CONFIG_DEFAULTS["method_config"]["rho0"],
        ),
        "method_config.rho0",
    )
    server_lr = _finite_real(
        _read_mapping_value(
            method_config,
            "server_lr",
            METHOD_CONFIG_DEFAULTS["method_config"]["server_lr"],
        ),
        "method_config.server_lr",
    )

    return {
        "local_lr": local_lr,
        "local_steps": int(local_steps),
        "num_experts": int(num_experts),
        "lambda_div": lambda_div,
        "rho0": rho0,
        "server_lr": server_lr,
    }


def qr_retraction(matrix: torch.Tensor) -> torch.Tensor:
    """Return the column-orthonormal factor of a reduced QR decomposition."""
    if not torch.is_tensor(matrix) or matrix.ndim != 2:
        raise ValueError("qr_retraction 输入必须是二维 Tensor。")
    if not matrix.is_floating_point():
        raise TypeError("qr_retraction 只支持浮点 Tensor。")
    rows, columns = matrix.shape
    if rows < columns:
        raise ValueError(
            "列正交 Stiefel 参数要求矩阵行数不少于列数，"
            f"当前 shape={tuple(matrix.shape)}。"
        )

    work_dtype = torch.float64 if matrix.dtype == torch.float64 else torch.float32
    q, r = torch.linalg.qr(matrix.to(dtype=work_dtype), mode="reduced")
    diagonal = torch.diagonal(r, offset=0)
    signs = torch.where(
        diagonal < 0.0,
        -torch.ones_like(diagonal),
        torch.ones_like(diagonal),
    )
    q = q * signs.unsqueeze(0)
    return q.to(dtype=matrix.dtype)


def stiefel_tangent_projection(
    point: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """Project an ambient matrix onto the embedded Stiefel tangent space."""
    if point.ndim != 2 or vector.shape != point.shape:
        raise ValueError("Stiefel 点和切向量必须是形状相同的二维 Tensor。")
    sym_part = 0.5 * (
        point.transpose(0, 1) @ vector
        + vector.transpose(0, 1) @ point
    )
    return vector - point @ sym_part


def stiefel_riemannian_gradient(
    point: torch.Tensor,
    euclidean_gradient: torch.Tensor,
) -> torch.Tensor:
    """Compute the Stiefel Riemannian gradient for the Frobenius metric."""
    return stiefel_tangent_projection(point, euclidean_gradient)


def grassmann_overlap(
    basis_a: torch.Tensor,
    basis_b: torch.Tensor,
) -> torch.Tensor:
    """Return the squared Frobenius overlap of two equal-rank subspaces."""
    if basis_a.ndim != 2 or basis_b.ndim != 2:
        raise ValueError("Grassmann 子空间基必须是二维 Tensor。")
    if basis_a.shape != basis_b.shape:
        raise ValueError(
            "Grassmann 子空间基必须具有相同 shape，"
            f"当前为 {tuple(basis_a.shape)} 和 {tuple(basis_b.shape)}。"
        )
    if basis_a.shape[0] < basis_a.shape[1]:
        raise ValueError("列正交子空间基要求行数不少于列数。")
    cross_gram = basis_a.transpose(0, 1) @ basis_b
    return cross_gram.square().sum()


def grassmann_projection_distance(
    basis_a: torch.Tensor,
    basis_b: torch.Tensor,
) -> torch.Tensor:
    """Return the Grassmann projection distance used by GrMoE."""
    overlap = grassmann_overlap(basis_a, basis_b)
    rank = int(basis_a.shape[1])
    return torch.sqrt(
        torch.clamp(overlap.new_tensor(float(rank)) - overlap, min=0.0)
    )


def compute_subspace_overlap_loss(
    weights_by_layer: Mapping[str, Sequence[torch.Tensor]],
    rho0: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the rank-normalized, layer-balanced Grassmann hinge penalty.

    For each expert pair in layer l, the normalized overlap is

        s_ij = ||U_i^T U_j||_F^2 / r_l.

    The loss averages the hinge penalties over expert pairs within each
    layer, then averages those layer losses. This preserves the GrMoE hinge
    form while making the regularizer scale independent of subspace rank,
    expert-pair count per layer, and the number of MoE layers.

    Active-overlap ratio is reported as the fraction of all expert pairs
    whose normalized overlap is strictly greater than rho0.
    """
    rho0_value = _finite_real(rho0, "rho0")
    if not 0.0 <= rho0_value < 1.0:
        raise ValueError("rho0 必须满足 0 <= rho0 < 1。")
    if not weights_by_layer:
        raise ValueError("weights_by_layer 不能为空。")

    layer_losses = []
    layer_mean_overlaps = []
    layer_mean_distances = []
    normalized_overlaps = []
    projection_distances = []
    active_pair_flags = []

    for layer_name in sorted(weights_by_layer):
        layer_weights = list(weights_by_layer[layer_name])
        if len(layer_weights) < 2:
            raise ValueError(
                f"MoE 层 {layer_name!r} 至少需要两个专家才能计算重叠正则。"
            )
        reference_shape = tuple(layer_weights[0].shape)
        if (
            len(reference_shape) != 2
            or reference_shape[0] < reference_shape[1]
            or reference_shape[1] <= 0
        ):
            raise ValueError(
                f"MoE 层 {layer_name!r} 的 fc1.weight 不满足列正交 Stiefel 形状："
                f"{reference_shape}。"
            )
        rank = int(reference_shape[1])

        for expert_index, weight in enumerate(layer_weights):
            if tuple(weight.shape) != reference_shape:
                raise ValueError(
                    f"MoE 层 {layer_name!r} 的专家 fc1.weight shape 不一致："
                    f"expert_index={expert_index}, shape={tuple(weight.shape)}, "
                    f"expected={reference_shape}。"
                )

        layer_penalties = []
        layer_overlaps = []
        layer_distances = []
        for left_index in range(len(layer_weights)):
            for right_index in range(left_index + 1, len(layer_weights)):
                overlap = grassmann_overlap(
                    layer_weights[left_index],
                    layer_weights[right_index],
                )
                normalized_overlap = overlap / float(rank)
                layer_penalties.append(
                    F.relu(normalized_overlap - rho0_value)
                )
                layer_overlaps.append(normalized_overlap)
                layer_distances.append(
                    torch.sqrt(
                        torch.clamp(
                            overlap.new_tensor(float(rank)) - overlap,
                            min=0.0,
                        )
                    )
                )
                normalized_overlaps.append(normalized_overlap)
                projection_distances.append(layer_distances[-1])
                active_pair_flags.append(
                    (normalized_overlap.detach() > rho0_value).to(
                        dtype=torch.float32
                    )
                )

        layer_losses.append(torch.stack(layer_penalties).mean())
        layer_mean_overlaps.append(torch.stack(layer_overlaps).mean())
        layer_mean_distances.append(torch.stack(layer_distances).mean())

    loss = torch.stack(layer_losses).mean()
    overlap_values = torch.stack(normalized_overlaps).detach()
    distance_values = torch.stack(projection_distances).detach()
    active_values = torch.stack(active_pair_flags).detach()
    num_pairs = int(active_values.numel())
    num_active_pairs = int(active_values.sum().cpu().item())
    diagnostics = {
        "mean_normalized_overlap": float(
            torch.stack(layer_mean_overlaps).mean().cpu().item()
        ),
        "max_normalized_overlap": float(overlap_values.max().cpu().item()),
        "mean_projection_distance": float(
            torch.stack(layer_mean_distances).mean().cpu().item()
        ),
        "min_projection_distance": float(distance_values.min().cpu().item()),
        "active_overlap_ratio": num_active_pairs / num_pairs,
        "num_active_pairs": num_active_pairs,
        "num_pairs": num_pairs,
    }
    return loss, diagnostics


def orthogonality_error(matrix: torch.Tensor) -> torch.Tensor:
    """Return ||W^T W - I||_F for a column-Stiefel matrix."""
    if matrix.ndim != 2 or matrix.shape[0] < matrix.shape[1]:
        raise ValueError("orthogonality_error 输入不是合法的列正交矩阵形状。")
    identity = torch.eye(
        matrix.shape[1],
        device=matrix.device,
        dtype=matrix.dtype,
    )
    return torch.linalg.matrix_norm(
        matrix.transpose(0, 1) @ matrix - identity,
        ord="fro",
    )


def _build_parameter_layout(optimization_groups: Any) -> ExpertParameterLayout:
    """Resolve all expert parameters and each expert's first-layer weight."""
    expert_parameter_names = tuple(
        str(name) for name in optimization_groups.expert_parameters
    )
    by_layer_and_id = optimization_groups.expert_parameters_by_layer_and_id
    if not expert_parameter_names or not by_layer_and_id:
        raise ValueError("没有找到可训练专家参数或逐层专家分组。")

    expert_name_set = set(expert_parameter_names)
    mapped_names = set()
    stiefel_names_by_layer: Dict[str, Tuple[str, ...]] = {}

    for layer_name, experts in sorted(by_layer_and_id.items()):
        if not isinstance(experts, Mapping) or len(experts) < 2:
            raise ValueError(
                f"MoE 层 {layer_name!r} 至少需要两个可训练专家。"
            )
        layer_stiefel_names = []
        for expert_id, names in sorted(experts.items()):
            names = [str(name) for name in names]
            mapped_names.update(names)
            candidates = [name for name in names if name.endswith(".fc1.weight")]
            if len(candidates) != 1:
                raise ValueError(
                    f"MoE 层 {layer_name!r} 的专家 {expert_id} 必须且只能有一个 "
                    f"fc1.weight，当前找到：{candidates}。"
                )
            layer_stiefel_names.append(candidates[0])
        stiefel_names_by_layer[str(layer_name)] = tuple(layer_stiefel_names)

    if mapped_names != expert_name_set:
        missing = sorted(expert_name_set - mapped_names)
        extra = sorted(mapped_names - expert_name_set)
        raise ValueError(
            "逐层专家参数分组与 expert_parameters 不一致；"
            f"missing={missing[:10]}, extra={extra[:10]}。"
        )

    stiefel_names = tuple(
        name
        for layer_name in sorted(stiefel_names_by_layer)
        for name in stiefel_names_by_layer[layer_name]
    )
    if not stiefel_names:
        raise ValueError("没有找到任何专家 fc1.weight Stiefel 参数。")

    return ExpertParameterLayout(
        expert_parameter_names=expert_parameter_names,
        stiefel_parameter_names=stiefel_names,
        stiefel_names_by_layer=stiefel_names_by_layer,
    )


def _layer_weights_from_parameters(
    parameter_map: Mapping[str, torch.Tensor],
    layout: ExpertParameterLayout,
) -> Dict[str, Tuple[torch.Tensor, ...]]:
    return {
        layer_name: tuple(parameter_map[name] for name in names)
        for layer_name, names in layout.stiefel_names_by_layer.items()
    }


def _resolve_optimization_groups(
    model: nn.Module,
    method_context: Optional[base.MethodContext],
) -> Any:
    if (
        method_context is not None
        and method_context.optimization_groups is not None
    ):
        return method_context.optimization_groups
    return base.build_federated_optimization_groups(
        model=model,
        strict=True,
    )


def _compute_model_subspace_diagnostics(
    model: nn.Module,
    layout: ExpertParameterLayout,
    rho0: float,
) -> Dict[str, float]:
    parameter_map = dict(model.named_parameters())
    weights_by_layer = _layer_weights_from_parameters(parameter_map, layout)
    with torch.no_grad():
        _, overlap_metrics = compute_subspace_overlap_loss(weights_by_layer, rho0)
        errors = [
            float(orthogonality_error(parameter_map[name]).detach().cpu().item())
            for name in layout.stiefel_parameter_names
        ]
    return {
        **overlap_metrics,
        "max_orthogonality_error": max(errors),
        "mean_orthogonality_error": sum(errors) / len(errors),
    }


class RFedAGSGrassmannClientStrategy(base.ExpertClientTrainingStrategy):
    """Client-side product-manifold SGD and RFedAGS gradient-stream builder."""

    def __init__(self, cfg: Any, client_id: int) -> None:
        self.cfg = cfg
        self.client_id = int(client_id)
        self.settings = _validate_runtime_config(cfg)
        self._layout: Optional[ExpertParameterLayout] = None
        self._parameters: Dict[str, nn.Parameter] = {}
        self._anchors: Dict[str, torch.Tensor] = {}
        self._gradient_streams: Dict[str, torch.Tensor] = {}
        self._train_context: Optional[base.ExpertClientTrainContext] = None
        self._local_step_count = 0
        self._diversity_loss_sum = 0.0
        self._weighted_diversity_loss_sum = 0.0
        self._overlap_sum = 0.0
        self._max_overlap = 0.0
        self._distance_sum = 0.0
        self._active_overlap_ratio_sum = 0.0

    @property
    def strategy_name(self) -> str:
        return ALGORITHM_NAME

    @property
    def includes_expert_parameters_in_model_delta(self) -> bool:
        return False

    def on_train_start(
        self,
        context: base.ExpertClientTrainContext,
    ) -> None:
        if int(context.client_id) != self.client_id:
            raise ValueError(
                f"训练策略 client_id={self.client_id} 与上下文 client_id="
                f"{context.client_id} 不一致。"
            )
        if context.local_steps != self.settings["local_steps"]:
            raise ValueError(
                "训练上下文 local_steps 与 RFedAGS 配置不一致："
                f"context={context.local_steps}, "
                f"configured={self.settings['local_steps']}。"
            )

        self._layout = _build_parameter_layout(context.optimization_groups)
        self._parameters = dict(context.expert_parameters)
        if set(self._parameters) != set(self._layout.expert_parameter_names):
            raise ValueError("训练上下文中的专家参数与优化分组不一致。")

        self._anchors = {}
        for name in self._layout.stiefel_parameter_names:
            if name not in context.global_state:
                raise KeyError(f"全局状态中缺少 Stiefel 参数：{name}。")
            parameter = self._parameters[name]
            anchor = context.global_state[name].to(
                device=parameter.device,
                dtype=parameter.dtype,
            ).clone()
            error = float(orthogonality_error(anchor).detach().cpu().item())
            if error > ORTHOGONALITY_TOLERANCE:
                raise ValueError(
                    f"全局参数 {name} 不满足 Stiefel 约束，"
                    f"orthogonality_error={error:.6e}。"
                    "请检查服务端 initialize_model() 是否执行。"
                )
            self._anchors[name] = anchor

        self._gradient_streams = {
            name: torch.zeros_like(parameter, memory_format=torch.preserve_format)
            for name, parameter in self._parameters.items()
        }
        self._train_context = context
        self._local_step_count = 0
        self._diversity_loss_sum = 0.0
        self._weighted_diversity_loss_sum = 0.0
        self._overlap_sum = 0.0
        self._max_overlap = 0.0
        self._distance_sum = 0.0
        self._active_overlap_ratio_sum = 0.0

    def step(self, context: base.ExpertClientBatchContext) -> None:
        if self._train_context is None or self._layout is None:
            raise RuntimeError("必须先调用 on_train_start()。")
        if context.train_context is not self._train_context:
            raise ValueError("BatchContext 不属于当前客户端训练上下文。")
        expected_step = self._local_step_count + 1
        if int(context.local_step) != expected_step:
            raise ValueError(
                f"本地更新步数不连续：expected={expected_step}, "
                f"actual={context.local_step}。"
            )
        if expected_step > self.settings["local_steps"]:
            raise ValueError("实际本地更新步数超过配置的 local_steps。")

        weights_by_layer = _layer_weights_from_parameters(
            self._parameters,
            self._layout,
        )
        diversity_loss, overlap_metrics = compute_subspace_overlap_loss(
            weights_by_layer,
            rho0=self.settings["rho0"],
        )
        if self.settings["lambda_div"] > 0.0:
            (self.settings["lambda_div"] * diversity_loss).backward()

        local_lr = self.settings["local_lr"]
        stiefel_name_set = set(self._layout.stiefel_parameter_names)
        with torch.no_grad():
            for name in self._layout.expert_parameter_names:
                parameter = self._parameters[name]
                gradient = parameter.grad
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                else:
                    gradient = gradient.detach()
                if not bool(torch.isfinite(gradient).all().item()):
                    raise ValueError(f"专家参数 {name} 的本地梯度包含 NaN 或 Inf。")

                stream = self._gradient_streams[name]
                if name in stiefel_name_set:
                    riemannian_gradient = stiefel_riemannian_gradient(
                        parameter.detach(),
                        gradient,
                    )
                    transported_gradient = stiefel_tangent_projection(
                        self._anchors[name],
                        riemannian_gradient,
                    )
                    stream.add_(transported_gradient, alpha=local_lr)
                    parameter.copy_(
                        qr_retraction(
                            parameter.detach() - local_lr * riemannian_gradient
                        )
                    )
                else:
                    stream.add_(gradient, alpha=local_lr)
                    parameter.add_(gradient, alpha=-local_lr)

        self._diversity_loss_sum += float(diversity_loss.detach().cpu().item())
        self._weighted_diversity_loss_sum += float(
            (self.settings["lambda_div"] * diversity_loss).detach().cpu().item()
        )
        self._overlap_sum += overlap_metrics["mean_normalized_overlap"]
        self._max_overlap = max(
            self._max_overlap,
            overlap_metrics["max_normalized_overlap"],
        )
        self._distance_sum += overlap_metrics["mean_projection_distance"]
        self._active_overlap_ratio_sum += overlap_metrics["active_overlap_ratio"]
        self._local_step_count += 1

    def on_train_end(
        self,
        context: base.ExpertClientTrainContext,
    ) -> Mapping[str, Any]:
        if context is not self._train_context or self._layout is None:
            raise ValueError("on_train_end() 收到的上下文与当前训练不一致。")
        expected_steps = self.settings["local_steps"]
        if self._local_step_count != expected_steps:
            raise ValueError(
                "客户端实际本地更新次数与 RFedAGS 配置不一致："
                f"actual={self._local_step_count}, expected={expected_steps}。"
            )

        with torch.no_grad():
            final_metrics = compute_subspace_overlap_loss(
                _layer_weights_from_parameters(self._parameters, self._layout),
                rho0=self.settings["rho0"],
            )[1]
            final_errors = [
                float(
                    orthogonality_error(self._parameters[name])
                    .detach()
                    .cpu()
                    .item()
                )
                for name in self._layout.stiefel_parameter_names
            ]
        diagnostics = {
            "avg_diversity_loss": self._diversity_loss_sum / expected_steps,
            "avg_weighted_diversity_loss": (
                self._weighted_diversity_loss_sum / expected_steps
            ),
            "avg_normalized_overlap": self._overlap_sum / expected_steps,
            "max_normalized_overlap": self._max_overlap,
            "avg_projection_distance": self._distance_sum / expected_steps,
            "avg_active_overlap_ratio": (
                self._active_overlap_ratio_sum / expected_steps
            ),
            "final_active_overlap_ratio": final_metrics[
                "active_overlap_ratio"
            ],
            "final_num_active_pairs": final_metrics["num_active_pairs"],
            "final_num_pairs": final_metrics["num_pairs"],
            "final_avg_normalized_overlap": final_metrics[
                "mean_normalized_overlap"
            ],
            "final_max_normalized_overlap": final_metrics[
                "max_normalized_overlap"
            ],
            "final_avg_projection_distance": final_metrics[
                "mean_projection_distance"
            ],
            "max_orthogonality_error": max(final_errors),
            "mean_orthogonality_error": sum(final_errors) / len(final_errors),
        }

        for name, stream in self._gradient_streams.items():
            if not bool(torch.isfinite(stream).all().item()):
                raise ValueError(f"专家参数 {name} 的梯度流包含 NaN 或 Inf。")

        return {
            "algorithm": ALGORITHM_NAME,
            "version": PAYLOAD_VERSION,
            "num_local_steps": int(self._local_step_count),
            "gradient_streams": {
                name: stream.detach().clone()
                for name, stream in self._gradient_streams.items()
            },
            "diagnostics": diagnostics,
        }


class RFedAGSGrassmannExpertAggregator(base.Aggregator):
    """Server-side RFedAGS aggregation on the expert product manifold."""

    def __init__(self, cfg: Any, param_group_name: str = "expert") -> None:
        super().__init__(cfg=cfg, param_group_name=param_group_name)
        if param_group_name != "expert":
            raise ValueError("RFedAGS-Grassmann 聚合器只能负责 expert 参数组。")
        self.settings = _validate_runtime_config(cfg)
        self._layout: Optional[ExpertParameterLayout] = None

    @property
    def method_name(self) -> str:
        return ALGORITHM_NAME

    @property
    def consumes_algorithm_payload(self) -> bool:
        return True

    @property
    def handles_manifold_retraction(self) -> bool:
        return True

    def compute_weights(
        self,
        client_updates: Sequence[base.ClientUpdate],
    ) -> Dict[int, float]:
        return base.build_uniform_weights(client_updates)

    def _resolve_layout(self, model: Optional[nn.Module] = None) -> ExpertParameterLayout:
        if self._layout is not None:
            return self._layout
        if model is not None:
            groups = _resolve_optimization_groups(model, self.method_context)
        elif (
            self.method_context is not None
            and self.method_context.optimization_groups is not None
        ):
            groups = self.method_context.optimization_groups
        else:
            raise RuntimeError(
                "无法解析专家参数布局；请先注入 MethodContext 或调用 initialize_model()。"
            )
        self._layout = _build_parameter_layout(groups)
        return self._layout

    def initialize_model(self, model: nn.Module) -> None:
        """Project every expert's full fc1 weight onto the Stiefel manifold."""
        groups = _resolve_optimization_groups(model, self.method_context)
        self._layout = _build_parameter_layout(groups)
        parameter_map = dict(model.named_parameters())
        with torch.no_grad():
            for name in self._layout.stiefel_parameter_names:
                if name not in parameter_map:
                    raise KeyError(f"模型中不存在 Stiefel 参数：{name}。")
                parameter = parameter_map[name]
                if parameter.ndim != 2 or parameter.shape[0] < parameter.shape[1]:
                    raise ValueError(
                        f"专家参数 {name} 不满足列正交 Stiefel 形状，"
                        f"shape={tuple(parameter.shape)}。"
                    )
                parameter.copy_(qr_retraction(parameter))

        errors = [
            float(orthogonality_error(parameter_map[name]).cpu().item())
            for name in self._layout.stiefel_parameter_names
        ]
        if max(errors) > ORTHOGONALITY_TOLERANCE:
            raise RuntimeError(
                "全局专家第一层权重初始化后仍未满足 Stiefel 约束："
                f"max_error={max(errors):.6e}。"
            )

    def aggregate(
        self,
        global_state: Mapping[str, torch.Tensor],
        client_updates: Sequence[base.ClientUpdate],
        param_names: Optional[Sequence[str]] = None,
        base_state: Optional[Mapping[str, torch.Tensor]] = None,
        strict: bool = True,
    ) -> base.AggregationResult:
        self._validate_client_updates(client_updates)
        if not client_updates:
            raise ValueError("client_updates 不能为空。")

        round_ids = {int(update.round_id) for update in client_updates}
        if len(round_ids) != 1:
            raise ValueError(f"客户端更新轮数不一致：{sorted(round_ids)}。")

        layout = self._resolve_layout()
        expected_names = set(layout.expert_parameter_names)
        if param_names is not None:
            param_names = list(param_names)
            selected_names = set(param_names)
            missing_selected = sorted(expected_names - selected_names)
            if missing_selected:
                raise ValueError(
                    "服务端传入的 expert param_names 缺少可训练专家参数："
                    f"{missing_selected[:10]}。"
                )

        weights = self.compute_weights(client_updates)
        total_weight = sum(float(value) for value in weights.values())
        if not math.isclose(total_weight, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"RFedAGS 客户端等权权重之和必须为 1，当前为 {total_weight}。"
            )

        payloads: Dict[int, Mapping[str, Any]] = {}
        for update in client_updates:
            payload = update.algorithm_payload
            if not isinstance(payload, Mapping):
                raise TypeError(
                    f"客户端 {update.client_id} 的 algorithm_payload 必须是 Mapping。"
                )
            if payload.get("algorithm") != ALGORITHM_NAME:
                raise ValueError(
                    f"客户端 {update.client_id} 上传的算法标识不正确："
                    f"{payload.get('algorithm')!r}。"
                )
            if payload.get("version") != PAYLOAD_VERSION:
                raise ValueError(
                    f"客户端 {update.client_id} 的梯度流载荷版本不受支持："
                    f"{payload.get('version')!r}。"
                )
            if payload.get("num_local_steps") != self.settings["local_steps"]:
                raise ValueError(
                    f"客户端 {update.client_id} 的 local_steps 与配置不一致："
                    f"{payload.get('num_local_steps')!r}。"
                )
            streams = payload.get("gradient_streams")
            if not isinstance(streams, Mapping):
                raise TypeError(
                    f"客户端 {update.client_id} 缺少 gradient_streams Mapping。"
                )
            if set(streams) != expected_names:
                missing = sorted(expected_names - set(streams))
                extra = sorted(set(streams) - expected_names)
                raise ValueError(
                    f"客户端 {update.client_id} 的梯度流参数集合不匹配；"
                    f"missing={missing[:10]}, extra={extra[:10]}。"
                )
            diagnostics = payload.get("diagnostics", {})
            if not isinstance(diagnostics, Mapping):
                raise TypeError(
                    f"客户端 {update.client_id} 的 diagnostics 必须是 Mapping。"
                )
            payloads[int(update.client_id)] = payload

        result_state = base.clone_state_dict(
            global_state if base_state is None else base_state
        )
        stiefel_name_set = set(layout.stiefel_parameter_names)
        aggregated_stream_norms = {"stiefel": [], "euclidean": []}

        for name in layout.expert_parameter_names:
            if name not in global_state:
                raise KeyError(f"全局模型状态中不存在专家参数：{name}。")
            global_tensor = global_state[name]
            if not torch.is_tensor(global_tensor) or not global_tensor.is_floating_point():
                raise TypeError(f"专家参数 {name} 必须是浮点 Tensor。")

            average_stream = torch.zeros_like(global_tensor)
            for update in client_updates:
                stream = payloads[int(update.client_id)]["gradient_streams"][name]
                if not torch.is_tensor(stream):
                    raise TypeError(
                        f"客户端 {update.client_id} 的 {name} 梯度流不是 Tensor。"
                    )
                if tuple(stream.shape) != tuple(global_tensor.shape):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 {name} 梯度流 shape 不匹配："
                        f"stream={tuple(stream.shape)}, "
                        f"global={tuple(global_tensor.shape)}。"
                    )
                if not stream.is_floating_point():
                    raise TypeError(
                        f"客户端 {update.client_id} 的 {name} 梯度流必须是浮点 Tensor。"
                    )
                if not bool(torch.isfinite(stream).all().item()):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 {name} 梯度流包含 NaN 或 Inf。"
                    )
                average_stream.add_(
                    stream.to(device=global_tensor.device, dtype=global_tensor.dtype),
                    alpha=float(weights[int(update.client_id)]),
                )

            if name in stiefel_name_set:
                current_error = float(
                    orthogonality_error(global_tensor).detach().cpu().item()
                )
                if current_error > ORTHOGONALITY_TOLERANCE:
                    raise ValueError(
                        f"全局 Stiefel 参数 {name} 已偏离流形："
                        f"orthogonality_error={current_error:.6e}。"
                    )
                tangent_stream = stiefel_tangent_projection(
                    global_tensor,
                    average_stream,
                )
                updated = qr_retraction(
                    global_tensor - self.settings["server_lr"] * tangent_stream
                )
                aggregated_stream_norms["stiefel"].append(
                    float(torch.linalg.vector_norm(tangent_stream).cpu().item())
                )
            else:
                updated = (
                    global_tensor
                    - self.settings["server_lr"] * average_stream
                )
                aggregated_stream_norms["euclidean"].append(
                    float(torch.linalg.vector_norm(average_stream).cpu().item())
                )

            result_state[name] = updated.detach().clone()

        payload_diagnostics = [
            payloads[int(update.client_id)].get("diagnostics", {})
            for update in client_updates
        ]

        def mean_payload_metric(key: str) -> Optional[float]:
            values = [
                float(item[key])
                for item in payload_diagnostics
                if key in item
            ]
            if not values:
                return None
            return sum(values) / len(values)

        def max_payload_metric(key: str) -> Optional[float]:
            values = [
                float(item[key])
                for item in payload_diagnostics
                if key in item
            ]
            if not values:
                return None
            return max(values)

        diagnostics: Dict[str, Any] = {
            "method": self.method_name,
            "param_group": self.param_group_name,
            "num_clients": len(client_updates),
            "num_state_entries": (
                len(param_names) if param_names is not None else None
            ),
            "num_expert_parameters": len(layout.expert_parameter_names),
            "num_stiefel_parameters": len(layout.stiefel_parameter_names),
            "client_weighting": "uniform",
            "aggregation": "rfedags_gradient_stream",
            "local_steps": self.settings["local_steps"],
            "local_lr": self.settings["local_lr"],
            "server_lr": self.settings["server_lr"],
            "lambda_div": self.settings["lambda_div"],
            "rho0": self.settings["rho0"],
            "mean_client_diversity_loss": mean_payload_metric("avg_diversity_loss"),
            "mean_client_weighted_diversity_loss": mean_payload_metric(
                "avg_weighted_diversity_loss"
            ),
            "mean_client_normalized_overlap": mean_payload_metric(
                "avg_normalized_overlap"
            ),
            "mean_client_active_overlap_ratio": mean_payload_metric(
                "avg_active_overlap_ratio"
            ),
            "max_client_active_overlap_ratio": max_payload_metric(
                "avg_active_overlap_ratio"
            ),
            "max_client_normalized_overlap": max_payload_metric(
                "max_normalized_overlap"
            ),
            "mean_client_projection_distance": mean_payload_metric(
                "avg_projection_distance"
            ),
            "max_client_orthogonality_error": max_payload_metric(
                "max_orthogonality_error"
            ),
            "mean_stiefel_aggregate_stream_norm": (
                sum(aggregated_stream_norms["stiefel"])
                / len(aggregated_stream_norms["stiefel"])
                if aggregated_stream_norms["stiefel"]
                else 0.0
            ),
            "mean_euclidean_aggregate_stream_norm": (
                sum(aggregated_stream_norms["euclidean"])
                / len(aggregated_stream_norms["euclidean"])
                if aggregated_stream_norms["euclidean"]
                else 0.0
            ),
        }
        return base.AggregationResult(
            new_state_dict=result_state,
            weights=weights,
            diagnostics=diagnostics,
        )

    def after_aggregate(self, model: nn.Module) -> None:
        """Validate the server-side QR retraction already performed in aggregate()."""
        layout = self._resolve_layout(model)
        parameter_map = dict(model.named_parameters())
        errors = [
            float(orthogonality_error(parameter_map[name]).detach().cpu().item())
            for name in layout.stiefel_parameter_names
        ]
        if max(errors) > ORTHOGONALITY_TOLERANCE:
            raise RuntimeError(
                "RFedAGS 服务端聚合后的 Stiefel 回缩校验失败："
                f"max_error={max(errors):.6e}。"
            )

    def build_post_aggregate_diagnostics(
        self,
        model: nn.Module,
    ) -> Dict[str, Any]:
        layout = self._resolve_layout(model)
        metrics = _compute_model_subspace_diagnostics(
            model=model,
            layout=layout,
            rho0=self.settings["rho0"],
        )
        return {
            "post_avg_normalized_overlap": metrics["mean_normalized_overlap"],
            "post_max_normalized_overlap": metrics["max_normalized_overlap"],
            "post_avg_grassmann_projection_distance": metrics[
                "mean_projection_distance"
            ],
            "post_active_overlap_ratio": metrics["active_overlap_ratio"],
            "post_num_active_overlap_pairs": metrics["num_active_pairs"],
            "post_num_overlap_pairs": metrics["num_pairs"],
            "post_min_grassmann_projection_distance": metrics[
                "min_projection_distance"
            ],
            "post_max_stiefel_orthogonality_error": metrics[
                "max_orthogonality_error"
            ],
            "post_mean_stiefel_orthogonality_error": metrics[
                "mean_orthogonality_error"
            ],
        }


def build_expert_aggregator(cfg: Any) -> base.Aggregator:
    """Build the RFedAGS-Grassmann expert-only aggregator."""
    return RFedAGSGrassmannExpertAggregator(
        cfg=cfg,
        param_group_name="expert",
    )


def build_expert_client_training_strategy(
    cfg: Any,
    client_id: int,
) -> base.ExpertClientTrainingStrategy:
    """Build one RFedAGS-Grassmann client training strategy."""
    return RFedAGSGrassmannClientStrategy(cfg=cfg, client_id=client_id)


def register_method_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the algorithm-specific command-line hyperparameters."""
    parser.add_argument("--lambda-div", type=float, default=None)
    parser.add_argument("--rho0", type=float, default=None)
    parser.add_argument("--server-lr", type=float, default=None)


def build_method_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert algorithm-specific CLI arguments into method_config overrides."""
    method_overrides: Dict[str, Any] = {}
    if args.lambda_div is not None:
        method_overrides["lambda_div"] = args.lambda_div
    if args.rho0 is not None:
        method_overrides["rho0"] = args.rho0
    if args.server_lr is not None:
        method_overrides["server_lr"] = args.server_lr
    return {"method_config": method_overrides} if method_overrides else {}


def main(model_builder: Optional[base.ModelBuilder] = None) -> int:
    """Run the RFedAGS-Grassmann federated experiment."""
    return base.main(
        expert_aggregator_builder=build_expert_aggregator,
        embedded_method_config=EMBEDDED_METHOD_CONFIG,
        expert_method_name=ALGORITHM_NAME,
        method_config_defaults=METHOD_CONFIG_DEFAULTS,
        method_config_validator=validate_method_config,
        method_cli_argument_registrar=register_method_cli_arguments,
        method_cli_overrides_builder=build_method_cli_overrides,
        expert_client_training_strategy_builder=(
            build_expert_client_training_strategy
        ),
        model_builder=model_builder,
    )


if __name__ == "__main__":
    raise SystemExit(main())

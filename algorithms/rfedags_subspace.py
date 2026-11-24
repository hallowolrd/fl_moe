"""RFedAGS with client-side expert subspace diversity on Stiefel manifolds.

The first linear layer of every Switch expert is parameterized as W1 = A P^T.
The P factors are optimized on Stiefel manifolds and aggregated with RFedAGS
gradient streams. All remaining expert parameters use the base optimizer and
client-uniform FedAvg. Non-expert parameters remain on base.py's fixed uniform
aggregation path.
"""

from __future__ import annotations

# Import base before torch so deterministic CUDA environment setup is applied
# before PyTorch initializes CUDA/cuBLAS.
import base

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


ALGORITHM_NAME = "rfedags_subspace"
PAYLOAD_KEY = ALGORITHM_NAME
PAYLOAD_VERSION = 1

# ---------------------------------------------------------------------------
# Algorithm-specific hyperparameters.
# Common training/model/optimizer defaults continue to come from base.py.
# ---------------------------------------------------------------------------
SUBSPACE_RANK = 128
LAMBDA_DIV = 1e-3
TAU_MARGIN = 0.05
SUBSPACE_OVERLAP_THRESHOLD_CAP = 0.95
LOCAL_MANIFOLD_LR = None  # None -> use base.py's optimizer.lr.
SERVER_MANIFOLD_LR = 1.0

METHOD_CONFIG_DEFAULTS: Dict[str, Any] = {
    "method_config": {
        "subspace_rank": SUBSPACE_RANK,
        "lambda_div": LAMBDA_DIV,
        "tau_margin": TAU_MARGIN,
        "subspace_overlap_threshold_cap": SUBSPACE_OVERLAP_THRESHOLD_CAP,
        "local_manifold_lr": LOCAL_MANIFOLD_LR,
        "server_manifold_lr": SERVER_MANIFOLD_LR,
    },
}

EMBEDDED_METHOD_CONFIG = {
    "agg": {
        "non_expert": {"method": "uniform"},
        "expert": {"method": ALGORITHM_NAME},
    },
}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    return base._cfg_get(cfg, key, default)


def _method_cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    return _cfg_get(cfg, f"method_config.{key}", default)


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数，当前值：{value!r}")
    return int(value)


def _validate_finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限非负数，当前值：{value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限非负数，当前值：{value!r}") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} 必须是有限非负数，当前值：{value!r}")
    return result


def _validate_finite_positive(value: Any, name: str) -> float:
    result = _validate_finite_nonnegative(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} 必须是有限正数，当前值：{value!r}")
    return result


def validate_method_config(cfg: Mapping[str, Any]) -> None:
    """Validate the full-participation and fixed-local-step assumptions."""
    local_steps = _cfg_get(cfg, "local_steps", None)
    _validate_positive_int(local_steps, "local_steps")

    raw_frac = _cfg_get(cfg, "frac", 1.0)
    if isinstance(raw_frac, bool):
        raise ValueError("frac 必须为数值 1.0。")
    try:
        frac = float(raw_frac)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"frac 必须为数值 1.0，当前值：{raw_frac!r}"
        ) from exc
    if not math.isfinite(frac) or frac != 1.0:
        raise ValueError(
            f"{ALGORITHM_NAME} 首版要求全客户端参与，必须设置 frac=1.0；"
            f"当前值：{frac}"
        )

    num_experts = _validate_positive_int(
        _cfg_get(cfg, "num_experts", base.DEFAULT_CONFIG["num_experts"]),
        "num_experts",
    )
    if num_experts < 2:
        raise ValueError("专家子空间差异性约束至少需要 2 个专家。")

    rank = _validate_positive_int(
        _method_cfg_get(cfg, "subspace_rank", SUBSPACE_RANK),
        "method_config.subspace_rank",
    )
    hidden_dim = _validate_positive_int(
        _cfg_get(
            cfg,
            "model_cfg.hidden_dim",
            base.DEFAULT_CONFIG["model_cfg"]["hidden_dim"],
        ),
        "model_cfg.hidden_dim",
    )
    expert_hidden_dim = _validate_positive_int(
        _cfg_get(
            cfg,
            "model_cfg.expert_hidden_dim",
            base.DEFAULT_CONFIG["model_cfg"]["expert_hidden_dim"],
        ),
        "model_cfg.expert_hidden_dim",
    )
    if rank >= hidden_dim:
        raise ValueError(
            "subspace_rank 必须小于 hidden_dim，否则所有专家子空间都等于完整输入空间；"
            f"当前 rank={rank}, hidden_dim={hidden_dim}。"
        )
    if rank > expert_hidden_dim:
        raise ValueError(
            "subspace_rank 不能大于 expert_hidden_dim；"
            f"当前 rank={rank}, expert_hidden_dim={expert_hidden_dim}。"
        )

    _validate_finite_nonnegative(
        _method_cfg_get(cfg, "lambda_div", LAMBDA_DIV),
        "method_config.lambda_div",
    )
    tau_margin = _validate_finite_nonnegative(
        _method_cfg_get(cfg, "tau_margin", TAU_MARGIN),
        "method_config.tau_margin",
    )
    if tau_margin >= 1.0:
        raise ValueError("method_config.tau_margin 必须小于 1。")

    threshold_cap = _validate_finite_positive(
        _method_cfg_get(
            cfg,
            "subspace_overlap_threshold_cap",
            SUBSPACE_OVERLAP_THRESHOLD_CAP,
        ),
        "method_config.subspace_overlap_threshold_cap",
    )
    if threshold_cap >= 1.0:
        raise ValueError(
            "method_config.subspace_overlap_threshold_cap 必须小于 1。"
        )

    local_lr = _method_cfg_get(
        cfg,
        "local_manifold_lr",
        LOCAL_MANIFOLD_LR,
    )
    if local_lr is None:
        local_lr = _cfg_get(
            cfg,
            "optimizer.lr",
            base.DEFAULT_CONFIG["optimizer"]["lr"],
        )
    _validate_finite_positive(local_lr, "local_manifold_lr / optimizer.lr")

    _validate_finite_positive(
        _method_cfg_get(cfg, "server_manifold_lr", SERVER_MANIFOLD_LR),
        "method_config.server_manifold_lr",
    )


def _accumulation_dtype(reference: torch.Tensor) -> torch.dtype:
    if reference.dtype in (torch.float64, torch.complex128):
        return reference.dtype
    if reference.is_complex():
        return torch.complex64
    return torch.float32


def _tensor_is_finite(value: torch.Tensor) -> bool:
    if value.is_floating_point() or value.is_complex():
        return bool(torch.isfinite(value).all().item())
    return True


def _clip_product_manifold_gradients(
    riemannian_gradients: Mapping[str, torch.Tensor],
    euclidean_parameters: Sequence[nn.Parameter],
    max_norm: Optional[float],
) -> float:
    """Clip the combined Euclidean/Stiefel gradient in the product metric.

    Stiefel gradients must already be projected onto their tangent spaces.
    The common clipping factor is applied to those Riemannian gradients and
    to the ordinary gradients consumed by the Euclidean optimizer.
    """
    if max_norm is not None:
        max_norm = _validate_finite_positive(max_norm, "grad_clip")

    gradients = list(riemannian_gradients.values())
    gradients.extend(
        parameter.grad
        for parameter in euclidean_parameters
        if parameter.grad is not None
    )
    if not gradients:
        return 0.0

    device = gradients[0].device
    squared_norm = torch.zeros((), device=device, dtype=torch.float64)
    for gradient in gradients:
        if gradient.is_sparse:
            raise NotImplementedError(
                "RFedAGS 子空间策略暂不支持稀疏专家梯度。"
            )
        if gradient.device != device:
            raise ValueError("产品流形梯度裁剪要求所有专家梯度位于同一设备。")
        if not _tensor_is_finite(gradient):
            raise ValueError("产品流形专家梯度包含 NaN 或 Inf。")
        squared_norm.add_(
            gradient.detach().abs().to(dtype=torch.float64).square().sum()
        )

    total_norm = torch.sqrt(squared_norm)
    total_norm_value = float(total_norm.item())
    if max_norm is not None and total_norm_value > max_norm:
        clip_coefficient = max_norm / (total_norm_value + 1e-6)
        with torch.no_grad():
            for gradient in riemannian_gradients.values():
                gradient.mul_(clip_coefficient)
            for parameter in euclidean_parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(clip_coefficient)

    return total_norm_value


def stiefel_tangent_projection(
    point: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """Project an ambient matrix onto T_point St(d, r), embedded metric."""
    if point.ndim != 2 or vector.shape != point.shape:
        raise ValueError(
            "Stiefel tangent projection requires two same-shaped matrices; "
            f"point={tuple(point.shape)}, vector={tuple(vector.shape)}"
        )
    sym = 0.5 * (point.transpose(0, 1) @ vector + vector.transpose(0, 1) @ point)
    return vector - point @ sym


def qr_retraction(matrix: torch.Tensor) -> torch.Tensor:
    """QR retraction with a deterministic positive-diagonal convention."""
    if matrix.ndim != 2:
        raise ValueError(f"QR retraction expects a matrix, got shape={tuple(matrix.shape)}")
    q, r = torch.linalg.qr(matrix, mode="reduced")
    diagonal = torch.diagonal(r, dim1=-2, dim2=-1)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    return q * signs.unsqueeze(0)


class LowRankStiefelLinear(nn.Module):
    """Linear layer with W = A P^T and P^T P = I."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        _initialize: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0 or rank >= in_features or rank > out_features:
            raise ValueError(
                "LowRankStiefelLinear 要求 0 < rank < in_features 且 "
                f"rank <= out_features；当前 in={in_features}, "
                f"out={out_features}, rank={rank}。"
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)

        self.A = nn.Parameter(torch.empty(out_features, rank))
        self.P = nn.Parameter(torch.empty(in_features, rank))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        if _initialize:
            nn.init.xavier_uniform_(self.A)
            nn.init.orthogonal_(self.P)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
    ) -> "LowRankStiefelLinear":
        """Initialize from the truncated SVD of an existing nn.Linear."""
        weight = linear.weight.detach()
        out_features, in_features = weight.shape
        if rank <= 0 or rank >= in_features or rank > out_features:
            raise ValueError(
                "SVD 低秩初始化要求 0 < rank < in_features 且 "
                f"rank <= out_features；当前 in={in_features}, "
                f"out={out_features}, rank={rank}。"
            )

        replacement = cls(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            bias=linear.bias is not None,
            _initialize=False,
        ).to(device=weight.device, dtype=weight.dtype)

        u, singular_values, vh = torch.linalg.svd(weight, full_matrices=False)
        truncated_a = u[:, :rank] * singular_values[:rank].unsqueeze(0)
        approximation = truncated_a @ vh[:rank, :]
        denominator = torch.linalg.norm(weight).clamp_min(
            torch.finfo(weight.dtype).eps
        )
        replacement.initial_reconstruction_error = float(
            (torch.linalg.norm(weight - approximation) / denominator).item()
        )
        with torch.no_grad():
            replacement.A.copy_(truncated_a)
            replacement.P.copy_(vh[:rank, :].transpose(0, 1))
            if replacement.bias is not None:
                replacement.bias.copy_(linear.bias.detach())
        return replacement

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = F.linear(inputs, self.P.transpose(0, 1))
        return F.linear(latent, self.A, self.bias)


def _collect_layer_records(
    model: nn.Module,
) -> Dict[str, list[Dict[str, Any]]]:
    """Discover each Stiefel P/A pair and group pairs by MoE layer."""
    records: Dict[str, list[Dict[str, Any]]] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, LowRankStiefelLinear):
            continue
        if ".experts." not in module_name:
            raise ValueError(
                "低秩 Stiefel 层必须位于 MoE experts 子模块内："
                f"{module_name!r}"
            )

        layer_name, suffix = module_name.split(".experts.", 1)
        expert_key = suffix.split(".", 1)[0]
        match = re.fullmatch(r"(?:expert_)?(\d+)", expert_key)
        if match is None:
            raise ValueError(
                f"无法从专家模块路径解析 expert id：{module_name!r}"
            )

        expert_id = int(match.group(1))
        records.setdefault(layer_name, []).append(
            {
                "expert_id": expert_id,
                "module_name": module_name,
                "p_name": f"{module_name}.P",
                "a_name": f"{module_name}.A",
            }
        )

    for layer_name, layer_records in records.items():
        layer_records.sort(key=lambda item: item["expert_id"])
        expert_ids = [int(item["expert_id"]) for item in layer_records]
        if len(expert_ids) != len(set(expert_ids)):
            raise ValueError(f"MoE 层 {layer_name} 存在重复专家编号。")
    return records


def _get_layer_records(model: nn.Module) -> Dict[str, list[Dict[str, Any]]]:
    records = getattr(model, "_rfedags_subspace_layer_records", None)
    if records is None:
        records = _collect_layer_records(model)
    if not isinstance(records, Mapping) or not records:
        raise ValueError("模型中没有发现可用的 Stiefel 专家子空间参数。")
    return {
        str(layer_name): [dict(item) for item in layer_records]
        for layer_name, layer_records in records.items()
    }


def align_stiefel_factor_gauge(
    local_a: torch.Tensor,
    local_p: torch.Tensor,
    reference_p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Procrustes-align (A, P) to reference_p without changing A P^T."""
    if (
        local_a.ndim != 2
        or local_p.ndim != 2
        or reference_p.shape != local_p.shape
        or local_a.shape[1] != local_p.shape[1]
    ):
        raise ValueError(
            "因子对齐要求 A、P、reference_P 维度兼容；"
            f"A={tuple(local_a.shape)}, P={tuple(local_p.shape)}, "
            f"reference_P={tuple(reference_p.shape)}"
        )
    cross = local_p.transpose(0, 1) @ reference_p
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    rotation = u @ vh
    return local_a @ rotation, local_p @ rotation


def _apply_factor_gauge_average(
    model: nn.Module,
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    pending_updates: Mapping[
        str,
        Sequence[tuple[float, torch.Tensor, torch.Tensor]],
    ],
) -> int:
    """Average client A factors after aligning them to the final global P gauge."""
    if not pending_updates:
        return 0

    parameter_map = dict(model.named_parameters())
    expected_a_names = {
        str(item["a_name"])
        for layer_records in records.values()
        for item in layer_records
    }
    if set(pending_updates) != expected_a_names:
        raise ValueError(
            "待对齐的 A 因子集合与 Stiefel P 因子记录不一致；"
            f"expected={sorted(expected_a_names)}, "
            f"actual={sorted(pending_updates)}"
        )

    aligned_count = 0
    with torch.no_grad():
        for layer_records in records.values():
            for item in layer_records:
                p_name = str(item["p_name"])
                a_name = str(item["a_name"])
                if p_name not in parameter_map or a_name not in parameter_map:
                    raise KeyError(
                        f"全局模型缺少低秩因子参数：P={p_name}, A={a_name}"
                    )
                reference_p = parameter_map[p_name].detach()
                reference_a = parameter_map[a_name]
                accumulator = torch.zeros(
                    reference_a.shape,
                    device=reference_a.device,
                    dtype=_accumulation_dtype(reference_a),
                )
                total_weight = 0.0
                for weight, local_a, local_p in pending_updates[a_name]:
                    if not math.isfinite(float(weight)) or float(weight) < 0.0:
                        raise ValueError(f"{a_name} 的客户端聚合权重无效：{weight!r}")
                    aligned_a, aligned_p = align_stiefel_factor_gauge(
                        local_a=local_a.to(
                            device=reference_a.device,
                            dtype=reference_a.dtype,
                        ),
                        local_p=local_p.to(
                            device=reference_p.device,
                            dtype=reference_p.dtype,
                        ),
                        reference_p=reference_p,
                    )
                    if not _tensor_is_finite(aligned_a) or not _tensor_is_finite(aligned_p):
                        raise ValueError(f"{a_name}/{p_name} Gauge 对齐结果包含 NaN 或 Inf。")
                    accumulator.add_(
                        aligned_a.to(dtype=accumulator.dtype),
                        alpha=float(weight),
                    )
                    total_weight += float(weight)
                if not math.isclose(total_weight, 1.0, rel_tol=1e-6, abs_tol=1e-8):
                    raise ValueError(
                        f"{a_name} 的客户端聚合权重之和必须为 1，当前为 {total_weight}"
                    )
                averaged_a = accumulator.to(dtype=reference_a.dtype)
                if not _tensor_is_finite(averaged_a):
                    raise ValueError(f"{a_name} 的 Gauge 对齐聚合结果包含 NaN 或 Inf。")
                reference_a.copy_(averaged_a)
                aligned_count += 1
    return aligned_count


def _collect_pending_factor_updates(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    client_updates: Sequence[base.ClientUpdate],
    weights: Mapping[int, float],
    global_state: Mapping[str, torch.Tensor],
) -> Dict[str, list[tuple[float, torch.Tensor, torch.Tensor]]]:
    """Reconstruct raw local (A, P) factors for final server-side alignment."""
    pending: Dict[str, list[tuple[float, torch.Tensor, torch.Tensor]]] = {
        str(item["a_name"]): []
        for layer_records in records.values()
        for item in layer_records
    }
    for update in client_updates:
        client_id = int(update.client_id)
        if client_id not in weights:
            raise KeyError(f"Gauge 对齐权重缺少客户端 {client_id}。")
        for layer_records in records.values():
            for item in layer_records:
                a_name = str(item["a_name"])
                p_name = str(item["p_name"])
                local_a = global_state[a_name] + update.model_delta[a_name].to(
                    device=global_state[a_name].device,
                    dtype=global_state[a_name].dtype,
                )
                local_p = global_state[p_name] + update.model_delta[p_name].to(
                    device=global_state[p_name].device,
                    dtype=global_state[p_name].dtype,
                )
                pending[a_name].append(
                    (
                        float(weights[client_id]),
                        local_a.detach().clone(),
                        local_p.detach().clone(),
                    )
                )
    return pending


def _collect_pairwise_overlap_metrics(
    model: nn.Module,
) -> Dict[str, Dict[str, float]]:
    """Return each within-layer expert-pair overlap for final-model diagnostics."""
    parameter_map = dict(model.named_parameters())
    pairwise: Dict[str, Dict[str, float]] = {}
    for layer_name, layer_records in _get_layer_records(model).items():
        layer_values: Dict[str, float] = {}
        for left_index in range(len(layer_records)):
            for right_index in range(left_index + 1, len(layer_records)):
                left_item = layer_records[left_index]
                right_item = layer_records[right_index]
                overlap = _pairwise_subspace_overlap(
                    parameter_map[left_item["p_name"]].detach(),
                    parameter_map[right_item["p_name"]].detach(),
                )
                pair_name = (
                    f"expert_{int(left_item['expert_id'])}_"
                    f"expert_{int(right_item['expert_id'])}"
                )
                layer_values[pair_name] = float(overlap.item())
        pairwise[layer_name] = layer_values
    return pairwise


def _pairwise_subspace_overlap(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError(
            "子空间重叠要求同形状的二维 Stiefel 参数；"
            f"left={tuple(left.shape)}, right={tuple(right.shape)}"
        )
    rank = int(left.shape[1])
    return (left.transpose(0, 1) @ right).square().sum() / float(rank)


def compute_subspace_diversity_loss(
    model: nn.Module,
    thresholds: Optional[Mapping[str, float]] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Compute the mean squared hinge penalty over all within-layer pairs."""
    records = _get_layer_records(model)
    if thresholds is None:
        thresholds = getattr(model, "_rfedags_subspace_thresholds", None)
    if not isinstance(thresholds, Mapping):
        raise ValueError("模型缺少初始化时确定的逐层子空间重叠阈值。")

    parameter_map = dict(model.named_parameters())
    penalties: list[torch.Tensor] = []
    overlaps_by_layer: Dict[str, list[torch.Tensor]] = {}

    for layer_name, layer_records in records.items():
        if layer_name not in thresholds:
            raise KeyError(f"缺少 MoE 层 {layer_name} 的子空间重叠阈值。")
        threshold = float(thresholds[layer_name])
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"MoE 层 {layer_name} 的阈值必须位于 [0, 1]，当前值：{threshold}"
            )

        for left_index in range(len(layer_records)):
            for right_index in range(left_index + 1, len(layer_records)):
                left_name = layer_records[left_index]["p_name"]
                right_name = layer_records[right_index]["p_name"]
                left = parameter_map[left_name]
                right = parameter_map[right_name]
                overlap = _pairwise_subspace_overlap(left, right)
                penalties.append(torch.relu(overlap - threshold).square())
                overlaps_by_layer.setdefault(layer_name, []).append(
                    overlap.detach().clamp(min=0.0, max=1.0)
                )

    if not penalties:
        raise ValueError("子空间差异性损失没有可计算的专家配对。")

    loss = torch.stack(penalties).mean()
    mean_overlap_by_layer = {
        layer_name: float(torch.stack(values).mean().item())
        for layer_name, values in overlaps_by_layer.items()
    }
    return loss, mean_overlap_by_layer


def build_rfedags_subspace_model(cfg: Any) -> nn.Module:
    """Build the base SwitchViT and replace every expert fc1 with W=A P^T."""
    model = base.build_switch_vit_model(cfg)
    rank = _validate_positive_int(
        _method_cfg_get(cfg, "subspace_rank", SUBSPACE_RANK),
        "method_config.subspace_rank",
    )
    num_experts = _validate_positive_int(
        _cfg_get(cfg, "num_experts", base.DEFAULT_CONFIG["num_experts"]),
        "num_experts",
    )

    expert_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, base.SwitchExpert)
    ]
    if not expert_modules:
        raise ValueError("默认 SwitchViT 中没有发现 SwitchExpert 模块。")

    reconstruction_errors: Dict[str, float] = {}
    for module_name, expert in expert_modules:
        expert.fc1 = LowRankStiefelLinear.from_linear(expert.fc1, rank)
        reconstruction_errors[f"{module_name}.fc1"] = float(
            expert.fc1.initial_reconstruction_error
        )

    records = _collect_layer_records(model)
    if not records:
        raise ValueError("低秩参数化后没有发现 Stiefel 专家子空间。")

    thresholds: Dict[str, float] = {}
    initial_overlaps: Dict[str, float] = {}
    parameter_map = dict(model.named_parameters())
    tau_margin = float(_method_cfg_get(cfg, "tau_margin", TAU_MARGIN))
    threshold_cap = _validate_finite_positive(
        _method_cfg_get(
            cfg,
            "subspace_overlap_threshold_cap",
            SUBSPACE_OVERLAP_THRESHOLD_CAP,
        ),
        "method_config.subspace_overlap_threshold_cap",
    )
    if threshold_cap >= 1.0:
        raise ValueError(
            "method_config.subspace_overlap_threshold_cap 必须小于 1。"
        )

    for layer_name, layer_records in records.items():
        expert_ids = [int(item["expert_id"]) for item in layer_records]
        if expert_ids != list(range(num_experts)):
            raise ValueError(
                f"MoE 层 {layer_name} 的专家编号不完整；"
                f"expected={list(range(num_experts))}, actual={expert_ids}"
            )

        pair_overlaps = []
        for left_index in range(len(layer_records)):
            for right_index in range(left_index + 1, len(layer_records)):
                left = parameter_map[layer_records[left_index]["p_name"]]
                right = parameter_map[layer_records[right_index]["p_name"]]
                pair_overlaps.append(
                    float(_pairwise_subspace_overlap(left, right).detach().item())
                )

        if not pair_overlaps:
            raise ValueError(f"MoE 层 {layer_name} 不存在专家配对。")
        initial_mean = sum(pair_overlaps) / len(pair_overlaps)
        initial_overlaps[layer_name] = initial_mean
        thresholds[layer_name] = min(initial_mean + tau_margin, threshold_cap)

    model._rfedags_subspace_layer_records = records
    model._rfedags_subspace_thresholds = thresholds
    model._rfedags_subspace_initial_overlaps = initial_overlaps
    model._rfedags_subspace_reconstruction_errors = reconstruction_errors
    model._rfedags_subspace_rank = rank
    model._rfedags_subspace_threshold_cap = threshold_cap
    return model


class RFedAGSSubspaceClientStrategy(base.ExpertClientTrainingStrategy):
    """Accumulate Stiefel gradient streams and update Euclidean expert factors."""

    @property
    def strategy_name(self) -> str:
        return ALGORITHM_NAME

    @property
    def includes_expert_parameters_in_model_delta(self) -> bool:
        # Euclidean expert parameters (A, fc2, bias) are FedAvg-aggregated
        # from model_delta. P is ignored in model_delta and uses its stream.
        return True

    @property
    def handles_expert_grad_clip(self) -> bool:
        # The framework clips non-expert gradients; this strategy clips the
        # complete expert gradient group after adding the diversity gradient.
        return True

    def __init__(self, cfg: Any, client_id: int) -> None:
        self.cfg = cfg
        self.client_id = int(client_id)
        self.lambda_div = float(
            _method_cfg_get(cfg, "lambda_div", LAMBDA_DIV)
        )
        configured_local_lr = _method_cfg_get(
            cfg,
            "local_manifold_lr",
            LOCAL_MANIFOLD_LR,
        )
        self.local_lr = float(
            configured_local_lr
            if configured_local_lr is not None
            else _cfg_get(
                cfg,
                "optimizer.lr",
                base.DEFAULT_CONFIG["optimizer"]["lr"],
            )
        )

        self._parameter_map: Dict[str, nn.Parameter] = {}
        self._records: Dict[str, list[Dict[str, Any]]] = {}
        self._p_names: list[str] = []
        self._expert_parameter_names: list[str] = []
        self._expert_parameters: list[nn.Parameter] = []
        self._euclidean_expert_names: list[str] = []
        self._euclidean_expert_parameters: list[nn.Parameter] = []
        self._expert_buffer_names: list[str] = []
        self._global_points: Dict[str, torch.Tensor] = {}
        self._gradient_stream: Dict[str, torch.Tensor] = {}
        self._thresholds: Dict[str, float] = {}
        self._euclidean_optimizer: Optional[torch.optim.Optimizer] = None
        self._expected_local_steps: Optional[int] = None
        self._local_steps = 0
        self._diversity_loss_sum = 0.0
        self._overlap_sums: Dict[str, float] = {}

    def on_train_start(self, context: base.ExpertClientTrainContext) -> None:
        model = context.model
        self._parameter_map = dict(model.named_parameters())
        self._records = _get_layer_records(model)
        self._thresholds = {
            str(name): float(value)
            for name, value in getattr(
                model, "_rfedags_subspace_thresholds", {}
            ).items()
        }
        if set(self._thresholds) != set(self._records):
            raise ValueError("模型逐层子空间阈值与 Stiefel 参数分组不一致。")

        if context.local_steps is None:
            raise ValueError(
                f"{ALGORITHM_NAME} 要求固定本地步数，请设置 --local-steps K。"
            )
        self._expected_local_steps = _validate_positive_int(
            context.local_steps,
            "local_steps",
        )

        self._p_names = [
            item["p_name"]
            for layer_records in self._records.values()
            for item in layer_records
        ]
        self._expert_parameter_names = list(
            context.optimization_groups.expert_parameter_names
        )
        missing_p_names = set(self._p_names) - set(self._expert_parameter_names)
        if missing_p_names:
            raise ValueError(
                "Stiefel 参数没有全部归入 expert 参数组："
                f"{sorted(missing_p_names)}"
            )

        self._euclidean_expert_names = [
            name
            for name in self._expert_parameter_names
            if name not in set(self._p_names)
        ]
        if not self._euclidean_expert_names:
            raise ValueError("没有可由普通优化器更新的欧氏专家参数。")

        # Keep the full expert-parameter list for subclasses that implement
        # their own expert clipping (e.g. the server-diversity variant).
        self._expert_parameters = [
            self._parameter_map[name]
            for name in self._expert_parameter_names
        ]
        self._euclidean_expert_parameters = [
            self._parameter_map[name]
            for name in self._euclidean_expert_names
        ]
        self._expert_buffer_names = sorted(
            set(context.optimization_groups.expert_state_names)
            - set(self._parameter_map)
        )

        self._global_points = {}
        self._gradient_stream = {}
        for name in self._p_names:
            if name not in context.global_state:
                raise KeyError(f"全局 state_dict 缺少 Stiefel 参数：{name}")
            parameter = self._parameter_map[name]
            global_point = context.global_state[name].detach().to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            self._global_points[name] = global_point.clone()
            self._gradient_stream[name] = torch.zeros(
                parameter.shape,
                device=parameter.device,
                dtype=_accumulation_dtype(parameter),
            )

        self._euclidean_optimizer = base.build_optimizer(
            model=model,
            cfg=self.cfg,
            param_names=self._euclidean_expert_names,
        )
        self._local_steps = 0
        self._diversity_loss_sum = 0.0
        self._overlap_sums = {}

    def step(self, context: base.ExpertClientBatchContext) -> None:
        if self._euclidean_optimizer is None:
            raise RuntimeError("客户端欧氏专家优化器尚未初始化。")
        expected_step_index = self._local_steps + 1
        if int(context.local_step) != expected_step_index:
            raise ValueError(
                "RFedAGS 子空间策略收到非连续本地步数："
                f"expected={expected_step_index}, actual={context.local_step}"
            )

        model = context.train_context.model
        diversity_loss, overlap_by_layer = compute_subspace_diversity_loss(
            model,
            thresholds=self._thresholds,
        )
        if not _tensor_is_finite(diversity_loss):
            raise ValueError("专家子空间差异性损失包含 NaN 或 Inf。")
        if self.lambda_div > 0.0:
            (self.lambda_div * diversity_loss).backward()

        grad_clip = context.train_context.non_expert_grad_clip
        max_grad_norm = (
            float(grad_clip)
            if grad_clip is not None and float(grad_clip) > 0.0
            else None
        )

        with torch.no_grad():
            riemannian_gradients: Dict[str, torch.Tensor] = {}
            for name in self._p_names:
                parameter = self._parameter_map[name]
                gradient = parameter.grad
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                if gradient.is_sparse:
                    raise NotImplementedError(
                        f"当前不支持稀疏 Stiefel 梯度：{name}"
                    )
                if not _tensor_is_finite(gradient):
                    raise ValueError(f"Stiefel 参数 {name} 的梯度包含 NaN 或 Inf。")

                riemannian_gradient = stiefel_tangent_projection(
                    parameter.detach(),
                    gradient.detach(),
                )
                if not _tensor_is_finite(riemannian_gradient):
                    raise ValueError(
                        f"Stiefel 参数 {name} 的黎曼梯度包含 NaN 或 Inf。"
                    )
                riemannian_gradients[name] = riemannian_gradient

            # Clip in the product metric: tangent-projected P gradients plus
            # ordinary Euclidean gradients. Normal components of ambient P
            # gradients must not influence the clipping coefficient.
            _clip_product_manifold_gradients(
                riemannian_gradients=riemannian_gradients,
                euclidean_parameters=self._euclidean_expert_parameters,
                max_norm=max_grad_norm,
            )

            for name in self._p_names:
                parameter = self._parameter_map[name]
                scaled_direction = self.local_lr * riemannian_gradients[name]

                transported = stiefel_tangent_projection(
                    self._global_points[name],
                    scaled_direction,
                )
                if not _tensor_is_finite(transported):
                    raise ValueError(f"Stiefel 参数 {name} 的传输梯度包含 NaN 或 Inf。")
                self._gradient_stream[name].add_(
                    transported.to(dtype=self._gradient_stream[name].dtype)
                )

                updated = qr_retraction(parameter.detach() - scaled_direction)
                parameter.copy_(updated)

        self._euclidean_optimizer.step()
        self._euclidean_optimizer.zero_grad(set_to_none=True)

        self._diversity_loss_sum += float(diversity_loss.detach().item())
        for layer_name, overlap in overlap_by_layer.items():
            self._overlap_sums[layer_name] = (
                self._overlap_sums.get(layer_name, 0.0) + float(overlap)
            )
        self._local_steps += 1

    def on_train_end(
        self,
        context: base.ExpertClientTrainContext,
    ) -> Mapping[str, Any]:
        if self._expected_local_steps is None:
            raise RuntimeError("RFedAGS 子空间策略没有初始化固定本地步数。")
        if self._local_steps != self._expected_local_steps:
            raise ValueError(
                "客户端实际更新步数与配置不一致："
                f"expected={self._expected_local_steps}, actual={self._local_steps}"
            )

        # Keep raw local (A, P) factors in model_delta. The server can only
        # align A correctly after the RFedAGS update produces final global P.
        state_dict = context.model.state_dict()
        expert_buffers = {
            name: state_dict[name].detach().clone()
            for name in self._expert_buffer_names
        }
        mean_overlap_by_layer = {
            layer_name: value / self._local_steps
            for layer_name, value in self._overlap_sums.items()
        }
        return {
            PAYLOAD_KEY: {
                "version": PAYLOAD_VERSION,
                "local_steps": int(self._local_steps),
                "gradient_stream": {
                    name: value.detach().clone()
                    for name, value in self._gradient_stream.items()
                },
                "expert_buffers": expert_buffers,
                "mean_diversity_loss": (
                    self._diversity_loss_sum / self._local_steps
                ),
                "mean_overlap_by_layer": mean_overlap_by_layer,
            }
        }


class RFedAGSSubspaceExpertAggregator(base.Aggregator):
    """RFedAGS for P plus client-uniform FedAvg for remaining expert state."""

    @property
    def method_name(self) -> str:
        return ALGORITHM_NAME

    @property
    def consumes_algorithm_payload(self) -> bool:
        return True

    @property
    def handles_manifold_retraction(self) -> bool:
        return True

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg=cfg, param_group_name="expert")
        self.server_lr = float(
            _method_cfg_get(cfg, "server_manifold_lr", SERVER_MANIFOLD_LR)
        )
        self._pending_factor_updates: Dict[
            str, list[tuple[float, torch.Tensor, torch.Tensor]]
        ] = {}
        self._last_factor_gauge_diagnostics: Dict[str, Any] = {}

    def compute_weights(
        self,
        client_updates: Sequence[base.ClientUpdate],
    ) -> Dict[int, float]:
        return base.build_uniform_weights(client_updates)

    def _get_context_resources(
        self,
    ) -> tuple[nn.Module, Any]:
        if self.method_context is None:
            raise RuntimeError("RFedAGS 子空间聚合器尚未注入 MethodContext。")
        model = self.method_context.global_model
        groups = self.method_context.optimization_groups
        if model is None or groups is None:
            raise RuntimeError(
                "RFedAGS 子空间聚合器需要 MethodContext.global_model 和 "
                "MethodContext.optimization_groups。"
            )
        return model, groups

    def initialize_model(self, model: nn.Module) -> None:
        records = _get_layer_records(model)
        parameter_map = dict(model.named_parameters())
        for layer_records in records.values():
            for item in layer_records:
                p = parameter_map[item["p_name"]].detach()
                identity = torch.eye(
                    p.shape[1],
                    device=p.device,
                    dtype=p.dtype,
                )
                error = torch.linalg.norm(p.transpose(0, 1) @ p - identity)
                if not torch.isfinite(error) or float(error.item()) > 1e-4:
                    raise ValueError(
                        f"初始化 Stiefel 参数 {item['p_name']} 不满足正交约束；"
                        f"orthogonality_error={float(error.item())}"
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
        model, groups = self._get_context_resources()

        configured_num_clients = int(_cfg_get(self.cfg, "num_clients", 0))
        expected_client_ids = set(range(configured_num_clients))
        actual_client_ids = {int(update.client_id) for update in client_updates}
        if actual_client_ids != expected_client_ids:
            raise ValueError(
                f"{ALGORITHM_NAME} 首版要求每轮全部客户端参与；"
                f"expected={sorted(expected_client_ids)}, "
                f"actual={sorted(actual_client_ids)}"
            )

        configured_local_steps = _validate_positive_int(
            _cfg_get(self.cfg, "local_steps", None),
            "local_steps",
        )
        expert_state_names = set(groups.expert_state_names)
        named_parameters = dict(model.named_parameters())
        trainable_expert_parameter_names = set(groups.expert_parameter_names)
        parameter_state_names = expert_state_names.intersection(named_parameters)
        records = _get_layer_records(model)
        p_names = {
            item["p_name"]
            for layer_records in records.values()
            for item in layer_records
        }
        euclidean_parameter_names = parameter_state_names - p_names
        factor_a_names = {
            str(item["a_name"])
            for layer_records in records.values()
            for item in layer_records
        }
        if not factor_a_names.issubset(euclidean_parameter_names):
            raise ValueError("低秩 A 因子必须全部属于欧氏 expert 参数组。")
        fedavg_euclidean_parameter_names = euclidean_parameter_names - factor_a_names
        expert_buffer_names = expert_state_names - parameter_state_names

        if not p_names:
            raise ValueError("RFedAGS 子空间聚合器没有发现 Stiefel 参数。")
        if not p_names.issubset(trainable_expert_parameter_names):
            raise ValueError("Stiefel 参数必须全部属于可训练的 expert 参数组。")
        if parameter_state_names.intersection(expert_buffer_names):
            raise ValueError("expert 参数与 buffer 分组发生重叠。")
        if parameter_state_names | expert_buffer_names != expert_state_names:
            raise ValueError("expert 参数与 buffer 分组不能完整覆盖 expert state。")
        if param_names is not None and set(param_names) != expert_state_names:
            raise ValueError("服务端传入的专家 state 名称与优化分组不一致。")

        source_state = base_state if base_state is not None else global_state
        base.validate_state_dict_shapes(
            reference_state=global_state,
            candidate_state=source_state,
            context="RFedAGS 子空间聚合基础 state_dict",
        )
        new_state_dict = {
            name: value.detach().clone()
            for name, value in source_state.items()
        }

        weights = base.normalize_weights(self.compute_weights(client_updates))
        payloads_by_client: Dict[int, Mapping[str, Any]] = {}
        local_steps_by_client: Dict[int, int] = {}
        diversity_loss_by_client: Dict[int, float] = {}
        overlap_by_client: Dict[int, Mapping[str, Any]] = {}

        for update in client_updates:
            root = update.algorithm_payload.get(PAYLOAD_KEY)
            if not isinstance(root, Mapping):
                raise TypeError(
                    f"客户端 {update.client_id} 缺少 {PAYLOAD_KEY} 算法载荷。"
                )
            if root.get("version") != PAYLOAD_VERSION:
                raise ValueError(
                    f"客户端 {update.client_id} 的 payload version 不受支持："
                    f"{root.get('version')!r}"
                )

            local_steps = _validate_positive_int(
                root.get("local_steps"),
                f"client[{update.client_id}].local_steps",
            )
            if local_steps != configured_local_steps:
                raise ValueError(
                    f"客户端 {update.client_id} 的本地步数不等于配置值："
                    f"expected={configured_local_steps}, actual={local_steps}"
                )
            local_steps_by_client[int(update.client_id)] = local_steps

            stream = root.get("gradient_stream")
            buffers = root.get("expert_buffers")
            if not isinstance(stream, Mapping):
                raise TypeError("gradient_stream 必须是 Mapping。")
            if not isinstance(buffers, Mapping):
                raise TypeError("expert_buffers 必须是 Mapping。")
            if set(stream) != p_names:
                raise ValueError(
                    f"客户端 {update.client_id} 的 Stiefel gradient_stream 参数集合不完整。"
                )
            if set(buffers) != expert_buffer_names:
                raise ValueError(
                    f"客户端 {update.client_id} 的 expert_buffers 参数集合不完整。"
                )

            for name, value in stream.items():
                if not torch.is_tensor(value):
                    raise TypeError(
                        f"客户端 {update.client_id} 的 gradient_stream[{name}] 必须是 Tensor。"
                    )
                reference = named_parameters[name]
                if tuple(value.shape) != tuple(reference.shape):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 {name} stream shape 不匹配。"
                    )
                if not _tensor_is_finite(value):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 {name} stream 包含 NaN 或 Inf。"
                    )

            for name, value in buffers.items():
                if not torch.is_tensor(value):
                    raise TypeError(
                        f"客户端 {update.client_id} 的 expert buffer {name} 必须是 Tensor。"
                    )
                reference = global_state[name]
                if tuple(value.shape) != tuple(reference.shape):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 expert buffer {name} shape 不匹配。"
                    )
                if not _tensor_is_finite(value):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 expert buffer {name} 包含 NaN 或 Inf。"
                    )

            for name in p_names:
                value = update.model_delta.get(name)
                if not torch.is_tensor(value):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 model_delta 缺少 Stiefel 参数 {name}。"
                    )
                if tuple(value.shape) != tuple(global_state[name].shape):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 {name} delta shape 不匹配。"
                    )
                if not _tensor_is_finite(value):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 {name} delta 包含 NaN 或 Inf。"
                    )

            for name in euclidean_parameter_names:
                value = update.model_delta.get(name)
                if not torch.is_tensor(value):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 model_delta 缺少欧氏专家参数 {name}。"
                    )
                if tuple(value.shape) != tuple(global_state[name].shape):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 {name} delta shape 不匹配。"
                    )
                if not _tensor_is_finite(value):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 {name} delta 包含 NaN 或 Inf。"
                    )

            raw_diversity_loss = root.get("mean_diversity_loss")
            if (
                isinstance(raw_diversity_loss, bool)
                or not isinstance(raw_diversity_loss, (int, float))
                or not math.isfinite(float(raw_diversity_loss))
                or float(raw_diversity_loss) < 0.0
            ):
                raise ValueError(
                    f"客户端 {update.client_id} 的 mean_diversity_loss 无效。"
                )
            raw_overlaps = root.get("mean_overlap_by_layer")
            if not isinstance(raw_overlaps, Mapping):
                raise TypeError("mean_overlap_by_layer 必须是 Mapping。")
            for layer_name, value in raw_overlaps.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0
                ):
                    raise ValueError(
                        f"客户端 {update.client_id} 的层 {layer_name} overlap 无效。"
                    )

            payloads_by_client[int(update.client_id)] = root
            diversity_loss_by_client[int(update.client_id)] = float(
                raw_diversity_loss
            )
            overlap_by_client[int(update.client_id)] = raw_overlaps

        if set(local_steps_by_client.values()) != {configured_local_steps}:
            raise ValueError(
                "RFedAGS 子空间聚合要求所有客户端执行相同的固定本地步数；"
                f"当前各客户端步数：{local_steps_by_client}"
            )

        # Ordinary Euclidean expert parameters use client-uniform FedAvg.
        # A factors are averaged only after the final global P is available.
        for name in sorted(fedavg_euclidean_parameter_names):
            reference = global_state[name]
            accumulator = torch.zeros(
                reference.shape,
                device=reference.device,
                dtype=_accumulation_dtype(reference),
            )
            for update in client_updates:
                delta = update.model_delta[name]
                accumulator.add_(
                    delta.to(device=reference.device, dtype=accumulator.dtype),
                    alpha=float(weights[update.client_id]),
                )
            new_state_dict[name] = reference + accumulator.to(dtype=reference.dtype)

        # Expert buffers, if any, are averaged from their final client values.
        for name in sorted(expert_buffer_names):
            reference = global_state[name]
            if reference.is_floating_point() or reference.is_complex():
                accumulator = torch.zeros(
                    reference.shape,
                    device=reference.device,
                    dtype=_accumulation_dtype(reference),
                )
                for update in client_updates:
                    value = payloads_by_client[update.client_id]["expert_buffers"][name]
                    accumulator.add_(
                        value.to(device=reference.device, dtype=accumulator.dtype),
                        alpha=float(weights[update.client_id]),
                    )
                new_state_dict[name] = accumulator.to(dtype=reference.dtype)
            else:
                accumulator = torch.zeros(
                    reference.shape,
                    device=reference.device,
                    dtype=torch.float64,
                )
                for update in client_updates:
                    value = payloads_by_client[update.client_id]["expert_buffers"][name]
                    accumulator.add_(
                        value.to(device=reference.device, dtype=torch.float64),
                        alpha=float(weights[update.client_id]),
                    )
                if reference.dtype == torch.bool:
                    new_state_dict[name] = (accumulator >= 0.5).to(reference.dtype)
                else:
                    new_state_dict[name] = torch.round(accumulator).to(reference.dtype)

        # RFedAGS AGS-AP for full participation: each client has probability 1,
        # so the gradient streams are averaged with weight 1 / num_clients.
        aggregated_stream_norms = []
        for name in sorted(p_names):
            reference = global_state[name]
            accumulator = torch.zeros(
                reference.shape,
                device=reference.device,
                dtype=_accumulation_dtype(reference),
            )
            for update in client_updates:
                stream = payloads_by_client[update.client_id]["gradient_stream"][name]
                accumulator.add_(
                    stream.to(device=reference.device, dtype=accumulator.dtype),
                    alpha=float(weights[update.client_id]),
                )
            aggregate_stream = accumulator.to(dtype=reference.dtype)
            aggregated_stream_norms.append(float(torch.linalg.norm(aggregate_stream).item()))
            new_state_dict[name] = qr_retraction(
                reference - self.server_lr * aggregate_stream
            )

        if set(new_state_dict) != set(global_state):
            missing = sorted(set(global_state) - set(new_state_dict))
            extra = sorted(set(new_state_dict) - set(global_state))
            raise ValueError(
                "RFedAGS 子空间聚合必须返回完整模型 state_dict；"
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        base.validate_state_dict_shapes(
            reference_state=global_state,
            candidate_state=new_state_dict,
            context="RFedAGS 子空间聚合结果",
        )
        base.check_finite_state_dict(new_state_dict)

        # Retain raw local factors until after_aggregate(), when the final
        # global P (including algorithm-specific post-processing) is known.
        self._pending_factor_updates = _collect_pending_factor_updates(
            records=records,
            client_updates=client_updates,
            weights=weights,
            global_state=global_state,
        )

        # Diagnostics cover only parameters averaged directly in Euclidean
        # coordinates. A is aggregated through the aligned factor path.
        diagnostics = self.build_diagnostics(
            client_updates=client_updates,
            weights=weights,
            param_names=sorted(fedavg_euclidean_parameter_names),
        )
        diagnostics.update(
            {
                "aggregation_rule": "client_uniform",
                "client_participation": "all_clients",
                "update_space": "stiefel_riemannian_gradient_stream_and_euclidean_delta",
                "local_steps": configured_local_steps,
                "subspace_rank": int(_method_cfg_get(self.cfg, "subspace_rank", SUBSPACE_RANK)),
                "lambda_div": float(_method_cfg_get(self.cfg, "lambda_div", LAMBDA_DIV)),
                "server_manifold_lr": self.server_lr,
                "num_stiefel_parameters": len(p_names),
                "num_euclidean_expert_parameters": len(euclidean_parameter_names),
                "num_fedavg_euclidean_expert_parameters": len(fedavg_euclidean_parameter_names),
                "num_gauge_aligned_factor_parameters": len(factor_a_names),
                "mean_diversity_loss": sum(
                    float(weights[client_id]) * value
                    for client_id, value in diversity_loss_by_client.items()
                ),
                "mean_weighted_diversity_loss": (
                    float(_method_cfg_get(self.cfg, "lambda_div", LAMBDA_DIV))
                    * sum(
                        float(weights[client_id]) * value
                        for client_id, value in diversity_loss_by_client.items()
                    )
                ),
                "mean_overlap_by_layer": {
                    layer_name: sum(
                        float(weights[client_id])
                        * float(overlap_by_client[client_id].get(layer_name, 0.0))
                        for client_id in overlap_by_client
                    )
                    for layer_name in sorted(
                        {
                            name
                            for values in overlap_by_client.values()
                            for name in values
                        }
                    )
                },
                "mean_aggregated_stream_norm": (
                    sum(aggregated_stream_norms) / len(aggregated_stream_norms)
                    if aggregated_stream_norms
                    else 0.0
                ),
            }
        )
        return base.AggregationResult(
            new_state_dict=new_state_dict,
            weights=weights,
            diagnostics=diagnostics,
        )

    def after_aggregate(self, model: nn.Module) -> None:
        """Align and average client A factors to the final global P gauge."""
        aligned_count = _apply_factor_gauge_average(
            model=model,
            records=_get_layer_records(model),
            pending_updates=self._pending_factor_updates,
        )
        self._pending_factor_updates = {}
        self._last_factor_gauge_diagnostics = {
            "factor_gauge_alignment_applied": bool(aligned_count),
            "num_factor_gauge_aligned_parameters": int(aligned_count),
        }

    def build_post_aggregate_diagnostics(
        self,
        model: nn.Module,
    ) -> Dict[str, Any]:
        parameter_map = dict(model.named_parameters())
        errors = []
        for layer_records in _get_layer_records(model).values():
            for item in layer_records:
                p = parameter_map[item["p_name"]].detach()
                identity = torch.eye(
                    p.shape[1],
                    device=p.device,
                    dtype=p.dtype,
                )
                errors.append(
                    float(
                        torch.linalg.norm(
                            p.transpose(0, 1) @ p - identity
                        ).item()
                    )
                )
        reconstruction_errors = getattr(
            model,
            "_rfedags_subspace_reconstruction_errors",
            {},
        )
        return {
            "max_stiefel_orthogonality_error": max(errors) if errors else 0.0,
            "mean_stiefel_orthogonality_error": (
                sum(errors) / len(errors) if errors else 0.0
            ),
            "max_initial_reconstruction_error": (
                max(reconstruction_errors.values())
                if reconstruction_errors
                else 0.0
            ),
            "mean_initial_reconstruction_error": (
                sum(reconstruction_errors.values()) / len(reconstruction_errors)
                if reconstruction_errors
                else 0.0
            ),
            "initial_subspace_overlap_by_layer": dict(
                getattr(model, "_rfedags_subspace_initial_overlaps", {})
            ),
            "subspace_thresholds_by_layer": dict(
                getattr(model, "_rfedags_subspace_thresholds", {})
            ),
            "global_subspace_overlap_by_layer": {
                layer_name: (
                    sum(pair_values.values()) / len(pair_values)
                    if pair_values else 0.0
                )
                for layer_name, pair_values in _collect_pairwise_overlap_metrics(model).items()
            },
            "global_subspace_pair_overlap_by_layer": _collect_pairwise_overlap_metrics(model),
            **self._last_factor_gauge_diagnostics,
        }


def build_expert_aggregator(cfg: Any) -> base.Aggregator:
    return RFedAGSSubspaceExpertAggregator(cfg=cfg)


def build_expert_client_training_strategy(
    cfg: Any,
    client_id: int,
) -> base.ExpertClientTrainingStrategy:
    return RFedAGSSubspaceClientStrategy(cfg=cfg, client_id=client_id)


def main() -> int:
    return base.main(
        expert_aggregator_builder=build_expert_aggregator,
        embedded_method_config=EMBEDDED_METHOD_CONFIG,
        expert_method_name=ALGORITHM_NAME,
        method_config_defaults=METHOD_CONFIG_DEFAULTS,
        method_config_validator=validate_method_config,
        expert_client_training_strategy_builder=(
            build_expert_client_training_strategy
        ),
        model_builder=build_rfedags_subspace_model,
    )


if __name__ == "__main__":
    raise SystemExit(main())

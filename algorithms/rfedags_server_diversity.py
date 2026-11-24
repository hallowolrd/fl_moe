"""RFedAGS with server-side expert subspace diversity correction.

Clients optimize task and Switch auxiliary objectives, accumulate Stiefel
gradient streams, and upload Euclidean expert deltas. The server performs
RFedAGS aggregation followed by one diversity-aware Stiefel correction step.
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


ALGORITHM_NAME = "rfedags_server_diversity"
PAYLOAD_KEY = ALGORITHM_NAME
PAYLOAD_VERSION = 1

# ---------------------------------------------------------------------------
# Algorithm-specific hyperparameters.
# All common training/model/optimizer defaults continue to come from base.py.
# ---------------------------------------------------------------------------
SUBSPACE_RANK = 128
TAU_MARGIN = 0.05
LOCAL_MANIFOLD_LR = None  # None -> use base.py's optimizer.lr.
SERVER_MANIFOLD_LR = 1.0
SERVER_DIVERSITY_LR = 1e-3
SERVER_DIVERSITY_GRAD_CLIP = None
SUBSPACE_OVERLAP_THRESHOLD_CAP = 0.95

METHOD_CONFIG_DEFAULTS: Dict[str, Any] = {
    "method_config": {
        "subspace_rank": SUBSPACE_RANK,
        "tau_margin": TAU_MARGIN,
        "local_manifold_lr": LOCAL_MANIFOLD_LR,
        "server_manifold_lr": SERVER_MANIFOLD_LR,
        "server_diversity_lr": SERVER_DIVERSITY_LR,
        "server_diversity_grad_clip": SERVER_DIVERSITY_GRAD_CLIP,
        "subspace_overlap_threshold_cap": SUBSPACE_OVERLAP_THRESHOLD_CAP,
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
    """Validate the fixed-step, full-participation RFedAGS configuration."""
    local_steps = _cfg_get(cfg, "local_steps", None)
    _validate_positive_int(local_steps, "local_steps")

    raw_frac = _cfg_get(cfg, "frac", 1.0)
    if isinstance(raw_frac, bool):
        raise ValueError("frac 必须为数值 1.0。")
    try:
        frac = float(raw_frac)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"frac 必须为数值 1.0，当前值：{raw_frac!r}") from exc
    if not math.isfinite(frac) or not math.isclose(
        frac, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"{ALGORITHM_NAME} 首版要求全客户端参与，必须设置 frac=1.0；"
            f"当前值：{frac}"
        )

    num_experts = _validate_positive_int(
        _cfg_get(cfg, "num_experts", base.DEFAULT_CONFIG["num_experts"]),
        "num_experts",
    )
    if num_experts < 2:
        raise ValueError("服务器端专家子空间校正至少需要 2 个专家。")

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
            "subspace_rank 必须小于 hidden_dim；"
            f"当前 rank={rank}, hidden_dim={hidden_dim}。"
        )
    if rank > expert_hidden_dim:
        raise ValueError(
            "subspace_rank 不能大于 expert_hidden_dim；"
            f"当前 rank={rank}, expert_hidden_dim={expert_hidden_dim}。"
        )

    tau_margin = _validate_finite_nonnegative(
        _method_cfg_get(cfg, "tau_margin", TAU_MARGIN),
        "method_config.tau_margin",
    )
    if tau_margin >= 1.0:
        raise ValueError("method_config.tau_margin 必须小于 1。")

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
    _validate_finite_positive(
        local_lr,
        "local_manifold_lr / optimizer.lr",
    )
    _validate_finite_positive(
        _method_cfg_get(cfg, "server_manifold_lr", SERVER_MANIFOLD_LR),
        "method_config.server_manifold_lr",
    )
    _validate_finite_positive(
        _method_cfg_get(cfg, "server_diversity_lr", SERVER_DIVERSITY_LR),
        "method_config.server_diversity_lr",
    )

    grad_clip = _method_cfg_get(
        cfg,
        "server_diversity_grad_clip",
        SERVER_DIVERSITY_GRAD_CLIP,
    )
    if grad_clip is not None:
        _validate_finite_positive(
            grad_clip,
            "method_config.server_diversity_grad_clip",
        )

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


class ServerDiversityLowRankStiefelLinear(nn.Module):
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
                "ServerDiversityLowRankStiefelLinear 要求 0 < rank < in_features 且 "
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
    ) -> "ServerDiversityLowRankStiefelLinear":
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
        if not isinstance(module, ServerDiversityLowRankStiefelLinear):
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
    records = getattr(model, "_rfedags_server_diversity_layer_records", None)
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
        thresholds = getattr(model, "_rfedags_server_diversity_thresholds", None)
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


def build_rfedags_server_diversity_model(cfg: Any) -> nn.Module:
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
        expert.fc1 = ServerDiversityLowRankStiefelLinear.from_linear(expert.fc1, rank)
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

    model._rfedags_server_diversity_layer_records = records
    model._rfedags_server_diversity_thresholds = thresholds
    model._rfedags_server_diversity_initial_overlaps = initial_overlaps
    model._rfedags_server_diversity_reconstruction_errors = reconstruction_errors
    model._rfedags_server_diversity_rank = rank
    model._rfedags_server_diversity_threshold_cap = threshold_cap
    return model




class RFedAGSServerDiversityClientStrategy(base.ExpertClientTrainingStrategy):
    """Accumulate task-gradient Stiefel streams; diversity is server-only."""

    @property
    def strategy_name(self) -> str:
        return ALGORITHM_NAME

    @property
    def includes_expert_parameters_in_model_delta(self) -> bool:
        return True

    @property
    def handles_expert_grad_clip(self) -> bool:
        return True

    def __init__(self, cfg: Any, client_id: int) -> None:
        self.cfg = cfg
        self.client_id = int(client_id)
        configured_local_lr = _method_cfg_get(cfg, "local_manifold_lr", LOCAL_MANIFOLD_LR)
        self.local_lr = float(
            configured_local_lr
            if configured_local_lr is not None
            else _cfg_get(cfg, "optimizer.lr", base.DEFAULT_CONFIG["optimizer"]["lr"])
        )
        self._parameter_map: Dict[str, nn.Parameter] = {}
        self._records: Dict[str, list[Dict[str, Any]]] = {}
        self._p_names: list[str] = []
        self._expert_parameter_names: list[str] = []
        self._expert_parameters: list[nn.Parameter] = []
        self._euclidean_expert_names: list[str] = []
        self._expert_buffer_names: list[str] = []
        self._global_points: Dict[str, torch.Tensor] = {}
        self._gradient_stream: Dict[str, torch.Tensor] = {}
        self._euclidean_optimizer: Optional[torch.optim.Optimizer] = None
        self._expected_local_steps: Optional[int] = None
        self._local_steps = 0

    def on_train_start(self, context: base.ExpertClientTrainContext) -> None:
        model = context.model
        self._parameter_map = dict(model.named_parameters())
        self._records = _get_layer_records(model)
        if context.local_steps is None:
            raise ValueError(
                f"{ALGORITHM_NAME} 要求固定本地步数，请设置 --local-steps K。"
            )
        self._expected_local_steps = _validate_positive_int(context.local_steps, "local_steps")
        self._p_names = [
            item["p_name"]
            for layer_records in self._records.values()
            for item in layer_records
        ]
        self._expert_parameter_names = list(context.optimization_groups.expert_parameter_names)
        missing_p_names = set(self._p_names) - set(self._expert_parameter_names)
        if missing_p_names:
            raise ValueError(
                "Stiefel 参数没有全部归入 expert 参数组："
                f"{sorted(missing_p_names)}"
            )
        p_name_set = set(self._p_names)
        self._euclidean_expert_names = [
            name for name in self._expert_parameter_names if name not in p_name_set
        ]
        if not self._euclidean_expert_names:
            raise ValueError("没有可由普通优化器更新的欧氏专家参数。")
        self._expert_parameters = [
            self._parameter_map[name] for name in self._expert_parameter_names
        ]
        self._expert_buffer_names = sorted(
            set(context.optimization_groups.expert_state_names) - set(self._parameter_map)
        )
        self._global_points = {}
        self._gradient_stream = {}
        for name in self._p_names:
            if name not in context.global_state:
                raise KeyError(f"全局 state_dict 缺少 Stiefel 参数：{name}")
            parameter = self._parameter_map[name]
            global_point = context.global_state[name].detach().to(
                device=parameter.device, dtype=parameter.dtype
            )
            self._global_points[name] = global_point.clone()
            self._gradient_stream[name] = torch.zeros(
                parameter.shape,
                device=parameter.device,
                dtype=_accumulation_dtype(parameter),
            )
        self._euclidean_optimizer = base.build_optimizer(
            model=model, cfg=self.cfg, param_names=self._euclidean_expert_names
        )
        self._local_steps = 0

    def step(self, context: base.ExpertClientBatchContext) -> None:
        if self._euclidean_optimizer is None:
            raise RuntimeError("客户端欧氏专家优化器尚未初始化。")
        expected_step_index = self._local_steps + 1
        if int(context.local_step) != expected_step_index:
            raise ValueError(
                "RFedAGS 服务器端多样性策略收到非连续本地步数："
                f"expected={expected_step_index}, actual={context.local_step}"
            )

        # base.py has already backpropagated task and Switch auxiliary losses.
        # Scheme two deliberately adds no client-side diversity loss.
        grad_clip = context.train_context.non_expert_grad_clip
        if grad_clip is not None and float(grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self._expert_parameters, max_norm=float(grad_clip)
            )

        with torch.no_grad():
            for name in self._p_names:
                parameter = self._parameter_map[name]
                gradient = parameter.grad
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                if gradient.is_sparse:
                    raise NotImplementedError(f"当前不支持稀疏 Stiefel 梯度：{name}")
                if not _tensor_is_finite(gradient):
                    raise ValueError(f"Stiefel 参数 {name} 的梯度包含 NaN 或 Inf。")
                riemannian_gradient = stiefel_tangent_projection(
                    parameter.detach(), gradient.detach()
                )
                if not _tensor_is_finite(riemannian_gradient):
                    raise ValueError(f"Stiefel 参数 {name} 的黎曼梯度包含 NaN 或 Inf。")
                scaled_direction = self.local_lr * riemannian_gradient
                transported = stiefel_tangent_projection(
                    self._global_points[name], scaled_direction
                )
                if not _tensor_is_finite(transported):
                    raise ValueError(f"Stiefel 参数 {name} 的传输梯度包含 NaN 或 Inf。")
                self._gradient_stream[name].add_(
                    transported.to(dtype=self._gradient_stream[name].dtype)
                )
                parameter.copy_(qr_retraction(parameter.detach() - scaled_direction))

        self._euclidean_optimizer.step()
        self._euclidean_optimizer.zero_grad(set_to_none=True)
        self._local_steps += 1

    def on_train_end(
        self, context: base.ExpertClientTrainContext
    ) -> Mapping[str, Any]:
        if self._expected_local_steps is None:
            raise RuntimeError("RFedAGS 服务器端多样性策略没有初始化固定本地步数。")
        if self._local_steps != self._expected_local_steps:
            raise ValueError(
                "客户端实际更新步数与配置不一致："
                f"expected={self._expected_local_steps}, actual={self._local_steps}"
            )
        with torch.no_grad():
            for layer_records in self._records.values():
                for item in layer_records:
                    p_name, a_name = item["p_name"], item["a_name"]
                    aligned_a, aligned_p = align_stiefel_factor_gauge(
                        local_a=self._parameter_map[a_name],
                        local_p=self._parameter_map[p_name],
                        reference_p=self._global_points[p_name],
                    )
                    self._parameter_map[p_name].copy_(aligned_p)
                    self._parameter_map[a_name].copy_(aligned_a)

        state_dict = context.model.state_dict()
        expert_buffers = {
            name: state_dict[name].detach().clone()
            for name in self._expert_buffer_names
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
            }
        }


def _collect_pairwise_overlap_metrics(
    model: nn.Module,
) -> Dict[str, Dict[str, float]]:
    """Return each within-layer expert-pair overlap for server diagnostics."""
    parameter_map = dict(model.named_parameters())
    pairwise: Dict[str, Dict[str, float]] = {}
    for layer_name, layer_records in _get_layer_records(model).items():
        layer_values: Dict[str, float] = {}
        for left_index in range(len(layer_records)):
            for right_index in range(left_index + 1, len(layer_records)):
                left_item = layer_records[left_index]
                right_item = layer_records[right_index]
                left = parameter_map[left_item["p_name"]].detach()
                right = parameter_map[right_item["p_name"]].detach()
                overlap = _pairwise_subspace_overlap(left, right)
                pair_name = (
                    f"expert_{int(left_item['expert_id'])}_"
                    f"expert_{int(right_item['expert_id'])}"
                )
                layer_values[pair_name] = float(overlap.item())
        pairwise[layer_name] = layer_values
    return pairwise


def _count_pairs_above_threshold(
    pairwise: Mapping[str, Mapping[str, float]],
    thresholds: Mapping[str, float],
) -> int:
    return sum(
        1
        for layer_name, pairs in pairwise.items()
        for overlap in pairs.values()
        if float(overlap) > float(thresholds[layer_name])
    )


class RFedAGSServerDiversityExpertAggregator(base.Aggregator):
    """Independent RFedAGS aggregation followed by server-side diversity correction."""

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
        self.diversity_lr = float(
            _method_cfg_get(cfg, "server_diversity_lr", SERVER_DIVERSITY_LR)
        )
        configured_grad_clip = _method_cfg_get(
            cfg, "server_diversity_grad_clip", SERVER_DIVERSITY_GRAD_CLIP
        )
        self.diversity_grad_clip = (
            None if configured_grad_clip is None else float(configured_grad_clip)
        )
        self.subspace_overlap_threshold_cap = float(
            _method_cfg_get(
                cfg, "subspace_overlap_threshold_cap", SUBSPACE_OVERLAP_THRESHOLD_CAP
            )
        )
        self._last_server_correction_diagnostics: Dict[str, Any] = {}

    def compute_weights(
        self,
        client_updates: Sequence[base.ClientUpdate],
    ) -> Dict[int, float]:
        return base.build_uniform_weights(client_updates)

    def _get_context_resources(
        self,
    ) -> tuple[nn.Module, Any]:
        if self.method_context is None:
            raise RuntimeError("RFedAGS 服务器端多样性聚合器尚未注入 MethodContext。")
        model = self.method_context.global_model
        groups = self.method_context.optimization_groups
        if model is None or groups is None:
            raise RuntimeError(
                "RFedAGS 服务器端多样性聚合器需要 MethodContext.global_model 和 "
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
        expert_buffer_names = expert_state_names - parameter_state_names

        if not p_names:
            raise ValueError("RFedAGS 服务器端多样性聚合器没有发现 Stiefel 参数。")
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
            context="RFedAGS 服务器端多样性聚合基础 state_dict",
        )
        new_state_dict = {
            name: value.detach().clone()
            for name, value in source_state.items()
        }

        weights = base.normalize_weights(self.compute_weights(client_updates))
        payloads_by_client: Dict[int, Mapping[str, Any]] = {}
        local_steps_by_client: Dict[int, int] = {}

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

            payloads_by_client[int(update.client_id)] = root

        if set(local_steps_by_client.values()) != {configured_local_steps}:
            raise ValueError(
                "RFedAGS 服务器端多样性聚合要求所有客户端执行相同的固定本地步数；"
                f"当前各客户端步数：{local_steps_by_client}"
            )

        # Euclidean expert parameters use the same client-uniform delta rule
        # as FedAvg. P is deliberately excluded from this path.
        for name in sorted(euclidean_parameter_names):
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
                "RFedAGS 服务器端多样性聚合必须返回完整模型 state_dict；"
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        base.validate_state_dict_shapes(
            reference_state=global_state,
            candidate_state=new_state_dict,
            context="RFedAGS 服务器端多样性聚合结果",
        )
        base.check_finite_state_dict(new_state_dict)

        # model_delta diagnostics describe only the Euclidean expert path;
        # P is aggregated from gradient_stream, not from its endpoint delta.
        diagnostics = self.build_diagnostics(
            client_updates=client_updates,
            weights=weights,
            param_names=sorted(euclidean_parameter_names),
        )
        diagnostics.update(
            {
                "aggregation_rule": "client_uniform",
                "client_participation": "all_clients",
                "update_space": "stiefel_riemannian_gradient_stream_and_euclidean_delta",
                "local_steps": configured_local_steps,
                "subspace_rank": int(_method_cfg_get(self.cfg, "subspace_rank", SUBSPACE_RANK)),
                "server_manifold_lr": self.server_lr,
                "server_diversity_lr": self.diversity_lr,
                "server_diversity_grad_clip": self.diversity_grad_clip,
                "server_diversity_threshold_cap": self.subspace_overlap_threshold_cap,
                "num_stiefel_parameters": len(p_names),
                "num_euclidean_expert_parameters": len(euclidean_parameter_names),
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
        """Apply one diversity-aware Riemannian step to the RFedAGS result."""
        records = _get_layer_records(model)
        thresholds = getattr(model, "_rfedags_server_diversity_thresholds", None)
        if not isinstance(thresholds, Mapping) or set(thresholds) != set(records):
            raise ValueError("服务器端校正缺少与 Stiefel 参数分组匹配的逐层阈值。")

        parameter_map = dict(model.named_parameters())
        p_names = [
            item["p_name"]
            for layer_records in records.values()
            for item in layer_records
        ]
        if not p_names:
            raise ValueError("服务器端校正没有发现 Stiefel 专家子空间参数。")
        try:
            p_parameters = [parameter_map[name] for name in p_names]
        except KeyError as exc:
            raise ValueError(f"服务器端模型缺少 Stiefel 参数：{exc}") from exc

        # Compute all gradients at the same RFedAGS intermediate state before
        # changing any P block.
        with torch.enable_grad():
            loss_before, overlap_before = compute_subspace_diversity_loss(
                model,
                thresholds=thresholds,
            )
            if not _tensor_is_finite(loss_before):
                raise ValueError("服务器端差异性损失包含 NaN 或 Inf。")
            gradients = torch.autograd.grad(
                loss_before,
                p_parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )

        pairwise_overlap_before = _collect_pairwise_overlap_metrics(model)
        pairs_above_threshold_before = _count_pairs_above_threshold(
            pairwise_overlap_before,
            thresholds,
        )
        before_points = [parameter.detach().clone() for parameter in p_parameters]

        # Convert ambient Euclidean gradients into Riemannian gradients
        # before measuring/clipping their product-manifold norm.
        riemannian_gradients = [
            stiefel_tangent_projection(
                parameter.detach(),
                gradient.detach(),
            )
            for parameter, gradient in zip(p_parameters, gradients)
        ]
        if any(
            not _tensor_is_finite(gradient)
            for gradient in riemannian_gradients
        ):
            raise ValueError("服务器端黎曼差异性梯度包含 NaN 或 Inf。")

        gradient_norm = math.sqrt(
            sum(
                float(torch.sum(gradient.to(torch.float64).square()).item())
                for gradient in riemannian_gradients
            )
        )
        if not math.isfinite(gradient_norm):
            raise ValueError("服务器端差异性梯度范数包含 NaN 或 Inf。")

        clip_scale = 1.0
        if (
            self.diversity_grad_clip is not None
            and gradient_norm > self.diversity_grad_clip
            and gradient_norm > 0.0
        ):
            clip_scale = self.diversity_grad_clip / gradient_norm
        clipped_gradient_norm = gradient_norm * clip_scale

        loss_before_value = float(loss_before.detach().item())
        parameter_step_norm = 0.0
        applied = False
        skip_reason: Optional[str] = None

        if loss_before_value <= 0.0:
            skip_reason = "no_active_overlap_pairs"
        elif clipped_gradient_norm <= 0.0:
            skip_reason = "zero_riemannian_gradient"
        else:
            candidate_parameters = []
            with torch.no_grad():
                for parameter, gradient in zip(
                    p_parameters,
                    riemannian_gradients,
                ):
                    candidate = qr_retraction(
                        parameter.detach()
                        - self.diversity_lr * clip_scale * gradient
                    )
                    if not _tensor_is_finite(candidate):
                        raise ValueError(
                            "服务器端差异性校正生成了 NaN 或 Inf 参数。"
                        )
                    candidate_parameters.append(candidate)

                parameter_step_norm = math.sqrt(
                    sum(
                        float(
                            torch.sum(
                                (candidate - before).to(torch.float64).square()
                            ).item()
                        )
                        for candidate, before in zip(
                            candidate_parameters,
                            before_points,
                        )
                    )
                )

                if not math.isfinite(parameter_step_norm):
                    raise ValueError("服务器端参数更新范数包含 NaN 或 Inf。")

                applied = parameter_step_norm > 0.0
                if applied:
                    for parameter, candidate in zip(
                        p_parameters,
                        candidate_parameters,
                    ):
                        parameter.copy_(candidate)
                else:
                    skip_reason = "zero_effective_parameter_step"

        with torch.no_grad():
            loss_after, overlap_after = compute_subspace_diversity_loss(
                model,
                thresholds=thresholds,
            )
            loss_after_value = float(loss_after.item())

        if not math.isfinite(loss_after_value) or not math.isfinite(parameter_step_norm):
            raise ValueError("服务器端多样性校正产生了 NaN 或 Inf。")

        pairwise_overlap_after = _collect_pairwise_overlap_metrics(model)
        self._last_server_correction_diagnostics = {
            "server_diversity_applied": bool(applied),
            "server_diversity_skip_reason": skip_reason,
            "server_diversity_loss_before": loss_before_value,
            "server_diversity_loss_after": loss_after_value,
            "server_diversity_loss_reduction": (
                loss_before_value - loss_after_value
            ),
            "server_subspace_overlap_before_by_layer": dict(overlap_before),
            "server_subspace_overlap_after_by_layer": dict(overlap_after),
            "server_subspace_pair_overlap_before": pairwise_overlap_before,
            "server_subspace_pair_overlap_after": pairwise_overlap_after,
            "server_diversity_thresholds_by_layer": {
                str(name): float(value)
                for name, value in thresholds.items()
            },
            "server_diversity_pairs_above_threshold_before": (
                pairs_above_threshold_before
            ),
            "server_diversity_pairs_above_threshold_after": (
                _count_pairs_above_threshold(pairwise_overlap_after, thresholds)
            ),
            "server_diversity_gradient_norm": gradient_norm,
            "server_diversity_clipped_gradient_norm": clipped_gradient_norm,
            "server_diversity_parameter_step_norm": parameter_step_norm,
            "server_diversity_lr": self.diversity_lr,
            "server_diversity_grad_clip": self.diversity_grad_clip,
            "server_diversity_threshold_cap": self.subspace_overlap_threshold_cap,
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
                identity = torch.eye(p.shape[1], device=p.device, dtype=p.dtype)
                errors.append(
                    float(torch.linalg.norm(p.transpose(0, 1) @ p - identity).item())
                )
        reconstruction_errors = getattr(
            model, "_rfedags_server_diversity_reconstruction_errors", {}
        )
        diagnostics = {
            "max_stiefel_orthogonality_error": max(errors) if errors else 0.0,
            "mean_stiefel_orthogonality_error": (
                sum(errors) / len(errors) if errors else 0.0
            ),
            "max_initial_reconstruction_error": (
                max(reconstruction_errors.values()) if reconstruction_errors else 0.0
            ),
            "mean_initial_reconstruction_error": (
                sum(reconstruction_errors.values()) / len(reconstruction_errors)
                if reconstruction_errors else 0.0
            ),
            "initial_subspace_overlap_by_layer": dict(
                getattr(model, "_rfedags_server_diversity_initial_overlaps", {})
            ),
            "subspace_thresholds_by_layer": dict(
                getattr(model, "_rfedags_server_diversity_thresholds", {})
            ),
        }
        diagnostics.update(self._last_server_correction_diagnostics)
        return diagnostics



 
def build_expert_aggregator(cfg: Any) -> base.Aggregator:
    return RFedAGSServerDiversityExpertAggregator(cfg=cfg)


def build_expert_client_training_strategy(
    cfg: Any,
    client_id: int,
) -> base.ExpertClientTrainingStrategy:
    return RFedAGSServerDiversityClientStrategy(cfg=cfg, client_id=client_id)


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
        model_builder=build_rfedags_server_diversity_model,
    )


if __name__ == "__main__":
    raise SystemExit(main())

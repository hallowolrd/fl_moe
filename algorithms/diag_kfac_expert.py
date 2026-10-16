from __future__ import annotations

"""Diagonal K-FAC expert aggregation experiment.

All common experiment behavior is owned by base.py. This file preserves the
expert-only Diagonal K-FAC aggregation implementation from commit 77e980c.
"""

# base.py must run its deterministic pre-PyTorch bootstrap first.
import base

Aggregator = base.Aggregator
build_sample_weights = base.build_sample_weights
AggregationResult = base.AggregationResult
ClientUpdate = base.ClientUpdate
check_finite_state_dict = base.check_finite_state_dict
clone_state_dict = base.clone_state_dict
normalize_weights = base.normalize_weights

ALGORITHM_NAME = "diag_kfac_expert"

EMBEDDED_METHOD_CONFIG = {
    "agg": {
        "non_expert": {"method": "uniform"},
        "expert": {"method": ALGORITHM_NAME},
    },
    "diag_kfac": {
        "collect": True,
        "weight_mode": "sample_weighted",
        "solve_scope": "global_expert",
        "solve_mode": "adam",
        "use_server_validation": False,
        "model_selection": "final_step",
        "server_steps": 50,
        "server_lr": 0.003,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "adam_eps": 0.01,
        "damping": 0.01,
        "use_damping": True,
        "min_count": 512,
        "fallback": "none",
        "include_bias": True,
        "fisher_timing": "after_train",
        "model_mode": "eval",
        "log_detail": True,
    },
}

METHOD_CONFIG_DEFAULTS = {
    "diag_kfac": {
        "collect": False,
        "weight_mode": "sample_weighted",
        "solve_scope": "per_layer",
        "solve_mode": "cg",
        "server_steps": 5,
        "server_lr": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "adam_eps": 0.01,
        "cg_tol": 1.0e-8,
        "damping": 0.0,
        "use_damping": False,
        "min_count": 1,
        "fallback": "none",
        "include_bias": True,
        "fisher_timing": "after_train",
        "model_mode": "eval",
        "max_batches": 0,
        "expert_name_pattern": "experts.",
        "use_server_validation": False,
        "model_selection": "final_step",
        "log_detail": True,
    },
}

import argparse
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

unpack_batch = base.unpack_batch
extract_logits = base.extract_logits
ConfigError = base.ConfigError



# ============================================================================
# Bundled from fl/diag_kfac.py
# ============================================================================


DiagKFACLayerPayload = Dict[str, Any]
ExpertDiagKFACPayload = Dict[str, DiagKFACLayerPayload]


@dataclass

@dataclass
class _DiagKFACLayerBuffer:
    """
    单个 expert Linear 层的 Diagonal K-FAC 统计缓存。

    从标准 K-FAC 因子
        A = E[a a^T]
        B = E[g g^T]
    只保留对角项：
        A_diag = E[a ⊙ a]
        B_diag = E[g ⊙ g]

    因此该层的对角 K-FAC Fisher 为：
        D = B_diag[:, None] * A_diag[None, :]

    对任意参数矩阵 X：
        F_diag-kfac(X) = D ⊙ X

    include_bias=True 时仍使用 a_aug=[a,1]，因此 bias 作为 W_aug 最后一列
    与 weight 一起参与同一个 diagonal K-FAC FedFisher 求解。
    """

    module_name: str
    module: nn.Linear
    include_bias: bool
    A_diag_sum: Optional[torch.Tensor] = None
    B_diag_sum: Optional[torch.Tensor] = None
    a_count: int = 0
    b_count: int = 0

    def add_activation(self, activation: torch.Tensor) -> None:
        """累计 A_diag_sum += sum_items(a ⊙ a)。"""
        if activation is None:
            return

        a = _flatten_last_dim(
            tensor=activation,
            expected_dim=self.module.in_features,
            tensor_name=f"{self.module_name}.activation",
        )
        if a.numel() == 0 or a.size(0) <= 0:
            return

        a = a.detach().float()
        if self.include_bias and self.module.bias is not None:
            ones = torch.ones(a.size(0), 1, device=a.device, dtype=a.dtype)
            a = torch.cat([a, ones], dim=1)

        A_diag_batch = torch.sum(a * a, dim=0)
        if self.A_diag_sum is None:
            self.A_diag_sum = torch.zeros_like(A_diag_batch)
        self.A_diag_sum.add_(A_diag_batch)
        self.a_count += int(a.size(0))

    def add_grad_output(self, grad_output: torch.Tensor) -> None:
        """累计 B_diag_sum += sum_items(g ⊙ g)。"""
        if grad_output is None:
            return

        g = _flatten_last_dim(
            tensor=grad_output,
            expected_dim=self.module.out_features,
            tensor_name=f"{self.module_name}.grad_output",
        )
        if g.numel() == 0 or g.size(0) <= 0:
            return

        g = g.detach().float()
        B_diag_batch = torch.sum(g * g, dim=0)
        if self.B_diag_sum is None:
            self.B_diag_sum = torch.zeros_like(B_diag_batch)
        self.B_diag_sum.add_(B_diag_batch)
        self.b_count += int(g.size(0))

    def to_payload(self, min_count: int) -> Optional[DiagKFACLayerPayload]:
        """导出 A_diag / B_diag / count，不构造完整 A、B 矩阵。"""
        if self.A_diag_sum is None or self.B_diag_sum is None:
            return None
        if self.a_count <= 0 or self.b_count <= 0:
            return None

        count = min(int(self.a_count), int(self.b_count))
        if count < int(min_count):
            return None

        A_diag = self.A_diag_sum / float(self.a_count)
        B_diag = self.B_diag_sum / float(self.b_count)
        if not torch.isfinite(A_diag).all() or not torch.isfinite(B_diag).all():
            return None
        if torch.any(A_diag < 0) or torch.any(B_diag < 0):
            return None

        bias_name = None
        if self.module.bias is not None:
            bias_name = f"{self.module_name}.bias"

        return {
            "module_name": self.module_name,
            "weight_name": f"{self.module_name}.weight",
            "bias_name": bias_name,
            "A_diag": A_diag.detach().cpu(),
            "B_diag": B_diag.detach().cpu(),
            "count": int(count),
            "a_count": int(self.a_count),
            "b_count": int(self.b_count),
            "include_bias": bool(self.include_bias and self.module.bias is not None),
            "in_features": int(self.module.in_features),
            "out_features": int(self.module.out_features),
            # 保留与 full K-FAC 可直接比较的 trace 诊断：sum(diag(A)) == trace(A)。
            "trace_A": float(A_diag.detach().float().sum().cpu().item()),
            "trace_B": float(B_diag.detach().float().sum().cpu().item()),
            "mean_A_diag": float(A_diag.detach().float().mean().cpu().item()),
            "mean_B_diag": float(B_diag.detach().float().mean().cpu().item()),
            "max_A_diag": float(A_diag.detach().float().max().cpu().item()),
            "max_B_diag": float(B_diag.detach().float().max().cpu().item()),
        }



def collect_expert_diag_kfac(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: torch.device | str | None = None,
    cfg: Any = None,
) -> ExpertDiagKFACPayload:
    """
    在本地训练完成后的 local_model 上，额外跑一遍数据来采集 expert Linear 层的 Diagonal K-FAC 因子。

    返回格式：
        {
            "switch_layers.0.switch_ffn.experts.2.0": {
                "module_name": ...,
                "weight_name": "...weight",
                "bias_name": "...bias",
                "A": Tensor[in_dim(+1), in_dim(+1)],
                "B": Tensor[out_dim, out_dim],
                "count": int,
                ...
            },
            ...
        }

    设计约束：
        1. 只采集 module name 包含 experts. 的 nn.Linear。
        2. 默认使用 CrossEntropyLoss(reduction="sum")，避免 mean loss 缩放梯度。
        3. 默认 model.eval() 采集，避免 Dropout / BN 引入额外随机性。
        4. 不修改训练逻辑，不做 optimizer.step()。
        5. 这里只支持 after_train 采集时机，和 FedFisher 的“先得到本地模型再算 Fisher”流程对齐。
    """
    if device is None:
        device = _infer_model_device(model)

    device = torch.device(device)

    include_bias = bool(_cfg_get(cfg, "diag_kfac.include_bias", True))
    min_count = int(_cfg_get(cfg, "diag_kfac.min_count", 1))
    max_batches = int(_cfg_get(cfg, "diag_kfac.max_batches", 0))
    expert_name_pattern = str(_cfg_get(cfg, "diag_kfac.expert_name_pattern", "experts."))
    model_mode = str(_cfg_get(cfg, "diag_kfac.model_mode", "eval")).lower().strip()
    fisher_timing = str(
        _cfg_get(
            cfg,
            "diag_kfac.fisher_timing",
            _cfg_get(cfg, "diag_kfac.collect_timing", "after_train"),
        )
    ).lower().strip()

    if fisher_timing != "after_train":
        raise ValueError(
            "当前 collect_expert_diag_kfac 只支持 diag_kfac.fisher_timing=after_train。"
            f"当前值：{fisher_timing}。"
            "请在客户端本地训练完成后再单独采集 Diagonal K-FAC。"
        )

    if min_count <= 0:
        min_count = 1

    buffers: Dict[str, _DiagKFACLayerBuffer] = {}
    handles = []

    for module_name, module in model.named_modules():
        if not _is_expert_linear(
            module_name=module_name,
            module=module,
            expert_name_pattern=expert_name_pattern,
        ):
            continue

        buffers[module_name] = _DiagKFACLayerBuffer(
            module_name=module_name,
            module=module,
            include_bias=include_bias,
        )

    if len(buffers) == 0:
        return {}

    for module_name, module_buffer in buffers.items():
        module = module_buffer.module

        def forward_hook(
            layer: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            name: str = module_name,
        ) -> None:
            if len(inputs) == 0:
                return

            buffers[name].add_activation(inputs[0])

        def backward_hook(
            layer: nn.Module,
            grad_input: tuple[Optional[torch.Tensor], ...],
            grad_output: tuple[Optional[torch.Tensor], ...],
            name: str = module_name,
        ) -> None:
            if len(grad_output) == 0:
                return

            buffers[name].add_grad_output(grad_output[0])

        handles.append(module.register_forward_hook(forward_hook))
        handles.append(module.register_full_backward_hook(backward_hook))

    was_training = bool(model.training)
    model.to(device)

    if model_mode == "train":
        model.train()
    else:
        model.eval()

    sum_criterion = _build_sum_criterion(
        cfg=cfg,
        fallback_criterion=criterion,
    )
    sum_criterion.to(device)

    model.zero_grad(set_to_none=True)

    try:
        with torch.enable_grad():
            for batch_idx, batch in enumerate(train_loader):
                if max_batches > 0 and batch_idx >= max_batches:
                    break

                images, targets = unpack_batch(batch)

                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                model.zero_grad(set_to_none=True)

                outputs = model(images)
                logits = extract_logits(outputs)

                loss = sum_criterion(logits, targets)

                if not torch.isfinite(loss):
                    continue

                loss.backward()

                # 只采集 Fisher，不更新参数。
                model.zero_grad(set_to_none=True)
    finally:
        for handle in handles:
            handle.remove()

        model.zero_grad(set_to_none=True)
        model.train(was_training)

    payload: ExpertDiagKFACPayload = {}

    for module_name, module_buffer in buffers.items():
        layer_payload = module_buffer.to_payload(min_count=min_count)

        if layer_payload is None:
            continue

        layer_payload["fisher_timing"] = fisher_timing
        layer_payload["collect_timing"] = fisher_timing
        layer_payload["model_mode"] = model_mode
        layer_payload["max_batches"] = int(max_batches)
        layer_payload["expert_name_pattern"] = expert_name_pattern

        payload[module_name] = layer_payload

    return payload



def summarize_expert_diag_kfac(payload: ExpertDiagKFACPayload) -> Dict[str, Any]:
    """生成轻量 Diagonal K-FAC 客户端诊断，不包含因子 tensor 本体。"""
    if not payload:
        return {
            "num_layers": 0,
            "total_count": 0,
            "mean_count": 0.0,
            "mean_trace_A": 0.0,
            "mean_trace_B": 0.0,
            "max_trace_A": 0.0,
            "max_trace_B": 0.0,
            "mean_A_diag": 0.0,
            "mean_B_diag": 0.0,
            "max_A_diag": 0.0,
            "max_B_diag": 0.0,
            "fisher_timing": "",
            "model_mode": "",
        }

    counts = [int(item["count"]) for item in payload.values()]
    trace_A = [float(item["trace_A"]) for item in payload.values()]
    trace_B = [float(item["trace_B"]) for item in payload.values()]
    mean_A_diag = [float(item.get("mean_A_diag", 0.0)) for item in payload.values()]
    mean_B_diag = [float(item.get("mean_B_diag", 0.0)) for item in payload.values()]
    max_A_diag = [float(item.get("max_A_diag", 0.0)) for item in payload.values()]
    max_B_diag = [float(item.get("max_B_diag", 0.0)) for item in payload.values()]

    fisher_timings = sorted({
        str(item.get("fisher_timing", item.get("collect_timing", "")))
        for item in payload.values()
        if str(item.get("fisher_timing", item.get("collect_timing", ""))) != ""
    })
    model_modes = sorted({
        str(item.get("model_mode", ""))
        for item in payload.values()
        if str(item.get("model_mode", "")) != ""
    })

    return {
        "num_layers": int(len(payload)),
        "total_count": int(sum(counts)),
        "mean_count": float(sum(counts) / max(len(counts), 1)),
        "mean_trace_A": float(sum(trace_A) / max(len(trace_A), 1)),
        "mean_trace_B": float(sum(trace_B) / max(len(trace_B), 1)),
        "max_trace_A": float(max(trace_A)),
        "max_trace_B": float(max(trace_B)),
        "mean_A_diag": float(sum(mean_A_diag) / max(len(mean_A_diag), 1)),
        "mean_B_diag": float(sum(mean_B_diag) / max(len(mean_B_diag), 1)),
        "max_A_diag": float(max(max_A_diag)),
        "max_B_diag": float(max(max_B_diag)),
        "fisher_timing": fisher_timings[0] if len(fisher_timings) == 1 else ",".join(fisher_timings),
        "model_mode": model_modes[0] if len(model_modes) == 1 else ",".join(model_modes),
    }



def _is_expert_linear(
    module_name: str,
    module: nn.Module,
    expert_name_pattern: str,
) -> bool:
    """判断一个 module 是否是 expert 内部的 Linear 层。"""
    if not isinstance(module, nn.Linear):
        return False

    if expert_name_pattern not in module_name:
        return False

    return True


def _flatten_last_dim(
    tensor: torch.Tensor,
    expected_dim: int,
    tensor_name: str,
) -> torch.Tensor:
    """
    把 Linear 的输入或 grad_output 展平成 [num_items, feature_dim]。

    支持：
        [N, D]
        [B, N, D]
        [B, ..., D]
    """
    if tensor is None:
        raise ValueError(f"{tensor_name} 为空。")

    if tensor.dim() == 0:
        raise ValueError(f"{tensor_name} 维度错误：{tuple(tensor.shape)}")

    if tensor.size(-1) != int(expected_dim):
        raise ValueError(
            f"{tensor_name} 最后一维不匹配："
            f"实际={tensor.size(-1)}, 期望={expected_dim}, "
            f"shape={tuple(tensor.shape)}"
        )

    if tensor.dim() == 1:
        return tensor.reshape(1, -1)

    return tensor.reshape(-1, tensor.size(-1))


def _build_sum_criterion(
    cfg: Any,
    fallback_criterion: Optional[nn.Module] = None,
) -> nn.Module:
    """
    构建 Diagonal K-FAC 采集用 loss。

    这里强制 reduction='sum'。
    如果直接复用训练时 CrossEntropyLoss 的 mean reduction，
    backward 得到的 delta 会被 batch size 缩小，Diagonal K-FAC 尺度会不稳定。
    """
    label_smoothing = float(_cfg_get(cfg, "label_smooth", 0.0))

    if isinstance(fallback_criterion, nn.CrossEntropyLoss):
        label_smoothing = float(
            getattr(fallback_criterion, "label_smoothing", label_smoothing)
        )
        ignore_index = int(getattr(fallback_criterion, "ignore_index", -100))
        weight = getattr(fallback_criterion, "weight", None)

        return nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction="sum",
            label_smoothing=label_smoothing,
        )

    return nn.CrossEntropyLoss(
        reduction="sum",
        label_smoothing=label_smoothing,
    )


def _infer_model_device(model: nn.Module) -> torch.device:
    """从模型参数推断 device。"""
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("模型没有参数，无法推断 device。") from exc



def validate_method_config(cfg: Mapping[str, Any]) -> None:
    """检查 Diagonal K-FAC / FedFisher expert 聚合配置。"""
    diag_kfac_cfg = cfg.get("diag_kfac", {})

    if not isinstance(diag_kfac_cfg, Mapping):
        raise ConfigError("diag_kfac 必须是 dict。")

    weight_mode = str(diag_kfac_cfg.get("weight_mode", "sample_weighted")).lower().strip()
    if weight_mode not in {"routed_count", "sample_weighted", "uniform"}:
        raise ConfigError(
            f"不支持的 diag_kfac.weight_mode：{weight_mode}。"
            "当前支持：routed_count, sample_weighted, uniform"
        )

    solve_scope = str(diag_kfac_cfg.get("solve_scope", "per_layer")).lower().strip()
    if solve_scope not in {"per_layer", "global_expert"}:
        raise ConfigError(
            f"不支持的 diag_kfac.solve_scope：{solve_scope}。"
            "当前支持：per_layer, global_expert"
        )

    solve_mode = str(diag_kfac_cfg.get("solve_mode", "cg")).lower().strip()
    if solve_mode not in {"cg", "gd", "adam"}:
        raise ConfigError(
            f"不支持的 diag_kfac.solve_mode：{solve_mode}。"
            "当前支持：cg, gd, adam"
        )

    if solve_scope == "global_expert" and solve_mode == "cg":
        raise ConfigError(
            "diag_kfac.solve_scope=global_expert 时不建议使用 solve_mode=cg。"
            "请使用 gd 或 adam。"
        )

    if solve_scope == "per_layer" and solve_mode in {"gd", "adam"}:
        raise ConfigError(
            "diag_kfac.solve_scope=per_layer 当前只支持 solve_mode=cg。"
            "如果要使用 gd/adam，请设置 solve_scope=global_expert。"
        )

    server_steps = int(diag_kfac_cfg.get("server_steps", 5))
    if server_steps < 0:
        raise ConfigError(
            f"diag_kfac.server_steps 不能小于 0，当前值：{server_steps}"
        )

    server_lr = float(diag_kfac_cfg.get("server_lr", 0.01))
    if server_lr <= 0:
        raise ConfigError(
            f"diag_kfac.server_lr 必须大于 0，当前值：{server_lr}"
        )

    adam_beta1 = float(diag_kfac_cfg.get("adam_beta1", 0.9))
    adam_beta2 = float(diag_kfac_cfg.get("adam_beta2", 0.99))

    if not (0.0 <= adam_beta1 < 1.0):
        raise ConfigError(
            f"diag_kfac.adam_beta1 必须在 [0, 1) 范围内，当前值：{adam_beta1}"
        )

    if not (0.0 <= adam_beta2 < 1.0):
        raise ConfigError(
            f"diag_kfac.adam_beta2 必须在 [0, 1) 范围内，当前值：{adam_beta2}"
        )

    adam_eps = float(diag_kfac_cfg.get("adam_eps", 0.01))
    if adam_eps <= 0:
        raise ConfigError(
            f"diag_kfac.adam_eps 必须大于 0，当前值：{adam_eps}"
        )

    cg_tol = float(diag_kfac_cfg.get("cg_tol", 1.0e-8))
    if cg_tol < 0:
        raise ConfigError(
            f"diag_kfac.cg_tol 不能小于 0，当前值：{cg_tol}"
        )

    damping = float(diag_kfac_cfg.get("damping", 0.0))
    if damping < 0:
        raise ConfigError(
            f"diag_kfac.damping 不能小于 0，当前值：{damping}"
        )

    min_count = int(diag_kfac_cfg.get("min_count", 1))
    if min_count <= 0:
        raise ConfigError(
            f"diag_kfac.min_count 必须大于 0，当前值：{min_count}"
        )

    max_batches = int(diag_kfac_cfg.get("max_batches", 0))
    if max_batches < 0:
        raise ConfigError(
            f"diag_kfac.max_batches 不能小于 0，当前值：{max_batches}"
        )

    fallback = str(diag_kfac_cfg.get("fallback", "none")).lower().strip()
    if fallback not in {"none", "sample_weighted"}:
        raise ConfigError(
            f"不支持的 diag_kfac.fallback：{fallback}。"
            "当前支持：none, sample_weighted"
        )

    fisher_timing = str(diag_kfac_cfg.get("fisher_timing", "after_train")).lower().strip()
    if fisher_timing != "after_train":
        raise ConfigError(
            f"当前只支持 diag_kfac.fisher_timing=after_train，当前值：{fisher_timing}"
        )

    model_mode = str(diag_kfac_cfg.get("model_mode", "eval")).lower().strip()
    if model_mode not in {"eval", "train"}:
        raise ConfigError(
            f"不支持的 diag_kfac.model_mode：{model_mode}。"
            "当前支持：eval, train"
        )

    model_selection = str(diag_kfac_cfg.get("model_selection", "final_step")).lower().strip()
    if model_selection != "final_step":
        raise ConfigError(
            "当前主实验不支持 server validation 选 best，"
            f"diag_kfac.model_selection 必须是 final_step，当前值：{model_selection}"
        )

    use_server_validation = bool(diag_kfac_cfg.get("use_server_validation", False))
    if use_server_validation:
        raise ConfigError(
            "当前主实验不使用 server validation，"
            "请设置 diag_kfac.use_server_validation=false。"
        )



class DiagKFACExpertAggregator(Aggregator):
    """
    基于 Diagonal K-FAC Fisher 的专家参数聚合器。

    这个聚合器只用于 expert 参数聚合，不用于 non_expert 参数。

    paper-like FedFisher 目标：
        min_W sum_i p_i / 2 * <W - W_i, F_i(W - W_i)>

    其中：
        F_i ≈ diag(B_i) ⊗ diag(A_i)

    对 Linear 层，Diagonal K-FAC matvec 为：
        F_i(W) ≈ (B_diag_i[:,None] * A_diag_i[None,:]) ⊙ W

    默认 paper-like 模式不再把 routed count 当聚合权重，也不再默认加入
    damping 软正则。routed count 只用于判断该 expert layer 的 Diagonal K-FAC 是否有效。

    支持两种求解范围：
        1. per_layer：逐个 expert Linear layer 求解，兼容旧实现。
        2. global_expert：把所有 expert layer 放进同一个服务端优化过程，
           等价于在 expert 参数空间上做一个 block-diagonal Diagonal K-FAC FedFisher 求解。

    支持三种求解方式：
        1. cg：Conjugate Gradient 求解线性系统。
        2. gd：FedFisher Algorithm 1 风格固定步数梯度下降。
        3. adam：作者实践中使用的 Adam-like 服务端优化，但这里不使用
           server validation 选 best，固定返回最后一步。
    """

    @property
    def method_name(self) -> str:
        """返回当前聚合方法名称。"""
        return ALGORITHM_NAME

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """
        为了满足 Aggregator 接口，返回样本数权重。

        注意：
            diag_kfac_expert 的主聚合逻辑不走普通加权 delta。
            这里的权重主要用于 fallback=sample_weighted，以及
            diag_kfac.weight_mode=sample_weighted 时的客户端级权重。
        """
        return build_sample_weights(client_updates)

    def aggregate(
        self,
        global_state: Mapping[str, torch.Tensor],
        client_updates: Sequence[ClientUpdate],
        param_names: Optional[Iterable[str]] = None,
        base_state: Optional[Mapping[str, torch.Tensor]] = None,
        strict: bool = True,
    ) -> AggregationResult:
        """
        执行迭代版 Diagonal K-FAC expert 聚合。

        参数：
            global_state:
                本轮聚合前的全局参数。

            client_updates:
                本轮客户端更新。

            param_names:
                expert 参数名列表。

            base_state:
                上一步 non_expert 聚合后的 state_dict。
                expert 聚合结果会写到这个基础 state_dict 上。

            strict:
                是否严格检查缺失字段。
        """
        self._validate_client_updates(client_updates)

        if self.param_group_name != "expert":
            raise ValueError("diag_kfac_expert 只能用于 expert 参数聚合。")

        target_param_names = _resolve_param_names(
            global_state=global_state,
            param_names=param_names,
        )
        target_param_set = set(target_param_names)

        raw_weights = self.compute_weights(client_updates)
        sample_weights = normalize_weights(raw_weights)

        if base_state is None:
            new_state_dict = clone_state_dict(global_state)
        else:
            new_state_dict = clone_state_dict(base_state)

        min_count = int(_cfg_get(self.cfg, "diag_kfac.min_count", 512))
        solver_steps = int(_cfg_get(self.cfg, "diag_kfac.server_steps", 50))
        cg_tol = float(_cfg_get(self.cfg, "diag_kfac.cg_tol", 1.0e-8))
        server_lr = float(_cfg_get(self.cfg, "diag_kfac.server_lr", 0.003))
        adam_beta1 = float(_cfg_get(self.cfg, "diag_kfac.adam_beta1", 0.9))
        adam_beta2 = float(_cfg_get(self.cfg, "diag_kfac.adam_beta2", 0.99))
        adam_eps = float(_cfg_get(self.cfg, "diag_kfac.adam_eps", 0.01))
        damping = float(_cfg_get(self.cfg, "diag_kfac.damping", 0.01))
        use_damping = bool(_cfg_get(self.cfg, "diag_kfac.use_damping", True))
        fallback = str(_cfg_get(self.cfg, "diag_kfac.fallback", "none")).lower().strip()
        weight_mode = str(_cfg_get(self.cfg, "diag_kfac.weight_mode", "sample_weighted")).lower().strip()
        solve_scope = str(_cfg_get(self.cfg, "diag_kfac.solve_scope", "global_expert")).lower().strip()
        solve_mode = str(_cfg_get(self.cfg, "diag_kfac.solve_mode", "adam")).lower().strip()
        fisher_timing = str(_cfg_get(self.cfg, "diag_kfac.fisher_timing", "after_train")).lower().strip()

        if min_count <= 0:
            min_count = 1

        if solver_steps < 0:
            raise ValueError(f"diag_kfac.server_steps 不能小于 0，当前值：{solver_steps}")

        if cg_tol < 0:
            raise ValueError(f"diag_kfac.cg_tol 不能小于 0，当前值：{cg_tol}")

        if server_lr < 0:
            raise ValueError(f"diag_kfac.server_lr 不能小于 0，当前值：{server_lr}")

        if damping < 0:
            raise ValueError(f"diag_kfac.damping 不能小于 0，当前值：{damping}")

        if not use_damping:
            damping = 0.0

        _validate_choice(
            name="diag_kfac.weight_mode",
            value=weight_mode,
            choices=("routed_count", "sample_weighted", "uniform"),
        )
        _validate_choice(
            name="diag_kfac.solve_scope",
            value=solve_scope,
            choices=("per_layer", "global_expert"),
        )
        _validate_choice(
            name="diag_kfac.solve_mode",
            value=solve_mode,
            choices=("cg", "gd", "adam"),
        )

        layer_names = _collect_diag_kfac_layer_names(client_updates)
        layer_groups: List[Dict[str, Any]] = []
        skipped_layers: List[str] = []

        for layer_name in layer_names:
            entries = _collect_valid_layer_entries(
                layer_name=layer_name,
                client_updates=client_updates,
                global_state=global_state,
                target_param_set=target_param_set,
                min_count=min_count,
                strict=False,
            )

            if len(entries) == 0:
                skipped_layers.append(layer_name)
                continue

            reference = entries[0]
            weight_name = reference["weight_name"]
            bias_name = reference["bias_name"]
            include_bias = bool(reference["include_bias"])

            if weight_name not in target_param_set:
                skipped_layers.append(layer_name)
                continue

            if bias_name is not None and bias_name not in target_param_set:
                include_bias = False
                bias_name = None

            layer_groups.append(
                {
                    "layer_name": layer_name,
                    "entries": entries,
                    "weight_name": weight_name,
                    "bias_name": bias_name,
                    "include_bias": include_bias,
                }
            )

        solved_params = set()
        fallback_params = set()
        valid_client_ids = set()
        diag_kfac_client_counts: Dict[int, int] = {}
        diag_kfac_layer_weights: Dict[str, Dict[int, float]] = {}

        valid_layers = 0
        valid_client_layers = 0
        total_count = 0
        global_expert_param_count = 0

        trace_A_values: List[float] = []
        trace_B_values: List[float] = []
        residual_norm_values: List[float] = []
        delta_norm_values: List[float] = []
        solver_delta_norm_values: List[float] = []
        solver_grad_norm_values: List[float] = []
        solver_update_norm_values: List[float] = []

        if len(layer_groups) > 0:
            if solve_scope == "global_expert":
                try:
                    global_result = _solve_global_expert_layers(
                        global_state=global_state,
                        client_updates=client_updates,
                        sample_weights=sample_weights,
                        layer_groups=layer_groups,
                        weight_mode=weight_mode,
                        solve_mode=solve_mode,
                        solver_steps=solver_steps,
                        cg_tol=cg_tol,
                        server_lr=server_lr,
                        adam_beta1=adam_beta1,
                        adam_beta2=adam_beta2,
                        adam_eps=adam_eps,
                        damping=damping,
                        use_damping=use_damping,
                        strict=strict,
                    )
                except Exception:
                    if strict:
                        raise

                    global_result = {
                        "solutions": {},
                        "diagnostics": [],
                        "skipped_layers": [
                            str(group["layer_name"])
                            for group in layer_groups
                        ],
                        "solver_grad_norm_values": [],
                        "solver_update_norm_values": [],
                    }

                skipped_layers.extend(global_result.get("skipped_layers", []))
                solver_grad_norm_values.extend(
                    global_result.get("solver_grad_norm_values", [])
                )
                solver_update_norm_values.extend(
                    global_result.get("solver_update_norm_values", [])
                )

                for layer_result in global_result.get("diagnostics", []):
                    weight_name = str(layer_result["weight_name"])
                    bias_name = layer_result.get("bias_name", None)
                    solved_weight = layer_result["solved_weight"]
                    solved_bias = layer_result.get("solved_bias", None)

                    new_state_dict[weight_name] = solved_weight.detach().cpu()
                    solved_params.add(weight_name)

                    if bias_name is not None and solved_bias is not None:
                        new_state_dict[str(bias_name)] = solved_bias.detach().cpu()
                        solved_params.add(str(bias_name))

                    _accumulate_layer_diagnostics(
                        layer_diag=layer_result,
                        valid_client_ids=valid_client_ids,
                        diag_kfac_client_counts=diag_kfac_client_counts,
                        diag_kfac_layer_weights=diag_kfac_layer_weights,
                        trace_A_values=trace_A_values,
                        trace_B_values=trace_B_values,
                        residual_norm_values=residual_norm_values,
                        delta_norm_values=delta_norm_values,
                        solver_delta_norm_values=solver_delta_norm_values,
                    )

                    valid_layers += 1
                    valid_client_layers += int(layer_result["valid_clients"])
                    total_count += int(layer_result["total_count"])
                    global_expert_param_count += int(layer_result["param_count"])
            else:
                for group in layer_groups:
                    layer_name = str(group["layer_name"])
                    weight_name = str(group["weight_name"])
                    bias_name = group.get("bias_name", None)
                    include_bias = bool(group["include_bias"])

                    try:
                        solved_weight, solved_bias, layer_diag = _solve_diag_kfac_linear_layer(
                            global_state=global_state,
                            client_updates=client_updates,
                            sample_weights=sample_weights,
                            entries=group["entries"],
                            weight_name=weight_name,
                            bias_name=bias_name,
                            include_bias=include_bias,
                            weight_mode=weight_mode,
                            solve_mode=solve_mode,
                            solver_steps=solver_steps,
                            cg_tol=cg_tol,
                            server_lr=server_lr,
                            adam_beta1=adam_beta1,
                            adam_beta2=adam_beta2,
                            adam_eps=adam_eps,
                            damping=damping,
                            use_damping=use_damping,
                        )
                    except Exception:
                        if strict:
                            raise

                        skipped_layers.append(layer_name)
                        continue

                    new_state_dict[weight_name] = solved_weight.detach().cpu()
                    solved_params.add(weight_name)

                    if bias_name is not None and solved_bias is not None:
                        new_state_dict[str(bias_name)] = solved_bias.detach().cpu()
                        solved_params.add(str(bias_name))

                    _accumulate_layer_diagnostics(
                        layer_diag=layer_diag,
                        valid_client_ids=valid_client_ids,
                        diag_kfac_client_counts=diag_kfac_client_counts,
                        diag_kfac_layer_weights=diag_kfac_layer_weights,
                        trace_A_values=trace_A_values,
                        trace_B_values=trace_B_values,
                        residual_norm_values=residual_norm_values,
                        delta_norm_values=delta_norm_values,
                        solver_delta_norm_values=solver_delta_norm_values,
                    )

                    solver_grad_norm_values.extend(layer_diag.get("solver_grad_norm_values", []))
                    solver_update_norm_values.extend(layer_diag.get("solver_update_norm_values", []))
                    valid_layers += 1
                    valid_client_layers += int(layer_diag["valid_clients"])
                    total_count += int(layer_diag["total_count"])
                    global_expert_param_count += int(layer_diag["param_count"])

        for name in target_param_names:
            if name in solved_params:
                continue

            if fallback == "none":
                continue

            if fallback != "sample_weighted":
                raise ValueError(
                    f"不支持的 diag_kfac.fallback：{fallback}。"
                    "当前支持：sample_weighted, none"
                )

            if not torch.is_tensor(global_state[name]):
                continue

            if not torch.is_floating_point(global_state[name]):
                continue

            new_state_dict[name] = _sample_weighted_param(
                name=name,
                global_state=global_state,
                client_updates=client_updates,
                weights=sample_weights,
                strict=strict,
            ).detach().cpu()
            fallback_params.add(name)

        check_finite_state_dict(
            state_dict=new_state_dict,
            param_names=target_param_names,
        )

        mean_count = float(total_count / max(valid_client_layers, 1))
        result_weights = _build_result_client_weights(
            weight_mode=weight_mode,
            client_counts=diag_kfac_client_counts,
            client_updates=client_updates,
            sample_weights=sample_weights,
        )
        cos_diag_kfac_uniform = _cos_diag_kfac_uniform(
            global_state=global_state,
            new_state_dict=new_state_dict,
            client_updates=client_updates,
            param_names=sorted(solved_params),
        )

        diagnostics = {
            "method": self.method_name,
            "param_group": self.param_group_name,
            "num_clients": len(client_updates),
            "param_count": len(target_param_names),
            "weights": {
                int(client_id): float(weight)
                for client_id, weight in result_weights.items()
            },
            "diag_kfac_weight_mode": weight_mode,
            "weight_mode": weight_mode,
            "solve_scope": solve_scope,
            "solve_mode": solve_mode,
            "diag_kfac_client_sample_weights": {
                int(client_id): float(weight)
                for client_id, weight in sample_weights.items()
            },
            "diag_kfac_client_counts": {
                int(client_id): int(count)
                for client_id, count in diag_kfac_client_counts.items()
            },
            "diag_kfac_layer_weights": diag_kfac_layer_weights,
            "valid_layers": int(valid_layers),
            "valid_clients": int(len(valid_client_ids)),
            "skipped_layers": int(len(skipped_layers)),
            "skipped_layer_names": list(skipped_layers[:20]),
            "valid_client_layers": int(valid_client_layers),
            "total_count": int(total_count),
            "mean_count": float(mean_count),
            "mean_trace_A": _safe_mean(trace_A_values),
            "mean_trace_B": _safe_mean(trace_B_values),
            "max_trace_A": _safe_max(trace_A_values),
            "max_trace_B": _safe_max(trace_B_values),
            "mean_residual_norm": _safe_mean(residual_norm_values),
            "max_residual_norm": _safe_max(residual_norm_values),
            # 兼容旧字段名：cg 时表示残差范数，gd/adam 时表示最终 FedFisher 梯度范数。
            "mean_grad_norm": _safe_mean(residual_norm_values),
            "max_grad_norm": _safe_max(residual_norm_values),
            "mean_solver_grad_norm": _safe_mean(solver_grad_norm_values),
            "max_solver_grad_norm": _safe_max(solver_grad_norm_values),
            "mean_solver_update_norm": _safe_mean(solver_update_norm_values),
            "max_solver_update_norm": _safe_max(solver_update_norm_values),
            # mean_delta_norm 表示最终 Diagonal K-FAC 参数相对上一轮 global 参数的真实更新幅度。
            "mean_delta_norm": _safe_mean(delta_norm_values),
            "mean_global_delta_norm": _safe_mean(delta_norm_values),
            # mean_solver_delta_norm 表示 Diagonal K-FAC 解相对 FedAvg 初始化点的修正幅度。
            "mean_solver_delta_norm": _safe_mean(solver_delta_norm_values),
            "cos_diag_kfac_uniform": float(cos_diag_kfac_uniform),
            "solver_steps": int(solver_steps),
            "server_steps": int(solver_steps),
            "server_lr": float(server_lr),
            "adam_beta1": float(adam_beta1),
            "adam_beta2": float(adam_beta2),
            "adam_eps": float(adam_eps),
            "cg_tol": float(cg_tol),
            "damping": float(damping),
            "use_damping": bool(use_damping),
            "min_count": int(min_count),
            "fallback": fallback,
            "fisher_timing": fisher_timing,
            "model_selection": "final_step",
            "use_server_validation": False,
            "global_expert_param_count": int(global_expert_param_count),
            "solved_params": int(len(solved_params)),
            "fallback_params": int(len(fallback_params)),
        }

        if bool(_cfg_get(self.cfg, "diag_kfac.log_detail", True)):
            print(
                "[ExpertDiagKFAC] "
                f"weight_mode={diagnostics['weight_mode']} "
                f"solve_scope={diagnostics['solve_scope']} "
                f"solve_mode={diagnostics['solve_mode']} "
                f"valid_layers={diagnostics['valid_layers']} "
                f"valid_clients={diagnostics['valid_clients']} "
                f"skipped_layers={diagnostics['skipped_layers']} "
                f"total_count={diagnostics['total_count']} "
                f"mean_count={diagnostics['mean_count']:.2f} "
                f"mean_trace_A={diagnostics['mean_trace_A']:.6e} "
                f"mean_trace_B={diagnostics['mean_trace_B']:.6e} "
                f"server_steps={diagnostics['server_steps']} "
                f"server_lr={diagnostics['server_lr']:.6e} "
                f"damping={diagnostics['damping']:.6e} "
                f"use_damping={diagnostics['use_damping']} "
                f"mean_residual_norm={diagnostics['mean_residual_norm']:.6e} "
                f"mean_solver_grad_norm={diagnostics['mean_solver_grad_norm']:.6e} "
                f"mean_solver_update_norm={diagnostics['mean_solver_update_norm']:.6e} "
                f"mean_delta_norm={diagnostics['mean_delta_norm']:.6e} "
                f"mean_solver_delta_norm={diagnostics['mean_solver_delta_norm']:.6e} "
                f"global_expert_param_count={diagnostics['global_expert_param_count']} "
                f"fallback_params={diagnostics['fallback_params']} "
                f"cos_diag_kfac_uniform={diagnostics['cos_diag_kfac_uniform']:.6f}",
                flush=True,
            )

        return AggregationResult(
            new_state_dict=new_state_dict,
            weights=result_weights,
            diagnostics=diagnostics,
        )


def _solve_diag_kfac_linear_layer(
    global_state: Mapping[str, torch.Tensor],
    client_updates: Sequence[ClientUpdate],
    sample_weights: Mapping[int, float],
    entries: Sequence[Dict[str, Any]],
    weight_name: str,
    bias_name: Optional[str],
    include_bias: bool,
    weight_mode: str,
    solve_mode: str,
    solver_steps: int,
    cg_tol: float,
    server_lr: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    damping: float,
    use_damping: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """
    对一个 Linear block 求解 Diagonal K-FAC/Fisher 加权聚合结果。

    paper-like 方程：
        sum_i p_i * D_i ⊙ W = sum_i p_i * D_i ⊙ W_i,  D_i=B_diag_i A_diag_i^T

    只有 use_damping=True 且 damping>0 时才会额外加入：
        + damping * W = + damping * W_avg
    """
    system = _prepare_layer_system(
        global_state=global_state,
        client_updates=client_updates,
        sample_weights=sample_weights,
        entries=entries,
        weight_name=weight_name,
        bias_name=bias_name,
        include_bias=include_bias,
        weight_mode=weight_mode,
        damping=damping,
        use_damping=use_damping,
    )

    if solve_mode == "cg":
        solutions, residual_norm_values = _run_cg_on_systems(
            systems=[system],
            max_steps=solver_steps,
            tol=cg_tol,
        )
        W_aug = solutions[system["layer_name"]]
        solver_grad_norm_values = list(residual_norm_values)
        solver_update_norm_values: List[float] = []
    else:
        solutions, solver_grad_norm_values, solver_update_norm_values = _run_optimizer_on_systems(
            systems=[system],
            solve_mode=solve_mode,
            server_steps=solver_steps,
            server_lr=server_lr,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_eps=adam_eps,
        )
        W_aug = solutions[system["layer_name"]]
        residual_norm_values = _compute_system_residual_norms(
            systems=[system],
            solutions=solutions,
        )

    layer_diag = _build_solution_diagnostics(
        system=system,
        W_aug=W_aug,
        residual_norm_values=residual_norm_values,
    )
    layer_diag["solver_grad_norm_values"] = list(solver_grad_norm_values)
    layer_diag["solver_update_norm_values"] = list(solver_update_norm_values)

    solved_weight = layer_diag.pop("solved_weight")
    solved_bias = layer_diag.pop("solved_bias")

    return solved_weight, solved_bias, layer_diag


def _solve_global_expert_layers(
    global_state: Mapping[str, torch.Tensor],
    client_updates: Sequence[ClientUpdate],
    sample_weights: Mapping[int, float],
    layer_groups: Sequence[Dict[str, Any]],
    weight_mode: str,
    solve_mode: str,
    solver_steps: int,
    cg_tol: float,
    server_lr: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    damping: float,
    use_damping: bool,
    strict: bool,
) -> Dict[str, Any]:
    """
    在所有 expert layer 上执行一个统一的 FedFisher Diagonal K-FAC 服务端求解。

    实现上仍然按 layer 做 Diagonal K-FAC matvec，但 CG/GD/Adam 的梯度范数、
    更新范数和迭代过程是在所有 expert layer 的联合参数空间上完成的。
    """
    systems: List[Dict[str, Any]] = []
    skipped_layers: List[str] = []

    for group in layer_groups:
        try:
            system = _prepare_layer_system(
                global_state=global_state,
                client_updates=client_updates,
                sample_weights=sample_weights,
                entries=group["entries"],
                weight_name=str(group["weight_name"]),
                bias_name=group.get("bias_name", None),
                include_bias=bool(group["include_bias"]),
                weight_mode=weight_mode,
                damping=damping,
                use_damping=use_damping,
            )
        except Exception:
            if strict:
                raise

            skipped_layers.append(str(group["layer_name"]))
            continue

        systems.append(system)

    if len(systems) == 0:
        return {
            "solutions": {},
            "diagnostics": [],
            "skipped_layers": skipped_layers,
            "solver_grad_norm_values": [],
            "solver_update_norm_values": [],
        }

    if solve_mode == "cg":
        solutions, solver_grad_norm_values = _run_cg_on_systems(
            systems=systems,
            max_steps=solver_steps,
            tol=cg_tol,
        )
        solver_update_norm_values: List[float] = []
    else:
        solutions, solver_grad_norm_values, solver_update_norm_values = _run_optimizer_on_systems(
            systems=systems,
            solve_mode=solve_mode,
            server_steps=solver_steps,
            server_lr=server_lr,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_eps=adam_eps,
        )

    diagnostics = []
    for system in systems:
        layer_name = str(system["layer_name"])
        residual = system["rhs"] - _layer_matvec(system, solutions[layer_name])
        layer_residual_norm_values = [
            float(residual.detach().float().norm().item())
        ]
        layer_diag = _build_solution_diagnostics(
            system=system,
            W_aug=solutions[layer_name],
            residual_norm_values=layer_residual_norm_values,
        )
        diagnostics.append(layer_diag)

    return {
        "solutions": solutions,
        "diagnostics": diagnostics,
        "skipped_layers": skipped_layers,
        "solver_grad_norm_values": solver_grad_norm_values,
        "solver_update_norm_values": solver_update_norm_values,
    }



def _prepare_layer_system(
    global_state: Mapping[str, torch.Tensor],
    client_updates: Sequence[ClientUpdate],
    sample_weights: Mapping[int, float],
    entries: Sequence[Dict[str, Any]],
    weight_name: str,
    bias_name: Optional[str],
    include_bias: bool,
    weight_mode: str,
    damping: float,
    use_damping: bool,
) -> Dict[str, Any]:
    """把某个 expert Linear layer 的 diagonal K-FAC entries 转成可迭代求解系统。"""
    if len(entries) == 0:
        raise ValueError(f"{weight_name} 没有有效 Diagonal K-FAC entries。")

    device = global_state[weight_name].device
    dtype = global_state[weight_name].dtype
    processed_entries = []
    total_count = 0

    for entry in entries:
        count = int(entry["count"])
        if count <= 0:
            continue

        A_diag = entry["A_diag"].to(device=device, dtype=dtype)
        B_diag = entry["B_diag"].to(device=device, dtype=dtype)
        if A_diag.dim() != 1 or B_diag.dim() != 1:
            raise ValueError(
                f"{entry.get('layer_name', weight_name)} 的 diagonal K-FAC 因子必须是一维向量："
                f"A_diag={tuple(A_diag.shape)}, B_diag={tuple(B_diag.shape)}"
            )
        if not torch.isfinite(A_diag).all() or not torch.isfinite(B_diag).all():
            raise ValueError(f"{entry.get('layer_name', weight_name)} 的 diagonal K-FAC 因子含 NaN/Inf。")
        if torch.any(A_diag < 0) or torch.any(B_diag < 0):
            raise ValueError(f"{entry.get('layer_name', weight_name)} 的 diagonal K-FAC 因子出现负值。")

        local_weight = entry["local_weight"].to(device=device, dtype=dtype)
        local_bias = None
        if include_bias and bias_name is not None and entry.get("local_bias") is not None:
            local_bias = entry["local_bias"].to(device=device, dtype=dtype)

        local_aug = _make_augmented_weight(
            weight=local_weight,
            bias=local_bias,
            include_bias=include_bias,
        )
        _validate_diag_kfac_shapes(
            A_diag=A_diag,
            B_diag=B_diag,
            W_aug=local_aug,
            layer_name=str(entry.get("layer_name", weight_name)),
        )

        processed_entries.append({
            "client_id": int(entry["client_id"]),
            "layer_name": str(entry.get("layer_name", weight_name)),
            "count": count,
            "A_diag": A_diag,
            "B_diag": B_diag,
            "local_aug": local_aug,
            "trace_A": float(A_diag.detach().float().sum().item()),
            "trace_B": float(B_diag.detach().float().sum().item()),
        })
        total_count += count

    if len(processed_entries) == 0 or total_count <= 0:
        raise ValueError(f"{weight_name} 没有 count > 0 的有效 Diagonal K-FAC entries。")

    weights = _compute_entry_weights(
        processed_entries=processed_entries,
        client_updates=client_updates,
        sample_weights=sample_weights,
        weight_mode=weight_mode,
    )

    W_avg = torch.zeros_like(processed_entries[0]["local_aug"])
    for weight, entry in zip(weights, processed_entries):
        W_avg = W_avg + float(weight) * entry["local_aug"]

    global_weight = global_state[weight_name].to(device=device, dtype=dtype)
    global_bias = None
    if include_bias and bias_name is not None and bias_name in global_state:
        global_bias = global_state[bias_name].to(device=device, dtype=dtype)
    W_global_aug = _make_augmented_weight(
        weight=global_weight,
        bias=global_bias,
        include_bias=include_bias,
    )

    # FedFisher RHS: sum_i p_i F_i W_i. 这里 F_i 是 diagonal K-FAC。
    rhs = torch.zeros_like(W_avg)
    for weight, entry in zip(weights, processed_entries):
        rhs = rhs + float(weight) * _diag_kfac_matvec(
            delta=entry["local_aug"],
            A_diag=entry["A_diag"],
            B_diag=entry["B_diag"],
            damping=0.0,
        )

    if use_damping and damping > 0:
        rhs = rhs + float(damping) * W_avg

    layer_weights = {
        int(entry["client_id"]): float(weight)
        for entry, weight in zip(processed_entries, weights)
    }

    return {
        "layer_name": str(processed_entries[0].get("layer_name", weight_name)),
        "weight_name": str(weight_name),
        "bias_name": bias_name,
        "include_bias": bool(include_bias),
        "processed_entries": processed_entries,
        "weights": weights,
        "layer_weights": layer_weights,
        "total_count": int(total_count),
        "W_avg": W_avg,
        "W_global_aug": W_global_aug,
        "rhs": rhs,
        "damping": float(damping),
        "use_damping": bool(use_damping),
        "param_count": int(W_avg.numel()),
    }



def _compute_entry_weights(
    processed_entries: Sequence[Dict[str, Any]],
    client_updates: Sequence[ClientUpdate],
    sample_weights: Mapping[int, float],
    weight_mode: str,
) -> List[float]:
    """根据 weight_mode 计算当前 layer 的客户端权重。"""
    if len(processed_entries) == 0:
        return []

    if weight_mode == "routed_count":
        counts = [float(max(int(entry["count"]), 0)) for entry in processed_entries]
        total = float(sum(counts))
        if total <= 0:
            return [1.0 / float(len(processed_entries)) for _ in processed_entries]
        return [float(count) / total for count in counts]

    if weight_mode == "sample_weighted":
        raw = [
            float(sample_weights.get(int(entry["client_id"]), 0.0))
            for entry in processed_entries
        ]
        total = float(sum(raw))
        if total <= 0:
            return [1.0 / float(len(processed_entries)) for _ in processed_entries]
        return [float(value) / total for value in raw]

    if weight_mode == "uniform":
        return [1.0 / float(len(processed_entries)) for _ in processed_entries]

    raise ValueError(f"不支持的 diag_kfac.weight_mode：{weight_mode}")


def _layer_matvec(system: Mapping[str, Any], x: torch.Tensor) -> torch.Tensor:
    """计算当前 layer 系统的 sum_i p_i F_i x。"""
    result = torch.zeros_like(x)

    for weight, entry in zip(system["weights"], system["processed_entries"]):
        result = result + float(weight) * _diag_kfac_matvec(
            delta=x,
            A_diag=entry["A_diag"],
            B_diag=entry["B_diag"],
            damping=0.0,
        )

    if bool(system.get("use_damping", False)) and float(system.get("damping", 0.0)) > 0:
        result = result + float(system["damping"]) * x

    return result


def _run_optimizer_on_systems(
    systems: Sequence[Dict[str, Any]],
    solve_mode: str,
    server_steps: int,
    server_lr: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
) -> Tuple[Dict[str, torch.Tensor], List[float], List[float]]:
    """在多个 expert layer 系统上执行统一的 GD/Adam-like 服务端优化。"""
    if solve_mode not in {"gd", "adam"}:
        raise ValueError(f"_run_optimizer_on_systems 不支持 solve_mode={solve_mode}")

    current = {
        str(system["layer_name"]): system["W_avg"].detach().clone()
        for system in systems
    }
    first_moment = {
        str(system["layer_name"]): torch.zeros_like(system["W_avg"])
        for system in systems
    }
    second_moment = {
        str(system["layer_name"]): torch.zeros_like(system["W_avg"])
        for system in systems
    }

    grad_norm_values: List[float] = []
    update_norm_values: List[float] = []

    if server_steps == 0:
        grad_norm_values.append(
            _compute_global_grad_norm(
                systems=systems,
                current=current,
            )
        )
        return current, grad_norm_values, update_norm_values

    for _ in range(int(server_steps)):
        grads: Dict[str, torch.Tensor] = {}
        grad_sq_sum = 0.0

        for system in systems:
            layer_name = str(system["layer_name"])
            grad = _layer_matvec(system, current[layer_name]) - system["rhs"]

            if not torch.isfinite(grad).all():
                raise ValueError(f"{layer_name} 的 FedFisher 梯度出现 NaN 或 Inf。")

            grads[layer_name] = grad
            grad_sq_sum += float(torch.sum(grad.detach().float() * grad.detach().float()).item())

        grad_norm_values.append(float(grad_sq_sum ** 0.5))

        update_sq_sum = 0.0
        for system in systems:
            layer_name = str(system["layer_name"])
            grad = grads[layer_name]

            if solve_mode == "gd":
                update = float(server_lr) * grad
            else:
                # 对齐 FedFisher 作者实践中的 Adam-like 写法：不做 bias correction，
                # 且一阶/二阶动量不乘 (1-beta)。
                first_moment[layer_name] = (
                    float(adam_beta1) * first_moment[layer_name] + grad
                )
                second_moment[layer_name] = (
                    float(adam_beta2) * second_moment[layer_name] + grad * grad
                )
                update = float(server_lr) * first_moment[layer_name] / (
                    torch.sqrt(second_moment[layer_name]) + float(adam_eps)
                )

            if not torch.isfinite(update).all():
                raise ValueError(f"{layer_name} 的 FedFisher 更新出现 NaN 或 Inf。")

            current[layer_name] = current[layer_name] - update
            update_sq_sum += float(torch.sum(update.detach().float() * update.detach().float()).item())

            if not torch.isfinite(current[layer_name]).all():
                raise ValueError(f"{layer_name} 的 FedFisher 解出现 NaN 或 Inf。")

        update_norm_values.append(float(update_sq_sum ** 0.5))

    return current, grad_norm_values, update_norm_values


def _run_cg_on_systems(
    systems: Sequence[Dict[str, Any]],
    max_steps: int,
    tol: float,
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    """在多个 expert layer 系统上执行一个联合 CG 求解。"""
    x = {
        str(system["layer_name"]): system["W_avg"].detach().clone()
        for system in systems
    }
    r = {}
    p = {}

    for system in systems:
        layer_name = str(system["layer_name"])
        residual = system["rhs"] - _layer_matvec(system, x[layer_name])

        if not torch.isfinite(residual).all():
            raise ValueError(f"{layer_name} 的 Diagonal K-FAC 初始残差出现 NaN 或 Inf。")

        r[layer_name] = residual
        p[layer_name] = residual.detach().clone()

    rs_old = _dict_dot(r, r)
    residual_norm_values = [float(max(rs_old, 0.0) ** 0.5)]

    if residual_norm_values[-1] <= float(tol):
        return x, residual_norm_values

    if max_steps == 0:
        return x, residual_norm_values

    for _ in range(int(max_steps)):
        Ap = {}
        for system in systems:
            layer_name = str(system["layer_name"])
            value = _layer_matvec(system, p[layer_name])

            if not torch.isfinite(value).all():
                raise ValueError(f"{layer_name} 的 Diagonal K-FAC matvec 出现 NaN 或 Inf。")

            Ap[layer_name] = value

        denom = _dict_dot(p, Ap)

        if not math.isfinite(float(denom)):
            raise ValueError("Diagonal K-FAC CG denom 出现 NaN 或 Inf。")

        if abs(float(denom)) <= 1.0e-30:
            break

        alpha = float(rs_old) / (float(denom) + 1.0e-30)

        for system in systems:
            layer_name = str(system["layer_name"])
            x[layer_name] = x[layer_name] + alpha * p[layer_name]
            r[layer_name] = r[layer_name] - alpha * Ap[layer_name]

            if not torch.isfinite(x[layer_name]).all():
                raise ValueError(f"{layer_name} 的 Diagonal K-FAC CG 解出现 NaN 或 Inf。")

            if not torch.isfinite(r[layer_name]).all():
                raise ValueError(f"{layer_name} 的 Diagonal K-FAC CG 残差出现 NaN 或 Inf。")

        rs_new = _dict_dot(r, r)
        residual_norm = float(max(rs_new, 0.0) ** 0.5)
        residual_norm_values.append(residual_norm)

        if residual_norm <= float(tol):
            break

        beta = float(rs_new) / (float(rs_old) + 1.0e-30)
        for system in systems:
            layer_name = str(system["layer_name"])
            p[layer_name] = r[layer_name] + beta * p[layer_name]

        rs_old = rs_new

    return x, residual_norm_values


def _compute_global_grad_norm(
    systems: Sequence[Dict[str, Any]],
    current: Mapping[str, torch.Tensor],
) -> float:
    """计算所有 expert layer 上的 FedFisher 梯度范数。"""
    grad_sq_sum = 0.0
    for system in systems:
        layer_name = str(system["layer_name"])
        grad = _layer_matvec(system, current[layer_name]) - system["rhs"]
        grad_sq_sum += float(torch.sum(grad.detach().float() * grad.detach().float()).item())

    return float(grad_sq_sum ** 0.5)


def _compute_system_residual_norms(
    systems: Sequence[Dict[str, Any]],
    solutions: Mapping[str, torch.Tensor],
) -> List[float]:
    """计算每个 layer 最终方程残差范数。"""
    residuals = []
    for system in systems:
        layer_name = str(system["layer_name"])
        residual = system["rhs"] - _layer_matvec(system, solutions[layer_name])
        residuals.append(float(residual.detach().float().norm().item()))

    return residuals


def _build_solution_diagnostics(
    system: Mapping[str, Any],
    W_aug: torch.Tensor,
    residual_norm_values: Sequence[float],
) -> Dict[str, Any]:
    """把某个 layer 的最终解和诊断信息打包。"""
    if not torch.isfinite(W_aug).all():
        raise ValueError(f"{system['weight_name']} 的 Diagonal K-FAC 解出现 NaN 或 Inf。")

    solved_weight, solved_bias = _split_augmented_weight(
        W_aug=W_aug,
        include_bias=bool(system["include_bias"]),
    )

    global_delta_norm = float(
        (W_aug.detach().float() - system["W_global_aug"].detach().float()).norm().item()
    )
    solver_delta_norm = float(
        (W_aug.detach().float() - system["W_avg"].detach().float()).norm().item()
    )

    processed_entries = system["processed_entries"]

    return {
        "layer_name": str(system["layer_name"]),
        "weight_name": str(system["weight_name"]),
        "bias_name": system.get("bias_name", None),
        "include_bias": bool(system["include_bias"]),
        "solved_weight": solved_weight,
        "solved_bias": solved_bias,
        "valid_clients": int(len(processed_entries)),
        "client_ids": [int(entry["client_id"]) for entry in processed_entries],
        "client_counts": {
            int(entry["client_id"]): int(entry["count"])
            for entry in processed_entries
        },
        "layer_weights": {
            int(client_id): float(weight)
            for client_id, weight in system["layer_weights"].items()
        },
        "total_count": int(system["total_count"]),
        "trace_A_values": [
            float(entry["trace_A"])
            for entry in processed_entries
        ],
        "trace_B_values": [
            float(entry["trace_B"])
            for entry in processed_entries
        ],
        "residual_norm_values": list(float(value) for value in residual_norm_values),
        "delta_norm": float(global_delta_norm),
        "global_delta_norm": float(global_delta_norm),
        "solver_delta_norm": float(solver_delta_norm),
        "param_count": int(system["param_count"]),
    }


def _accumulate_layer_diagnostics(
    layer_diag: Mapping[str, Any],
    valid_client_ids: set[int],
    diag_kfac_client_counts: Dict[int, int],
    diag_kfac_layer_weights: Dict[str, Dict[int, float]],
    trace_A_values: List[float],
    trace_B_values: List[float],
    residual_norm_values: List[float],
    delta_norm_values: List[float],
    solver_delta_norm_values: List[float],
) -> None:
    """汇总单个 layer 的诊断信息。"""
    layer_name = str(layer_diag["layer_name"])

    for client_id in layer_diag.get("client_ids", []):
        valid_client_ids.add(int(client_id))

    for client_id, count in layer_diag.get("client_counts", {}).items():
        client_id = int(client_id)
        diag_kfac_client_counts[client_id] = int(diag_kfac_client_counts.get(client_id, 0)) + int(count)

    diag_kfac_layer_weights[layer_name] = {
        int(client_id): float(weight)
        for client_id, weight in layer_diag.get("layer_weights", {}).items()
    }

    trace_A_values.extend(layer_diag.get("trace_A_values", []))
    trace_B_values.extend(layer_diag.get("trace_B_values", []))
    residual_norm_values.extend(layer_diag.get("residual_norm_values", []))
    delta_norm_values.append(float(layer_diag.get("delta_norm", 0.0)))
    solver_delta_norm_values.append(float(layer_diag.get("solver_delta_norm", 0.0)))


def _dict_dot(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    """计算多个 tensor 组成的向量点积。"""
    value = 0.0
    for key in left.keys():
        value += float(torch.sum(left[key].detach().float() * right[key].detach().float()).item())
    return float(value)



def _diag_kfac_matvec(
    delta: torch.Tensor,
    A_diag: torch.Tensor,
    B_diag: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    """
    Diagonal K-FAC 矩阵向量乘法。

    diag(A)=A_diag, diag(B)=B_diag 时：
        diag(B ⊗ A) reshape 到 W 的形状后为
        D = B_diag[:, None] * A_diag[None, :]

    因而：
        F_diag-kfac(delta) = D ⊙ delta
    """
    curvature = B_diag.reshape(-1, 1) * A_diag.reshape(1, -1)
    if curvature.shape != delta.shape:
        raise ValueError(
            "Diagonal K-FAC curvature 与参数 shape 不匹配："
            f"curvature={tuple(curvature.shape)}, delta={tuple(delta.shape)}"
        )

    result = curvature * delta
    if damping > 0:
        result = result + float(damping) * delta
    return result



def _symmetrize_square(matrix: torch.Tensor) -> torch.Tensor:
    """兼容保留的方阵对称化 helper；Diagonal K-FAC 主路径不使用。"""
    if matrix.dim() == 2 and matrix.size(0) == matrix.size(1):
        return 0.5 * (matrix + matrix.transpose(0, 1))

    return matrix


def _collect_diag_kfac_layer_names(
    client_updates: Sequence[ClientUpdate],
) -> List[str]:
    """收集本轮所有客户端上传过的 Diagonal K-FAC layer_name。"""
    layer_names = set()

    for update in client_updates:
        payload = update.extra.get("expert_diag_kfac", None)

        if not isinstance(payload, Mapping):
            continue

        for layer_name in payload.keys():
            layer_names.add(str(layer_name))

    return sorted(layer_names)


def _collect_valid_layer_entries(
    layer_name: str,
    client_updates: Sequence[ClientUpdate],
    global_state: Mapping[str, torch.Tensor],
    target_param_set: set[str],
    min_count: int,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """收集某个 Diagonal K-FAC layer 在所有客户端上的有效条目。"""
    entries: List[Dict[str, Any]] = []

    for update in client_updates:
        payload = update.extra.get("expert_diag_kfac", None)

        if not isinstance(payload, Mapping):
            continue

        if layer_name not in payload:
            continue

        item = payload[layer_name]

        if not isinstance(item, Mapping):
            continue

        try:
            entry = _build_layer_entry(
                layer_name=layer_name,
                item=item,
                update=update,
                global_state=global_state,
                target_param_set=target_param_set,
                min_count=min_count,
            )
        except Exception:
            if strict:
                raise

            continue

        if entry is not None:
            entries.append(entry)

    return entries



def _build_layer_entry(
    layer_name: str,
    item: Mapping[str, Any],
    update: ClientUpdate,
    global_state: Mapping[str, torch.Tensor],
    target_param_set: set[str],
    min_count: int,
) -> Optional[Dict[str, Any]]:
    """把客户端上传的单层 Diagonal K-FAC payload 转成服务端 entry。"""
    count = int(item.get("count", 0))
    if count < int(min_count):
        return None

    weight_name = str(item.get("weight_name", ""))
    bias_name_raw = item.get("bias_name", None)
    bias_name = None if bias_name_raw is None else str(bias_name_raw)
    if weight_name == "" or weight_name not in target_param_set:
        return None
    if weight_name not in global_state or weight_name not in update.model_delta:
        return None

    A_diag = item.get("A_diag", None)
    B_diag = item.get("B_diag", None)
    if not torch.is_tensor(A_diag) or not torch.is_tensor(B_diag):
        return None
    if A_diag.dim() != 1 or B_diag.dim() != 1:
        return None
    if not torch.isfinite(A_diag).all() or not torch.isfinite(B_diag).all():
        return None
    if torch.any(A_diag < 0) or torch.any(B_diag < 0):
        return None

    global_weight = global_state[weight_name]
    local_weight = global_weight.detach().cpu() + update.model_delta[weight_name].detach().cpu()

    local_bias = None
    include_bias = bool(item.get("include_bias", False))
    if include_bias and bias_name is not None:
        if bias_name in target_param_set and bias_name in global_state and bias_name in update.model_delta:
            global_bias = global_state[bias_name]
            local_bias = global_bias.detach().cpu() + update.model_delta[bias_name].detach().cpu()
        else:
            include_bias = False
            bias_name = None

    return {
        "client_id": int(update.client_id),
        "layer_name": str(layer_name),
        "weight_name": weight_name,
        "bias_name": bias_name,
        "include_bias": include_bias,
        "count": int(count),
        "A_diag": A_diag.detach().cpu(),
        "B_diag": B_diag.detach().cpu(),
        "local_weight": local_weight,
        "local_bias": local_bias,
    }



def _make_augmented_weight(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    include_bias: bool,
) -> torch.Tensor:
    """
    把 Linear 的 weight 和 bias 合成 W_aug。

    weight: [out_features, in_features]
    bias: [out_features]

    include_bias=True 时：
        W_aug = [W, b]
        shape: [out_features, in_features + 1]
    """
    if include_bias and bias is not None:
        return torch.cat(
            [
                weight,
                bias.reshape(-1, 1),
            ],
            dim=1,
        )

    return weight


def _split_augmented_weight(
    W_aug: torch.Tensor,
    include_bias: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """把 W_aug 拆回 weight 和 bias。"""
    if include_bias:
        weight = W_aug[:, :-1]
        bias = W_aug[:, -1]
        return weight, bias

    return W_aug, None



def _validate_diag_kfac_shapes(
    A_diag: torch.Tensor,
    B_diag: torch.Tensor,
    W_aug: torch.Tensor,
    layer_name: str,
) -> None:
    """检查 A_diag、B_diag 与 W_aug 的维度是否匹配。"""
    if A_diag.dim() != 1:
        raise ValueError(f"{layer_name} 的 A_diag 不是一维向量，shape={tuple(A_diag.shape)}")
    if B_diag.dim() != 1:
        raise ValueError(f"{layer_name} 的 B_diag 不是一维向量，shape={tuple(B_diag.shape)}")
    if W_aug.dim() != 2:
        raise ValueError(f"{layer_name} 的 W_aug 不是二维矩阵，shape={tuple(W_aug.shape)}")
    if B_diag.numel() != W_aug.size(0):
        raise ValueError(
            f"{layer_name} 的 B_diag 和 W_aug 输出维度不匹配："
            f"B_diag={tuple(B_diag.shape)}, W_aug={tuple(W_aug.shape)}"
        )
    if A_diag.numel() != W_aug.size(1):
        raise ValueError(
            f"{layer_name} 的 A_diag 和 W_aug 输入维度不匹配："
            f"A_diag={tuple(A_diag.shape)}, W_aug={tuple(W_aug.shape)}"
        )



def _sample_weighted_param(
    name: str,
    global_state: Mapping[str, torch.Tensor],
    client_updates: Sequence[ClientUpdate],
    weights: Mapping[int, float],
    strict: bool,
) -> torch.Tensor:
    """对单个参数执行 sample_weighted fallback。"""
    global_tensor = global_state[name]
    total_delta = torch.zeros_like(global_tensor)

    for update in client_updates:
        client_id = int(update.client_id)

        if client_id not in weights:
            if strict:
                raise KeyError(f"weights 缺少客户端 {client_id} 的权重。")

            continue

        if name not in update.model_delta:
            if strict:
                raise KeyError(
                    f"客户端 {client_id} 的 model_delta 缺少参数：{name}"
                )

            continue

        delta = update.model_delta[name].to(global_tensor.device)
        total_delta = total_delta + float(weights[client_id]) * delta

    return global_tensor + total_delta


def _resolve_param_names(
    global_state: Mapping[str, torch.Tensor],
    param_names: Optional[Iterable[str]],
) -> List[str]:
    """解析 expert 参数名列表。"""
    if param_names is None:
        return list(global_state.keys())

    names = list(param_names)

    for name in names:
        if name not in global_state:
            raise KeyError(f"global_state 中不存在参数：{name}")

    return names


def _build_result_client_weights(
    weight_mode: str,
    client_counts: Mapping[int, int],
    client_updates: Sequence[ClientUpdate],
    sample_weights: Mapping[int, float],
) -> Dict[int, float]:
    """构造 AggregationResult.weights 中展示的客户端权重。"""
    if weight_mode == "sample_weighted":
        return {
            int(update.client_id): float(sample_weights.get(int(update.client_id), 0.0))
            for update in client_updates
        }

    if weight_mode == "uniform":
        if len(client_updates) == 0:
            return {}
        weight = 1.0 / float(len(client_updates))
        return {
            int(update.client_id): float(weight)
            for update in client_updates
        }

    return _normalize_diag_kfac_client_counts(
        client_counts=client_counts,
        client_updates=client_updates,
    )


def _normalize_diag_kfac_client_counts(
    client_counts: Mapping[int, int],
    client_updates: Sequence[ClientUpdate],
) -> Dict[int, float]:
    """
    把所有 solved Diagonal K-FAC layer 的 routed count 汇总成 client 级别权重。

    注意：
        这个是 routed_count 模式下的 Diagonal K-FAC evidence 汇总权重，
        不是 sample_weighted / uniform 模式下的真实权重。
    """
    result = {
        int(update.client_id): 0.0
        for update in client_updates
    }

    total_count = int(sum(int(count) for count in client_counts.values()))

    if total_count <= 0:
        if len(client_updates) == 0:
            return result

        uniform_weight = 1.0 / float(len(client_updates))
        return {
            int(update.client_id): float(uniform_weight)
            for update in client_updates
        }

    for client_id, count in client_counts.items():
        result[int(client_id)] = float(count) / float(total_count)

    return result


def _cos_diag_kfac_uniform(
    global_state: Mapping[str, torch.Tensor],
    new_state_dict: Mapping[str, torch.Tensor],
    client_updates: Sequence[ClientUpdate],
    param_names: Sequence[str],
) -> float:
    """
    计算 Diagonal K-FAC 聚合方向和 uniform 直接平均方向的余弦相似度。

    cos 接近 1：
        Diagonal K-FAC 基本退化成 uniform 直接平均。

    cos 明显小于 1：
        Diagonal K-FAC 改变了专家聚合方向。
    """
    if len(param_names) == 0:
        return 0.0

    if len(client_updates) == 0:
        return 0.0

    uniform_weight = 1.0 / float(len(client_updates))

    dot = 0.0
    norm_kfac = 0.0
    norm_uniform = 0.0

    for name in param_names:
        if name not in global_state or name not in new_state_dict:
            continue

        if not torch.is_tensor(global_state[name]):
            continue

        if not torch.is_floating_point(global_state[name]):
            continue

        kfac_delta = (
            new_state_dict[name].detach().cpu().float()
            - global_state[name].detach().cpu().float()
        )

        uniform_delta = torch.zeros_like(kfac_delta)

        for update in client_updates:
            if name not in update.model_delta:
                continue

            uniform_delta = uniform_delta + uniform_weight * update.model_delta[
                name
            ].detach().cpu().float()

        kfac_flat = kfac_delta.reshape(-1)
        uniform_flat = uniform_delta.reshape(-1)

        dot += float(torch.dot(kfac_flat, uniform_flat).item())
        norm_kfac += float(torch.dot(kfac_flat, kfac_flat).item())
        norm_uniform += float(torch.dot(uniform_flat, uniform_flat).item())

    if norm_kfac <= 0 or norm_uniform <= 0:
        return 0.0

    return float(dot / ((norm_kfac ** 0.5) * (norm_uniform ** 0.5) + 1.0e-12))


def _safe_mean(values: Sequence[float]) -> float:
    """安全计算均值。"""
    finite_values = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]

    if len(finite_values) == 0:
        return 0.0

    return float(sum(finite_values) / len(finite_values))


def _safe_max(values: Sequence[float]) -> float:
    """安全计算最大值。"""
    finite_values = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]

    if len(finite_values) == 0:
        return 0.0

    return float(max(finite_values))


def _validate_choice(
    name: str,
    value: str,
    choices: Sequence[str],
) -> None:
    """检查配置枚举值。"""
    if value not in set(choices):
        raise ValueError(
            f"{name} 必须是 {sorted(choices)} 之一，当前值：{value}"
        )


def _cfg_get(
    cfg: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    兼容 dict / ConfigNode / 普通对象的读取。

    dict 或 ConfigNode:
        cfg.get(key, default)

    普通对象:
        getattr(cfg, key, default)
    """
    if hasattr(cfg, "get"):
        return cfg.get(key, default)

    return getattr(cfg, key, default)


def collect_method_evidence(
    *,
    model: nn.Module,
    evidence_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str,
    cfg: Any,
) -> Dict[str, Any]:
    """Collect the same post-train expert evidence previously owned by base.py."""
    expert_diag_kfac_timing = str(
        _cfg_get(
            cfg,
            "diag_kfac.fisher_timing",
            _cfg_get(cfg, "diag_kfac.collect_timing", "after_train"),
        )
    ).lower().strip()

    if expert_diag_kfac_timing != "after_train":
        raise ValueError(
            "当前 Diagonal K-FAC 采集只支持 diag_kfac.fisher_timing=after_train。"
            f"当前值：{expert_diag_kfac_timing}。"
            "请不要在本地训练过程中混合统计 Diagonal K-FAC。"
        )

    expert_diag_kfac = collect_expert_diag_kfac(
        model=model,
        train_loader=evidence_loader,
        criterion=criterion,
        device=device,
        cfg=cfg,
    )
    expert_diag_kfac_summary = summarize_expert_diag_kfac(expert_diag_kfac)

    return {
        "expert_diag_kfac": expert_diag_kfac,
        "expert_diag_kfac_summary": expert_diag_kfac_summary,
        "expert_diag_kfac_timing": expert_diag_kfac_timing,
    }


def build_method_client_diagnostics(
    update: ClientUpdate,
) -> Dict[str, Any]:
    """Expose only lightweight method diagnostics to the shared server summary."""
    extra = dict(update.extra or {})
    return {
        "expert_diag_kfac_summary": extra.get("expert_diag_kfac_summary", None),
    }



def register_method_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Register Diagonal K-FAC-only command-line overrides."""
    parser.add_argument("--server-lr", type=float, default=None)
    parser.add_argument("--server-steps", type=int, default=None)
    parser.add_argument(
        "--weight-mode",
        choices=("routed_count", "sample_weighted", "uniform"),
        default=None,
    )
    parser.add_argument(
        "--solve-scope",
        choices=("per_layer", "global_expert"),
        default=None,
    )
    parser.add_argument(
        "--solve-mode",
        choices=("cg", "gd", "adam"),
        default=None,
    )
    parser.add_argument("--adam-beta1", type=float, default=None)
    parser.add_argument("--adam-beta2", type=float, default=None)
    parser.add_argument("--adam-eps", type=float, default=None)
    parser.add_argument("--cg-tol", type=float, default=None)
    parser.add_argument("--damping", type=float, default=None)
    parser.add_argument("--min-count", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--fallback",
        choices=("none", "sample_weighted"),
        default=None,
    )
    parser.add_argument(
        "--model-mode",
        choices=("eval", "train"),
        default=None,
    )


def build_method_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Map explicit Diagonal K-FAC CLI values to the nested diag_kfac configuration."""
    diag_kfac_overrides: Dict[str, Any] = {}

    mappings = (
        ("server_lr", "server_lr"),
        ("server_steps", "server_steps"),
        ("weight_mode", "weight_mode"),
        ("solve_scope", "solve_scope"),
        ("solve_mode", "solve_mode"),
        ("adam_beta1", "adam_beta1"),
        ("adam_beta2", "adam_beta2"),
        ("adam_eps", "adam_eps"),
        ("cg_tol", "cg_tol"),
        ("damping", "damping"),
        ("min_count", "min_count"),
        ("max_batches", "max_batches"),
        ("fallback", "fallback"),
        ("model_mode", "model_mode"),
    )

    for arg_name, config_key in mappings:
        value = getattr(args, arg_name, None)
        if value is not None:
            diag_kfac_overrides[config_key] = value

    if not diag_kfac_overrides:
        return {}

    return {"diag_kfac": diag_kfac_overrides}


def build_expert_aggregator(cfg: Any) -> base.Aggregator:
    """Build the expert-only Diagonal K-FAC aggregator injected into base.py."""
    return DiagKFACExpertAggregator(
        cfg=cfg,
        param_group_name="expert",
    )


def main() -> int:
    return base.main(
        expert_aggregator_builder=build_expert_aggregator,
        embedded_method_config=EMBEDDED_METHOD_CONFIG,
        expert_method_name=ALGORITHM_NAME,
        method_config_defaults=METHOD_CONFIG_DEFAULTS,
        method_config_validator=validate_method_config,
        expert_evidence_collector=collect_method_evidence,
        method_client_diagnostics_builder=build_method_client_diagnostics,
        method_cli_argument_registrar=register_method_cli_arguments,
        method_cli_overrides_builder=build_method_cli_overrides,
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""RFedAGS with server-side expert subspace diversity correction.

Clients optimize only the task objective and the existing Switch auxiliary loss.
The server first performs RFedAGS aggregation, then applies one Stiefel
Riemannian diversity-correction step to the aggregated expert subspaces.
"""

from __future__ import annotations

# Import base before torch so deterministic CUDA environment setup is applied
# before PyTorch initializes CUDA/cuBLAS.
import base
import rfedags_subspace as subspace

import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


ALGORITHM_NAME = "rfedags_server_diversity"

# Reuse the established RFedAGS/Stiefel payload contract. The client-side
# diversity fields are retained only for compatibility with the shared
# RFedAGS aggregator and are always empty/zero in this algorithm.
PAYLOAD_KEY = subspace.PAYLOAD_KEY
PAYLOAD_VERSION = subspace.PAYLOAD_VERSION

# ---------------------------------------------------------------------------
# Algorithm-specific hyperparameters.
# All common training/model/optimizer defaults continue to come from base.py.
# ---------------------------------------------------------------------------
SUBSPACE_RANK = 128
TAU_MARGIN = 0.05
LOCAL_MANIFOLD_LR = None  # None -> use base.py's optimizer.lr.
SERVER_MANIFOLD_LR = 1.0  # RFedAGS server step.
SERVER_DIVERSITY_LR = 1e-3  # Additional server-side Stiefel correction step.
SERVER_DIVERSITY_GRAD_CLIP = None  # None -> no product-manifold gradient clipping.

METHOD_CONFIG_DEFAULTS: Dict[str, Any] = {
    "method_config": {
        "subspace_rank": SUBSPACE_RANK,
        "tau_margin": TAU_MARGIN,
        "local_manifold_lr": LOCAL_MANIFOLD_LR,
        "server_manifold_lr": SERVER_MANIFOLD_LR,
        "server_diversity_lr": SERVER_DIVERSITY_LR,
        "server_diversity_grad_clip": SERVER_DIVERSITY_GRAD_CLIP,
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


def validate_method_config(cfg: Mapping[str, Any]) -> None:
    """Validate the fixed-step, full-participation RFedAGS configuration."""
    local_steps = _cfg_get(cfg, "local_steps", None)
    subspace._validate_positive_int(local_steps, "local_steps")

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

    num_experts = subspace._validate_positive_int(
        _cfg_get(cfg, "num_experts", base.DEFAULT_CONFIG["num_experts"]),
        "num_experts",
    )
    if num_experts < 2:
        raise ValueError("服务器端专家子空间校正至少需要 2 个专家。")

    rank = subspace._validate_positive_int(
        _method_cfg_get(cfg, "subspace_rank", SUBSPACE_RANK),
        "method_config.subspace_rank",
    )
    hidden_dim = subspace._validate_positive_int(
        _cfg_get(
            cfg,
            "model_cfg.hidden_dim",
            base.DEFAULT_CONFIG["model_cfg"]["hidden_dim"],
        ),
        "model_cfg.hidden_dim",
    )
    expert_hidden_dim = subspace._validate_positive_int(
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

    tau_margin = subspace._validate_finite_nonnegative(
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
    subspace._validate_finite_positive(
        local_lr,
        "local_manifold_lr / optimizer.lr",
    )
    subspace._validate_finite_positive(
        _method_cfg_get(cfg, "server_manifold_lr", SERVER_MANIFOLD_LR),
        "method_config.server_manifold_lr",
    )
    subspace._validate_finite_positive(
        _method_cfg_get(cfg, "server_diversity_lr", SERVER_DIVERSITY_LR),
        "method_config.server_diversity_lr",
    )

    grad_clip = _method_cfg_get(
        cfg,
        "server_diversity_grad_clip",
        SERVER_DIVERSITY_GRAD_CLIP,
    )
    if grad_clip is not None:
        subspace._validate_finite_positive(
            grad_clip,
            "method_config.server_diversity_grad_clip",
        )


def build_rfedags_server_diversity_model(cfg: Any) -> nn.Module:
    """Build the shared low-rank Stiefel Switch-ViT parameterization."""
    return subspace.build_rfedags_subspace_model(cfg)


class RFedAGSServerDiversityClientStrategy(
    subspace.RFedAGSSubspaceClientStrategy
):
    """Use task/auxiliary gradients locally, without client-side diversity loss."""

    @property
    def strategy_name(self) -> str:
        return ALGORITHM_NAME

    def step(self, context: base.ExpertClientBatchContext) -> None:
        if self._euclidean_optimizer is None:
            raise RuntimeError("客户端欧氏专家优化器尚未初始化。")

        expected_step_index = self._local_steps + 1
        if int(context.local_step) != expected_step_index:
            raise ValueError(
                "RFedAGS 服务器端多样性策略收到非连续本地步数："
                f"expected={expected_step_index}, actual={context.local_step}"
            )

        # The task loss and the existing Switch auxiliary loss have already
        # been backpropagated by base.py. Deliberately do not compute or add
        # any expert-subspace diversity loss on the client.
        grad_clip = context.train_context.non_expert_grad_clip
        if grad_clip is not None and float(grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self._expert_parameters,
                max_norm=float(grad_clip),
            )

        with torch.no_grad():
            for name in self._p_names:
                parameter = self._parameter_map[name]
                gradient = parameter.grad
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                if gradient.is_sparse:
                    raise NotImplementedError(
                        f"当前不支持稀疏 Stiefel 梯度：{name}"
                    )
                if not subspace._tensor_is_finite(gradient):
                    raise ValueError(
                        f"Stiefel 参数 {name} 的梯度包含 NaN 或 Inf。"
                    )

                riemannian_gradient = subspace.stiefel_tangent_projection(
                    parameter.detach(),
                    gradient.detach(),
                )
                scaled_direction = self.local_lr * riemannian_gradient

                transported = subspace.stiefel_tangent_projection(
                    self._global_points[name],
                    scaled_direction,
                )
                if not subspace._tensor_is_finite(transported):
                    raise ValueError(
                        f"Stiefel 参数 {name} 的传输梯度包含 NaN 或 Inf。"
                    )
                self._gradient_stream[name].add_(
                    transported.to(dtype=self._gradient_stream[name].dtype)
                )

                parameter.copy_(
                    subspace.qr_retraction(
                        parameter.detach() - scaled_direction
                    )
                )

        self._euclidean_optimizer.step()
        self._euclidean_optimizer.zero_grad(set_to_none=True)
        self._local_steps += 1


def _collect_pairwise_overlap_metrics(
    model: nn.Module,
) -> Dict[str, Dict[str, float]]:
    """Return every within-layer expert-pair overlap for server diagnostics."""
    parameter_map = dict(model.named_parameters())
    pairwise: Dict[str, Dict[str, float]] = {}
    for layer_name, layer_records in subspace._get_layer_records(model).items():
        layer_values: Dict[str, float] = {}
        for left_index in range(len(layer_records)):
            for right_index in range(left_index + 1, len(layer_records)):
                left_item = layer_records[left_index]
                right_item = layer_records[right_index]
                left = parameter_map[left_item["p_name"]].detach()
                right = parameter_map[right_item["p_name"]].detach()
                overlap = subspace._pairwise_subspace_overlap(left, right)
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


class RFedAGSServerDiversityExpertAggregator(
    subspace.RFedAGSSubspaceExpertAggregator
):
    """RFedAGS aggregation followed by one server-side Stiefel correction."""

    @property
    def method_name(self) -> str:
        return ALGORITHM_NAME

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg=cfg)
        self.diversity_lr = float(
            _method_cfg_get(cfg, "server_diversity_lr", SERVER_DIVERSITY_LR)
        )
        configured_grad_clip = _method_cfg_get(
            cfg,
            "server_diversity_grad_clip",
            SERVER_DIVERSITY_GRAD_CLIP,
        )
        self.diversity_grad_clip = (
            None
            if configured_grad_clip is None
            else float(configured_grad_clip)
        )
        self._last_server_correction_diagnostics: Dict[str, Any] = {}

    def aggregate(
        self,
        global_state: Mapping[str, torch.Tensor],
        client_updates: Sequence[base.ClientUpdate],
        param_names: Optional[Sequence[str]] = None,
        base_state: Optional[Mapping[str, torch.Tensor]] = None,
        strict: bool = True,
    ) -> base.AggregationResult:
        # Stage 1 is the existing RFedAGS aggregation. Stage 2 is intentionally
        # deferred to after_aggregate(), which base.py calls after loading this
        # intermediate state into the global model.
        result = super().aggregate(
            global_state=global_state,
            client_updates=client_updates,
            param_names=param_names,
            base_state=base_state,
            strict=strict,
        )
        diagnostics = dict(result.diagnostics)
        diagnostics.pop("mean_diversity_loss", None)
        diagnostics.pop("mean_weighted_diversity_loss", None)
        diagnostics.pop("mean_overlap_by_layer", None)
        diagnostics.pop("lambda_div", None)
        diagnostics["aggregation_method"] = ALGORITHM_NAME
        diagnostics["server_diversity_lr"] = self.diversity_lr
        diagnostics["server_diversity_grad_clip"] = self.diversity_grad_clip
        result.diagnostics = diagnostics
        return result

    def after_aggregate(self, model: nn.Module) -> None:
        """Apply one diversity-aware Riemannian step to the RFedAGS result."""
        records = subspace._get_layer_records(model)
        thresholds = getattr(model, "_rfedags_subspace_thresholds", None)
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
            loss_before, overlap_before = subspace.compute_subspace_diversity_loss(
                model,
                thresholds=thresholds,
            )
            if not subspace._tensor_is_finite(loss_before):
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
            subspace.stiefel_tangent_projection(
                parameter.detach(),
                gradient.detach(),
            )
            for parameter, gradient in zip(p_parameters, gradients)
        ]
        if any(
            not subspace._tensor_is_finite(gradient)
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
        applied = loss_before_value > 0.0 and clipped_gradient_norm > 0.0
        parameter_step_norm = 0.0

        if applied:
            with torch.no_grad():
                for parameter, gradient in zip(
                    p_parameters,
                    riemannian_gradients,
                ):
                    parameter.copy_(
                        subspace.qr_retraction(
                            parameter.detach()
                            - self.diversity_lr * clip_scale * gradient
                        )
                    )
            parameter_step_norm = math.sqrt(
                sum(
                    float(
                        torch.sum(
                            (parameter.detach() - before).to(torch.float64).square()
                        ).item()
                    )
                    for parameter, before in zip(p_parameters, before_points)
                )
            )

        with torch.no_grad():
            loss_after, overlap_after = subspace.compute_subspace_diversity_loss(
                model,
                thresholds=thresholds,
            )
            loss_after_value = float(loss_after.item())

        if not math.isfinite(loss_after_value) or not math.isfinite(parameter_step_norm):
            raise ValueError("服务器端多样性校正产生了 NaN 或 Inf。")

        pairwise_overlap_after = _collect_pairwise_overlap_metrics(model)
        self._last_server_correction_diagnostics = {
            "server_diversity_applied": bool(applied),
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
        }

    def build_post_aggregate_diagnostics(
        self,
        model: nn.Module,
    ) -> Dict[str, Any]:
        diagnostics = super().build_post_aggregate_diagnostics(model)
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

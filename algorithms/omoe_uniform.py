"""OMoE-inspired Top-1 expert-output orthogonalization with uniform aggregation.

Each MoE token is evaluated by every expert. Their output vectors are
orthogonalized per token with a regularized, two-pass modified Gram-Schmidt
procedure, placing the actually selected expert last in the orthogonalization
order. The original Top-1 router, capacity handling, auxiliary loss, and
uniform federated aggregation remain unchanged.

This module injects the forward behavior through base.ModelBuilder and the
expert aggregation policy through base.Aggregator. It does not modify base.py.
"""

from __future__ import annotations

# Import base before torch so deterministic CUDA environment setup happens first.
import base

from typing import Any, Dict, Optional, Sequence, Tuple

import torch


ALGORITHM_NAME = "omoe_uniform"

# Relative Tikhonov regularization used by the orthogonalization. The scale
# floor avoids an excessively large derivative when an expert output itself
# has a very small norm. With this value, the normalization denominator is
# never smaller than 5e-3 in FP32.
GS_NORMALIZATION_RELATIVE_EPS = 5e-3
GS_PROJECTION_RELATIVE_EPS = 1e-4
GS_PROJECTION_EPS_FP32 = 1e-12
GS_PROJECTION_EPS_FP64 = 1e-24

Aggregator = base.Aggregator
ClientUpdate = base.ClientUpdate
build_uniform_weights = base.build_uniform_weights


EMBEDDED_METHOD_CONFIG = {
    "agg": {
        "non_expert": {"method": "uniform"},
        "expert": {"method": ALGORITHM_NAME},
    },
}


def _gram_schmidt_expert_outputs(
    expert_outputs: torch.Tensor,
    selected_experts: torch.Tensor,
    *,
    return_diagnostics: bool = False,
) -> (
    torch.Tensor
    | Tuple[torch.Tensor, Dict[str, Any]]
):
    """Orthogonalize per-token expert outputs with regularized MGS.

    Args:
        expert_outputs:
            Tensor with shape [num_tokens, num_experts, hidden_dim].
        selected_experts:
            Actually dispatched expert ID for each token, shape [num_tokens].
        return_diagnostics:
            If True, also return additive residual and regularization statistics.

    Returns:
        By default, the orthogonalized tensor with the same shape, device, and
        dtype as expert_outputs. When return_diagnostics=True, returns a pair
        containing that tensor and additive scalar diagnostics.

    The per-token order is deterministic: unselected experts retain their
    ascending expert-ID order, and the selected expert is processed last.
    Modified Gram-Schmidt uses two re-orthogonalization passes in FP32 (or
    FP64 for FP64 inputs).

    Projection uses a Tikhonov-regularized denominator. Normalization uses
    sqrt(residual_norm**2 + regularizer**2), rather than a hard clamp or a
    zero-producing torch.where branch. This keeps gradients finite near
    rank-deficient inputs. Well-conditioned outputs are approximately
    orthonormal; nearly dependent residuals are smoothly attenuated and
    therefore are not forced to have unit norm. Exact Stiefel orthonormality
    cannot be guaranteed when the input expert vectors are rank deficient.
    """
    if not torch.is_tensor(expert_outputs):
        raise TypeError("expert_outputs 必须是 Tensor。")
    if expert_outputs.ndim != 3:
        raise ValueError(
            "expert_outputs 必须具有 [num_tokens, num_experts, hidden_dim] "
            f"形状，当前 shape={tuple(expert_outputs.shape)}。"
        )
    if not expert_outputs.is_floating_point():
        raise TypeError(
            "expert_outputs 必须是浮点 Tensor，"
            f"当前 dtype={expert_outputs.dtype}。"
        )

    num_tokens, num_experts, hidden_dim = expert_outputs.shape
    if num_tokens <= 0 or num_experts <= 0 or hidden_dim <= 0:
        raise ValueError(
            "expert_outputs 的三个维度都必须大于 0，"
            f"当前 shape={tuple(expert_outputs.shape)}。"
        )
    if num_experts > hidden_dim:
        raise ValueError(
            "正交化要求 num_experts <= hidden_dim，"
            f"当前 num_experts={num_experts}, hidden_dim={hidden_dim}。"
        )

    if not torch.is_tensor(selected_experts):
        raise TypeError("selected_experts 必须是 Tensor。")
    if selected_experts.ndim != 1 or selected_experts.shape[0] != num_tokens:
        raise ValueError(
            "selected_experts 必须具有 [num_tokens] 形状，"
            f"当前 shape={tuple(selected_experts.shape)}。"
        )
    if selected_experts.dtype != torch.long:
        raise TypeError(
            "selected_experts 必须是 torch.long，"
            f"当前 dtype={selected_experts.dtype}。"
        )
    if selected_experts.device != expert_outputs.device:
        raise ValueError(
            "selected_experts 与 expert_outputs 必须位于同一设备。"
        )
    # The production caller receives IDs directly from torch.max in the
    # router, so they are guaranteed to be in range. Keep the defensive
    # value check for CPU-side direct calls without forcing a CUDA sync on
    # every training forward.
    if selected_experts.device.type == "cpu" and bool(
        (
            (selected_experts < 0)
            | (selected_experts >= num_experts)
        ).any().item()
    ):
        raise ValueError(
            f"selected_experts 的取值必须位于 [0, {num_experts - 1}]。"
        )

    output_dtype = expert_outputs.dtype
    work_dtype = (
        torch.float64
        if output_dtype == torch.float64
        else torch.float32
    )
    outputs = expert_outputs.to(dtype=work_dtype)

    # Keep all unselected experts in canonical order and move the selected
    # expert to the final Gram-Schmidt position for each individual token.
    expert_ids = torch.arange(
        num_experts,
        device=expert_outputs.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(num_tokens, -1)
    order_keys = torch.where(
        expert_ids == selected_experts.unsqueeze(1),
        torch.full_like(expert_ids, num_experts),
        expert_ids,
    )
    order = torch.argsort(order_keys, dim=1)
    ordered_outputs = outputs.gather(
        dim=1,
        index=order.unsqueeze(-1).expand(-1, -1, hidden_dim),
    )

    projection_eps = (
        GS_PROJECTION_EPS_FP64
        if work_dtype == torch.float64
        else GS_PROJECTION_EPS_FP32
    )

    # Store the unnormalized residual directions and their scale-aware
    # regularizers. Using the raw residual in the regularized projection
    # avoids dividing by the squared norm of a nearly-zero normalized vector.
    basis_residuals = []
    basis_projection_regularizers = []
    orthogonal_basis = []

    if return_diagnostics:
        residual_norm_sum = torch.zeros(
            (),
            device=expert_outputs.device,
            dtype=work_dtype,
        )
        relative_residual_norm_sum = torch.zeros_like(residual_norm_sum)
        regularized_vector_count = torch.zeros(
            (),
            device=expert_outputs.device,
            dtype=torch.long,
        )

    for expert_position in range(num_experts):
        raw_vector = ordered_outputs[:, expert_position, :]
        residual = raw_vector

        # Use a detached reference scale: the regularization level is a
        # numerical safeguard, not an additional gradient path.
        reference_norm = torch.linalg.vector_norm(
            raw_vector.detach(),
            ord=2,
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)
        normalization_regularizer = (
            GS_NORMALIZATION_RELATIVE_EPS * reference_norm
        )
        projection_regularizer = (
            GS_PROJECTION_RELATIVE_EPS * reference_norm
        )

        # Two-pass regularized modified Gram-Schmidt. For well-conditioned
        # basis vectors, the regularized projection approaches the ordinary
        # orthogonal projection. For near-zero basis vectors, its influence
        # smoothly vanishes instead of amplifying round-off and gradients.
        for _ in range(2):
            for previous_residual, previous_regularizer in zip(
                basis_residuals,
                basis_projection_regularizers,
            ):
                denominator = (
                    previous_residual.square().sum(
                        dim=-1,
                        keepdim=True,
                    )
                    + previous_regularizer.square()
                ).clamp_min(projection_eps)
                coefficient = (
                    previous_residual * residual
                ).sum(dim=-1, keepdim=True) / denominator
                residual = residual - coefficient * previous_residual

        residual_norm = torch.linalg.vector_norm(
            residual,
            ord=2,
            dim=-1,
            keepdim=True,
        )

        # Smooth normalization: no discontinuous zeroing branch and no
        # unbounded 1 / residual_norm derivative near rank deficiency.
        normalized = residual / torch.sqrt(
            residual_norm.square() + normalization_regularizer.square()
        )

        basis_residuals.append(residual)
        basis_projection_regularizers.append(projection_regularizer)
        orthogonal_basis.append(normalized)

        if return_diagnostics:
            detached_residual_norm = residual_norm.detach()
            residual_norm_sum = (
                residual_norm_sum + detached_residual_norm.sum()
            )
            relative_residual_norm_sum = (
                relative_residual_norm_sum
                + (
                    detached_residual_norm
                    / reference_norm
                ).sum()
            )
            regularized_vector_count = (
                regularized_vector_count
                + (
                    detached_residual_norm <= normalization_regularizer
                ).sum()
            )

    orthogonalized_ordered = torch.stack(orthogonal_basis, dim=1)

    # Restore the original expert-ID axis after token-dependent reordering.
    inverse_order = torch.argsort(order, dim=1)
    orthogonalized = orthogonalized_ordered.gather(
        dim=1,
        index=inverse_order.unsqueeze(-1).expand(-1, -1, hidden_dim),
    ).to(dtype=output_dtype)

    if not return_diagnostics:
        return orthogonalized

    diagnostics = {
        "orthogonalization_residual_norm_sum": residual_norm_sum,
        "orthogonalization_relative_residual_norm_sum": (
            relative_residual_norm_sum
        ),
        "orthogonalization_regularized_vectors": regularized_vector_count,
        "orthogonalization_vector_count": num_tokens * num_experts,
    }
    return orthogonalized, diagnostics


class OMoESwitchMoE(base.SwitchMoE):
    """SwitchMoE forward pass with per-token OMoE-style output orthogonalization."""

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Optional[torch.Tensor]]:
        if inputs.ndim != 3:
            raise ValueError(
                "SwitchMoE 输入必须是 [batch, tokens, hidden_dim]，"
                f"当前 shape={tuple(inputs.shape)}"
            )

        batch_size, token_count, hidden_dim = inputs.shape
        flat_tokens = inputs.reshape(batch_size * token_count, hidden_dim)
        num_tokens = int(flat_tokens.shape[0])
        if num_tokens == 0:
            raise ValueError("SwitchMoE 不支持空 Token 输入。")

        router_probabilities, top1_probabilities, top1_experts = self.router(
            flat_tokens
        )

        if self.expert_capacity:
            assigned_experts, capacity = self._assign_experts_with_capacity(
                router_probabilities,
                top1_experts,
            )
        else:
            assigned_experts = top1_experts
            capacity = None

        # Preserve the original Switch auxiliary-loss definition.
        aux_loss = None
        if self.auxiliary_loss:
            first_choice_counts = torch.bincount(
                top1_experts,
                minlength=self.num_experts,
            ).to(dtype=router_probabilities.dtype)
            expert_fraction = first_choice_counts / float(num_tokens)
            mean_router_probability = router_probabilities.mean(dim=0)
            aux_loss = (
                self.aux_loss_coef
                * self.num_experts
                * torch.sum(expert_fraction * mean_router_probability)
            )

        experts = list(self.experts.values())
        if len(experts) != self.num_experts:
            raise RuntimeError(
                "SwitchMoE 的专家数量与 num_experts 不一致："
                f"{len(experts)} != {self.num_experts}。"
            )
        if self.num_experts < 2:
            raise ValueError(
                "OMoE 输出正交化至少需要两个专家，"
                f"当前 num_experts={self.num_experts}。"
            )
        if self.num_experts > hidden_dim:
            raise ValueError(
                "OMoE 输出正交化要求 num_experts <= hidden_dim，"
                f"当前 num_experts={self.num_experts}, hidden_dim={hidden_dim}。"
            )

        # Deliberately evaluate every expert for every token. This is the
        # additional dense expert-compute cost required by this Top-1 variant.
        all_expert_outputs = torch.stack(
            [expert(flat_tokens) for expert in experts],
            dim=1,
        )

        # Capacity overflow may reroute a token to a fallback expert. Use the
        # actually dispatched expert as the final Gram-Schmidt column; dropped
        # tokens use their original Top-1 expert only to define a valid order.
        selected_for_orthogonalization = torch.where(
            assigned_experts >= 0,
            assigned_experts,
            top1_experts,
        )
        orthogonalization_result = _gram_schmidt_expert_outputs(
            expert_outputs=all_expert_outputs,
            selected_experts=selected_for_orthogonalization,
            return_diagnostics=self.collect_router_statistics,
        )
        if self.collect_router_statistics:
            (
                orthogonalized_outputs,
                orthogonalization_diagnostics,
            ) = orthogonalization_result
        else:
            orthogonalized_outputs = orthogonalization_result
            orthogonalization_diagnostics = {}

        # The router still dispatches exactly one expert per accepted token.
        # The selected orthogonalized vector is multiplied by the same router
        # probability as in the original SwitchMoE implementation.
        safe_assigned_experts = assigned_experts.clamp_min(0)
        selected_outputs = orthogonalized_outputs.gather(
            dim=1,
            index=safe_assigned_experts.view(-1, 1, 1).expand(
                -1,
                1,
                hidden_dim,
            ),
        ).squeeze(1)
        selected_gates = router_probabilities.gather(
            dim=1,
            index=safe_assigned_experts.unsqueeze(1),
        ).to(dtype=selected_outputs.dtype)
        accepted_mask = (assigned_experts >= 0).to(
            dtype=selected_outputs.dtype
        ).unsqueeze(-1)
        flat_outputs = selected_outputs * selected_gates * accepted_mask
        outputs = flat_outputs.reshape(batch_size, token_count, hidden_dim)

        stats: Dict[str, Any] = {}
        if self.collect_router_statistics:
            tokens_per_expert = torch.bincount(
                top1_experts,
                minlength=self.num_experts,
            )
            processed_tokens_per_expert = torch.bincount(
                assigned_experts[assigned_experts >= 0],
                minlength=self.num_experts,
            )
            rerouted_mask = (
                (assigned_experts >= 0)
                & (assigned_experts != top1_experts)
            )
            dropped_mask = assigned_experts < 0
            stats = {
                "tokens_per_expert": tokens_per_expert,
                "router_prob_sum": router_probabilities.sum(dim=0),
                "top1_prob_sum": top1_probabilities.sum(),
                "num_tokens": num_tokens,
                "processed_tokens_per_expert": processed_tokens_per_expert,
                "rerouted_tokens": int(rerouted_mask.sum().item()),
                "rerouted_tokens_per_expert": torch.bincount(
                    assigned_experts[rerouted_mask],
                    minlength=self.num_experts,
                ),
                "dropped_tokens": int(dropped_mask.sum().item()),
                **orthogonalization_diagnostics,
            }
            if capacity is not None:
                stats["overflow_tokens_per_expert"] = (
                    tokens_per_expert - capacity
                ).clamp_min(0)
            else:
                stats["overflow_tokens_per_expert"] = torch.zeros_like(
                    tokens_per_expert
                )

        return outputs, stats, aux_loss


def _inject_omoe_forward(model: torch.nn.Module) -> torch.nn.Module:
    """Replace only existing SwitchMoE forward implementations in-place.

    OMoESwitchMoE adds no parameters or buffers. Module identities, parameter
    identities, state_dict keys, and federated parameter groups therefore
    remain unchanged.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise TypeError("OMoE 模型必须具有 blocks 属性。")

    converted_layers = 0
    for block_index, block in enumerate(blocks, start=1):
        if not getattr(block, "use_moe", False):
            continue

        moe_module = getattr(block, "mlp", None)
        if not isinstance(moe_module, base.SwitchMoE):
            raise TypeError(
                f"第 {block_index} 个 MoE Block 的 mlp 不是 base.SwitchMoE，"
                f"当前类型：{type(moe_module).__name__}。"
            )

        if not isinstance(moe_module, OMoESwitchMoE):
            moe_module.__class__ = OMoESwitchMoE
        converted_layers += 1

    if converted_layers == 0:
        raise ValueError(
            "当前模型没有启用任何 MoE Block，无法应用 OMoE 输出正交化。"
        )

    return model


def build_omoe_switch_vit_model(cfg: Any) -> torch.nn.Module:
    """Build SwitchViT and inject OMoE behavior into its existing MoE blocks."""
    model = base.build_switch_vit_model(cfg)
    return _inject_omoe_forward(model)


class OMoEUniformExpertAggregator(Aggregator):
    """Client-equal expert Delta aggregation; identical weighting to Uniform."""

    @property
    def method_name(self) -> str:
        return ALGORITHM_NAME

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        return build_uniform_weights(client_updates)


def build_expert_aggregator(cfg: Any) -> base.Aggregator:
    """Build the expert-only Uniform aggregator."""
    return OMoEUniformExpertAggregator(
        cfg=cfg,
        param_group_name="expert",
    )


def main() -> int:
    return base.main(
        expert_aggregator_builder=build_expert_aggregator,
        model_builder=build_omoe_switch_vit_model,
        embedded_method_config=EMBEDDED_METHOD_CONFIG,
        expert_method_name=ALGORITHM_NAME,
    )


if __name__ == "__main__":
    raise SystemExit(main())

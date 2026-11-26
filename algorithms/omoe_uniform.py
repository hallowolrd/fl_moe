"""OMoE-inspired Top-1 expert-output orthogonalization with uniform aggregation.

Each MoE token is evaluated by every expert. Their output vectors are
orthogonalized per token with modified Gram-Schmidt, placing the actually
selected expert last in the orthogonalization order. The original Top-1
router, capacity handling, auxiliary loss, and uniform federated aggregation
remain unchanged.

This module injects the forward behavior through base.ModelBuilder and the
expert aggregation policy through base.Aggregator. It does not modify base.py.
"""

from __future__ import annotations

# Import base before torch so deterministic CUDA environment setup happens first.
import base

from typing import Any, Dict, Sequence

import torch


ALGORITHM_NAME = "omoe_uniform"
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
) -> torch.Tensor:
    """Orthogonalize per-token expert outputs, with the selected expert last.

    Args:
        expert_outputs:
            Tensor with shape [num_tokens, num_experts, hidden_dim].
        selected_experts:
            Actual dispatched expert ID for each token, shape [num_tokens].

    Returns:
        Orthogonalized expert outputs with the same shape and dtype.

    The per-token order is deterministic: unselected experts retain their
    ascending expert-ID order, and the selected expert is processed last.
    Modified Gram-Schmidt is evaluated in FP32 (or FP64 for FP64 inputs).
    Each non-degenerate residual is normalized, so the resulting columns
    satisfy the Stiefel orthonormality constraint up to floating-point error.

    If an expert output is linearly dependent on the preceding outputs, its
    residual is set to zero instead of dividing by a near-zero norm. This is
    numerically safe, but means a rank-deficient token does not produce a
    complete orthonormal frame.
    """
    if expert_outputs.ndim != 3:
        raise ValueError(
            "expert_outputs 必须具有 [num_tokens, num_experts, hidden_dim] "
            f"形状，当前 shape={tuple(expert_outputs.shape)}。"
        )

    num_tokens, num_experts, hidden_dim = expert_outputs.shape
    if num_tokens <= 0 or num_experts <= 0 or hidden_dim <= 0:
        raise ValueError(
            "expert_outputs 的三个维度都必须大于 0，"
            f"当前 shape={tuple(expert_outputs.shape)}。"
        )
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
    if bool(
        torch.any(
            (selected_experts < 0) | (selected_experts >= num_experts)
        )
    ):
        raise ValueError("selected_experts 中包含越界的专家编号。")

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

    eps = 1e-8 if work_dtype == torch.float32 else 1e-12
    orthogonal_basis = []

    for expert_position in range(num_experts):
        residual = ordered_outputs[:, expert_position, :]

        # Modified Gram-Schmidt with a second pass to reduce accumulated
        # floating-point loss of orthogonality.
        if orthogonal_basis:
            for _ in range(2):
                for previous_vector in orthogonal_basis:
                    projection = torch.sum(
                        previous_vector * residual,
                        dim=-1,
                        keepdim=True,
                    )
                    residual = residual - projection * previous_vector

        residual_norm = torch.linalg.vector_norm(
            residual,
            ord=2,
            dim=-1,
            keepdim=True,
        )
        valid = residual_norm > eps
        normalized = residual / residual_norm.clamp_min(eps)
        normalized = torch.where(
            valid,
            normalized,
            torch.zeros_like(normalized),
        )
        orthogonal_basis.append(normalized)

    orthogonalized_ordered = torch.stack(orthogonal_basis, dim=1)

    # Restore the original expert-ID axis after token-dependent reordering.
    inverse_order = torch.argsort(order, dim=1)
    orthogonalized = orthogonalized_ordered.gather(
        dim=1,
        index=inverse_order.unsqueeze(-1).expand(-1, -1, hidden_dim),
    )
    return orthogonalized.to(dtype=output_dtype)


class OMoESwitchMoE(base.SwitchMoE):
    """SwitchMoE forward pass with per-token OMoE-style output orthogonalization."""

    def forward(
        self,
        inputs: torch.Tensor,
    ):
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
        orthogonalized_outputs = _gram_schmidt_expert_outputs(
            expert_outputs=all_expert_outputs,
            selected_experts=selected_for_orthogonalization,
        )

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


def build_omoe_switch_vit_model(cfg: Any) -> torch.nn.Module:
    """Build the standard SwitchViT and inject OMoE behavior into MoE blocks.

    The module tree and parameter names remain unchanged. Only the forward
    implementation of each existing SwitchMoE instance is specialized.
    """
    model = base.build_switch_vit_model(cfg)
    converted_layers = 0

    for block_index, block in enumerate(model.blocks, start=1):
        if not getattr(block, "use_moe", False):
            continue

        moe_module = getattr(block, "mlp", None)
        if not isinstance(moe_module, base.SwitchMoE):
            raise TypeError(
                f"第 {block_index} 个 MoE Block 的 mlp 不是 base.SwitchMoE，"
                f"当前类型：{type(moe_module).__name__}。"
            )

        # The subclass adds no parameters or buffers, so state_dict keys,
        # optimizer parameter identities, and federated parameter groups stay
        # exactly the same as the original SwitchMoE model.
        moe_module.__class__ = OMoESwitchMoE
        converted_layers += 1

    if converted_layers == 0:
        raise ValueError(
            "当前模型没有启用任何 MoE Block，无法应用 OMoE 输出正交化。"
        )

    return model


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

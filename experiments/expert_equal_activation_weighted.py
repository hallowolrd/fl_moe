from __future__ import annotations

"""
Expert-Equal Local Loss + Activation-Weighted Expert Aggregation.

This file contains only method-specific behavior. Dataset handling, the model,
reproducibility, client training, evaluation, logging, and the federated round
loop are provided by base.py.
"""

import math

import base as base

# Import torch only after base has set CUBLAS_WORKSPACE_CONFIG.
import torch
import torch.nn as nn

ALGORITHM_NAME = "expert_equal_activation_weighted"
StateDict = base.StateDict
ClientUpdate = base.ClientUpdate


def validate_method_config(config: base.ExperimentConfig) -> None:
    if config.top_k != 1:
        raise ValueError(
            "This expert-equal loss experiment requires top_k=1 so that "
            "each sample belongs to exactly one expert."
        )


def aggregate_experts_activation_weighted(
    model: nn.Module,
    updates: list[ClientUpdate],
    num_experts: int,
) -> tuple[list[int], list[dict[int, float]]]:
    """
    Activation-Weighted 专家聚合：

    - 仅 route_count > 0 的客户端参与对应专家聚合；
    - 客户端权重为其该专家 route_count 占所有激活客户端总
      route_count 的比例；
    - 每个专家的客户端权重和固定为 1；
    - 某专家本轮无人激活时保留服务器原参数。

    Returns:
        participant_counts:
            每个专家本轮参与聚合的客户端数量。
        client_weights_by_expert:
            每个专家对应的 {client_id: normalized_route_count_weight}。
            无人激活的专家对应空字典。
    """
    participant_counts: list[int] = []
    client_weights_by_expert: list[dict[int, float]] = []

    for expert_idx in range(num_experts):
        old_state = model.get_expert_state_dict(expert_idx, to_cpu=True)
        active_updates = [
            update
            for update in updates
            if int(update.route_counts[expert_idx]) > 0
        ]
        participant_counts.append(len(active_updates))

        if not active_updates:
            client_weights_by_expert.append({})
            continue

        total_route_count = sum(
            int(update.route_counts[expert_idx])
            for update in active_updates
        )
        if total_route_count <= 0:
            raise RuntimeError(
                f"Expert {expert_idx} has active clients but "
                "non-positive total route_count."
            )

        client_weights = {
            update.client_id: (
                int(update.route_counts[expert_idx]) / float(total_route_count)
            )
            for update in active_updates
        }
        weight_sum = sum(client_weights.values())
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"Expert {expert_idx} activation weights sum to "
                f"{weight_sum}, expected 1.0."
            )
        client_weights_by_expert.append(client_weights)

        new_state: StateDict = {}
        for key, old_value in old_state.items():
            if torch.is_floating_point(old_value):
                accumulated = torch.zeros_like(old_value)
                for update in active_updates:
                    weight = client_weights[update.client_id]
                    accumulated.add_(
                        update.expert_deltas[expert_idx][key].to(
                            old_value.dtype
                        ),
                        alpha=weight,
                    )
                new_state[key] = old_value + accumulated
            else:
                new_state[key] = old_value

        model.load_expert_state_dict(
            expert_idx,
            new_state,
            strict=True,
        )

    return participant_counts, client_weights_by_expert


# =============================================================================
# 服务器测试
# =============================================================================

@torch.no_grad()


def server_aggregate(
    *,
    global_model: nn.Module,
    updates: list[ClientUpdate],
    config: base.ExperimentConfig,
    method_state: object | None,
    round_idx: int,
) -> base.AggregationResult:
    del method_state, round_idx
    # Keep the exact original server order:
    # 1) uniform aggregation of shared parameters;
    # 2) activation-weighted aggregation of experts.
    base.aggregate_shared_uniform(global_model, updates)
    participant_counts, client_weights_by_expert = (
        aggregate_experts_activation_weighted(
            global_model,
            updates,
            config.num_experts,
        )
    )
    return base.AggregationResult(
        expert_participants=participant_counts,
        expert_client_weights=client_weights_by_expert,
    )


def main() -> None:
    config = base.parse_config(
        description=(
            "Expert-equal local loss with activation-weighted federated "
            "Sparse-MoE aggregation."
        ),
        method_validator=validate_method_config,
    )
    base.configure_reproducibility(config)
    output_dir = base.create_output_dir(config, ALGORITHM_NAME)
    logger = base.create_logger(output_dir / "train.log", ALGORITHM_NAME)

    try:
        base.run_experiment(
            config,
            output_dir,
            logger,
            algorithm_name=ALGORITHM_NAME,
            local_train_fn=base.train_client,
            server_aggregate_fn=server_aggregate,
            local_objective_description=(
                "Local objective: equal mean over active experts; "
                "each expert loss is the mean CE of samples routed to it; top_k=1"
            ),
            aggregation_description=(
                "Expert aggregation: normalized raw route_count weights over "
                "clients activating each expert; per-expert weight sum=1"
            ),
        )
    except Exception:
        logger.exception("Experiment failed.")
        raise


if __name__ == "__main__":
    main()

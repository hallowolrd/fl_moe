from __future__ import annotations

"""
Expert-Equal Local Loss + Uniform Expert Aggregation.

This file contains only method-specific behavior. Dataset handling, the model,
reproducibility, client training, evaluation, logging, and the federated round
loop are provided by base.py.

The expert aggregation rule is kept identical to the original experiment:
- only clients with route_count > 0 for an expert contribute that expert delta;
- the denominator is always the number of all valid client updates in the round;
- an expert with no active client keeps the current server parameters.
"""

import base as base

# Import torch only after base has set CUBLAS_WORKSPACE_CONFIG.
import torch
import torch.nn as nn


ALGORITHM_NAME = "expert_equal_uniform"
StateDict = base.StateDict
ClientUpdate = base.ClientUpdate


def validate_method_config(config: base.ExperimentConfig) -> None:
    if config.top_k != 1:
        raise ValueError(
            "This expert-equal loss experiment requires top_k=1 so that "
            "each sample belongs to exactly one expert."
        )


def aggregate_experts_uniform(
    model: nn.Module,
    updates: list[ClientUpdate],
    num_experts: int,
) -> list[int]:
    """
    当前项目的 Uniform 专家聚合：
    - 仅激活专家的客户端增量进入求和；
    - 分母始终为本轮全部有效客户端数。
    """
    denominator = float(len(updates))
    participant_counts: list[int] = []

    for expert_idx in range(num_experts):
        old_state = model.get_expert_state_dict(expert_idx, to_cpu=True)
        active_updates = [
            update
            for update in updates
            if int(update.route_counts[expert_idx]) > 0
        ]
        participant_counts.append(len(active_updates))

        if not active_updates:
            continue

        new_state: StateDict = {}
        for key, old_value in old_state.items():
            if torch.is_floating_point(old_value):
                accumulated = torch.zeros_like(old_value)
                for update in active_updates:
                    accumulated.add_(
                        update.expert_deltas[expert_idx][key].to(old_value.dtype)
                    )
                new_state[key] = old_value + accumulated / denominator
            else:
                new_state[key] = old_value

        model.load_expert_state_dict(expert_idx, new_state, strict=True)

    return participant_counts


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
    # 2) uniform aggregation of expert parameters.
    base.aggregate_shared_uniform(global_model, updates)
    participant_counts = aggregate_experts_uniform(
        global_model,
        updates,
        config.num_experts,
    )

    # Logging-only representation of the effective expert coefficients.
    # The original uniform rule divides every active client's expert delta by
    # len(updates), so these coefficients need not sum to 1 when some clients
    # did not activate the expert. This does not participate in aggregation.
    denominator = float(len(updates))
    client_weights_by_expert = [
        {
            update.client_id: 1.0 / denominator
            for update in updates
            if int(update.route_counts[expert_idx]) > 0
        }
        for expert_idx in range(config.num_experts)
    ]

    return base.AggregationResult(
        expert_participants=participant_counts,
        expert_client_weights=client_weights_by_expert,
    )


def main() -> None:
    config = base.parse_config(
        description=(
            "Expert-equal local loss with uniform federated "
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
                "Expert aggregation: only active-client expert deltas are summed; "
                "the denominator is the number of all valid clients in the round"
            ),
        )
    except Exception:
        logger.exception("Experiment failed.")
        raise


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
Uniform Expert Aggregation with active-client denominator.

Method-specific behavior only:
- local training uses base.train_client (standard CE defined in base.py);
- shared parameters use base.aggregate_shared_uniform;
- expert aggregation only operates on expert parameters;
- for each expert, only clients with route_count > 0 contribute expert deltas;
- the denominator is the number of clients that activated that expert;
- an expert with no active client keeps the current server parameters.
"""

import base as base

# Import torch only after base has set CUBLAS_WORKSPACE_CONFIG.
import torch
import torch.nn as nn

ALGORITHM_NAME = "expert_uniform_active_denominator"
StateDict = base.StateDict
ClientUpdate = base.ClientUpdate


def aggregate_experts_uniform(
    model: nn.Module,
    updates: list[ClientUpdate],
    num_experts: int,
) -> list[int]:
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

        denominator = float(len(active_updates))
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

    base.aggregate_shared_uniform(global_model, updates)

    participant_counts = aggregate_experts_uniform(
        global_model,
        updates,
        config.num_experts,
    )

    client_weights_by_expert: list[dict[int, float]] = []
    for expert_idx in range(config.num_experts):
        active_updates = [
            update
            for update in updates
            if int(update.route_counts[expert_idx]) > 0
        ]

        if not active_updates:
            client_weights_by_expert.append({})
            continue

        weight = 1.0 / float(len(active_updates))
        client_weights_by_expert.append(
            {
                update.client_id: weight
                for update in active_updates
            }
        )

    return base.AggregationResult(
        expert_participants=participant_counts,
        expert_client_weights=client_weights_by_expert,
    )


def main() -> None:
    config = base.parse_config(
        description=(
            "Standard-CE federated Sparse-MoE with uniform expert aggregation "
            "over the clients that activated each expert."
        ),
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
                "Local objective: standard sample-mean cross-entropy; "
                "optional balance loss follows base.py configuration"
            ),
            aggregation_description=(
                "Shared aggregation: uniform average over all valid clients; "
                "expert aggregation: equal average over clients that activated "
                "each expert"
            ),
        )
    except Exception:
        logger.exception("Experiment failed.")
        raise


if __name__ == "__main__":
    main()

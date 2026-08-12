from __future__ import annotations

"""
Activation-Count-Weighted Expert Aggregation.

Method-specific behavior only:
- local training uses base.train_client (standard CE defined in base.py);
- shared parameters use base.aggregate_shared_uniform;
- expert aggregation only operates on expert parameters;
- for each expert, only clients with route_count > 0 participate;
- client weights are normalized raw route_count values;
- per-expert client weights sum to 1;
- an expert with no active client keeps the current server parameters.
"""

import math

import base as base

# Import torch only after base has set CUBLAS_WORKSPACE_CONFIG.
import torch
import torch.nn as nn


ALGORITHM_NAME = "expert_activation_count_weighted"
StateDict = base.StateDict
ClientUpdate = base.ClientUpdate


def aggregate_experts_activation_count_weighted(
    model: nn.Module,
    updates: list[ClientUpdate],
    num_experts: int,
) -> tuple[list[int], list[dict[int, float]]]:
    """
    Aggregate each expert using normalized raw activation counts.

    For expert e and active client i:
        w_i,e = route_count_i,e / sum_j route_count_j,e

    Only clients with route_count_i,e > 0 participate.
    """
    participant_counts: list[int] = []
    client_weights_by_expert: list[dict[int, float]] = []

    for expert_idx in range(num_experts):
        old_state = model.get_expert_state_dict(
            expert_idx,
            to_cpu=True,
        )

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
                int(update.route_counts[expert_idx])
                / float(total_route_count)
            )
            for update in active_updates
        }

        weight_sum = sum(client_weights.values())
        if not math.isclose(
            weight_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Expert {expert_idx} activation-count weights sum to "
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

    # Shared parameters always use the common uniform aggregation in base.py.
    base.aggregate_shared_uniform(
        global_model,
        updates,
    )

    # Method-specific aggregation acts only on expert parameters.
    (
        participant_counts,
        client_weights_by_expert,
    ) = aggregate_experts_activation_count_weighted(
        global_model,
        updates,
        config.num_experts,
    )

    return base.AggregationResult(
        expert_participants=participant_counts,
        expert_client_weights=client_weights_by_expert,
    )


def main() -> None:
    config = base.parse_config(
        description=(
            "Standard-CE federated Sparse-MoE with expert aggregation "
            "weighted by normalized raw activation counts."
        ),
    )

    base.configure_reproducibility(config)
    output_dir = base.create_output_dir(
        config,
        ALGORITHM_NAME,
    )
    logger = base.create_logger(
        output_dir / "train.log",
        ALGORITHM_NAME,
    )

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
                "expert aggregation: normalized raw route_count weights over "
                "clients activating each expert; per-expert weight sum=1"
            ),
        )
    except Exception:
        logger.exception("Experiment failed.")
        raise


if __name__ == "__main__":
    main()

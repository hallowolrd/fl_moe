from __future__ import annotations

"""
Activation-Frequency-Weighted Expert Aggregation.

Method-specific behavior only:
- local training uses base.train_client (standard CE defined in base.py);
- shared parameters use base.aggregate_shared_uniform;
- expert aggregation only operates on expert parameters;
- for each client and expert, activation frequency is:
      route_count_i,e / sum_k route_count_i,k
- for each expert, client frequencies are normalized across active clients;
- per-expert client weights sum to 1;
- an expert with no active client keeps the current server parameters.
"""

import math

import base as base

# Import torch only after base has set CUBLAS_WORKSPACE_CONFIG.
import torch
import torch.nn as nn


ALGORITHM_NAME = "expert_activation_frequency_weighted"
StateDict = base.StateDict
ClientUpdate = base.ClientUpdate


def aggregate_experts_activation_frequency_weighted(
    model: nn.Module,
    updates: list[ClientUpdate],
    num_experts: int,
) -> tuple[list[int], list[dict[int, float]]]:
    """
    Aggregate each expert using normalized client-side activation frequencies.

    Client i activation frequency for expert e:
        f_i,e = route_count_i,e / sum_k route_count_i,k

    Expert aggregation weight:
        w_i,e = f_i,e / sum_j f_j,e

    Only clients with route_count_i,e > 0 participate.
    """
    participant_counts: list[int] = []
    client_weights_by_expert: list[dict[int, float]] = []

    total_routes_by_client: dict[int, int] = {}
    for update in updates:
        total_routes = int(update.route_counts.sum().item())
        if total_routes <= 0:
            raise RuntimeError(
                f"Client {update.client_id} has non-positive total route count."
            )
        total_routes_by_client[update.client_id] = total_routes

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

        activation_frequencies = {
            update.client_id: (
                int(update.route_counts[expert_idx])
                / float(total_routes_by_client[update.client_id])
            )
            for update in active_updates
        }

        total_activation_frequency = sum(
            activation_frequencies.values()
        )
        if total_activation_frequency <= 0.0:
            raise RuntimeError(
                f"Expert {expert_idx} has active clients but "
                "non-positive total activation frequency."
            )

        client_weights = {
            client_id: (
                frequency / total_activation_frequency
            )
            for client_id, frequency in activation_frequencies.items()
        }

        weight_sum = sum(client_weights.values())
        if not math.isclose(
            weight_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Expert {expert_idx} activation-frequency weights sum to "
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
    ) = aggregate_experts_activation_frequency_weighted(
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
            "weighted by normalized client-side activation frequencies."
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
                "expert aggregation: normalized client-side activation-frequency "
                "weights, where frequency=expert route_count / total client "
                "route_count; per-expert weight sum=1"
            ),
        )
    except Exception:
        logger.exception("Experiment failed.")
        raise


if __name__ == "__main__":
    main()

from __future__ import annotations

"""pFedMoE-style sample-weighted expert aggregation experiment.

This is a controlled adaptation of the server-side weighted aggregation rule
used by pFedMoE to the expert parameter group of the shared Sparse-MoE model.
All common experiment behavior remains owned by base.py.

Only expert aggregation changes:
    w_k = n_k / sum_j n_j

where n_k is ClientUpdate.num_samples for client k in the current round.
The same client weights are used for every expert parameter. Non-expert
aggregation remains the fixed uniform aggregation implemented by base.py.

This module does not use router/expert usage, routed counts, Fisher/K-FAC
statistics, extra evidence collection, or client-persistent personalized state.
"""

import base
from typing import Any, Dict, Sequence


ALGORITHM_NAME = "pfedmoe_style_expert"

Aggregator = base.Aggregator
ClientUpdate = base.ClientUpdate
build_sample_weights = base.build_sample_weights


EMBEDDED_METHOD_CONFIG = {
    "agg": {
        "non_expert": {"method": "uniform"},
        "expert": {"method": ALGORITHM_NAME},
    },
}


class PFedMoEStyleExpertAggregator(Aggregator):
    """Sample-size-weighted aggregation for the expert parameter group only."""

    @property
    def method_name(self) -> str:
        return ALGORITHM_NAME

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """Return raw client sample counts; base.Aggregator normalizes them."""
        return build_sample_weights(client_updates)


def build_expert_aggregator(cfg: Any) -> base.Aggregator:
    """Build the expert-only pFedMoE-style aggregator injected into base.py."""
    return PFedMoEStyleExpertAggregator(
        cfg=cfg,
        param_group_name="expert",
    )


def main() -> int:
    return base.main(
        expert_aggregator_builder=build_expert_aggregator,
        embedded_method_config=EMBEDDED_METHOD_CONFIG,
        expert_method_name=ALGORITHM_NAME,
    )


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

"""Client-uniform FedAvg with QR-based cross-expert orthogonality.

Compatibility target:
    algorithms/base.py at commit
    8eab05f485f72a00412e9db4f9b845097452175d

This file implements the expert-parameter aggregation policy:
    w_i = 1 / K

The QR retraction lifecycle in this specific base.py revision is owned by
OrthogonalSparseMoEClassifier and is invoked by base.py:
    1. once during orthogonal model construction;
    2. after every local optimizer.step();
    3. after server-side aggregation.

Consequently, this version requires model="sparse_moe_orthogonal".
The later architecture in which the selected expert aggregation method itself
controls QR requires lifecycle hooks to be added to base.py first.
"""

# Import base before importing torch so base.py can prepare the deterministic
# CUDA environment before PyTorch initializes CUDA/cuBLAS.
import base

from typing import Any, Dict, Mapping, Sequence


ALGORITHM_NAME = "fedavg_orthogonal"

Aggregator = base.Aggregator
ClientUpdate = base.ClientUpdate


# These defaults are applied both when using the embedded configuration and
# when a user supplies a YAML file. Explicit YAML / CLI values take priority.
METHOD_CONFIG_DEFAULTS: Dict[str, Any] = {
    "model": "sparse_moe_orthogonal",
    "model_cfg": {
        "backbone": "resnet_cifar",
        "moe_hidden_dim": 128,
    },
    "agg": {
        "non_expert": {"method": "uniform"},
        "expert": {"method": ALGORITHM_NAME},
    },
}

EMBEDDED_METHOD_CONFIG = METHOD_CONFIG_DEFAULTS


def _get_nested(
    cfg: Mapping[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    """Read a dotted key from a nested mapping."""
    current: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def validate_method_config(cfg: Mapping[str, Any]) -> None:
    """Validate that the selected model actually enables this revision's QR hooks."""
    model_name = str(cfg.get("model", "")).strip().lower()
    if model_name != "sparse_moe_orthogonal":
        raise ValueError(
            "fedavg_orthogonal requires model='sparse_moe_orthogonal' "
            "with the current base.py revision. In this revision, QR is "
            "triggered by OrthogonalSparseMoEClassifier methods, not by the "
            "expert Aggregator lifecycle."
        )

    model_cfg = cfg.get("model_cfg", {})
    if not isinstance(model_cfg, Mapping):
        raise ValueError("model_cfg must be a mapping.")

    # Match build_orthogonal_sparse_moe_classifier_from_cfg() precedence:
    # moe_hidden_dim overrides hidden_dim; otherwise hidden_dim defaults to 128.
    hidden_dim_value = model_cfg.get(
        "moe_hidden_dim",
        model_cfg.get("hidden_dim", 128),
    )
    num_experts_value = cfg.get("num_experts", 4)

    if isinstance(hidden_dim_value, bool) or isinstance(num_experts_value, bool):
        raise ValueError("num_experts and moe_hidden_dim must be positive integers.")

    try:
        hidden_dim = int(hidden_dim_value)
        num_experts = int(num_experts_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "num_experts and model_cfg.moe_hidden_dim must be integers."
        ) from exc

    if hidden_dim <= 0:
        raise ValueError(
            f"model_cfg.moe_hidden_dim must be positive; got {hidden_dim}."
        )
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive; got {num_experts}.")

    feature_dim = int(getattr(base, "MOE_FEATURE_DIM", 512))
    total_rows = num_experts * hidden_dim
    if total_rows > feature_dim:
        raise ValueError(
            "The cross-expert QR constraint requires "
            "num_experts * moe_hidden_dim <= backbone feature dimension; "
            f"got {num_experts} * {hidden_dim} = {total_rows}, "
            f"but feature dimension is {feature_dim}. "
            "Reduce num_experts or moe_hidden_dim."
        )


class FedAvgOrthogonalExpertAggregator(Aggregator):
    """Client-uniform expert Delta aggregation for the orthogonal Sparse MoE."""

    @property
    def method_name(self) -> str:
        return ALGORITHM_NAME

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """Assign every participating client the same aggregation weight."""
        return base.build_uniform_weights(client_updates)

    def build_diagnostics(
        self,
        client_updates: Sequence[ClientUpdate],
        weights: Mapping[int, float],
        param_names: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        """Add method metadata to the common Aggregator diagnostics."""
        diagnostics = super().build_diagnostics(
            client_updates=client_updates,
            weights=weights,
            param_names=param_names,
        )
        diagnostics.update(
            {
                "aggregation_rule": "client_uniform",
                "orthogonal_constraint": "cross_expert_fc1_row_orthogonality",
                "orthogonal_projection": "QR",
                "orthogonal_projection_owner": (
                    "base.OrthogonalSparseMoEClassifier"
                ),
                "orthogonal_projection_stages": [
                    "model_initialization",
                    "after_each_local_optimizer_step",
                    "after_server_aggregation",
                ],
            }
        )
        return diagnostics


def build_expert_aggregator(cfg: Any) -> base.Aggregator:
    """Build the expert-only FedAvg-orthogonal aggregator injected into base.py."""
    return FedAvgOrthogonalExpertAggregator(
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
    )


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

"""Client-uniform FedAvg with algorithm-owned Stiefel orthogonality.

The model name selects only the MoE architecture. This algorithm selects the
expert parameter blocks to constrain, retracts them after local optimizer
steps, and retracts them again after server aggregation.
"""

# Import base first so deterministic CUDA environment setup happens before
# importing PyTorch in this entrypoint.
import base

from collections.abc import Mapping
from typing import Any, Dict, Sequence

import torch

from expert_geometry import (
    StiefelParameterBlock,
    build_stiefel_parameter_blocks,
    stiefel_orthogonality_error,
)


ALGORITHM_NAME = "fedavg_orthogonal"

DEFAULT_STIEFEL_BLOCKS = {
    "expert_fc1": {
        "parameter_patterns": ["moe_head.experts.*.fc1.weight"],
    }
}

METHOD_CONFIG_DEFAULTS: Dict[str, Any] = {
    "model": "sparse_moe_classifier",
    "model_cfg": {
        "backbone": "resnet_cifar",
        "moe_hidden_dim": 128,
    },
    "orthogonal": {
        "stiefel_blocks": DEFAULT_STIEFEL_BLOCKS,
    },
    "agg": {
        "non_expert": {"method": "uniform"},
        "expert": {"method": ALGORITHM_NAME},
    },
}

EMBEDDED_METHOD_CONFIG = METHOD_CONFIG_DEFAULTS


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    return base._cfg_get(cfg, key, default)


def _resolve_blocks(
    model: torch.nn.Module,
    expert_parameter_names: Sequence[str],
    cfg: Any,
) -> Dict[str, StiefelParameterBlock]:
    block_cfg = _cfg_get(
        cfg,
        "orthogonal.stiefel_blocks",
        DEFAULT_STIEFEL_BLOCKS,
    )
    blocks = build_stiefel_parameter_blocks(
        model=model,
        expert_parameter_names=expert_parameter_names,
        block_config=block_cfg,
    )
    if not blocks:
        raise ValueError(
            "fedavg_orthogonal 至少需要配置一个 orthogonal.stiefel_blocks。"
        )
    return blocks


def validate_method_config(cfg: Mapping[str, Any]) -> None:
    """Validate algorithm-level orthogonality configuration."""
    model_cfg = _cfg_get(cfg, "model_cfg", {})
    if not isinstance(model_cfg, Mapping):
        raise ValueError("model_cfg 必须是 mapping。")

    block_cfg = _cfg_get(
        cfg,
        "orthogonal.stiefel_blocks",
        DEFAULT_STIEFEL_BLOCKS,
    )
    if not isinstance(block_cfg, Mapping) or not block_cfg:
        raise ValueError(
            "fedavg_orthogonal 要求 orthogonal.stiefel_blocks 是非空 mapping。"
        )


class FedAvgOrthogonalClientStrategy(base.ExpertClientTrainingStrategy):
    """Use the configured optimizer for all expert parameters, then retract."""

    @property
    def includes_expert_parameters_in_model_delta(self) -> bool:
        # The strategy updates expert parameters locally; the server still
        # aggregates their ordinary model deltas with the FedAvg aggregator.
        return True

    def __init__(self, cfg: Any, client_id: int) -> None:
        self.cfg = cfg
        self.client_id = int(client_id)
        self._optimizer: torch.optim.Optimizer | None = None
        self._parameter_map: Dict[str, torch.nn.Parameter] = {}
        self._blocks: Dict[str, StiefelParameterBlock] = {}

    def on_train_start(self, context: base.ExpertClientTrainContext) -> None:
        groups = context.optimization_groups
        self._parameter_map = dict(context.model.named_parameters())
        self._blocks = _resolve_blocks(
            model=context.model,
            expert_parameter_names=groups.expert_parameter_names,
            cfg=self.cfg,
        )
        self._optimizer = base.build_optimizer(
            model=context.model,
            cfg=self.cfg,
            param_names=groups.expert_parameter_names,
        )

        # The server initializes the global model before broadcasting it.
        # Fail early if a client somehow receives a non-Stiefel starting point.
        for name, block in self._blocks.items():
            error = stiefel_orthogonality_error(
                block.get_point(self._parameter_map)
            )
            if error > 5e-4:
                raise ValueError(
                    f"客户端初始 Stiefel block {name!r} 不满足正交约束："
                    f"error={error:.6e}"
                )

    def step(self, context: base.ExpertClientBatchContext) -> None:
        if self._optimizer is None:
            raise RuntimeError("FedAvgOrthogonalClientStrategy 尚未初始化。")

        expert_parameters = [
            self._parameter_map[name]
            for name in context.train_context.optimization_groups.expert_parameter_names
        ]
        grad_clip = context.train_context.non_expert_grad_clip
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                expert_parameters,
                max_norm=float(grad_clip),
            )

        # The framework's optimizer updates non-expert parameters separately.
        self._optimizer.step()

        # Orthogonality is an algorithm policy, not a model side effect.
        with torch.no_grad():
            for block in self._blocks.values():
                block.retract_(self._parameter_map)

    def on_train_end(
        self,
        context: base.ExpertClientTrainContext,
    ) -> Mapping[str, Any]:
        return {}


class FedAvgOrthogonalExpertAggregator(base.Aggregator):
    """Client-uniform FedAvg for experts with algorithm-owned QR retraction."""

    @property
    def method_name(self) -> str:
        return ALGORITHM_NAME

    @property
    def handles_manifold_retraction(self) -> bool:
        return True

    def compute_weights(
        self,
        client_updates: Sequence[base.ClientUpdate],
    ) -> Dict[int, float]:
        return base.build_uniform_weights(client_updates)

    def _blocks_for_model(
        self,
        model: torch.nn.Module,
    ) -> Dict[str, StiefelParameterBlock]:
        if self.method_context is not None and self.method_context.optimization_groups is not None:
            expert_names = self.method_context.optimization_groups.expert_parameter_names
        else:
            groups = base.build_federated_optimization_groups(model)
            expert_names = groups.expert_parameter_names
        return _resolve_blocks(model, expert_names, self.cfg)

    @torch.no_grad()
    def initialize_model(self, model: torch.nn.Module) -> None:
        parameter_map = dict(model.named_parameters())
        for block in self._blocks_for_model(model).values():
            block.retract_(parameter_map)

    @torch.no_grad()
    def after_aggregate(self, model: torch.nn.Module) -> None:
        parameter_map = dict(model.named_parameters())
        for block in self._blocks_for_model(model).values():
            block.retract_(parameter_map)

    def build_diagnostics(
        self,
        client_updates: Sequence[base.ClientUpdate],
        weights: Mapping[int, float],
        param_names: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        diagnostics = super().build_diagnostics(
            client_updates=client_updates,
            weights=weights,
            param_names=param_names,
        )
        block_info: Dict[str, Any] = {}
        if self.method_context is not None and self.method_context.global_model is not None:
            model = self.method_context.global_model
            parameter_map = dict(model.named_parameters())
            for name, block in self._blocks_for_model(model).items():
                block_info[name] = {
                    "parameter_names": list(block.parameter_names),
                    "shape": list(block.shape),
                    "orthogonality_error": stiefel_orthogonality_error(
                        block.get_point(parameter_map)
                    ),
                }

        diagnostics.update(
            {
                "aggregation_rule": "client_uniform",
                "orthogonal_constraint": "algorithm_selected_stiefel_parameter_blocks",
                "orthogonal_projection": "QR",
                "orthogonal_projection_owner": ALGORITHM_NAME,
                "orthogonal_projection_stages": [
                    "global_model_initialization",
                    "after_each_client_expert_optimizer_step",
                    "after_server_expert_aggregation",
                ],
                "stiefel_blocks": block_info,
            }
        )
        return diagnostics


def build_expert_aggregator(cfg: Any) -> base.Aggregator:
    return FedAvgOrthogonalExpertAggregator(
        cfg=cfg,
        param_group_name="expert",
    )


def build_expert_client_training_strategy(
    cfg: Any,
    client_id: int,
) -> base.ExpertClientTrainingStrategy:
    return FedAvgOrthogonalClientStrategy(
        cfg=cfg,
        client_id=client_id,
    )


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
    )


if __name__ == "__main__":
    raise SystemExit(main())

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
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Dict, Iterable, Sequence, Tuple

import torch


def _stiefel_orthogonality_error(point: torch.Tensor) -> float:
    """Return ||X^T X - I||_F using a safe accumulation dtype."""
    if not torch.is_tensor(point) or point.ndim != 2:
        raise TypeError("Stiefel point 必须是二维 Tensor。")

    work = point.detach()
    if work.dtype in (torch.float16, torch.bfloat16):
        work = work.float()
    elif not work.is_floating_point():
        raise TypeError(
            f"Stiefel point 必须是浮点 Tensor，当前 dtype={work.dtype}。"
        )

    if not bool(torch.isfinite(work).all().item()):
        raise ValueError("Stiefel point 包含 NaN 或 Inf。")

    identity = torch.eye(
        work.shape[1],
        device=work.device,
        dtype=work.dtype,
    )
    residual = work.transpose(0, 1) @ work - identity
    return float(torch.linalg.norm(residual, ord="fro").item())


@torch.no_grad()
def _build_stiefel_orthogonality_diagnostics(
    model: torch.nn.Module,
    blocks: Mapping[str, Any],
) -> Dict[str, Any]:
    """Measure each configured Stiefel block on the final server model.

    mean_error is the unweighted arithmetic mean across configured blocks.
    Empty block mappings use null aggregate metrics rather than implying a
    measured zero constraint error.
    """
    named_parameters = dict(model.named_parameters())
    block_diagnostics: Dict[str, Any] = {}
    errors = []

    for raw_name, block in blocks.items():
        name = str(raw_name)
        point = block.get_point(named_parameters)
        error = _stiefel_orthogonality_error(point)
        errors.append(error)

        block_diagnostics[name] = {
            "orthogonality_error": error,
            "shape": list(point.shape),
            "parameter_names": list(block.parameter_names),
        }

    return {
        "metric": "frobenius_norm_XtX_minus_I",
        "measurement_stage": "after_server_aggregation_and_retraction",
        "max_error": max(errors) if errors else None,
        "mean_error": sum(errors) / len(errors) if errors else None,
        "blocks": block_diagnostics,
    }


@dataclass(frozen=True)
class _StiefelParameterBlock:
    """A Stiefel variable formed by stacking row-oriented Parameters.

    Each source Parameter has shape [rows_i, columns]. The block coordinate is
    X = cat(parameters, dim=0).T, so X has shape [columns, sum(rows_i)] and
    satisfies X.T @ X = I.
    """

    name: str
    parameter_names: Tuple[str, ...]
    shape: Tuple[int, int]

    def get_point(
        self,
        named_parameters: Mapping[str, torch.nn.Parameter],
    ) -> torch.Tensor:
        weights = [named_parameters[name] for name in self.parameter_names]
        return torch.cat([weight.detach() for weight in weights], dim=0).transpose(0, 1).clone()

    def get_gradient(
        self,
        named_parameters: Mapping[str, torch.nn.Parameter],
    ) -> torch.Tensor:
        gradients = []
        for name in self.parameter_names:
            parameter = named_parameters[name]
            gradient = parameter.grad
            if gradient is None:
                gradient = torch.zeros_like(parameter)
            elif gradient.is_sparse:
                raise NotImplementedError(
                    f"Stiefel block {self.name!r} does not support sparse gradient: {name}"
                )
            gradients.append(gradient.detach())
        return torch.cat(gradients, dim=0).transpose(0, 1).clone()

    @torch.no_grad()
    def set_point_(
        self,
        named_parameters: Mapping[str, torch.nn.Parameter],
        value: torch.Tensor,
    ) -> None:
        if not torch.is_tensor(value) or value.ndim != 2:
            raise TypeError(f"Stiefel block {self.name!r} must be a 2-D Tensor.")
        if tuple(value.shape) != self.shape:
            raise ValueError(
                f"Stiefel block {self.name!r} shape mismatch: "
                f"expected={self.shape}, actual={tuple(value.shape)}"
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"Stiefel block {self.name!r} contains NaN or Inf.")

        parameters = [named_parameters[name] for name in self.parameter_names]
        rows = value.detach().transpose(0, 1)
        offset = 0
        for parameter in parameters:
            count = int(parameter.shape[0])
            parameter.copy_(
                rows[offset:offset + count].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )
            offset += count

    @torch.no_grad()
    def retract_(
        self,
        named_parameters: Mapping[str, torch.nn.Parameter],
    ) -> torch.Tensor:
        point = self.get_point(named_parameters)
        compute_dtype = (
            torch.float64
            if point.dtype == torch.float64
            else torch.float32
            if point.dtype in (torch.float16, torch.bfloat16)
            else point.dtype
        )
        q, r = torch.linalg.qr(point.to(dtype=compute_dtype), mode="reduced")
        diagonal = torch.diagonal(r)
        signs = torch.where(
            diagonal < 0,
            -torch.ones_like(diagonal),
            torch.ones_like(diagonal),
        )
        retracted = (q * signs.unsqueeze(0)).to(dtype=point.dtype)
        self.set_point_(named_parameters, retracted)
        return retracted


def _build_stiefel_parameter_blocks(
    model: torch.nn.Module,
    expert_parameter_names: Iterable[str],
    block_config: Mapping[str, object] | None,
) -> Dict[str, _StiefelParameterBlock]:
    """Resolve algorithm-selected expert parameter patterns into Stiefel blocks.

    Configuration format:
        {
            "expert_fc1": {
                "parameter_patterns": ["moe_head.experts.*.fc1.weight"]
            }
        }

    A block may contain one matrix or multiple matrices. All selected matrices
    are stacked along rows and must share the same input-column dimension.
    """
    if block_config is None:
        block_config = {}
    elif isinstance(block_config, base.ConfigNode):
        # ConfigNode is the framework's nested config wrapper, not a Mapping.
        # Convert it at this algorithm boundary so the block parser receives
        # the same plain-dict structure as it does from YAML validation.
        block_config = block_config.to_dict()
    if not isinstance(block_config, Mapping):
        raise TypeError("orthogonal.stiefel_blocks 必须是 mapping。")

    named_parameters = dict(model.named_parameters())
    expert_names = set(expert_parameter_names)
    result: Dict[str, _StiefelParameterBlock] = {}
    assigned_names: set[str] = set()

    for raw_name, raw_spec in block_config.items():
        block_name = str(raw_name).strip()
        if not block_name:
            raise ValueError("Stiefel block 名称不能为空。")

        if isinstance(raw_spec, Mapping):
            patterns = raw_spec.get("parameter_patterns", raw_spec.get("patterns", ()))
        else:
            patterns = raw_spec

        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, Sequence) or not patterns:
            raise ValueError(
                f"Stiefel block {block_name!r} 必须提供非空 parameter_patterns。"
            )
        normalized_patterns = [str(pattern).strip() for pattern in patterns]
        if any(not pattern for pattern in normalized_patterns):
            raise ValueError(
                f"Stiefel block {block_name!r} 包含空参数匹配模式。"
            )

        matched_names = [
            name
            for name in named_parameters
            if name in expert_names
            and any(fnmatchcase(name, pattern) for pattern in normalized_patterns)
        ]
        if not matched_names:
            raise ValueError(
                f"Stiefel block {block_name!r} 未匹配到任何可训练专家参数；"
                f"patterns={normalized_patterns}"
            )

        overlap = assigned_names.intersection(matched_names)
        if overlap:
            raise ValueError(
                f"专家参数被多个 Stiefel block 重复选择：{sorted(overlap)}"
            )

        parameters = [named_parameters[name] for name in matched_names]
        for name, parameter in zip(matched_names, parameters):
            if not parameter.requires_grad:
                raise ValueError(
                    f"Stiefel block {block_name!r} 选择了被冻结的参数：{name}"
                )
            if parameter.ndim != 2:
                raise ValueError(
                    f"Stiefel block {block_name!r} 只支持二维参数矩阵；"
                    f"{name} shape={tuple(parameter.shape)}"
                )
            if not parameter.is_floating_point():
                raise TypeError(
                    f"Stiefel block {block_name!r} 要求浮点参数：{name}"
                )

        column_dims = {int(parameter.shape[1]) for parameter in parameters}
        if len(column_dims) != 1:
            raise ValueError(
                f"Stiefel block {block_name!r} 中的矩阵列数必须一致；"
                f"shapes={[tuple(parameter.shape) for parameter in parameters]}"
            )

        total_rows = sum(int(parameter.shape[0]) for parameter in parameters)
        column_dim = next(iter(column_dims))
        if total_rows > column_dim:
            raise ValueError(
                f"Stiefel block {block_name!r} 要求总行数不超过列数；"
                f"total_rows={total_rows}, columns={column_dim}。"
            )

        shape = (column_dim, total_rows)
        result[block_name] = _StiefelParameterBlock(
            name=block_name,
            parameter_names=tuple(matched_names),
            shape=shape,
        )
        assigned_names.update(matched_names)

    return result


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
) -> Dict[str, _StiefelParameterBlock]:
    block_cfg = _cfg_get(
        cfg,
        "orthogonal.stiefel_blocks",
        DEFAULT_STIEFEL_BLOCKS,
    )
    blocks = _build_stiefel_parameter_blocks(
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
        self._blocks: Dict[str, _StiefelParameterBlock] = {}

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
            initial_point = block.get_point(self._parameter_map)
            error = _stiefel_orthogonality_error(initial_point)
            tolerance = (
                5e-3
                if initial_point.dtype in (torch.float16, torch.bfloat16)
                else 5e-4
            )
            if error > tolerance:
                raise ValueError(
                    f"客户端初始 Stiefel block {name!r} 不满足正交约束："
                    f"error={error:.6e}"
                )

    def step(self, context: base.ExpertClientBatchContext) -> None:
        if self._optimizer is None:
            raise RuntimeError("FedAvgOrthogonalClientStrategy 尚未初始化。")

        # The base training loop clips all model gradients together before
        # this strategy step, preserving the ordinary optimizer path's semantics.
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
    ) -> Dict[str, _StiefelParameterBlock]:
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

    @torch.no_grad()
    def build_post_aggregate_diagnostics(
        self,
        model: torch.nn.Module,
    ) -> Dict[str, Any]:
        return {
            "orthogonality": _build_stiefel_orthogonality_diagnostics(
                model=model,
                blocks=self._blocks_for_model(model),
            )
        }

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
                "server_retraction_hook": "algorithm_owned_after_aggregate",
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

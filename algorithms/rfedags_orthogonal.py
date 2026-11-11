from __future__ import annotations

"""Full-client RFedAGS for all MoE expert parameters on a product manifold.

The model only defines the MoE architecture and exposes ordinary Parameters.
This algorithm selects Stiefel-constrained expert parameter blocks through
configuration; every other expert Parameter remains an Euclidean factor.
Non-expert parameters continue to use base.py's fixed client-uniform FedAvg.
"""

# Import base first so deterministic CUDA environment setup happens before
# importing PyTorch in this entrypoint.
import base

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

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


def _stiefel_project_tangent(
    point: torch.Tensor,
    ambient_vector: torch.Tensor,
) -> torch.Tensor:
    """Orthogonally project an ambient matrix onto T_point St(n, p)."""
    if point.ndim != 2 or ambient_vector.shape != point.shape:
        raise ValueError(
            "Stiefel tangent projection requires equally shaped 2-D tensors; "
            f"got point={tuple(point.shape)}, vector={tuple(ambient_vector.shape)}."
        )
    if point.shape[0] < point.shape[1]:
        raise ValueError(
            f"Stiefel matrix must have n >= p, got {tuple(point.shape)}."
        )

    compute_dtype = (
        torch.float64
        if point.dtype == torch.float64 or ambient_vector.dtype == torch.float64
        else torch.float32
        if point.dtype in (torch.float16, torch.bfloat16)
        or ambient_vector.dtype in (torch.float16, torch.bfloat16)
        else point.dtype
    )
    x = point.to(dtype=compute_dtype)
    g = ambient_vector.to(device=point.device, dtype=compute_dtype)
    xtg = x.transpose(0, 1) @ g
    sym_xtg = 0.5 * (xtg + xtg.transpose(0, 1))
    return (g - x @ sym_xtg).to(dtype=point.dtype)


def _stiefel_orthogonality_error(point: torch.Tensor) -> float:
    """Return ||X.T X - I||_F in a numerically safe accumulation dtype."""
    work = (
        point.float()
        if point.dtype in (torch.float16, torch.bfloat16)
        else point
    )
    identity = torch.eye(
        work.shape[1],
        device=work.device,
        dtype=work.dtype,
    )
    return float(
        torch.linalg.norm(
            work.transpose(0, 1) @ work - identity,
            ord="fro",
        ).item()
    )



ALGORITHM_NAME = "rfedags_orthogonal"
PAYLOAD_KEY = "rfedags"
PAYLOAD_VERSION = 2

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
    "optimizer": {
        "type": "sgd",
        "lr": 0.01,
        "momentum": 0.0,
        "nesterov": False,
        "weight_decay": 0.0,
    },
    "orthogonal": {
        "stiefel_blocks": DEFAULT_STIEFEL_BLOCKS,
    },
    "rfedags": {
        # None: follow local_epochs; all clients must consequently execute
        # the same number of batches. Set a positive integer to use fixed K.
        "local_steps": None,
    },
    "agg": {
        "non_expert": {"method": "uniform"},
        "expert": {"method": ALGORITHM_NAME},
    },
}
EMBEDDED_METHOD_CONFIG = METHOD_CONFIG_DEFAULTS


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    return base._cfg_get(cfg, key, default)


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数，当前值：{value!r}")
    return int(value)


def _build_stiefel_blocks(
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
            "rfedags_orthogonal 至少需要配置一个 orthogonal.stiefel_blocks。"
        )
    return blocks


def validate_method_config(cfg: Mapping[str, Any]) -> None:
    """Validate the full-client RSGD assumptions used by RFedAGS."""
    frac = float(_cfg_get(cfg, "frac", 1.0))
    if not math.isfinite(frac) or not math.isclose(
        frac,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "rfedags_orthogonal 当前只实现全客户端参与，要求 frac=1.0；"
            f"当前值：{frac}"
        )

    optimizer_type = str(_cfg_get(cfg, "optimizer.type", "sgd")).lower().strip()
    if optimizer_type != "sgd":
        raise ValueError(
            "rfedags_orthogonal 的本地 RSGD 第一版要求 optimizer.type='sgd'。"
        )

    lr = float(_cfg_get(cfg, "optimizer.lr", 0.01))
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError(f"optimizer.lr 必须是有限正数，当前值：{lr}")

    momentum = float(_cfg_get(cfg, "optimizer.momentum", 0.0))
    if not math.isfinite(momentum) or momentum != 0.0:
        raise ValueError(
            "RFedAGS 第一版要求 momentum=0，以累积真实的 SGD 切向更新。"
        )

    weight_decay = float(_cfg_get(cfg, "optimizer.weight_decay", 0.0))
    if not math.isfinite(weight_decay) or weight_decay != 0.0:
        raise ValueError(
            "RFedAGS 第一版要求 weight_decay=0；如需正则化，应另行定义其"
            "流形梯度与梯度流处理方式。"
        )

    if bool(_cfg_get(cfg, "optimizer.nesterov", False)):
        raise ValueError("RFedAGS 第一版不支持 Nesterov。")

    if base._get_grad_clip(cfg) is not None:
        raise ValueError(
            "RFedAGS 第一版不启用梯度裁剪，避免客户端累积的梯度流与"
            "未裁剪的 RSGD 更新定义不一致。"
        )

    block_cfg = _cfg_get(
        cfg,
        "orthogonal.stiefel_blocks",
        DEFAULT_STIEFEL_BLOCKS,
    )
    if not isinstance(block_cfg, Mapping) or not block_cfg:
        raise ValueError(
            "rfedags_orthogonal 要求 orthogonal.stiefel_blocks 是非空 mapping。"
        )

    local_steps = _cfg_get(cfg, "rfedags.local_steps", None)
    if local_steps is not None:
        _validate_positive_int(local_steps, "rfedags.local_steps")
    else:
        _validate_positive_int(
            _cfg_get(cfg, "local_epochs", 5),
            "local_epochs",
        )


def _tensor_is_finite(value: torch.Tensor) -> bool:
    if value.is_floating_point() or value.is_complex():
        return bool(torch.isfinite(value).all().item())
    return True


def _accumulation_dtype(reference: torch.Tensor) -> torch.dtype:
    if reference.dtype in (torch.float64, torch.complex128):
        return reference.dtype
    if reference.is_complex():
        return torch.complex64
    return torch.float32


def _weighted_tensor_sum(
    client_updates: Sequence[base.ClientUpdate],
    weights: Mapping[int, float],
    values_by_client: Mapping[int, Mapping[str, torch.Tensor]],
    key: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Weighted sum with FP32 accumulation for low-precision model tensors."""
    accumulator = torch.zeros(
        reference.shape,
        device=reference.device,
        dtype=_accumulation_dtype(reference),
    )
    for update in client_updates:
        value = values_by_client[update.client_id][key]
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(
                f"客户端 {update.client_id} 的 {key} shape 不匹配："
                f"expected={tuple(reference.shape)}, actual={tuple(value.shape)}"
            )
        if not _tensor_is_finite(value):
            raise ValueError(
                f"客户端 {update.client_id} 的 {key} 包含 NaN 或 Inf。"
            )
        accumulator.add_(
            value.to(device=reference.device, dtype=accumulator.dtype),
            alpha=float(weights[update.client_id]),
        )
    return accumulator.to(dtype=reference.dtype)


def _weighted_buffer_average(
    client_updates: Sequence[base.ClientUpdate],
    weights: Mapping[int, float],
    buffers_by_client: Mapping[int, Mapping[str, torch.Tensor]],
    name: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Uniformly aggregate expert buffers; integer counters are rounded."""
    if reference.is_floating_point() or reference.is_complex():
        return _weighted_tensor_sum(
            client_updates,
            weights,
            buffers_by_client,
            name,
            reference,
        )

    accumulator = torch.zeros(
        reference.shape,
        device=reference.device,
        dtype=torch.float64,
    )
    for update in client_updates:
        value = buffers_by_client[update.client_id][name]
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(
                f"客户端 {update.client_id} 的 buffer {name} shape 不匹配。"
            )
        accumulator.add_(
            value.to(device=reference.device, dtype=torch.float64),
            alpha=float(weights[update.client_id]),
        )

    if reference.dtype == torch.bool:
        return (accumulator >= 0.5).to(dtype=reference.dtype)
    return torch.round(accumulator).to(dtype=reference.dtype)


class RFedAGSOrthogonalClientStrategy(base.ExpertClientTrainingStrategy):
    """Accumulate RFedAGS streams for every trainable expert parameter."""

    def __init__(self, cfg: Any, client_id: int) -> None:
        self.cfg = cfg
        self.client_id = int(client_id)
        self.lr = float(_cfg_get(cfg, "optimizer.lr", 0.01))
        self.configured_local_steps = _cfg_get(cfg, "rfedags.local_steps", None)

        self._parameter_map: Dict[str, torch.nn.Parameter] = {}
        self._expert_names: list[str] = []
        self._euclidean_names: list[str] = []
        self._blocks: Dict[str, _StiefelParameterBlock] = {}
        self._initial_points: Dict[str, torch.Tensor] = {}
        self._euclidean_stream: Dict[str, torch.Tensor] = {}
        self._stiefel_stream: Dict[str, torch.Tensor] = {}
        self._expert_buffer_names: list[str] = []
        self._local_steps = 0

    def on_train_start(self, context: base.ExpertClientTrainContext) -> None:
        model = context.model
        groups = context.optimization_groups
        self._parameter_map = dict(model.named_parameters())
        self._expert_names = list(groups.expert_parameter_names)
        self._blocks = _build_stiefel_blocks(
            model=model,
            expert_parameter_names=self._expert_names,
            cfg=self.cfg,
        )

        stiefel_parameter_names: set[str] = set()
        self._initial_points = {}
        self._stiefel_stream = {}
        for name, block in self._blocks.items():
            overlap = stiefel_parameter_names.intersection(block.parameter_names)
            if overlap:
                raise ValueError(
                    f"多个 Stiefel block 重复选择专家参数：{sorted(overlap)}"
                )
            stiefel_parameter_names.update(block.parameter_names)

            point = block.get_point(self._parameter_map)
            error = _stiefel_orthogonality_error(point)
            tolerance = (
                5e-3
                if point.dtype in (torch.float16, torch.bfloat16)
                else 1e-4
            )
            if error > tolerance:
                raise ValueError(
                    f"客户端初始 Stiefel block {name!r} 不满足正交约束："
                    f"error={error:.6e}"
                )

            self._initial_points[name] = point.detach().clone()
            self._stiefel_stream[name] = torch.zeros(
                point.shape,
                device=point.device,
                dtype=_accumulation_dtype(point),
            )

        self._euclidean_names = [
            name
            for name in self._expert_names
            if name not in stiefel_parameter_names
        ]
        covered_names = set(self._euclidean_names) | stiefel_parameter_names
        if covered_names != set(self._expert_names):
            missing = sorted(set(self._expert_names) - covered_names)
            extra = sorted(covered_names - set(self._expert_names))
            raise ValueError(
                "RFedAGS 参数几何分组未完整覆盖专家参数："
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

        self._euclidean_stream = {}
        for name in self._euclidean_names:
            parameter = self._parameter_map[name]
            self._euclidean_stream[name] = torch.zeros(
                parameter.shape,
                device=parameter.device,
                dtype=_accumulation_dtype(parameter),
            )

        all_parameter_names = set(self._parameter_map)
        self._expert_buffer_names = sorted(
            set(groups.expert_state_names) - all_parameter_names
        )
        self._local_steps = 0

    def local_step_limit(
        self,
        context: base.ExpertClientTrainContext,
    ) -> Optional[int]:
        if self.configured_local_steps is None:
            return None
        return _validate_positive_int(
            self.configured_local_steps,
            "rfedags.local_steps",
        )

    def step(self, context: base.ExpertClientBatchContext) -> None:
        model = context.train_context.model
        self._parameter_map = dict(model.named_parameters())

        # Euclidean expert factors: the Euclidean vector transport is identity.
        with torch.no_grad():
            for name in self._euclidean_names:
                parameter = self._parameter_map[name]
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise NotImplementedError(
                        f"RFedAGS 第一版不支持稀疏专家梯度：{name}"
                    )
                if not _tensor_is_finite(gradient):
                    raise ValueError(f"专家参数 {name} 的梯度包含 NaN 或 Inf。")

                local_step = gradient.detach().mul(-self.lr)
                self._euclidean_stream[name].add_(
                    local_step.to(dtype=self._euclidean_stream[name].dtype)
                )
                parameter.add_(local_step)

            # Stiefel expert factors: Riemannian SGD + projection transport
            # back to the common round-start tangent space.
            for name, block in self._blocks.items():
                point = block.get_point(self._parameter_map)
                ambient_gradient = block.get_gradient(self._parameter_map)
                if not _tensor_is_finite(ambient_gradient):
                    raise ValueError(
                        f"Stiefel block {name!r} 的欧氏梯度包含 NaN 或 Inf。"
                    )

                riemannian_gradient = _stiefel_project_tangent(
                    point,
                    ambient_gradient,
                )
                local_step = riemannian_gradient.mul(-self.lr)

                transported_step = _stiefel_project_tangent(
                    self._initial_points[name],
                    local_step,
                )
                self._stiefel_stream[name].add_(
                    transported_step.to(dtype=self._stiefel_stream[name].dtype)
                )

                block.set_point_(
                    self._parameter_map,
                    point + local_step,
                )
                block.retract_(self._parameter_map)

        self._local_steps += 1

    def on_train_end(
        self,
        context: base.ExpertClientTrainContext,
    ) -> Mapping[str, Any]:
        if self._local_steps <= 0:
            raise ValueError("RFedAGS 客户端本地更新步数必须大于 0。")

        state_dict = context.model.state_dict()
        expert_buffers = {
            name: state_dict[name].detach().clone()
            for name in self._expert_buffer_names
        }
        return {
            PAYLOAD_KEY: {
                "version": PAYLOAD_VERSION,
                "local_steps": int(self._local_steps),
                # Streams are sums of actual negative local update vectors.
                "euclidean_stream": {
                    name: value.detach().clone()
                    for name, value in self._euclidean_stream.items()
                },
                "stiefel_stream": {
                    name: value.detach().clone()
                    for name, value in self._stiefel_stream.items()
                },
                "expert_buffers": expert_buffers,
            }
        }


class RFedAGSOrthogonalAggregator(base.Aggregator):
    """Aggregate all expert Gradient Streams with client-uniform RFedAGS."""

    @property
    def method_name(self) -> str:
        return ALGORITHM_NAME

    @property
    def consumes_algorithm_payload(self) -> bool:
        return True

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
        if self.method_context is None or self.method_context.optimization_groups is None:
            groups = base.build_federated_optimization_groups(model)
            expert_names = groups.expert_parameter_names
        else:
            expert_names = self.method_context.optimization_groups.expert_parameter_names
        return _build_stiefel_blocks(
            model=model,
            expert_parameter_names=expert_names,
            cfg=self.cfg,
        )

    @torch.no_grad()
    def initialize_model(self, model: torch.nn.Module) -> None:
        parameter_map = dict(model.named_parameters())
        for block in self._blocks_for_model(model).values():
            block.retract_(parameter_map)

    def aggregate(
        self,
        global_state: Mapping[str, torch.Tensor],
        client_updates: Sequence[base.ClientUpdate],
        param_names: Optional[Sequence[str]] = None,
        base_state: Optional[Mapping[str, torch.Tensor]] = None,
        strict: bool = True,
    ) -> base.AggregationResult:
        self._validate_client_updates(client_updates)

        if self.method_context is None:
            raise RuntimeError("RFedAGS 聚合器尚未注入 MethodContext。")
        model = self.method_context.global_model
        groups = self.method_context.optimization_groups
        if model is None or groups is None:
            raise RuntimeError(
                "RFedAGS 聚合器需要 MethodContext.global_model 和 "
                "MethodContext.optimization_groups。"
            )

        configured_num_clients = int(_cfg_get(self.cfg, "num_clients", 0))
        expected_client_ids = set(range(configured_num_clients))
        actual_client_ids = {int(update.client_id) for update in client_updates}
        if actual_client_ids != expected_client_ids:
            raise ValueError(
                "RFedAGS-Orthogonal 要求每轮全部客户端参与。"
                f" expected={sorted(expected_client_ids)}, "
                f"actual={sorted(actual_client_ids)}"
            )

        expected_state_names = set(groups.expert_state_names)
        if param_names is not None and set(param_names) != expected_state_names:
            raise ValueError(
                "服务器传入的专家 state 名称与优化分组不一致。"
            )

        named_parameters = dict(model.named_parameters())
        expert_names = list(groups.expert_parameter_names)
        blocks = self._blocks_for_model(model)
        stiefel_parameter_names = {
            name
            for block in blocks.values()
            for name in block.parameter_names
        }
        euclidean_names = [
            name for name in expert_names if name not in stiefel_parameter_names
        ]
        expert_buffer_names = sorted(
            expected_state_names - set(named_parameters)
        )

        weights = base.normalize_weights(self.compute_weights(client_updates))
        euclidean_streams: Dict[int, Mapping[str, torch.Tensor]] = {}
        stiefel_streams: Dict[int, Mapping[str, torch.Tensor]] = {}
        buffers_by_client: Dict[int, Mapping[str, torch.Tensor]] = {}
        local_steps_by_client: Dict[int, int] = {}

        for update in client_updates:
            root = update.algorithm_payload.get(PAYLOAD_KEY)
            if not isinstance(root, Mapping):
                raise TypeError(
                    f"客户端 {update.client_id} 缺少 {PAYLOAD_KEY} 算法载荷。"
                )
            if root.get("version") != PAYLOAD_VERSION:
                raise ValueError(
                    f"客户端 {update.client_id} 的 RFedAGS payload version "
                    f"不受支持：{root.get('version')!r}"
                )

            local_steps = _validate_positive_int(
                root.get("local_steps"),
                f"client[{update.client_id}].local_steps",
            )
            local_steps_by_client[update.client_id] = local_steps

            euclidean_stream = root.get("euclidean_stream")
            stiefel_stream = root.get("stiefel_stream")
            expert_buffers = root.get("expert_buffers")
            if not isinstance(euclidean_stream, Mapping):
                raise TypeError("euclidean_stream 必须是 Mapping。")
            if not isinstance(stiefel_stream, Mapping):
                raise TypeError("stiefel_stream 必须是 Mapping。")
            if not isinstance(expert_buffers, Mapping):
                raise TypeError("expert_buffers 必须是 Mapping。")

            if set(euclidean_stream) != set(euclidean_names):
                raise ValueError(
                    f"客户端 {update.client_id} 的欧氏专家 stream 参数集合不完整。"
                )
            if set(stiefel_stream) != set(blocks):
                raise ValueError(
                    f"客户端 {update.client_id} 的 Stiefel stream block 集合不完整。"
                )
            if set(expert_buffers) != set(expert_buffer_names):
                raise ValueError(
                    f"客户端 {update.client_id} 的专家 buffer 集合不完整。"
                )

            for name, value in euclidean_stream.items():
                if not torch.is_tensor(value):
                    raise TypeError(
                        f"客户端 {update.client_id} 的欧氏 stream {name} 必须是 Tensor。"
                    )
                reference = named_parameters[name]
                if tuple(value.shape) != tuple(reference.shape):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 {name} stream shape 不匹配。"
                    )
                if not _tensor_is_finite(value):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 {name} stream 包含 NaN 或 Inf。"
                    )

            for name, value in stiefel_stream.items():
                if not torch.is_tensor(value):
                    raise TypeError(
                        f"客户端 {update.client_id} 的 Stiefel stream {name} 必须是 Tensor。"
                    )
                if tuple(value.shape) != tuple(blocks[name].shape):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 Stiefel block {name} shape 不匹配。"
                    )
                if not _tensor_is_finite(value):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 Stiefel stream {name} 包含 NaN 或 Inf。"
                    )

            for name, value in expert_buffers.items():
                if not torch.is_tensor(value):
                    raise TypeError(
                        f"客户端 {update.client_id} 的专家 buffer {name} 必须是 Tensor。"
                    )
                reference = global_state[name]
                if tuple(value.shape) != tuple(reference.shape):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 buffer {name} shape 不匹配。"
                    )
                if not _tensor_is_finite(value):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 buffer {name} 包含 NaN 或 Inf。"
                    )

            euclidean_streams[update.client_id] = euclidean_stream
            stiefel_streams[update.client_id] = stiefel_stream
            buffers_by_client[update.client_id] = expert_buffers

        unique_local_steps = set(local_steps_by_client.values())
        if len(unique_local_steps) != 1:
            raise ValueError(
                "RFedAGS 全客户端版本要求每个客户端执行相同的本地更新步数；"
                f"当前各客户端步数：{local_steps_by_client}。"
                "请设置统一的 rfedags.local_steps，或确保各客户端 DataLoader "
                "具有相同 batch 数。"
            )
        local_steps = unique_local_steps.pop()

        source_state = base_state if base_state is not None else global_state
        new_state_dict = {
            name: value.detach().clone()
            for name, value in source_state.items()
        }
        if set(new_state_dict) != set(global_state):
            raise ValueError("base_state 与 global_state 的 state_dict keys 不一致。")

        # Euclidean expert factors: average their actual local update streams.
        for name in euclidean_names:
            reference = global_state[name]
            aggregated_stream = _weighted_tensor_sum(
                client_updates,
                weights,
                euclidean_streams,
                name,
                reference,
            )
            new_state_dict[name] = reference + aggregated_stream

        # Rebuild the Stiefel blocks on a CPU copy of the round-start model.
        model_copy = copy.deepcopy(model).to("cpu")
        model_copy.load_state_dict(global_state, strict=True)
        copy_parameters = dict(model_copy.named_parameters())

        tangent_residuals: Dict[str, float] = {}
        for name, block in blocks.items():
            point = block.get_point(copy_parameters)
            aggregation_reference = (
                point.float()
                if point.dtype in (torch.float16, torch.bfloat16)
                else point
            )
            aggregated_stream = _weighted_tensor_sum(
                client_updates,
                weights,
                stiefel_streams,
                name,
                aggregation_reference,
            )

            point_work = (
                point.float()
                if point.dtype in (torch.float16, torch.bfloat16)
                else point
            )
            stream_work = (
                aggregated_stream.float()
                if aggregated_stream.dtype in (torch.float16, torch.bfloat16)
                else aggregated_stream
            )
            tangent_residual = (
                point_work.transpose(0, 1) @ stream_work
                + stream_work.transpose(0, 1) @ point_work
            )
            residual_norm = float(torch.linalg.norm(tangent_residual).item())
            stream_norm = float(torch.linalg.norm(stream_work).item())
            relative_residual = residual_norm / max(1.0, stream_norm)
            tangent_residuals[name] = relative_residual
            tangent_tolerance = (
                5e-3
                if point.dtype in (torch.float16, torch.bfloat16)
                else 5e-4
            )
            if relative_residual > tangent_tolerance:
                raise ValueError(
                    f"聚合后的 Stiefel block {name!r} Gradient Stream "
                    "不属于全局初始点的切空间："
                    f"relative_residual={relative_residual:.6e}"
                )

            block.set_point_(
                copy_parameters,
                point + aggregated_stream,
            )
            block.retract_(copy_parameters)

        model_copy_state = model_copy.state_dict()
        for block in blocks.values():
            for parameter_name in block.parameter_names:
                new_state_dict[parameter_name] = (
                    model_copy_state[parameter_name].detach().clone()
                )

        # Expert buffers are not differentiable manifold coordinates. Preserve
        # their normal client-uniform state aggregation separately.
        for name in expert_buffer_names:
            new_state_dict[name] = _weighted_buffer_average(
                client_updates,
                weights,
                buffers_by_client,
                name,
                global_state[name],
            )

        if set(new_state_dict) != set(global_state):
            missing = sorted(set(global_state) - set(new_state_dict))
            extra = sorted(set(new_state_dict) - set(global_state))
            raise ValueError(
                "RFedAGS 必须返回完整模型 state_dict。"
                f" missing={missing[:10]}, extra={extra[:10]}"
            )
        base.check_finite_state_dict(new_state_dict)

        stiefel_diagnostics: Dict[str, Any] = {}
        for name, block in blocks.items():
            point = block.get_point(copy_parameters)
            stiefel_diagnostics[name] = {
                "shape": list(block.shape),
                "parameter_names": list(block.parameter_names),
                "tangent_relative_residual": tangent_residuals[name],
                "orthogonality_error": _stiefel_orthogonality_error(point),
            }

        diagnostics = {
            "aggregation_rule": "client_uniform",
            "client_participation": "all_clients",
            "gradient_stream": "sum_of_negative_local_update_vectors",
            "local_steps": int(local_steps),
            "num_euclidean_expert_parameters": len(euclidean_names),
            "stiefel_blocks": stiefel_diagnostics,
            "vector_transport": "Stiefel tangent projection to round-start point",
            "server_retraction": "algorithm_owned_QR",
            "weights": {
                int(client_id): float(weight)
                for client_id, weight in weights.items()
            },
        }

        return base.AggregationResult(
            new_state_dict=new_state_dict,
            weights=weights,
            diagnostics=diagnostics,
        )


def build_expert_aggregator(cfg: Any) -> base.Aggregator:
    return RFedAGSOrthogonalAggregator(
        cfg=cfg,
        param_group_name="expert",
    )


def build_expert_client_training_strategy(
    cfg: Any,
    client_id: int,
) -> base.ExpertClientTrainingStrategy:
    return RFedAGSOrthogonalClientStrategy(
        cfg=cfg,
        client_id=client_id,
    )


def register_method_cli_arguments(parser: Any) -> None:
    parser.add_argument(
        "--rfedags-local-steps",
        type=int,
        default=None,
        help=(
            "RFedAGS 每个客户端每轮固定本地更新步数 K；"
            "不指定时沿用 local_epochs。"
        ),
    )


def build_method_cli_overrides(args: Any) -> Dict[str, Any]:
    local_steps = getattr(args, "rfedags_local_steps", None)
    if local_steps is None:
        return {}
    return {"rfedags": {"local_steps": int(local_steps)}}


def main() -> int:
    return base.main(
        expert_aggregator_builder=build_expert_aggregator,
        embedded_method_config=EMBEDDED_METHOD_CONFIG,
        expert_method_name=ALGORITHM_NAME,
        method_config_defaults=METHOD_CONFIG_DEFAULTS,
        method_config_validator=validate_method_config,
        method_cli_argument_registrar=register_method_cli_arguments,
        method_cli_overrides_builder=build_method_cli_overrides,
        expert_client_training_strategy_builder=(
            build_expert_client_training_strategy
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())

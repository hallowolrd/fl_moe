from __future__ import annotations

"""Full-client RFedAGS for MoE expert parameters on a product manifold.

All expert parameters are handled by this algorithm:
- Euclidean expert parameters accumulate ordinary negative SGD steps.
- Stiefel expert variables accumulate Riemannian steps transported to the
  round-start tangent space.
Non-expert parameters continue to use base.py's fixed client-uniform FedAvg.
"""

# Import base first so deterministic CUDA environment setup happens before
# importing PyTorch in this entrypoint.
import base

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

import torch


ALGORITHM_NAME = "rfedags_orthogonal"
PAYLOAD_KEY = "rfedags"
PAYLOAD_VERSION = 1


METHOD_CONFIG_DEFAULTS: Dict[str, Any] = {
    "model": "sparse_moe_orthogonal",
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


def validate_method_config(cfg: Mapping[str, Any]) -> None:
    """Validate the full-client RSGD assumptions used by RFedAGS."""
    frac = float(_cfg_get(cfg, "frac", 1.0))
    if not math.isfinite(frac) or not math.isclose(frac, 1.0, rel_tol=0.0, abs_tol=1e-12):
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

    local_steps = _cfg_get(cfg, "rfedags.local_steps", None)
    if local_steps is not None:
        _validate_positive_int(local_steps, "rfedags.local_steps")
    else:
        local_epochs = _validate_positive_int(
            _cfg_get(cfg, "local_epochs", 5),
            "local_epochs",
        )
        if local_epochs <= 0:
            raise ValueError("未设置 rfedags.local_steps 时，local_epochs 必须大于 0。")


def stiefel_project_tangent(
    point: torch.Tensor,
    ambient_vector: torch.Tensor,
) -> torch.Tensor:
    """Project an ambient matrix onto T_point St(n, p), using the Euclidean metric."""
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


def stiefel_qr_retraction(
    point: torch.Tensor,
    tangent_vector: torch.Tensor,
) -> torch.Tensor:
    """First-order QR retraction with a deterministic diagonal-sign convention."""
    if point.ndim != 2 or tangent_vector.shape != point.shape:
        raise ValueError(
            "Stiefel QR retraction requires equally shaped 2-D tensors; "
            f"got point={tuple(point.shape)}, vector={tuple(tangent_vector.shape)}."
        )
    if point.shape[0] < point.shape[1]:
        raise ValueError(
            f"Stiefel matrix must have n >= p, got {tuple(point.shape)}."
        )

    compute_dtype = (
        torch.float64
        if point.dtype == torch.float64 or tangent_vector.dtype == torch.float64
        else torch.float32
        if point.dtype in (torch.float16, torch.bfloat16)
        or tangent_vector.dtype in (torch.float16, torch.bfloat16)
        else point.dtype
    )
    matrix = point.to(dtype=compute_dtype) + tangent_vector.to(
        device=point.device,
        dtype=compute_dtype,
    )
    q, r = torch.linalg.qr(matrix, mode="reduced")
    diagonal = torch.diagonal(r)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    return (q * signs.unsqueeze(0)).to(dtype=point.dtype)


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
        self._euclidean_names: list[str] = []
        self._manifold_specs: Dict[str, base.ManifoldParameterSpec] = {}
        self._manifold_names: list[str] = []
        self._initial_manifold_points: Dict[str, torch.Tensor] = {}
        self._euclidean_stream: Dict[str, torch.Tensor] = {}
        self._manifold_stream: Dict[str, torch.Tensor] = {}
        self._expert_buffer_names: list[str] = []
        self._local_steps = 0

    def on_train_start(self, context: base.ExpertClientTrainContext) -> None:
        model = context.model
        groups = context.optimization_groups
        if not isinstance(model, base.ManifoldParameterProvider):
            raise TypeError(
                "RFedAGS-Orthogonal 要求模型实现 ManifoldParameterProvider。"
            )
        if not groups.expert_manifold:
            raise ValueError(
                "RFedAGS-Orthogonal 要求模型至少声明一个 expert 流形变量。"
            )

        self._parameter_map = dict(model.named_parameters())
        self._euclidean_names = list(groups.expert_euclidean_parameters)
        self._manifold_specs = dict(groups.expert_manifold)
        self._manifold_names = list(self._manifold_specs)
        self._initial_manifold_points = {}
        self._euclidean_stream = {}
        self._manifold_stream = {}
        self._local_steps = 0

        covered_parameter_names = set(self._euclidean_names)
        for name, spec in self._manifold_specs.items():
            if str(spec.manifold).lower().strip() != "stiefel":
                raise NotImplementedError(
                    f"RFedAGS-Orthogonal 当前只实现 Stiefel 流形，"
                    f"变量 {name!r} 声明为 {spec.manifold!r}。"
                )
            if covered_parameter_names.intersection(spec.parameter_names):
                raise ValueError(
                    f"专家参数被重复分配到欧氏组和流形组：{name!r}"
                )
            covered_parameter_names.update(spec.parameter_names)

            point = model.get_manifold_parameter(name).detach().clone()
            if tuple(point.shape) != tuple(spec.shape):
                raise ValueError(
                    f"流形变量 {name!r} shape 不匹配："
                    f"spec={spec.shape}, actual={tuple(point.shape)}"
                )
            if point.ndim != 2 or point.shape[0] < point.shape[1]:
                raise ValueError(
                    f"Stiefel 变量 {name!r} 必须满足二维矩阵 n >= p，"
                    f"当前 shape={tuple(point.shape)}。"
                )
            point_work = (
                point.float()
                if point.dtype in (torch.float16, torch.bfloat16)
                else point
            )
            identity = torch.eye(
                point_work.shape[1],
                device=point_work.device,
                dtype=point_work.dtype,
            )
            orthogonality_error = torch.linalg.norm(
                point_work.transpose(0, 1) @ point_work - identity,
                ord="fro",
            )
            tolerance = 5e-3 if point.dtype in (torch.float16, torch.bfloat16) else 1e-4
            if float(orthogonality_error.item()) > tolerance:
                raise ValueError(
                    f"流形变量 {name!r} 初始点不满足 Stiefel 约束："
                    f"error={float(orthogonality_error.item()):.6e}"
                )

            self._initial_manifold_points[name] = point
            self._manifold_stream[name] = torch.zeros(
                point.shape,
                device=point.device,
                dtype=_accumulation_dtype(point),
            )

        expected_trainable_expert_names = set(groups.expert_parameter_names)
        if covered_parameter_names != expected_trainable_expert_names:
            missing = sorted(expected_trainable_expert_names - covered_parameter_names)
            extra = sorted(covered_parameter_names - expected_trainable_expert_names)
            raise ValueError(
                "RFedAGS 参数几何分组没有完整覆盖可训练专家参数："
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

        for name in self._euclidean_names:
            if name not in self._parameter_map:
                raise KeyError(f"找不到专家欧氏参数：{name}")
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
        train_context = context.train_context
        model = train_context.model
        if not isinstance(model, base.ManifoldParameterProvider):
            raise TypeError("本地模型未实现 ManifoldParameterProvider。")

        # Euclidean expert factors: ordinary SGD steps are already tangent
        # vectors, and vector transport in Euclidean space is the identity.
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
                self._euclidean_stream[name].add_(local_step)
                parameter.add_(local_step)

            # Stiefel expert factors: Riemannian SGD + projection transport
            # back to the common round-start tangent space.
            for name, spec in self._manifold_specs.items():
                point = model.get_manifold_parameter(name)
                ambient_gradient = model.get_manifold_parameter_gradient(name)
                if tuple(ambient_gradient.shape) != tuple(point.shape):
                    raise ValueError(
                        f"流形变量 {name!r} 的梯度 shape 不匹配："
                        f"point={tuple(point.shape)}, "
                        f"gradient={tuple(ambient_gradient.shape)}"
                    )
                if not _tensor_is_finite(ambient_gradient):
                    raise ValueError(
                        f"流形变量 {name!r} 的欧氏梯度包含 NaN 或 Inf。"
                    )

                riemannian_gradient = stiefel_project_tangent(
                    point,
                    ambient_gradient,
                )
                local_step = riemannian_gradient.mul(-self.lr)

                transported_step = stiefel_project_tangent(
                    self._initial_manifold_points[name],
                    local_step,
                )
                self._manifold_stream[name].add_(transported_step)

                model.set_manifold_parameter_(
                    name,
                    point + local_step,
                )

        train_context.retract_manifold_parameters_(self._manifold_names)
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
                # Each stream stores the sum of actual negative local update
                # vectors, not the sum of positive gradients.
                "euclidean_stream": {
                    name: value.detach().clone()
                    for name, value in self._euclidean_stream.items()
                },
                "manifold_stream": {
                    name: value.detach().clone()
                    for name, value in self._manifold_stream.items()
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
        if not isinstance(model, base.ManifoldParameterProvider):
            raise TypeError(
                "RFedAGS-Orthogonal 要求全局模型实现 ManifoldParameterProvider。"
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
        euclidean_names = list(groups.expert_euclidean_parameters)
        manifold_specs = dict(groups.expert_manifold)
        manifold_names = list(manifold_specs)
        expert_buffer_names = sorted(
            expected_state_names - set(named_parameters)
        )

        weights = base.normalize_weights(self.compute_weights(client_updates))
        euclidean_streams: Dict[int, Mapping[str, torch.Tensor]] = {}
        manifold_streams: Dict[int, Mapping[str, torch.Tensor]] = {}
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
            manifold_stream = root.get("manifold_stream")
            expert_buffers = root.get("expert_buffers")
            if not isinstance(euclidean_stream, Mapping):
                raise TypeError("euclidean_stream 必须是 Mapping。")
            if not isinstance(manifold_stream, Mapping):
                raise TypeError("manifold_stream 必须是 Mapping。")
            if not isinstance(expert_buffers, Mapping):
                raise TypeError("expert_buffers 必须是 Mapping。")

            if set(euclidean_stream) != set(euclidean_names):
                raise ValueError(
                    f"客户端 {update.client_id} 的欧氏专家 stream 参数集合不完整。"
                )
            if set(manifold_stream) != set(manifold_names):
                raise ValueError(
                    f"客户端 {update.client_id} 的流形 stream 集合不完整。"
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

            for name, value in manifold_stream.items():
                if not torch.is_tensor(value):
                    raise TypeError(
                        f"客户端 {update.client_id} 的流形 stream {name} 必须是 Tensor。"
                    )
                reference = model.get_manifold_parameter(name)
                if tuple(value.shape) != tuple(reference.shape):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 {name} stream shape 不匹配。"
                    )
                if not _tensor_is_finite(value):
                    raise ValueError(
                        f"客户端 {update.client_id} 的 {name} stream 包含 NaN 或 Inf。"
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
            manifold_streams[update.client_id] = manifold_stream
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

        # Euclidean factors: vector transport is the identity and Retr_x(v)=x+v.
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

        # Stiefel factors: all client streams have already been transported
        # into the tangent space at the same round-start global point.
        model_copy = copy.deepcopy(model).to("cpu")
        model_copy.load_state_dict(global_state, strict=True)
        tangent_residuals: Dict[str, float] = {}
        for name, spec in manifold_specs.items():
            point = model_copy.get_manifold_parameter(name)
            aggregation_reference = (
                point.float()
                if point.dtype in (torch.float16, torch.bfloat16)
                else point
            )
            aggregated_stream = _weighted_tensor_sum(
                client_updates,
                weights,
                manifold_streams,
                name,
                aggregation_reference,
            )

            point_work = point.float() if point.dtype in (torch.float16, torch.bfloat16) else point
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
            if relative_residual > 5e-4:
                raise ValueError(
                    f"聚合后的 {name} Gradient Stream 不属于全局初始点的切空间："
                    f"relative_residual={relative_residual:.6e}"
                )

            updated_point = stiefel_qr_retraction(point, aggregated_stream)
            model_copy.set_manifold_parameter_(name, updated_point)

        model_state_after_retraction = model_copy.state_dict()
        for spec in manifold_specs.values():
            for parameter_name in spec.parameter_names:
                new_state_dict[parameter_name] = (
                    model_state_after_retraction[parameter_name].detach().clone()
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

        orthogonality_errors: Dict[str, float] = {}
        for name in manifold_names:
            point = model_copy.get_manifold_parameter(name)
            point_work = point.float() if point.dtype in (torch.float16, torch.bfloat16) else point
            identity = torch.eye(
                point_work.shape[1],
                device=point_work.device,
                dtype=point_work.dtype,
            )
            orthogonality_errors[name] = float(
                torch.linalg.norm(
                    point_work.transpose(0, 1) @ point_work - identity,
                    ord="fro",
                ).item()
            )

        diagnostics = {
            "aggregation_rule": "client_uniform",
            "client_participation": "all_clients",
            "gradient_stream": "sum_of_negative_local_update_vectors",
            "local_steps": int(local_steps),
            "num_euclidean_expert_parameters": len(euclidean_names),
            "manifold_variables": {
                name: {
                    "manifold": spec.manifold,
                    "shape": list(spec.shape),
                    "parameter_names": list(spec.parameter_names),
                    "tangent_relative_residual": tangent_residuals[name],
                    "orthogonality_error": orthogonality_errors[name],
                }
                for name, spec in manifold_specs.items()
            },
            "vector_transport": "Stiefel tangent projection to round-start point",
            "server_retraction": "QR",
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

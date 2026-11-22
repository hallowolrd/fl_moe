"""Full-client RFedAGS with Euclidean expert-gradient streams.

All trainable expert parameters are treated as ordinary Euclidean coordinates.
Non-expert parameters continue to use base.py's client-uniform FedAvg path.
This implementation intentionally contains no manifold constraint or retraction.
"""

from __future__ import annotations

# Import base before importing torch so deterministic CUDA environment setup
# happens before PyTorch initializes CUDA/cuBLAS.
import base

import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

import torch


ALGORITHM_NAME = "rfedags"
PAYLOAD_KEY = "rfedags"
PAYLOAD_VERSION = 1

METHOD_CONFIG_DEFAULTS: Dict[str, Any] = {
    "optimizer": {
        "type": "sgd",
        "lr": 0.01,
        "momentum": 0.0,
        "nesterov": False,
        "weight_decay": 0.0,
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
    """Validate the Euclidean gradient-stream assumptions used by RFedAGS."""
    frac = float(_cfg_get(cfg, "frac", 1.0))
    if not math.isfinite(frac) or not math.isclose(
        frac, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "rfedags 当前只实现全客户端参与，要求 frac=1.0；"
            f"当前值：{frac}"
        )

    optimizer_type = str(_cfg_get(cfg, "optimizer.type", "sgd")).lower().strip()
    if optimizer_type != "sgd":
        raise ValueError("rfedags 当前要求 optimizer.type='sgd'。")

    lr = float(_cfg_get(cfg, "optimizer.lr", 0.01))
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError(f"optimizer.lr 必须是有限正数，当前值：{lr}")

    momentum = float(_cfg_get(cfg, "optimizer.momentum", 0.0))
    if not math.isfinite(momentum) or momentum != 0.0:
        raise ValueError("rfedags 当前要求 momentum=0。")

    weight_decay = float(_cfg_get(cfg, "optimizer.weight_decay", 0.0))
    if not math.isfinite(weight_decay) or weight_decay != 0.0:
        raise ValueError("rfedags 当前要求 weight_decay=0。")

    if bool(_cfg_get(cfg, "optimizer.nesterov", False)):
        raise ValueError("rfedags 当前不支持 Nesterov。")

    if base._get_grad_clip(cfg) is not None:
        raise ValueError("rfedags 当前不启用梯度裁剪。")

    legacy_rfedags_cfg = _cfg_get(cfg, "rfedags", {})
    if (
        isinstance(legacy_rfedags_cfg, Mapping)
        and legacy_rfedags_cfg.get("local_steps") is not None
    ):
        raise ValueError(
            "rfedags.local_steps 已迁移为通用顶层配置 local_steps；"
            "请将 YAML 中的 rfedags.local_steps 改为 local_steps，"
            "或使用命令行参数 --local-steps。"
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
            client_updates, weights, buffers_by_client, name, reference
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


class RFedAGSClientStrategy(base.ExpertClientTrainingStrategy):
    """Accumulate Euclidean expert update streams during local SGD."""

    def __init__(self, cfg: Any, client_id: int) -> None:
        self.cfg = cfg
        self.client_id = int(client_id)
        self.lr = float(_cfg_get(cfg, "optimizer.lr", 0.01))
        self._parameter_map: Dict[str, torch.nn.Parameter] = {}
        self._expert_names: list[str] = []
        self._expert_stream: Dict[str, torch.Tensor] = {}
        self._expert_buffer_names: list[str] = []
        self._local_steps = 0

    def on_train_start(self, context: base.ExpertClientTrainContext) -> None:
        model = context.model
        groups = context.optimization_groups
        self._parameter_map = dict(model.named_parameters())
        self._expert_names = list(groups.expert_parameter_names)
        if not self._expert_names:
            raise ValueError("rfedags 没有找到可训练的专家参数。")

        self._expert_stream = {}
        for name in self._expert_names:
            parameter = self._parameter_map[name]
            self._expert_stream[name] = torch.zeros(
                parameter.shape,
                device=parameter.device,
                dtype=_accumulation_dtype(parameter),
            )

        self._expert_buffer_names = sorted(
            set(groups.expert_state_names) - set(self._parameter_map)
        )
        self._local_steps = 0

    def step(self, context: base.ExpertClientBatchContext) -> None:
        model = context.train_context.model
        self._parameter_map = dict(model.named_parameters())
        with torch.no_grad():
            for name in self._expert_names:
                parameter = self._parameter_map[name]
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise NotImplementedError(
                        f"rfedags 当前不支持稀疏专家梯度：{name}"
                    )
                if not _tensor_is_finite(gradient):
                    raise ValueError(
                        f"专家参数 {name} 的梯度包含 NaN 或 Inf。"
                    )

                local_step = gradient.detach().mul(-self.lr)
                self._expert_stream[name].add_(
                    local_step.to(dtype=self._expert_stream[name].dtype)
                )
                parameter.add_(local_step)
        self._local_steps += 1

    def on_train_end(
        self,
        context: base.ExpertClientTrainContext,
    ) -> Mapping[str, Any]:
        if self._local_steps <= 0:
            raise ValueError("rfedags 客户端本地更新步数必须大于 0。")
        state_dict = context.model.state_dict()
        expert_buffers = {
            name: state_dict[name].detach().clone()
            for name in self._expert_buffer_names
        }
        return {
            PAYLOAD_KEY: {
                "version": PAYLOAD_VERSION,
                "local_steps": int(self._local_steps),
                "expert_stream": {
                    name: value.detach().clone()
                    for name, value in self._expert_stream.items()
                },
                "expert_buffers": expert_buffers,
            }
        }


class RFedAGSExpertAggregator(base.Aggregator):
    """Aggregate Euclidean expert update streams with client-uniform weights."""

    @property
    def method_name(self) -> str:
        return ALGORITHM_NAME

    @property
    def consumes_algorithm_payload(self) -> bool:
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
            raise RuntimeError("rfedags 聚合器尚未注入 MethodContext。")
        model = self.method_context.global_model
        groups = self.method_context.optimization_groups
        if model is None or groups is None:
            raise RuntimeError(
                "rfedags 聚合器需要 MethodContext.global_model 和 "
                "MethodContext.optimization_groups。"
            )

        configured_num_clients = int(_cfg_get(self.cfg, "num_clients", 0))
        expected_client_ids = set(range(configured_num_clients))
        actual_client_ids = {int(update.client_id) for update in client_updates}
        if actual_client_ids != expected_client_ids:
            raise ValueError(
                "rfedags 当前要求每轮全部客户端参与。"
                f" expected={sorted(expected_client_ids)}, "
                f"actual={sorted(actual_client_ids)}"
            )

        expert_state_names = set(groups.expert_state_names)
        if param_names is not None and set(param_names) != expert_state_names:
            raise ValueError("服务器传入的专家 state 名称与优化分组不一致。")

        named_parameters = dict(model.named_parameters())
        expert_parameter_names = list(groups.expert_parameter_names)
        if not expert_parameter_names:
            raise ValueError("rfedags 没有找到可训练的专家参数。")
        missing_parameters = set(expert_parameter_names) - set(named_parameters)
        if missing_parameters:
            raise ValueError(
                f"rfedags 专家参数不存在于模型中：{sorted(missing_parameters)}"
            )

        expert_buffer_names = sorted(
            expert_state_names - set(named_parameters)
        )
        weights = base.normalize_weights(self.compute_weights(client_updates))
        streams_by_client: Dict[int, Mapping[str, torch.Tensor]] = {}
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
            expert_stream = root.get("expert_stream")
            expert_buffers = root.get("expert_buffers")
            if not isinstance(expert_stream, Mapping):
                raise TypeError("expert_stream 必须是 Mapping。")
            if not isinstance(expert_buffers, Mapping):
                raise TypeError("expert_buffers 必须是 Mapping。")
            if set(expert_stream) != set(expert_parameter_names):
                raise ValueError(
                    f"客户端 {update.client_id} 的专家 stream 参数集合不完整。"
                )
            if set(expert_buffers) != set(expert_buffer_names):
                raise ValueError(
                    f"客户端 {update.client_id} 的专家 buffer 集合不完整。"
                )

            for name, value in expert_stream.items():
                if not torch.is_tensor(value):
                    raise TypeError(
                        f"客户端 {update.client_id} 的专家 stream {name} 必须是 Tensor。"
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

            streams_by_client[update.client_id] = expert_stream
            buffers_by_client[update.client_id] = expert_buffers

        unique_local_steps = set(local_steps_by_client.values())
        if len(unique_local_steps) != 1:
            raise ValueError(
                "rfedags 全客户端版本要求每个客户端执行相同的本地更新步数；"
                f"当前各客户端步数：{local_steps_by_client}。"
                "请设置统一的顶层 local_steps（命令行使用 --local-steps K），"
                "或确保各客户端 DataLoader 具有相同 batch 数。"
            )
        local_steps = unique_local_steps.pop()

        source_state = base_state if base_state is not None else global_state
        base.validate_state_dict_shapes(
            reference_state=global_state,
            candidate_state=source_state,
            context="RFedAGS 聚合基础 state_dict",
        )
        new_state_dict = {
            name: value.detach().clone()
            for name, value in source_state.items()
        }
        for name in expert_parameter_names:
            reference = global_state[name]
            aggregated_stream = _weighted_tensor_sum(
                client_updates, weights, streams_by_client, name, reference
            )
            new_state_dict[name] = reference + aggregated_stream

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
                "rfedags 必须返回完整模型 state_dict。"
                f" missing={missing[:10]}, extra={extra[:10]}"
            )
        base.validate_state_dict_shapes(
            reference_state=global_state,
            candidate_state=new_state_dict,
            context="RFedAGS 聚合结果",
        )
        base.check_finite_state_dict(new_state_dict)

        diagnostic_param_names = (
            list(param_names)
            if param_names is not None
            else sorted(expert_state_names)
        )
        diagnostics = self.build_diagnostics(
            client_updates=client_updates,
            weights=weights,
            param_names=diagnostic_param_names,
        )
        diagnostics.update(
            {
                "aggregation_rule": "client_uniform",
                "client_participation": "all_clients",
                "update_space": "euclidean",
                "gradient_stream": "sum_of_negative_local_update_vectors",
                "local_steps": int(local_steps),
                "num_expert_parameters": len(expert_parameter_names),
                "num_expert_buffers": len(expert_buffer_names),
            }
        )
        return base.AggregationResult(
            new_state_dict=new_state_dict,
            weights=weights,
            diagnostics=diagnostics,
        )


def build_expert_aggregator(cfg: Any) -> base.Aggregator:
    return RFedAGSExpertAggregator(cfg=cfg, param_group_name="expert")


def build_expert_client_training_strategy(
    cfg: Any,
    client_id: int,
) -> base.ExpertClientTrainingStrategy:
    return RFedAGSClientStrategy(cfg=cfg, client_id=client_id)


def main(model_builder: base.ModelBuilder | None = None) -> int:
    return base.main(
        expert_aggregator_builder=build_expert_aggregator,
        model_builder=model_builder,
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

"""Client-side ECAI 2024 OMoE optimization with uniform expert aggregation.

The client optimizer alternates regular (R) epochs and orthogonal (O) epochs.
During R epochs, every model parameter is updated normally and each expert's
linear-layer inputs update a client-local OWM projector. During O epochs, only
expert parameters are updated; each expert's optimizer-proposed linear update
is right-projected by the average OWM projector of the other experts.

The OWM state is private to each federated client and persists across rounds.
The server uses the ordinary client-uniform delta aggregation for both expert
and non-expert parameter groups. No changes to base.py are required.
"""

from __future__ import annotations

# Import base before torch so deterministic CUDA environment setup happens first.
import base

import argparse
import math
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn


ALGORITHM_NAME = "omoe_optimizer_uniform"
_EXPERT_KEY_PATTERN = re.compile(r"expert_(\d+)\Z")
ExpertLinearKey = Tuple[str, int, str]

METHOD_CONFIG_DEFAULTS = {
    "method_config": {
        # Perform one orthogonal epoch after this many cumulative local epochs.
        "skip_epochs": 5,
        # The ECAI 2024 paper specifies a decaying alpha schedule but does not
        # report numeric alpha0/lambda values in its main experimental setup.
        "alpha0": 1.0,
        "alpha_decay": 0.99,
    },
}

EMBEDDED_METHOD_CONFIG = {
    "agg": {
        "non_expert": {"method": "uniform"},
        "expert": {"method": ALGORITHM_NAME},
    },
}


def _read_nested_config(cfg: Any, path: str, default: Any = None) -> Any:
    """Read a dotted path from ConfigNode, a mapping, or a namespace."""
    current = cfg
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return default
    return current


def _read_mapping_value(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    if hasattr(mapping, "get"):
        try:
            return mapping.get(key, default)
        except TypeError:
            pass
    if hasattr(mapping, key):
        return getattr(mapping, key)
    return default


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是有限实数，当前值：{value!r}。")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限实数，当前值：{value!r}。")
    return result


def validate_method_config(config: Mapping[str, Any]) -> None:
    """Validate OMoE-specific settings from the resolved configuration."""
    if not isinstance(config, Mapping):
        raise ValueError("配置必须是 Mapping。")
    method_config = config.get("method_config", config)
    if not isinstance(method_config, Mapping):
        raise ValueError("method_config 必须是 Mapping。")

    skip_epochs = _read_mapping_value(
        method_config,
        "skip_epochs",
        METHOD_CONFIG_DEFAULTS["method_config"]["skip_epochs"],
    )
    if (
        isinstance(skip_epochs, bool)
        or not isinstance(skip_epochs, int)
        or skip_epochs <= 0
    ):
        raise ValueError("method_config.skip_epochs 必须是正整数。")

    alpha0 = _finite_real(
        _read_mapping_value(
            method_config,
            "alpha0",
            METHOD_CONFIG_DEFAULTS["method_config"]["alpha0"],
        ),
        "method_config.alpha0",
    )
    if alpha0 <= 0.0:
        raise ValueError("method_config.alpha0 必须大于 0。")

    alpha_decay = _finite_real(
        _read_mapping_value(
            method_config,
            "alpha_decay",
            METHOD_CONFIG_DEFAULTS["method_config"]["alpha_decay"],
        ),
        "method_config.alpha_decay",
    )
    if not 0.0 < alpha_decay <= 1.0:
        raise ValueError(
            "method_config.alpha_decay 必须满足 0 < alpha_decay <= 1。"
        )


def _validate_runtime_config(cfg: Any) -> Dict[str, Any]:
    """Return validated OMoE settings from the complete runtime config."""
    method_config = _read_nested_config(cfg, "method_config", {})
    if not isinstance(method_config, Mapping):
        raise ValueError("method_config 必须是 Mapping。")
    validate_method_config({"method_config": method_config})
    return {
        "skip_epochs": int(
            _read_mapping_value(
                method_config,
                "skip_epochs",
                METHOD_CONFIG_DEFAULTS["method_config"]["skip_epochs"],
            )
        ),
        "alpha0": float(
            _read_mapping_value(
                method_config,
                "alpha0",
                METHOD_CONFIG_DEFAULTS["method_config"]["alpha0"],
            )
        ),
        "alpha_decay": float(
            _read_mapping_value(
                method_config,
                "alpha_decay",
                METHOD_CONFIG_DEFAULTS["method_config"]["alpha_decay"],
            )
        ),
    }


def update_owm_projector(
    projector: torch.Tensor,
    input_vector: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Apply one recursive-least-squares OWM projector update.

    P <- P - (P x x^T P) / (alpha + x^T P x)

    The input is expected to be the mean input vector for one expert and one
    linear layer in the current mini-batch. The caller may append a constant
    one to the vector when the corresponding Linear layer has a trainable bias.
    """
    if not torch.is_tensor(projector) or projector.ndim != 2:
        raise TypeError("projector 必须是二维 Tensor。")
    if projector.shape[0] != projector.shape[1]:
        raise ValueError(
            f"projector 必须是方阵，当前 shape={tuple(projector.shape)}。"
        )
    if not projector.is_floating_point():
        raise TypeError("projector 必须是浮点 Tensor。")
    if not torch.is_tensor(input_vector) or input_vector.ndim != 1:
        raise TypeError("input_vector 必须是一维 Tensor。")
    if input_vector.shape[0] != projector.shape[0]:
        raise ValueError(
            "input_vector 与 projector 维度不匹配："
            f"input={input_vector.shape[0]}, projector={projector.shape[0]}。"
        )
    alpha_value = _finite_real(alpha, "alpha")
    if alpha_value <= 0.0:
        raise ValueError("alpha 必须大于 0。")

    with torch.no_grad():
        vector = input_vector.detach().to(
            device=projector.device,
            dtype=projector.dtype,
        )
        if not bool(torch.isfinite(vector).all().item()):
            raise ValueError("OWM 输入向量包含 NaN 或 Inf。")
        if not bool(torch.isfinite(projector).all().item()):
            raise ValueError("OWM projector 包含 NaN 或 Inf。")

        projected_input = torch.mv(projector, vector)
        denominator = (
            torch.dot(vector, projected_input)
            + torch.as_tensor(
                alpha_value,
                device=projector.device,
                dtype=projector.dtype,
            )
        )
        if not bool(torch.isfinite(denominator).item()):
            raise ValueError("OWM 更新分母包含 NaN 或 Inf。")
        if float(denominator.item()) <= 0.0:
            raise ValueError(
                "OWM 更新分母必须大于 0，"
                f"当前值：{float(denominator.item())}。"
            )

        projector.sub_(
            torch.outer(projected_input, projected_input) / denominator
        )
        # The exact RLS update preserves symmetry; explicitly symmetrize to
        # suppress small floating-point asymmetries accumulated over many steps.
        projector.copy_(0.5 * (projector + projector.transpose(0, 1)))

        if not bool(torch.isfinite(projector).all().item()):
            raise ValueError("OWM projector 更新后出现 NaN 或 Inf。")
    return projector


def average_other_projectors(
    projectors: Mapping[int, torch.Tensor],
    current_expert_id: int,
) -> torch.Tensor:
    """Average all other experts' projectors using the paper's Eq. (8) scale.

    The printed ECAI 2024 equation uses 1/M while excluding the current expert
    from the sum. This function preserves that literal 1/M scaling.
    """
    if not isinstance(projectors, Mapping):
        raise TypeError("projectors 必须是 Mapping。")
    if current_expert_id not in projectors:
        raise KeyError(f"projectors 中不存在专家 {current_expert_id}。")
    num_experts = len(projectors)
    if num_experts < 2:
        raise ValueError("OMoE 至少需要两个专家。")

    reference = projectors[current_expert_id]
    if not torch.is_tensor(reference) or reference.ndim != 2:
        raise TypeError("专家 projector 必须是二维 Tensor。")
    result = torch.zeros_like(reference)
    for expert_id, projector in projectors.items():
        if expert_id == current_expert_id:
            continue
        if not torch.is_tensor(projector) or projector.shape != reference.shape:
            raise ValueError(
                f"专家 {expert_id} 的 projector shape 与专家 "
                f"{current_expert_id} 不一致。"
            )
        if projector.device != reference.device:
            raise ValueError("所有专家 projector 必须位于同一设备。")
        if projector.dtype != reference.dtype:
            raise TypeError("所有专家 projector 必须具有相同 dtype。")
        result.add_(projector)

    result.div_(float(num_experts))
    return result


def project_linear_delta(
    linear_delta: torch.Tensor,
    projector: torch.Tensor,
) -> torch.Tensor:
    """Project a PyTorch Linear update on its input-feature axis.

    PyTorch stores Linear weights as [out_features, in_features], so the OWM
    input-space projector is applied on the right: delta_W @ P.
    """
    if not torch.is_tensor(linear_delta) or linear_delta.ndim != 2:
        raise TypeError("linear_delta 必须是二维 Tensor。")
    if not torch.is_tensor(projector) or projector.ndim != 2:
        raise TypeError("projector 必须是二维 Tensor。")
    if projector.shape[0] != projector.shape[1]:
        raise ValueError("projector 必须是方阵。")
    if linear_delta.shape[1] != projector.shape[0]:
        raise ValueError(
            "linear_delta 与 projector 维度不匹配："
            f"delta={tuple(linear_delta.shape)}, "
            f"projector={tuple(projector.shape)}。"
        )
    if linear_delta.device != projector.device:
        raise ValueError("linear_delta 与 projector 必须位于同一设备。")

    output_dtype = linear_delta.dtype
    work_dtype = (
        torch.float64
        if output_dtype == torch.float64
        else torch.float32
    )
    projected = (
        linear_delta.to(dtype=work_dtype)
        @ projector.to(dtype=work_dtype)
    )
    return projected.to(dtype=output_dtype)


class OMoEUniformExpertAggregator(base.Aggregator):
    """Client-uniform delta aggregator for the expert parameter group."""

    @property
    def method_name(self) -> str:
        return ALGORITHM_NAME

    def compute_weights(
        self,
        client_updates: Sequence[base.ClientUpdate],
    ) -> Dict[int, float]:
        return base.build_uniform_weights(client_updates)


class OMoEOptimizerClientStrategy(base.ExpertClientTrainingStrategy):
    """Client-local alternating OMoE optimizer for Switch Transformer experts."""

    def __init__(self, cfg: Any, client_id: int) -> None:
        self.cfg = cfg
        self.client_id = int(client_id)
        self.settings = _validate_runtime_config(cfg)

        # Projectors are private client state and intentionally survive the
        # per-round local-model deepcopy/reinitialization.
        self._projectors: Dict[ExpertLinearKey, torch.Tensor] = {}
        self._linear_modules: Dict[ExpertLinearKey, nn.Linear] = {}
        self._expert_ids_by_layer: Dict[str, Tuple[int, ...]] = {}
        self._hooks: list[Any] = []

        self._context: Optional[base.ExpertClientTrainContext] = None
        self._expert_optimizer: Optional[torch.optim.Optimizer] = None
        self._non_expert_parameters: list[nn.Parameter] = []

        self._batches_per_epoch: Optional[int] = None
        self._total_planned_steps = 1
        self._total_local_steps = 0
        self._round_step_count = 0
        self._current_phase = "R"

        self._round_r_steps = 0
        self._round_o_steps = 0
        self._round_projector_updates = 0
        self._round_projected_parameter_groups = 0

    @property
    def strategy_name(self) -> str:
        return ALGORITHM_NAME

    @property
    def includes_expert_parameters_in_model_delta(self) -> bool:
        # The expert optimizer updates the actual local model parameters.
        # They therefore travel through the ordinary model_delta path.
        return True

    def _resolve_expert_linears(
        self,
        model: nn.Module,
        optimization_groups: base.FederatedOptimizationGroups,
    ) -> None:
        """Resolve every expert Linear module and verify parameter coverage."""
        linear_modules: Dict[ExpertLinearKey, nn.Linear] = {}
        expert_ids_by_layer: Dict[str, Tuple[int, ...]] = {}
        found_layers = set()

        for moe_name, moe_module in model.named_modules():
            if not isinstance(moe_module, base.SwitchMoE):
                continue
            found_layers.add(moe_name)
            if not isinstance(moe_module.experts, nn.ModuleDict):
                raise TypeError(
                    f"MoE 层 {moe_name!r} 的 experts 必须是 ModuleDict。"
                )

            experts_by_id: Dict[int, Tuple[str, nn.Module, Dict[str, nn.Linear]]] = {}
            for expert_key, expert_module in moe_module.experts.items():
                match = _EXPERT_KEY_PATTERN.fullmatch(str(expert_key))
                if match is None:
                    raise ValueError(
                        f"无法解析专家名称 {expert_key!r}；"
                        "当前 OMoE 实现要求 expert_<id> 命名。"
                    )
                expert_id = int(match.group(1))
                expert_linears = {
                    relative_name: submodule
                    for relative_name, submodule in expert_module.named_modules()
                    if isinstance(submodule, nn.Linear)
                }
                if not expert_linears:
                    raise ValueError(
                        f"MoE 层 {moe_name!r} 的专家 {expert_key!r} "
                        "没有 nn.Linear 子模块。"
                    )
                experts_by_id[expert_id] = (
                    str(expert_key),
                    expert_module,
                    expert_linears,
                )

            expert_ids = tuple(sorted(experts_by_id))
            if expert_ids != tuple(range(len(expert_ids))):
                raise ValueError(
                    f"MoE 层 {moe_name!r} 的专家 ID 必须从 0 连续编号，"
                    f"当前为 {expert_ids}。"
                )
            if len(expert_ids) < 2:
                raise ValueError(
                    f"MoE 层 {moe_name!r} 至少需要两个专家。"
                )
            if int(moe_module.num_experts) != len(expert_ids):
                raise ValueError(
                    f"MoE 层 {moe_name!r} 的 num_experts 与 ModuleDict 数量不一致。"
                )

            reference_names = set(
                experts_by_id[expert_ids[0]][2].keys()
            )
            for expert_id in expert_ids:
                expert_key, _, expert_linears = experts_by_id[expert_id]
                if set(expert_linears) != reference_names:
                    raise ValueError(
                        f"MoE 层 {moe_name!r} 的专家线性层结构不一致："
                        f"expert_id={expert_id}。"
                    )

                for relative_name in sorted(reference_names):
                    linear = expert_linears[relative_name]
                    reference_linear = experts_by_id[expert_ids[0]][2][
                        relative_name
                    ]
                    current_signature = (
                        linear.in_features,
                        linear.out_features,
                        linear.bias is not None and linear.bias.requires_grad,
                    )
                    reference_signature = (
                        reference_linear.in_features,
                        reference_linear.out_features,
                        reference_linear.bias is not None
                        and reference_linear.bias.requires_grad,
                    )
                    if current_signature != reference_signature:
                        raise ValueError(
                            f"MoE 层 {moe_name!r} 的专家线性层 "
                            f"{relative_name!r} 形状不一致。"
                        )
                    linear_modules[(moe_name, expert_id, relative_name)] = linear

            expert_ids_by_layer[moe_name] = expert_ids

        if not linear_modules:
            raise ValueError(
                "当前模型没有找到 base.SwitchMoE；"
                "OMoE Optimizer 仅支持当前 Switch Transformer 专家结构。"
            )

        expected_layers = set(
            optimization_groups.expert_parameters_by_moe_layer.keys()
        )
        if expected_layers and found_layers != expected_layers:
            raise ValueError(
                "模型中的 SwitchMoE 层与联邦专家参数分组不一致："
                f"model={sorted(found_layers)}, "
                f"groups={sorted(expected_layers)}。"
            )

        named_parameters = dict(model.named_parameters())
        covered_parameter_names = set()
        for (moe_name, expert_id, relative_name), linear in linear_modules.items():
            expert_key = f"expert_{expert_id}"
            module_prefix = (
                f"{moe_name}.experts.{expert_key}"
                if moe_name
                else f"experts.{expert_key}"
            )
            linear_prefix = (
                f"{module_prefix}.{relative_name}"
                if relative_name
                else module_prefix
            )
            if linear.weight.requires_grad:
                covered_parameter_names.add(f"{linear_prefix}.weight")
            if linear.bias is not None and linear.bias.requires_grad:
                covered_parameter_names.add(f"{linear_prefix}.bias")

        expected_parameter_names = set(
            optimization_groups.expert_parameter_names
        )
        missing = sorted(expected_parameter_names - covered_parameter_names)
        extra = sorted(covered_parameter_names - expected_parameter_names)
        if missing or extra:
            raise ValueError(
                "OMoE 当前仅支持由 nn.Linear weight/bias 构成的专家参数；"
                f"missing={missing[:10]}, extra={extra[:10]}。"
            )

        # Keep a reference to the current round's local model modules. The
        # projector tensors themselves remain client-owned across rounds.
        self._linear_modules = linear_modules
        self._expert_ids_by_layer = expert_ids_by_layer

    def _ensure_projectors(self, device: torch.device) -> None:
        """Create identity projectors once, then reuse them across FL rounds."""
        current_keys = set(self._linear_modules)
        existing_keys = set(self._projectors)
        if existing_keys and existing_keys != current_keys:
            raise ValueError(
                "客户端跨轮次的专家线性层布局发生变化："
                f"existing={len(existing_keys)}, current={len(current_keys)}。"
            )

        for key, linear in self._linear_modules.items():
            trainable_bias = (
                linear.bias is not None and linear.bias.requires_grad
            )
            projector_dim = linear.in_features + int(trainable_bias)
            projector = self._projectors.get(key)
            if projector is None:
                self._projectors[key] = torch.eye(
                    projector_dim,
                    device=device,
                    dtype=torch.float32,
                )
                continue
            if tuple(projector.shape) != (projector_dim, projector_dim):
                raise ValueError(
                    f"专家线性层 {key} 的 OWM projector shape 不匹配："
                    f"projector={tuple(projector.shape)}, "
                    f"expected={(projector_dim, projector_dim)}。"
                )
            if projector.device != device:
                self._projectors[key] = projector.to(device=device)

    def _phase_for_step(self, total_step: int) -> str:
        if self._batches_per_epoch is None:
            raise RuntimeError("尚未初始化每个 Epoch 的 Batch 数。")
        epoch_index = int(total_step) // self._batches_per_epoch
        epoch_number = epoch_index + 1
        if epoch_number % self.settings["skip_epochs"] == 0:
            return "O"
        return "R"

    def _alpha_for_current_step(self) -> float:
        progress = min(
            float(self._total_local_steps) / float(self._total_planned_steps),
            1.0,
        )
        return (
            self.settings["alpha0"]
            * self.settings["alpha_decay"] ** progress
        )

    def _register_input_hooks(self) -> None:
        """Update each expert/linear projector from its actual routed inputs."""
        self._remove_input_hooks()

        for key, linear in self._linear_modules.items():
            def capture_input(
                module: nn.Module,
                inputs: Tuple[Any, ...],
                *,
                projector_key: ExpertLinearKey = key,
            ) -> None:
                if self._current_phase != "R":
                    return
                if not inputs or not torch.is_tensor(inputs[0]):
                    raise TypeError(
                        f"专家线性层 {projector_key} 的输入必须是 Tensor。"
                    )

                layer_input = inputs[0].detach()
                current_linear = self._linear_modules[projector_key]
                if layer_input.shape[-1] != current_linear.in_features:
                    raise ValueError(
                        f"专家线性层 {projector_key} 输入维度不匹配："
                        f"input={tuple(layer_input.shape)}, "
                        f"in_features={current_linear.in_features}。"
                    )
                flattened = layer_input.reshape(
                    -1,
                    current_linear.in_features,
                )
                if flattened.shape[0] == 0:
                    return

                mean_input = flattened.to(dtype=torch.float32).mean(dim=0)
                trainable_bias = (
                    current_linear.bias is not None
                    and current_linear.bias.requires_grad
                )
                if trainable_bias:
                    mean_input = torch.cat(
                        (
                            mean_input,
                            mean_input.new_ones((1,)),
                        ),
                        dim=0,
                    )

                update_owm_projector(
                    projector=self._projectors[projector_key],
                    input_vector=mean_input,
                    alpha=self._alpha_for_current_step(),
                )
                self._round_projector_updates += 1

            self._hooks.append(
                linear.register_forward_pre_hook(capture_input)
            )

    def _remove_input_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def _average_other_expert_projectors(
        self,
        key: ExpertLinearKey,
    ) -> torch.Tensor:
        moe_name, expert_id, relative_name = key
        expert_ids = self._expert_ids_by_layer[moe_name]
        projectors = {
            other_id: self._projectors[
                (moe_name, other_id, relative_name)
            ]
            for other_id in expert_ids
        }
        return average_other_projectors(
            projectors=projectors,
            current_expert_id=expert_id,
        )

    def _suppress_non_expert_updates(self) -> None:
        """The base loop still calls optimizer.step(); None grads make it a no-op."""
        if self._context is None:
            raise RuntimeError("当前没有活动的客户端训练上下文。")
        parameter_map = dict(self._context.model.named_parameters())
        for name in self._context.optimization_groups.non_expert_parameters:
            parameter_map[name].grad = None

    def _orthogonal_optimizer_step(self) -> None:
        """Project the actual optimizer-proposed expert update onto other P spaces."""
        if self._expert_optimizer is None:
            raise RuntimeError("专家优化器尚未初始化。")

        snapshots: Dict[
            ExpertLinearKey,
            Tuple[torch.Tensor, Optional[torch.Tensor]],
        ] = {}
        for key, linear in self._linear_modules.items():
            weight_has_gradient = linear.weight.grad is not None
            bias_has_gradient = (
                linear.bias is not None
                and linear.bias.requires_grad
                and linear.bias.grad is not None
            )
            if not weight_has_gradient and not bias_has_gradient:
                continue

            before_weight = linear.weight.detach().clone()
            before_bias: Optional[torch.Tensor] = None
            if linear.bias is not None and linear.bias.requires_grad:
                before_bias = linear.bias.detach().clone()
            snapshots[key] = (before_weight, before_bias)

        if not snapshots:
            return

        # A single optimizer instance is used for both R and O updates. Thus
        # SGD momentum / Adam moments and optimizer hyperparameters are shared.
        # Projecting the proposed parameter delta (rather than only the raw
        # gradient) also projects momentum and weight-decay contributions.
        self._expert_optimizer.step()

        with torch.no_grad():
            for key, (before_weight, before_bias) in snapshots.items():
                linear = self._linear_modules[key]
                weight_delta = linear.weight.detach() - before_weight

                trainable_bias = (
                    linear.bias is not None and linear.bias.requires_grad
                )
                if trainable_bias:
                    if before_bias is None:
                        raise RuntimeError(
                            f"专家线性层 {key} 缺少 bias 参数快照。"
                        )
                    bias_delta = linear.bias.detach() - before_bias
                    linear_delta = torch.cat(
                        (weight_delta, bias_delta.unsqueeze(1)),
                        dim=1,
                    )
                else:
                    linear_delta = weight_delta

                other_projector = self._average_other_expert_projectors(key)
                projected_delta = project_linear_delta(
                    linear_delta=linear_delta,
                    projector=other_projector,
                )
                linear.weight.copy_(
                    before_weight + projected_delta[:, : linear.in_features]
                )
                if trainable_bias:
                    linear.bias.copy_(
                        before_bias + projected_delta[:, -1]
                    )
                self._round_projected_parameter_groups += 1

    def on_train_start(
        self,
        context: base.ExpertClientTrainContext,
    ) -> None:
        """Bind the current round's local model and retain client OWM state."""
        if int(context.client_id) != self.client_id:
            raise ValueError(
                f"策略 client_id={self.client_id} 与训练上下文 "
                f"client_id={context.client_id} 不一致。"
            )
        batches_per_epoch = len(context.train_loader)
        if batches_per_epoch <= 0:
            raise ValueError("客户端 DataLoader 必须至少包含一个 Batch。")
        if self._batches_per_epoch is None:
            self._batches_per_epoch = int(batches_per_epoch)
        elif self._batches_per_epoch != int(batches_per_epoch):
            raise ValueError(
                "客户端跨轮次 DataLoader 长度发生变化，无法继续累计 OMoE Epoch："
                f"previous={self._batches_per_epoch}, "
                f"current={batches_per_epoch}。"
            )

        self._remove_input_hooks()
        self._resolve_expert_linears(
            model=context.model,
            optimization_groups=context.optimization_groups,
        )
        self._ensure_projectors(device=context.device)

        expert_parameter_names = (
            context.optimization_groups.expert_parameter_names
        )
        if not expert_parameter_names:
            raise ValueError("当前模型没有可训练专家参数。")
        self._expert_optimizer = base.build_optimizer(
            model=context.model,
            cfg=context.cfg,
            param_names=expert_parameter_names,
        )
        parameter_map = dict(context.model.named_parameters())
        self._non_expert_parameters = [
            parameter_map[name]
            for name in context.optimization_groups.non_expert_parameters
        ]

        rounds = int(_read_nested_config(context.cfg, "rounds", 1))
        num_clients = int(
            _read_nested_config(context.cfg, "num_clients", 1)
        )
        frac = float(_read_nested_config(context.cfg, "frac", 1.0))
        selected_per_round = max(1, int(num_clients * frac))
        expected_participation = min(
            float(selected_per_round) / float(max(num_clients, 1)),
            1.0,
        )
        expected_rounds = max(
            1.0,
            float(rounds) * expected_participation,
        )
        if context.local_steps is not None:
            planned_steps_per_round = int(context.local_steps)
        else:
            planned_steps_per_round = (
                int(context.local_epochs) * int(batches_per_epoch)
            )
        self._total_planned_steps = max(
            1,
            int(math.ceil(expected_rounds * planned_steps_per_round)),
        )

        self._context = context
        self._round_step_count = 0
        self._round_r_steps = 0
        self._round_o_steps = 0
        self._round_projector_updates = 0
        self._round_projected_parameter_groups = 0
        self._current_phase = self._phase_for_step(
            self._total_local_steps
        )
        self._register_input_hooks()

    def step(self, context: base.ExpertClientBatchContext) -> None:
        """Execute one R or O step after the framework has backpropagated."""
        if self._context is None or context.train_context is not self._context:
            raise ValueError("BatchContext 不属于当前客户端训练上下文。")
        expected_local_step = self._round_step_count + 1
        if int(context.local_step) != expected_local_step:
            raise ValueError(
                "客户端本地 Batch 步数不连续："
                f"expected={expected_local_step}, actual={context.local_step}。"
            )

        if self._current_phase == "R":
            if self._expert_optimizer is None:
                raise RuntimeError("专家优化器尚未初始化。")
            self._expert_optimizer.step()
            self._round_r_steps += 1
        else:
            self._orthogonal_optimizer_step()
            self._suppress_non_expert_updates()
            self._round_o_steps += 1

        self._round_step_count += 1
        self._total_local_steps += 1
        self._current_phase = self._phase_for_step(
            self._total_local_steps
        )

    def on_train_end(
        self,
        context: base.ExpertClientTrainContext,
    ) -> Mapping[str, Any]:
        """Remove round hooks and return compact client-side OMoE diagnostics."""
        if context is not self._context:
            raise ValueError("on_train_end() 收到的训练上下文不匹配。")
        if self._batches_per_epoch is None:
            raise RuntimeError("客户端 Epoch 长度尚未初始化。")

        expected_steps = (
            int(context.local_steps)
            if context.local_steps is not None
            else int(context.local_epochs) * self._batches_per_epoch
        )
        if self._round_step_count != expected_steps:
            raise ValueError(
                "客户端实际更新次数与本地训练配置不一致："
                f"actual={self._round_step_count}, expected={expected_steps}。"
            )

        self._remove_input_hooks()
        projector_trace_ratios = [
            float(
                (
                    torch.diagonal(projector).mean()
                    .detach()
                    .cpu()
                    .item()
                )
            )
            for projector in self._projectors.values()
        ]
        diagnostics = {
            "client_id": int(self.client_id),
            "round_id": int(context.round_id),
            "r_steps": int(self._round_r_steps),
            "o_steps": int(self._round_o_steps),
            "total_local_steps": int(self._total_local_steps),
            "projector_updates": int(self._round_projector_updates),
            "projected_parameter_groups": int(
                self._round_projected_parameter_groups
            ),
            "num_projectors": int(len(self._projectors)),
            "mean_projector_diagonal": (
                sum(projector_trace_ratios) / len(projector_trace_ratios)
                if projector_trace_ratios
                else 0.0
            ),
            "skip_epochs": int(self.settings["skip_epochs"]),
            "alpha0": float(self.settings["alpha0"]),
            "alpha_decay": float(self.settings["alpha_decay"]),
        }

        self._context = None
        self._expert_optimizer = None
        self._non_expert_parameters = []
        self._linear_modules = {}
        return {"omoe_optimizer": diagnostics}


def build_expert_aggregator(cfg: Any) -> base.Aggregator:
    """Build the expert-only client-uniform aggregator."""
    return OMoEUniformExpertAggregator(
        cfg=cfg,
        param_group_name="expert",
    )


def build_expert_client_training_strategy(
    cfg: Any,
    client_id: int,
) -> base.ExpertClientTrainingStrategy:
    """Build one persistent OMoE client strategy per federated client."""
    return OMoEOptimizerClientStrategy(cfg=cfg, client_id=client_id)


def build_method_client_diagnostics(
    client_update: base.ClientUpdate,
) -> Dict[str, Any]:
    """Expose compact OMoE diagnostics in the per-client round summary."""
    payload = client_update.algorithm_payload
    if not isinstance(payload, Mapping):
        return {}
    diagnostics = payload.get("omoe_optimizer", {})
    if not isinstance(diagnostics, Mapping):
        return {}
    return {"omoe_optimizer": dict(diagnostics)}


def register_method_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Register OMoE-specific command-line hyperparameters."""
    parser.add_argument("--omoe-skip-epochs", type=int, default=None)
    parser.add_argument("--omoe-alpha0", type=float, default=None)
    parser.add_argument("--omoe-alpha-decay", type=float, default=None)


def build_method_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert OMoE command-line options into method_config overrides."""
    method_overrides: Dict[str, Any] = {}
    if args.omoe_skip_epochs is not None:
        method_overrides["skip_epochs"] = args.omoe_skip_epochs
    if args.omoe_alpha0 is not None:
        method_overrides["alpha0"] = args.omoe_alpha0
    if args.omoe_alpha_decay is not None:
        method_overrides["alpha_decay"] = args.omoe_alpha_decay
    return {"method_config": method_overrides} if method_overrides else {}


def main(model_builder: Optional[base.ModelBuilder] = None) -> int:
    """Run OMoE client optimization with server-side uniform aggregation."""
    return base.main(
        expert_aggregator_builder=build_expert_aggregator,
        embedded_method_config=EMBEDDED_METHOD_CONFIG,
        expert_method_name=ALGORITHM_NAME,
        method_config_defaults=METHOD_CONFIG_DEFAULTS,
        method_config_validator=validate_method_config,
        method_cli_argument_registrar=register_method_cli_arguments,
        method_cli_overrides_builder=build_method_cli_overrides,
        method_client_diagnostics_builder=build_method_client_diagnostics,
        expert_client_training_strategy_builder=(
            build_expert_client_training_strategy
        ),
        model_builder=model_builder,
    )


if __name__ == "__main__":
    raise SystemExit(main())

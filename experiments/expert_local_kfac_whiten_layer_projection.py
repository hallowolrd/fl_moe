from __future__ import annotations

"""
Local-KFAC Whitened Layer Projection for Federated Sparse-MoE.

This is the refactored method-only version of the original monolithic
experiment. The common model, datasets, Dirichlet partition, reproducibility,
client training, evaluation, logging and federated round loop are provided by
base.py.

Method-specific behavior kept here:
1. After standard local CE training, run one extra deterministic/no-augmentation,
   FP32 forward/backward pass with standard sample-mean CE to collect per-expert,
   per-Linear KFAC factors without updating parameters.
2. Convert expert deltas to pseudo-gradients D = -Delta / eta.
3. Whiten each client/layer pseudo-gradient with that client's damped local
   KFAC factors.
4. Build a self-included route-count-weighted reference from valid KFAC
   clients only.
5. If the whitened gradient conflicts with the reference (negative inner
   product), remove exactly its negative reference component.
6. Map the corrected whitened gradient back with the same local KFAC factors.
7. Aggregate all training-active clients by training-stage route counts;
   invalid KFAC clients keep their original pseudo-gradient.
8. Shared/non-expert parameters remain uniformly averaged across valid
   clients via base.aggregate_shared_uniform().

Run from the project root with the current shared base saved as
experiments/base.py:
    python experiments/expert_local_kfac_whiten_layer_projection.py
"""

import argparse
import copy
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Mapping

import base

# Import torch only after base has configured CUBLAS_WORKSPACE_CONFIG.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset


ALGORITHM_NAME = "expert_local_kfac_whiten_layer_projection"
StateDict = base.StateDict
ClientUpdate = base.ClientUpdate


# =============================================================================
# Method-specific configuration
# =============================================================================

@dataclass(frozen=True)
class KFACSettings:
    # None means reuse client_batch_size.
    fisher_batch_size: int | None = None
    minimum_kfac_samples: int = 8
    relative_damping: float = 1e-2
    max_whitening_gain: float = 5.0
    factor_scale_epsilon: float = 1e-12
    projection_epsilon: float = 1e-12
    reference_norm_warning_threshold: float = 1e-20
    # "training" reuses the local-training device. Eigendecomposition stays CPU FP64.
    kfac_server_device: str = "training"


def validate_method_config(config: base.ExperimentConfig) -> None:
    # The Local-KFAC projection itself is valid for any top_k accepted by base.py.
    # In the current shared base this includes both top_k=1 and top_k=2 when
    # num_experts=4, so no additional method-specific restriction is required.
    del config


def validate_kfac_settings(settings: KFACSettings) -> None:
    if settings.fisher_batch_size is not None and settings.fisher_batch_size <= 0:
        raise ValueError("fisher_batch_size must be positive or omitted.")
    if settings.minimum_kfac_samples <= 0:
        raise ValueError("minimum_kfac_samples must be greater than 0.")
    if settings.relative_damping <= 0.0:
        raise ValueError("relative_damping must be greater than 0.")
    if settings.max_whitening_gain <= 0.0:
        raise ValueError("max_whitening_gain must be greater than 0.")
    if settings.factor_scale_epsilon <= 0.0:
        raise ValueError("factor_scale_epsilon must be greater than 0.")
    if settings.projection_epsilon <= 0.0:
        raise ValueError("projection_epsilon must be greater than 0.")
    if settings.reference_norm_warning_threshold < 0.0:
        raise ValueError("reference_norm_warning_threshold must be non-negative.")
    if not settings.kfac_server_device.strip():
        raise ValueError("kfac_server_device must not be empty.")


def _parse_method_settings_from_argv() -> tuple[KFACSettings, list[str]]:
    """
    Parse only KFAC-specific options and return the remaining argv for base.py.

    This deliberately avoids changing base.py merely to add method-specific
    hyperparameters. All common CLI options continue to be parsed by base.py.
    """
    parser = argparse.ArgumentParser(add_help=False)

    def add(*names: str, **kwargs) -> None:
        kwargs.setdefault("default", argparse.SUPPRESS)
        parser.add_argument(*names, **kwargs)

    add("--fisher-batch-size", type=int)
    add("--minimum-kfac-samples", type=int)
    add("--relative-damping", type=float)
    add("--max-whitening-gain", type=float)
    add("--factor-scale-epsilon", type=float)
    add("--projection-epsilon", type=float)
    add("--reference-norm-warning-threshold", type=float)
    add(
        "--kfac-server-device",
        type=str,
        help='"training", cpu, cuda, cuda:0, etc.; eigendecomposition remains CPU FP64.',
    )

    namespace, remaining = parser.parse_known_args(sys.argv[1:])
    settings = replace(KFACSettings(), **vars(namespace))
    validate_kfac_settings(settings)
    return settings, remaining


def parse_configs() -> tuple[base.ExperimentConfig, KFACSettings]:
    settings, remaining = _parse_method_settings_from_argv()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        config = base.parse_config(
            description=(
                "Standard local CE with post-training local KFAC collection and "
                "layer-wise whitened conflict projection."
            ),
            method_validator=validate_method_config,
        )
    finally:
        sys.argv = original_argv
    return config, settings


def resolve_kfac_server_device(
    value: str,
    training_device: torch.device,
) -> torch.device:
    if value.strip().lower() == "training":
        return training_device
    return base.resolve_device(value)


# =============================================================================
# KFAC data structures
# =============================================================================

@dataclass
class KFACLayerFactors:
    activation: Tensor
    output_gradient: Tensor
    count: int


@dataclass
class FactorTransform:
    inverse_sqrt: Tensor
    sqrt: Tensor
    scale: float
    raw_eigen_min: float
    raw_eigen_max: float
    normalized_eigen_min: float
    normalized_eigen_max: float
    effective_eigen_min: float
    effective_eigen_max: float
    effective_condition: float
    max_inverse_sqrt_gain: float
    gain_cap_triggered: bool


@dataclass
class LocalLayerGeometry:
    client_id: int
    route_count: int
    final_weight: float
    valid_reference_weight: float
    gradient: Tensor
    whitened: Tensor
    activation_transform: FactorTransform
    output_transform: FactorTransform
    sample_count: int
    roundtrip_relative_error: float


@dataclass
class ExpertAggregationResult:
    participant_counts: list[int]
    client_weights_by_expert: list[dict[int, float]]
    diagnostics: list[dict]


# =============================================================================
# Post-local-training, no-augmentation KFAC collection
# =============================================================================

def build_fisher_dataset(
    train_dataset: Dataset,
    config: base.ExperimentConfig,
) -> Dataset:
    """
    Make an index-identical shallow copy of the training dataset but replace
    its random training transform with the deterministic evaluation transform.

    base.py owns the dataset-specific preprocessing, so this works for all
    datasets currently supported by the shared base rather than hard-coding
    CIFAR-10 normalization here.
    """
    if not hasattr(train_dataset, "transform"):
        raise TypeError(
            f"Dataset {type(train_dataset).__name__} does not expose a transform "
            "attribute required for the no-augmentation KFAC view."
        )

    # The current shared base exposes the dataset transform factory. The second
    # transform is deterministic and therefore suitable for the KFAC pass.
    _, deterministic_transform = base._dataset_transforms(config.dataset_name)
    fisher_dataset = copy.copy(train_dataset)
    setattr(fisher_dataset, "transform", deterministic_transform)

    if len(fisher_dataset) != len(train_dataset):
        raise RuntimeError("KFAC dataset copy changed dataset length.")
    return fisher_dataset


def make_fisher_loader(
    *,
    config: base.ExperimentConfig,
    settings: KFACSettings,
    fisher_dataset: Dataset,
    indices: list[int],
) -> DataLoader:
    batch_size = (
        config.client_batch_size
        if settings.fisher_batch_size is None
        else settings.fisher_batch_size
    )
    return DataLoader(
        Subset(fisher_dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )


class ExpertKFACCollector:
    """Collect augmented-input A and output-gradient S per expert Linear layer."""

    def __init__(self, model: nn.Module, num_experts: int) -> None:
        if not hasattr(model, "experts"):
            raise TypeError("Model must expose an `experts` attribute.")
        experts = getattr(model, "experts")
        if len(experts) != num_experts:
            raise RuntimeError(
                f"Expected {num_experts} experts, found {len(experts)}."
            )

        self.num_experts = int(num_experts)
        self._current_batch_size: int | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._batch_size_stacks: dict[tuple[int, str], list[int]] = {}
        self._activation_sums: dict[tuple[int, str], Tensor] = {}
        self._output_gradient_sums: dict[tuple[int, str], Tensor] = {}
        self._counts: dict[tuple[int, str], int] = {}
        self._layer_shapes: dict[tuple[int, str], tuple[int, int]] = {}

        for expert_idx, expert in enumerate(experts):
            linear_count = 0
            for layer_name, module in expert.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                if not layer_name:
                    raise RuntimeError(
                        "Expert Linear layer must have a non-empty relative name."
                    )
                linear_count += 1
                key = (expert_idx, layer_name)
                device = module.weight.device
                in_augmented = int(module.in_features) + 1
                out_features = int(module.out_features)
                self._layer_shapes[key] = (out_features, in_augmented)
                self._batch_size_stacks[key] = []
                self._counts[key] = 0
                self._activation_sums[key] = torch.zeros(
                    (in_augmented, in_augmented),
                    dtype=torch.float32,
                    device=device,
                )
                self._output_gradient_sums[key] = torch.zeros(
                    (out_features, out_features),
                    dtype=torch.float32,
                    device=device,
                )
                self._handles.append(
                    module.register_forward_pre_hook(
                        self._make_forward_pre_hook(expert_idx, layer_name)
                    )
                )
                self._handles.append(
                    module.register_full_backward_hook(
                        self._make_backward_hook(expert_idx, layer_name)
                    )
                )
            if linear_count == 0:
                raise RuntimeError(f"Expert {expert_idx} contains no Linear layer.")

    def set_batch_size(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("KFAC batch size must be positive.")
        self._current_batch_size = int(batch_size)

    def _make_forward_pre_hook(self, expert_idx: int, layer_name: str):
        key = (expert_idx, layer_name)

        def hook(module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            del module
            if self._current_batch_size is None:
                raise RuntimeError("Call set_batch_size before KFAC forward.")
            if not inputs or not isinstance(inputs[0], Tensor):
                raise RuntimeError(
                    f"Missing Linear input for expert={expert_idx}, layer={layer_name}."
                )
            activation = inputs[0].detach()
            if activation.ndim != 2:
                raise RuntimeError(
                    "KFAC expects Linear input [N, D], got "
                    f"{tuple(activation.shape)}."
                )
            activation = activation.float()
            sample_count = int(activation.shape[0])
            if sample_count <= 0:
                raise RuntimeError("An invoked expert layer received zero samples.")
            ones = torch.ones(
                (sample_count, 1),
                dtype=activation.dtype,
                device=activation.device,
            )
            augmented = torch.cat([activation, ones], dim=1)
            self._activation_sums[key].addmm_(
                augmented.transpose(0, 1), augmented
            )
            self._counts[key] += sample_count
            self._batch_size_stacks[key].append(self._current_batch_size)

        return hook

    def _make_backward_hook(self, expert_idx: int, layer_name: str):
        key = (expert_idx, layer_name)

        def hook(
            module: nn.Module,
            grad_input: tuple[Tensor | None, ...],
            grad_output: tuple[Tensor | None, ...],
        ) -> None:
            del module, grad_input
            stack = self._batch_size_stacks[key]
            if not stack:
                raise RuntimeError(
                    f"Missing forward record for expert={expert_idx}, layer={layer_name}."
                )
            original_batch_size = stack.pop()
            if not grad_output or grad_output[0] is None:
                raise RuntimeError(
                    f"Missing grad_output for expert={expert_idx}, layer={layer_name}."
                )
            output_gradient = grad_output[0].detach()
            if output_gradient.ndim != 2:
                raise RuntimeError(
                    "KFAC expects Linear grad_output [N, O], got "
                    f"{tuple(output_gradient.shape)}."
                )
            # Standard cross-entropy uses a full-batch mean. Multiplying by the
            # original batch size removes only that 1/B reduction so the
            # collected output-gradient second moment is on a per-sample scale.
            # Router/expert Jacobian factors from the actual model are preserved.
            output_gradient = output_gradient.float() * float(original_batch_size)
            self._output_gradient_sums[key].addmm_(
                output_gradient.transpose(0, 1), output_gradient
            )

        return hook

    def layer_names(self, expert_idx: int) -> list[str]:
        names = [
            layer_name
            for current_expert, layer_name in self._layer_shapes
            if current_expert == expert_idx
        ]
        names.sort()
        return names

    def layer_count(self, expert_idx: int, layer_name: str) -> int:
        return int(self._counts[(expert_idx, layer_name)])

    def mean_factors(self) -> list[dict[str, KFACLayerFactors]]:
        result: list[dict[str, KFACLayerFactors]] = []
        for expert_idx in range(self.num_experts):
            expert_result: dict[str, KFACLayerFactors] = {}
            for layer_name in self.layer_names(expert_idx):
                key = (expert_idx, layer_name)
                count = int(self._counts[key])
                if count > 0:
                    activation = (
                        self._activation_sums[key] / float(count)
                    ).detach().cpu()
                    output_gradient = (
                        self._output_gradient_sums[key] / float(count)
                    ).detach().cpu()
                else:
                    activation = torch.zeros_like(
                        self._activation_sums[key], device="cpu"
                    )
                    output_gradient = torch.zeros_like(
                        self._output_gradient_sums[key], device="cpu"
                    )
                expert_result[layer_name] = KFACLayerFactors(
                    activation=activation,
                    output_gradient=output_gradient,
                    count=count,
                )
            result.append(expert_result)
        return result

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        unmatched = {
            key: len(stack)
            for key, stack in self._batch_size_stacks.items()
            if stack
        }
        if unmatched:
            raise RuntimeError(
                f"Unmatched KFAC forward/backward hook stacks: {unmatched}"
            )


def estimate_expert_kfac_factors(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_experts: int,
) -> tuple[list[dict[str, KFACLayerFactors]], Tensor]:
    """Freeze the locally trained model and collect one FP32 KFAC pass."""
    was_training = model.training
    model.to(device)
    model.eval()
    route_counts = torch.zeros(num_experts, dtype=torch.long, device=device)
    collector = ExpertKFACCollector(model, num_experts)
    try:
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = int(targets.size(0))
            model.zero_grad(set_to_none=True)
            collector.set_batch_size(batch_size)
            with torch.autocast(device_type=device.type, enabled=False):
                output = model(images.float())
                classification_loss = F.cross_entropy(
                    output.logits.float(),
                    targets,
                )
            classification_loss.backward()
            route_counts.add_(
                output.route_counts.detach().to(device=device, dtype=torch.long)
            )

        factors = collector.mean_factors()
        route_counts_cpu = route_counts.cpu()
        for expert_idx in range(num_experts):
            expected = int(route_counts_cpu[expert_idx])
            for layer_name in collector.layer_names(expert_idx):
                observed = collector.layer_count(expert_idx, layer_name)
                if observed != expected:
                    raise RuntimeError(
                        "KFAC hook count does not match model route count: "
                        f"expert={expert_idx}, layer={layer_name}, "
                        f"hook_count={observed}, route_count={expected}."
                    )
    finally:
        collector.close()
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    return factors, route_counts_cpu


def _states_exactly_equal(left: Tensor, right: Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if torch.is_floating_point(left):
        return bool(torch.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True))
    return bool(torch.equal(left, right))


def collect_post_training_kfac_statistics(
    *,
    settings: KFACSettings,
    local_model: nn.Module,
    global_model: nn.Module,
    loader: DataLoader,
    train_dataset: Dataset,
    client_indices: list[int],
    config: base.ExperimentConfig,
    client_id: int,
    round_idx: int,
    device: torch.device,
    method_state: object | None,
) -> Mapping[str, object]:
    del global_model, loader, client_id, round_idx, method_state

    # base.train_client has already frozen shared_delta/expert_deltas before
    # invoking this hook. The state check below additionally verifies that the
    # statistics pass itself is parameter-state preserving.
    shared_before = local_model.get_shared_state_dict(to_cpu=True)
    experts_before = local_model.get_all_expert_state_dicts(to_cpu=True)

    fisher_dataset = build_fisher_dataset(train_dataset, config)
    fisher_loader = make_fisher_loader(
        config=config,
        settings=settings,
        fisher_dataset=fisher_dataset,
        indices=client_indices,
    )
    kfac_factors, fisher_route_counts = estimate_expert_kfac_factors(
        model=local_model,
        loader=fisher_loader,
        device=device,
        num_experts=config.num_experts,
    )

    shared_after = local_model.get_shared_state_dict(to_cpu=True)
    experts_after = local_model.get_all_expert_state_dicts(to_cpu=True)
    for key, before in shared_before.items():
        if not _states_exactly_equal(before, shared_after[key]):
            raise RuntimeError(
                f"KFAC pass unexpectedly changed shared state {key!r}."
            )
    for expert_idx in range(config.num_experts):
        for key, before in experts_before[expert_idx].items():
            if not _states_exactly_equal(before, experts_after[expert_idx][key]):
                raise RuntimeError(
                    "KFAC pass unexpectedly changed expert state: "
                    f"expert={expert_idx}, key={key!r}."
                )

    return {
        "kfac_factors": kfac_factors,
        "fisher_route_counts": fisher_route_counts,
    }


def make_local_train_fn(settings: KFACSettings) -> Callable[..., ClientUpdate | None]:
    def local_train_kfac(**kwargs) -> ClientUpdate | None:
        return base.train_client(
            **kwargs,
            post_local_train_statistics_fn=lambda **hook_kwargs: (
                collect_post_training_kfac_statistics(
                    settings=settings,
                    **hook_kwargs,
                )
            ),
        )

    return local_train_kfac


# =============================================================================
# Expert aggregation math (preserved from the original method)
# =============================================================================

def _kfac_factors(update: ClientUpdate) -> list[dict[str, KFACLayerFactors]]:
    value = update.method_payload.get("kfac_factors")
    if not isinstance(value, list):
        raise RuntimeError(
            f"Client {update.client_id} is missing list-valued kfac_factors payload."
        )
    return value


def _fisher_route_counts(update: ClientUpdate) -> Tensor:
    value = update.method_payload.get("fisher_route_counts")
    if not isinstance(value, Tensor):
        raise RuntimeError(
            f"Client {update.client_id} is missing Tensor fisher_route_counts payload."
        )
    return value


def floating_state_like(
    reference: Mapping[str, Tensor], fill: float = 0.0
) -> StateDict:
    return {
        key: torch.full_like(value, float(fill), device="cpu")
        for key, value in reference.items()
        if torch.is_floating_point(value)
    }


def pseudo_gradient_from_delta(
    delta: Mapping[str, Tensor], learning_rate: float
) -> StateDict:
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    return {
        key: -value.detach().cpu() / float(learning_rate)
        for key, value in delta.items()
        if torch.is_floating_point(value)
    }


def state_weighted_sum(
    states: list[Mapping[str, Tensor]],
    weights: list[float],
    reference: Mapping[str, Tensor],
) -> StateDict:
    if len(states) != len(weights):
        raise ValueError("states and weights must have equal length.")
    result = floating_state_like(reference)
    for state, weight in zip(states, weights):
        for key in result:
            result[key].add_(
                state[key].to(dtype=result[key].dtype, device="cpu"),
                alpha=float(weight),
            )
    return result


def augmented_layer_gradient(
    state: Mapping[str, Tensor],
    layer_name: str,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    weight_key = f"{layer_name}.weight"
    bias_key = f"{layer_name}.bias"
    if weight_key not in state:
        raise KeyError(f"Missing expert state key {weight_key!r}.")
    weight = state[weight_key].to(device=device, dtype=dtype)
    if weight.ndim != 2:
        raise RuntimeError(
            f"{weight_key} must be 2D, got {tuple(weight.shape)}."
        )
    if bias_key in state:
        bias = state[bias_key].to(device=device, dtype=dtype)
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise RuntimeError(
                f"{bias_key} shape is incompatible with {weight_key}."
            )
        return torch.cat([weight, bias.unsqueeze(1)], dim=1)
    zeros = torch.zeros((weight.shape[0], 1), dtype=dtype, device=device)
    return torch.cat([weight, zeros], dim=1)


def assign_augmented_layer_gradient(
    state: StateDict,
    layer_name: str,
    matrix: Tensor,
) -> None:
    weight_key = f"{layer_name}.weight"
    bias_key = f"{layer_name}.bias"
    if weight_key not in state:
        raise KeyError(f"Missing target state key {weight_key!r}.")
    weight_shape = state[weight_key].shape
    if matrix.ndim != 2 or matrix.shape[0] != weight_shape[0]:
        raise RuntimeError(
            f"Matrix shape {tuple(matrix.shape)} is incompatible with "
            f"{weight_key} shape {tuple(weight_shape)}."
        )
    if matrix.shape[1] != weight_shape[1] + 1:
        raise RuntimeError(
            f"Expected {weight_shape[1] + 1} augmented columns, "
            f"got {matrix.shape[1]}."
        )
    state[weight_key] = (
        matrix[:, : weight_shape[1]]
        .detach()
        .to(device="cpu", dtype=state[weight_key].dtype)
    )
    if bias_key in state:
        state[bias_key] = (
            matrix[:, -1]
            .detach()
            .to(device="cpu", dtype=state[bias_key].dtype)
        )


def expert_linear_layer_names(model: nn.Module, expert_idx: int) -> list[str]:
    expert = getattr(model, "experts")[expert_idx]
    names = [
        name
        for name, module in expert.named_modules()
        if isinstance(module, nn.Linear)
    ]
    names.sort()
    if not names:
        raise RuntimeError(f"Expert {expert_idx} has no Linear layer.")
    return names


def tensor_norm(value: Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().double()).item())


def tensor_inner(left: Tensor, right: Tensor) -> float:
    return float((left.detach().double() * right.detach().double()).sum().item())


def tensor_cosine(left: Tensor, right: Tensor, epsilon: float) -> float:
    denominator = tensor_norm(left) * tensor_norm(right)
    if denominator <= epsilon:
        return 0.0
    return tensor_inner(left, right) / denominator


def tensor_relative_error(actual: Tensor, expected: Tensor, epsilon: float) -> float:
    return tensor_norm(actual - expected) / max(tensor_norm(expected), epsilon)


def state_l2_norm(state: Mapping[str, Tensor]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for value in state.values():
        if torch.is_floating_point(value):
            total += value.detach().double().square().sum()
    return float(torch.sqrt(total).item())


def state_inner(left: Mapping[str, Tensor], right: Mapping[str, Tensor]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for key, left_value in left.items():
        if torch.is_floating_point(left_value):
            total += (
                left_value.detach().double()
                * right[key].detach().double()
            ).sum()
    return float(total.item())


def state_cosine(
    left: Mapping[str, Tensor],
    right: Mapping[str, Tensor],
    epsilon: float,
) -> float:
    denominator = state_l2_norm(left) * state_l2_norm(right)
    if denominator <= epsilon:
        return 0.0
    return state_inner(left, right) / denominator


def safe_stats(values: list[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    array = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def factor_transform_summary(transform: FactorTransform) -> dict[str, float | bool]:
    return {
        "scale_mean_eigenvalue": transform.scale,
        "raw_eigen_min": transform.raw_eigen_min,
        "raw_eigen_max": transform.raw_eigen_max,
        "normalized_eigen_min": transform.normalized_eigen_min,
        "normalized_eigen_max": transform.normalized_eigen_max,
        "effective_eigen_min": transform.effective_eigen_min,
        "effective_eigen_max": transform.effective_eigen_max,
        "effective_condition": transform.effective_condition,
        "max_inverse_sqrt_gain": transform.max_inverse_sqrt_gain,
        "gain_cap_triggered": transform.gain_cap_triggered,
    }


def build_factor_transform(
    factor: Tensor,
    *,
    settings: KFACSettings,
    device: torch.device,
) -> FactorTransform:
    """CPU/float64 eigendecomposition; sqrt matrices move to aggregation device."""
    matrix = factor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("factor_not_square")
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError("factor_non_finite")

    matrix = 0.5 * (matrix + matrix.transpose(0, 1))
    dimension = int(matrix.shape[0])
    scale = float((torch.trace(matrix) / float(dimension)).item())
    if not math.isfinite(scale) or scale <= settings.factor_scale_epsilon:
        raise ValueError("factor_scale_too_small")

    normalized = matrix / scale
    normalized = 0.5 * (normalized + normalized.transpose(0, 1))
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(
            normalized.contiguous(), UPLO="U"
        )
    except RuntimeError as error:
        raise ValueError("cpu_eigh_failed") from error

    if not bool(torch.isfinite(eigenvalues).all().item()):
        raise ValueError("eigenvalues_non_finite")
    if not bool(torch.isfinite(eigenvectors).all().item()):
        raise ValueError("eigenvectors_non_finite")

    raw_eigenvalues = eigenvalues * scale
    nonnegative = eigenvalues.clamp_min(0.0)
    gain_floor = 1.0 / float(settings.max_whitening_gain) ** 2
    damped = nonnegative + float(settings.relative_damping)
    effective = damped.clamp_min(gain_floor)

    inverse_values = effective.rsqrt()
    sqrt_values = effective.sqrt()
    inverse_sqrt_cpu = (
        eigenvectors * inverse_values.unsqueeze(0)
    ) @ eigenvectors.transpose(0, 1)
    sqrt_cpu = (
        eigenvectors * sqrt_values.unsqueeze(0)
    ) @ eigenvectors.transpose(0, 1)
    inverse_sqrt_cpu = 0.5 * (
        inverse_sqrt_cpu + inverse_sqrt_cpu.transpose(0, 1)
    )
    sqrt_cpu = 0.5 * (sqrt_cpu + sqrt_cpu.transpose(0, 1))
    if not bool(torch.isfinite(inverse_sqrt_cpu).all().item()):
        raise ValueError("inverse_sqrt_non_finite")
    if not bool(torch.isfinite(sqrt_cpu).all().item()):
        raise ValueError("sqrt_non_finite")

    effective_min = float(effective.min().item())
    effective_max = float(effective.max().item())
    inverse_sqrt = inverse_sqrt_cpu.to(device=device, dtype=torch.float32)
    sqrt = sqrt_cpu.to(device=device, dtype=torch.float32)
    return FactorTransform(
        inverse_sqrt=inverse_sqrt.contiguous(),
        sqrt=sqrt.contiguous(),
        scale=scale,
        raw_eigen_min=float(raw_eigenvalues.min().item()),
        raw_eigen_max=float(raw_eigenvalues.max().item()),
        normalized_eigen_min=float(eigenvalues.min().item()),
        normalized_eigen_max=float(eigenvalues.max().item()),
        effective_eigen_min=effective_min,
        effective_eigen_max=effective_max,
        effective_condition=(
            effective_max / max(effective_min, settings.factor_scale_epsilon)
        ),
        max_inverse_sqrt_gain=float(inverse_values.max().item()),
        gain_cap_triggered=bool((damped < gain_floor).any().item()),
    )


def build_local_layer_geometry(
    *,
    update: ClientUpdate,
    expert_idx: int,
    layer_name: str,
    gradient: Tensor,
    final_weight: float,
    valid_reference_weight: float,
    settings: KFACSettings,
    device: torch.device,
) -> LocalLayerGeometry:
    factors = _kfac_factors(update)[expert_idx]
    if layer_name not in factors:
        raise ValueError("missing_layer_factor")
    layer_factor = factors[layer_name]
    if layer_factor.count < settings.minimum_kfac_samples:
        raise ValueError("insufficient_kfac_samples")

    activation_transform = build_factor_transform(
        layer_factor.activation,
        settings=settings,
        device=device,
    )
    output_transform = build_factor_transform(
        layer_factor.output_gradient,
        settings=settings,
        device=device,
    )
    if output_transform.inverse_sqrt.shape[0] != gradient.shape[0]:
        raise ValueError("output_factor_shape_mismatch")
    if activation_transform.inverse_sqrt.shape[0] != gradient.shape[1]:
        raise ValueError("activation_factor_shape_mismatch")

    whitened = (
        output_transform.inverse_sqrt
        @ gradient
        @ activation_transform.inverse_sqrt
    )
    roundtrip = (
        output_transform.sqrt
        @ whitened
        @ activation_transform.sqrt
    )
    roundtrip_relative_error = tensor_relative_error(
        roundtrip, gradient, settings.projection_epsilon
    )
    if not bool(torch.isfinite(whitened).all().item()):
        raise ValueError("whitened_gradient_non_finite")
    if not math.isfinite(roundtrip_relative_error):
        raise ValueError("roundtrip_error_non_finite")

    return LocalLayerGeometry(
        client_id=update.client_id,
        route_count=int(update.route_counts[expert_idx]),
        final_weight=float(final_weight),
        valid_reference_weight=float(valid_reference_weight),
        gradient=gradient,
        whitened=whitened,
        activation_transform=activation_transform,
        output_transform=output_transform,
        sample_count=int(layer_factor.count),
        roundtrip_relative_error=roundtrip_relative_error,
    )


def aggregate_experts_local_kfac_layer_projection(
    *,
    model: nn.Module,
    updates: list[ClientUpdate],
    config: base.ExperimentConfig,
    settings: KFACSettings,
    learning_rate: float,
    device: torch.device,
) -> ExpertAggregationResult:
    """Self-included reference, strict negative projection, and no correction cap."""
    participant_counts: list[int] = []
    client_weights_by_expert: list[dict[int, float]] = []
    all_diagnostics: list[dict] = []

    for expert_idx in range(config.num_experts):
        old_state = model.get_expert_state_dict(expert_idx, to_cpu=True)
        active_updates = [
            update
            for update in updates
            if int(update.route_counts[expert_idx]) > 0
        ]
        participant_counts.append(len(active_updates))
        layer_names = expert_linear_layer_names(model, expert_idx)

        if not active_updates:
            client_weights_by_expert.append({})
            all_diagnostics.append(
                {
                    "expert": expert_idx,
                    "inactive": True,
                    "active_client_count": 0,
                    "training_route_count": 0,
                    "layers": {},
                }
            )
            continue

        total_routes = sum(
            int(update.route_counts[expert_idx]) for update in active_updates
        )
        if total_routes <= 0:
            raise RuntimeError(
                f"Expert {expert_idx} has active clients but zero routes."
            )
        final_weights = [
            float(int(update.route_counts[expert_idx])) / float(total_routes)
            for update in active_updates
        ]
        client_weights_by_expert.append(
            {
                update.client_id: float(weight)
                for update, weight in zip(active_updates, final_weights)
            }
        )

        client_gradients = [
            pseudo_gradient_from_delta(
                update.expert_deltas[expert_idx], learning_rate
            )
            for update in active_updates
        ]
        base_gradient = state_weighted_sum(
            client_gradients, final_weights, old_state
        )
        final_gradient = {
            key: value.detach().clone() for key, value in base_gradient.items()
        }
        expert_layer_diagnostics: dict[str, dict] = {}

        for layer_name in layer_names:
            client_layer_gradients = [
                augmented_layer_gradient(
                    gradient,
                    layer_name,
                    device=device,
                    dtype=torch.float32,
                )
                for gradient in client_gradients
            ]
            base_layer = sum(
                (
                    float(weight) * gradient
                    for weight, gradient in zip(
                        final_weights, client_layer_gradients
                    )
                ),
                torch.zeros_like(client_layer_gradients[0]),
            )

            geometries: dict[int, LocalLayerGeometry] = {}
            invalid_reasons: dict[int, str] = {}
            preliminary_positions: list[int] = []
            for position, update in enumerate(active_updates):
                factors = _kfac_factors(update)[expert_idx]
                factor = factors.get(layer_name)
                if factor is None:
                    invalid_reasons[position] = "missing_layer_factor"
                elif int(factor.count) < settings.minimum_kfac_samples:
                    invalid_reasons[position] = "insufficient_kfac_samples"
                else:
                    preliminary_positions.append(position)

            preliminary_routes = sum(
                int(active_updates[position].route_counts[expert_idx])
                for position in preliminary_positions
            )
            if preliminary_routes > 0:
                for position in preliminary_positions:
                    update = active_updates[position]
                    try:
                        geometries[position] = build_local_layer_geometry(
                            update=update,
                            expert_idx=expert_idx,
                            layer_name=layer_name,
                            gradient=client_layer_gradients[position],
                            final_weight=final_weights[position],
                            valid_reference_weight=(
                                float(int(update.route_counts[expert_idx]))
                                / float(preliminary_routes)
                            ),
                            settings=settings,
                            device=device,
                        )
                    except (ValueError, RuntimeError) as error:
                        invalid_reasons[position] = (
                            str(error).strip() or type(error).__name__
                        )

            valid_route_total = sum(
                geometry.route_count for geometry in geometries.values()
            )
            if valid_route_total > 0:
                for geometry in geometries.values():
                    geometry.valid_reference_weight = (
                        float(geometry.route_count) / float(valid_route_total)
                    )

            if geometries:
                reference = sum(
                    (
                        geometry.valid_reference_weight * geometry.whitened
                        for geometry in geometries.values()
                    ),
                    torch.zeros_like(next(iter(geometries.values())).whitened),
                )
                reference_norm = tensor_norm(reference)
                reference_norm_squared = tensor_inner(reference, reference)
                floor = max(
                    settings.reference_norm_warning_threshold,
                    settings.projection_epsilon,
                )
                reference_valid = (
                    math.isfinite(reference_norm_squared)
                    and reference_norm_squared > floor
                )
            else:
                reference = None
                reference_norm = 0.0
                reference_norm_squared = 0.0
                reference_valid = False

            corrected_layer_gradients = [
                gradient.detach().clone() for gradient in client_layer_gradients
            ]
            client_diagnostics: list[dict] = []
            conflict_count = 0
            projection_count = 0
            mapped_correction_ratios: list[float] = []
            whitened_correction_ratios: list[float] = []
            projection_scalars: list[float] = []
            whitened_cosines: list[float] = []
            gain_cap_flags: list[bool] = []
            roundtrip_errors: list[float] = []

            for position, (update, gradient, final_weight) in enumerate(
                zip(active_updates, client_layer_gradients, final_weights)
            ):
                factors = _kfac_factors(update)[expert_idx]
                factor = factors.get(layer_name)
                sample_count = 0 if factor is None else int(factor.count)
                fisher_route_counts = _fisher_route_counts(update)
                if position not in geometries:
                    client_diagnostics.append(
                        {
                            "client_id": update.client_id,
                            "training_route_count": int(
                                update.route_counts[expert_idx]
                            ),
                            "fisher_route_count": int(
                                fisher_route_counts[expert_idx]
                            ),
                            "final_aggregation_weight": float(final_weight),
                            "kfac_sample_count": sample_count,
                            "kfac_valid": False,
                            "projection_applied": False,
                            "fallback_reason": invalid_reasons.get(
                                position, "invalid_kfac"
                            ),
                        }
                    )
                    continue

                geometry = geometries[position]
                z = geometry.whitened
                inner = 0.0
                cosine = 0.0
                scalar = 0.0
                conflict = False
                corrected_z = z
                if reference is not None and reference_valid:
                    inner = tensor_inner(z, reference)
                    cosine = tensor_cosine(
                        z, reference, settings.projection_epsilon
                    )
                    scalar = inner / reference_norm_squared
                    conflict = bool(math.isfinite(scalar) and scalar < 0.0)
                    if conflict:
                        corrected_z = z - float(scalar) * reference

                corrected_gradient = (
                    geometry.output_transform.sqrt
                    @ corrected_z
                    @ geometry.activation_transform.sqrt
                )
                runtime_fallback_reason: str | None = None
                if not bool(torch.isfinite(corrected_gradient).all().item()):
                    corrected_gradient = gradient
                    corrected_z = z
                    conflict = False
                    scalar = 0.0
                    inner = 0.0
                    cosine = 0.0
                    runtime_fallback_reason = "unwhitened_gradient_non_finite"

                corrected_layer_gradients[position] = corrected_gradient
                gradient_norm = tensor_norm(gradient)
                z_norm = tensor_norm(z)
                mapped_ratio = tensor_norm(
                    corrected_gradient - gradient
                ) / max(gradient_norm, settings.projection_epsilon)
                whitened_ratio = tensor_norm(
                    corrected_z - z
                ) / max(z_norm, settings.projection_epsilon)
                post_inner = (
                    tensor_inner(corrected_z, reference)
                    if reference is not None
                    else 0.0
                )

                if conflict:
                    conflict_count += 1
                    projection_count += 1
                mapped_correction_ratios.append(mapped_ratio)
                whitened_correction_ratios.append(whitened_ratio)
                projection_scalars.append(float(scalar))
                whitened_cosines.append(float(cosine))
                gain_cap_flags.extend(
                    [
                        geometry.activation_transform.gain_cap_triggered,
                        geometry.output_transform.gain_cap_triggered,
                    ]
                )
                roundtrip_errors.append(geometry.roundtrip_relative_error)

                client_diagnostics.append(
                    {
                        "client_id": update.client_id,
                        "training_route_count": geometry.route_count,
                        "fisher_route_count": int(
                            fisher_route_counts[expert_idx]
                        ),
                        "final_aggregation_weight": geometry.final_weight,
                        "valid_reference_weight": geometry.valid_reference_weight,
                        "kfac_sample_count": geometry.sample_count,
                        "kfac_valid": True,
                        "projection_applied": bool(
                            conflict and runtime_fallback_reason is None
                        ),
                        "fallback_reason": runtime_fallback_reason,
                        "activation_factor": factor_transform_summary(
                            geometry.activation_transform
                        ),
                        "output_gradient_factor": factor_transform_summary(
                            geometry.output_transform
                        ),
                        "roundtrip_relative_error": (
                            geometry.roundtrip_relative_error
                        ),
                        "gradient_norm": gradient_norm,
                        "whitened_norm": z_norm,
                        "whitening_norm_ratio": z_norm
                        / max(gradient_norm, settings.projection_epsilon),
                        "projection_inner": float(inner),
                        "whitened_cosine": float(cosine),
                        "projection_scalar": float(scalar),
                        "conflict": bool(conflict),
                        "post_projection_inner": float(post_inner),
                        "whitened_correction_ratio": whitened_ratio,
                        "mapped_correction_ratio": mapped_ratio,
                        "gradient_corrected_cosine": tensor_cosine(
                            gradient,
                            corrected_gradient,
                            settings.projection_epsilon,
                        ),
                    }
                )

            final_layer = sum(
                (
                    float(weight) * corrected
                    for weight, corrected in zip(
                        final_weights, corrected_layer_gradients
                    )
                ),
                torch.zeros_like(base_layer),
            )
            assign_augmented_layer_gradient(
                final_gradient, layer_name, final_layer
            )
            valid_count = len(geometries)
            active_count = len(active_updates)
            layer_base_norm = tensor_norm(base_layer)
            layer_final_norm = tensor_norm(final_layer)
            expert_layer_diagnostics[layer_name] = {
                "active_client_count": active_count,
                "valid_kfac_client_count": valid_count,
                "fallback_client_count": active_count - valid_count,
                "conflict_count": conflict_count,
                "projection_count": projection_count,
                "conflict_rate_over_valid": (
                    float(conflict_count) / float(valid_count)
                    if valid_count > 0
                    else 0.0
                ),
                "projection_rate_over_active": (
                    float(projection_count) / float(active_count)
                    if active_count > 0
                    else 0.0
                ),
                "reference_includes_self": True,
                "reference_norm": reference_norm,
                "reference_norm_squared": reference_norm_squared,
                "reference_valid": reference_valid,
                "projection_scalar": safe_stats(projection_scalars),
                "whitened_cosine": safe_stats(whitened_cosines),
                "whitened_correction_ratio": safe_stats(
                    whitened_correction_ratios
                ),
                "mapped_correction_ratio": safe_stats(
                    mapped_correction_ratios
                ),
                "roundtrip_relative_error": safe_stats(roundtrip_errors),
                "gain_cap_trigger_rate": (
                    float(sum(bool(flag) for flag in gain_cap_flags))
                    / float(len(gain_cap_flags))
                    if gain_cap_flags
                    else 0.0
                ),
                "base_gradient_norm": layer_base_norm,
                "final_gradient_norm": layer_final_norm,
                "base_final_cosine": tensor_cosine(
                    base_layer,
                    final_layer,
                    settings.projection_epsilon,
                ),
                "base_final_norm_ratio": layer_final_norm
                / max(layer_base_norm, settings.projection_epsilon),
                "base_final_relative_change": tensor_norm(
                    final_layer - base_layer
                ) / max(layer_base_norm, settings.projection_epsilon),
                "fallback_reason_counts": dict(Counter(invalid_reasons.values())),
                "clients": client_diagnostics,
            }

        new_state: StateDict = {}
        for key, old_value in old_state.items():
            if torch.is_floating_point(old_value):
                new_state[key] = old_value - float(learning_rate) * (
                    final_gradient[key].to(dtype=old_value.dtype)
                )
            else:
                new_state[key] = old_value
        model.load_expert_state_dict(expert_idx, new_state, strict=True)

        base_norm = state_l2_norm(base_gradient)
        final_norm = state_l2_norm(final_gradient)
        all_diagnostics.append(
            {
                "expert": expert_idx,
                "inactive": False,
                "active_client_count": len(active_updates),
                "active_client_ids": [
                    update.client_id for update in active_updates
                ],
                "training_route_count": total_routes,
                "final_aggregation_weights": [
                    {
                        "client_id": update.client_id,
                        "route_count": int(update.route_counts[expert_idx]),
                        "weight": float(weight),
                    }
                    for update, weight in zip(active_updates, final_weights)
                ],
                "base_gradient_norm": base_norm,
                "final_gradient_norm": final_norm,
                "base_final_cosine": state_cosine(
                    base_gradient,
                    final_gradient,
                    settings.projection_epsilon,
                ),
                "base_final_norm_ratio": final_norm
                / max(base_norm, settings.projection_epsilon),
                "layers": expert_layer_diagnostics,
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return ExpertAggregationResult(
        participant_counts=participant_counts,
        client_weights_by_expert=client_weights_by_expert,
        diagnostics=all_diagnostics,
    )


def summarize_round_diagnostics(
    diagnostics: list[dict],
) -> dict[str, float | int]:
    active = 0
    valid = 0
    fallback = 0
    conflicts = 0
    projections = 0
    train_active_fisher_zero = 0
    mapped_mean_ratios: list[float] = []
    mapped_max_ratios: list[float] = []
    layer_cosines: list[float] = []
    gain_cap_rates: list[float] = []
    max_roundtrip = 0.0

    for expert in diagnostics:
        for layer in expert.get("layers", {}).values():
            active += int(layer.get("active_client_count", 0))
            valid += int(layer.get("valid_kfac_client_count", 0))
            fallback += int(layer.get("fallback_client_count", 0))
            conflicts += int(layer.get("conflict_count", 0))
            projections += int(layer.get("projection_count", 0))
            layer_cosines.append(float(layer.get("base_final_cosine", 1.0)))
            gain_cap_rates.append(float(layer.get("gain_cap_trigger_rate", 0.0)))
            mapped = layer.get("mapped_correction_ratio", {})
            if int(mapped.get("count", 0)) > 0:
                mapped_mean_ratios.append(float(mapped.get("mean", 0.0)))
                mapped_max_ratios.append(float(mapped.get("max", 0.0)))
            roundtrip = layer.get("roundtrip_relative_error", {})
            max_roundtrip = max(max_roundtrip, float(roundtrip.get("max", 0.0)))
            for client in layer.get("clients", []):
                if (
                    int(client.get("training_route_count", 0)) > 0
                    and int(client.get("fisher_route_count", 0)) == 0
                ):
                    train_active_fisher_zero += 1

    return {
        "active_client_layer_count": active,
        "valid_kfac_client_layer_count": valid,
        "fallback_client_layer_count": fallback,
        "conflict_client_layer_count": conflicts,
        "projection_client_layer_count": projections,
        "conflict_rate_over_valid": (
            float(conflicts) / float(valid) if valid > 0 else 0.0
        ),
        "projection_rate_over_active": (
            float(projections) / float(active) if active > 0 else 0.0
        ),
        "kfac_coverage_rate": (
            float(valid) / float(active) if active > 0 else 0.0
        ),
        "mean_layer_base_final_cosine": (
            float(np.mean(layer_cosines)) if layer_cosines else 1.0
        ),
        "mean_mapped_correction_ratio": (
            float(np.mean(mapped_mean_ratios)) if mapped_mean_ratios else 0.0
        ),
        "max_mapped_correction_ratio": max(mapped_max_ratios, default=0.0),
        "training_active_fisher_zero_client_layer_count": (
            train_active_fisher_zero
        ),
        "gain_cap_trigger_rate": (
            float(np.mean(gain_cap_rates)) if gain_cap_rates else 0.0
        ),
        "max_roundtrip_relative_error": max_roundtrip,
    }


# =============================================================================
# Adapter to the refactored server aggregation interface
# =============================================================================

def make_server_aggregate_fn(
    *,
    settings: KFACSettings,
    kfac_device: torch.device,
    diagnostics_path: Path,
    logger,
) -> Callable[..., base.AggregationResult]:
    def server_aggregate(
        *,
        global_model: nn.Module,
        updates: list[ClientUpdate],
        config: base.ExperimentConfig,
        method_state: object | None,
        round_idx: int,
    ) -> base.AggregationResult:
        del method_state

        # Preserve the original server order exactly:
        # 1) uniform shared/non-expert aggregation;
        # 2) KFAC expert aggregation.
        base.aggregate_shared_uniform(global_model, updates)
        expert_aggregation = aggregate_experts_local_kfac_layer_projection(
            model=global_model,
            updates=updates,
            config=config,
            settings=settings,
            learning_rate=config.learning_rate,
            device=kfac_device,
        )
        projection_summary = summarize_round_diagnostics(
            expert_aggregation.diagnostics
        )

        with diagnostics_path.open("a", encoding="utf-8") as diagnostics_file:
            diagnostics_file.write(
                json.dumps(
                    {
                        "round": round_idx + 1,
                        "learning_rate": config.learning_rate,
                        "summary": projection_summary,
                        "experts": expert_aggregation.diagnostics,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        logger.info(
            "Round %d KFAC | coverage=%.4f | conflicts=%d/%d (%.4f) | "
            "projection_over_active=%.4f | mean_layer_cos=%.6f | "
            "mapped_corr_mean=%.4f | mapped_corr_max=%.4f | "
            "train_active_fisher_zero=%d | max_roundtrip=%.3e",
            round_idx + 1,
            projection_summary["kfac_coverage_rate"],
            projection_summary["conflict_client_layer_count"],
            projection_summary["valid_kfac_client_layer_count"],
            projection_summary["conflict_rate_over_valid"],
            projection_summary["projection_rate_over_active"],
            projection_summary["mean_layer_base_final_cosine"],
            projection_summary["mean_mapped_correction_ratio"],
            projection_summary["max_mapped_correction_ratio"],
            projection_summary[
                "training_active_fisher_zero_client_layer_count"
            ],
            projection_summary["max_roundtrip_relative_error"],
            extra={"file_only": True},
        )

        return base.AggregationResult(
            expert_participants=expert_aggregation.participant_counts,
            expert_client_weights=expert_aggregation.client_weights_by_expert,
            method_metrics=projection_summary,
        )

    return server_aggregate


# =============================================================================
# Entrypoint
# =============================================================================

def main() -> None:
    config, settings = parse_configs()
    base.configure_reproducibility(config)
    training_device = base.resolve_device(config.device)
    kfac_device = resolve_kfac_server_device(
        settings.kfac_server_device,
        training_device,
    )

    output_dir = base.create_output_dir(config, ALGORITHM_NAME)
    logger = base.create_logger(output_dir / "train.log", ALGORITHM_NAME)
    diagnostics_path = output_dir / "kfac_projection_diagnostics.jsonl"

    base.save_json(
        output_dir / "kfac_config.json",
        {
            **asdict(settings),
            "resolved_kfac_server_device": str(kfac_device),
            "kfac_eigendecomposition_device": "cpu",
            "kfac_eigendecomposition_dtype": "float64",
            "kfac_collection_data_augmentation": False,
            "kfac_collection_updates_parameters": False,
            "kfac_collection_loss": "standard_sample_mean_cross_entropy",
            "supports_top_k_1_and_2": True,
            "kfac_reference_includes_self": True,
            "kfac_reference_valid_clients_only": True,
            "kfac_invalid_policy": (
                "exclude_from_reference_keep_original_in_final"
            ),
            "training_route_replay": False,
            "parameter_correction_cap": None,
        },
    )

    local_train_fn = make_local_train_fn(settings)
    server_aggregate_fn = make_server_aggregate_fn(
        settings=settings,
        kfac_device=kfac_device,
        diagnostics_path=diagnostics_path,
        logger=logger,
    )

    logger.info(
        "Local KFAC: post_training_extra_pass=True | fp32=True | "
        "loss=standard_sample_mean_ce | fisher_batch_size=%s | "
        "minimum_samples=%d | relative_damping=%s | max_whitening_gain=%s | "
        "kfac_server_device=%s | eigendecomposition=cpu_float64",
        settings.fisher_batch_size,
        settings.minimum_kfac_samples,
        settings.relative_damping,
        settings.max_whitening_gain,
        kfac_device,
    )

    try:
        base.run_experiment(
            config,
            output_dir,
            logger,
            algorithm_name=ALGORITHM_NAME,
            local_train_fn=local_train_fn,
            server_aggregate_fn=server_aggregate_fn,
            local_objective_description=(
                "Local objective: standard sample-mean cross-entropy; "
                "optional balance loss follows base.py configuration"
            ),
            aggregation_description=(
                "Expert aggregation: training-route-count-weighted local-KFAC "
                "layer whitening; valid-KFAC-only self-included reference; "
                "strict negative projection in whitened coordinates; "
                "client-specific unwhitening with the same damped KFAC map; "
                "invalid KFAC keeps original pseudo-gradient; final aggregation "
                "uses training-stage route counts; no replay; no correction cap"
            ),
        )
    except Exception:
        logger.exception("Experiment failed.")
        raise


if __name__ == "__main__":
    main()

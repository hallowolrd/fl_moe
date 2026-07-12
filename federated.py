from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# 该接口由后续 aggregation/base.py 提供。
from aggregation.base import (
    ExpertAggregator,
    ExpertClientUpdate,
)


StateDict = Dict[str, Tensor]
OptimizerFactory = Callable[[nn.Module], Optimizer]


__all__ = [
    "AggregationSummary",
    "ClientUpdate",
    "EvaluationMetrics",
    "OptimizerFactory",
    "aggregate_expert_parameters",
    "aggregate_global_model",
    "aggregate_shared_parameters_uniform",
    "evaluate_model",
    "sample_clients",
    "train_client",
]


@dataclass
class ClientUpdate:
    """
    单个客户端完成本地训练后上传到服务器的信息。

    所有 state delta 默认存放在 CPU，便于单机串行模拟时控制显存。

    Attributes:
        client_id:
            客户端编号。
        num_examples:
            客户端数据集中的唯一样本数，不乘 local epochs。
        num_processed_examples:
            本轮本地训练实际处理的样本总次数。
        shared_delta:
            共享模块参数增量：
            local_shared_state - global_shared_state。
        expert_deltas:
            每个专家的参数增量列表。
        route_counts:
            每个专家在本地训练期间被选中的累计次数，形状 [E]。
        route_weight_sums:
            每个专家累计获得的路由概率，形状 [E]。
        expert_update_norms:
            每个专家参数增量的 L2 范数，形状 [E]。
        train_loss:
            本地总损失的样本加权平均。
        classification_loss:
            本地分类损失的样本加权平均。
        balance_loss:
            本地负载均衡损失的样本加权平均。
        accuracy:
            本地训练准确率。
    """

    client_id: int
    num_examples: int
    num_processed_examples: int

    shared_delta: StateDict
    expert_deltas: list[StateDict]

    route_counts: Tensor
    route_weight_sums: Tensor
    expert_update_norms: Tensor

    train_loss: float
    classification_loss: float
    balance_loss: float
    accuracy: float


@dataclass
class EvaluationMetrics:
    """
    全局模型评估结果。
    """

    loss: float
    classification_loss: float
    balance_loss: float
    accuracy: float
    num_examples: int

    route_counts: Tensor
    route_weight_sums: Tensor
    route_distribution: Tensor

    def to_dict(self) -> dict:
        return {
            "loss": self.loss,
            "classification_loss": self.classification_loss,
            "balance_loss": self.balance_loss,
            "accuracy": self.accuracy,
            "num_examples": self.num_examples,
            "route_counts": self.route_counts.tolist(),
            "route_weight_sums": self.route_weight_sums.tolist(),
            "route_distribution": self.route_distribution.tolist(),
        }


@dataclass
class AggregationSummary:
    """
    一轮服务器聚合的摘要信息。
    """

    num_client_updates: int

    # 每个专家本轮有多少个客户端产生了有效更新。
    expert_participant_counts: Tensor

    def to_dict(self) -> dict:
        return {
            "num_client_updates": self.num_client_updates,
            "expert_participant_counts": (
                self.expert_participant_counts.tolist()
            ),
        }


def sample_clients(
    num_clients: int,
    *,
    participation_rate: Optional[float] = None,
    num_selected_clients: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> list[int]:
    """
    无放回采样客户端。

    `participation_rate` 和 `num_selected_clients` 必须二选一。

    当使用 participation_rate 时，采样数量采用:
        ceil(num_clients * participation_rate)

    Args:
        num_clients:
            客户端总数。
        participation_rate:
            参与比例，范围 (0, 1]。
        num_selected_clients:
            精确指定参与客户端数量。
        generator:
            由 train.py 创建并设置种子的 torch.Generator。

    Returns:
        排序后的客户端编号列表。
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be greater than 0.")

    exactly_one_is_set = (
        (participation_rate is None)
        != (num_selected_clients is None)
    )

    if not exactly_one_is_set:
        raise ValueError(
            "Exactly one of participation_rate and "
            "num_selected_clients must be provided."
        )

    if participation_rate is not None:
        if not 0.0 < participation_rate <= 1.0:
            raise ValueError(
                "participation_rate must be in the interval (0, 1]."
            )

        selected_count = max(
            1,
            math.ceil(num_clients * participation_rate),
        )
    else:
        selected_count = int(num_selected_clients)

        if not 1 <= selected_count <= num_clients:
            raise ValueError(
                "num_selected_clients must satisfy "
                "1 <= num_selected_clients <= num_clients."
            )

    permutation = torch.randperm(
        num_clients,
        generator=generator,
    )

    selected = permutation[:selected_count].tolist()
    selected.sort()

    return selected


def train_client(
    *,
    client_id: int,
    global_model: nn.Module,
    train_loader: DataLoader,
    optimizer_factory: OptimizerFactory,
    device: torch.device | str,
    local_epochs: int,
    balance_loss_weight: float = 0.01,
    use_amp: bool = False,
    max_grad_norm: Optional[float] = None,
    num_examples: Optional[int] = None,
) -> Optional[ClientUpdate]:
    """
    在单机上串行模拟一个客户端的本地训练。

    流程:
        1. deepcopy 全局模型；
        2. 保存本轮全局共享状态和专家状态；
        3. 创建本地优化器；
        4. 执行 local epochs；
        5. 计算参数增量和每个专家更新范数；
        6. 返回 CPU 上的 ClientUpdate。

    Args:
        optimizer_factory:
            由 train.py 提供。函数签名为:
                optimizer = optimizer_factory(local_model)

            这样 train.py 可以决定优化器类型、学习率和参数组。

        use_amp:
            仅在 CUDA 设备上真正启用。非 CUDA 设备会自动退化为
            普通精度训练。

        max_grad_norm:
            None 表示不裁剪；否则使用 clip_grad_norm_。

    Returns:
        空客户端返回 None，其余返回 ClientUpdate。
    """
    device = torch.device(device)

    if local_epochs <= 0:
        raise ValueError("local_epochs must be greater than 0.")

    if balance_loss_weight < 0.0:
        raise ValueError(
            "balance_loss_weight must be non-negative."
        )

    if max_grad_norm is not None and max_grad_norm <= 0.0:
        raise ValueError(
            "max_grad_norm must be positive or None."
        )

    resolved_num_examples = _resolve_num_examples(
        train_loader=train_loader,
        explicit_num_examples=num_examples,
    )

    # 按约定，空客户端直接跳过。
    if resolved_num_examples == 0:
        return None

    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    local_model.train()

    _validate_moe_model_interface(local_model)

    # 服务器下发时的状态，用于计算本地参数增量。
    initial_shared_state = local_model.get_shared_state_dict(
        clone=True,
        to_cpu=True,
    )
    initial_expert_states = (
        local_model.get_all_expert_state_dicts(
            clone=True,
            to_cpu=True,
        )
    )

    optimizer = optimizer_factory(local_model)

    if not isinstance(optimizer, Optimizer):
        raise TypeError(
            "optimizer_factory must return a torch.optim.Optimizer."
        )

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    route_counts = torch.zeros(
        local_model.num_experts,
        dtype=torch.long,
        device=device,
    )
    route_weight_sums = torch.zeros(
        local_model.num_experts,
        dtype=torch.float64,
        device=device,
    )

    total_loss_sum = 0.0
    classification_loss_sum = 0.0
    balance_loss_sum = 0.0

    total_correct = 0
    total_processed = 0

    for _ in range(local_epochs):
        for batch in train_loader:
            images, labels = _unpack_batch(batch)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            batch_size = int(labels.shape[0])

            if batch_size == 0:
                continue

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                output = local_model(images)

                classification_loss = F.cross_entropy(
                    output.logits,
                    labels,
                )

                total_loss = (
                    classification_loss
                    + balance_loss_weight * output.balance_loss
                )

            _assert_finite_scalar(
                total_loss,
                name=(
                    f"client {client_id} total loss"
                ),
            )

            scaler.scale(total_loss).backward()

            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    local_model.parameters(),
                    max_norm=max_grad_norm,
                )

            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                predictions = output.logits.argmax(dim=1)
                total_correct += int(
                    predictions.eq(labels).sum().item()
                )
                total_processed += batch_size

                total_loss_sum += (
                    float(total_loss.detach().item())
                    * batch_size
                )
                classification_loss_sum += (
                    float(
                        classification_loss.detach().item()
                    )
                    * batch_size
                )
                balance_loss_sum += (
                    float(
                        output.balance_loss.detach().item()
                    )
                    * batch_size
                )

                route_counts += (
                    output.route_counts.detach().to(
                        device=device,
                        dtype=torch.long,
                    )
                )
                route_weight_sums += (
                    output.route_weight_sums.detach().to(
                        device=device,
                        dtype=torch.float64,
                    )
                )

    if total_processed == 0:
        # DataLoader 没有产生有效 batch 时也按空客户端处理。
        return None

    final_shared_state = local_model.get_shared_state_dict(
        clone=True,
        to_cpu=True,
    )
    final_expert_states = (
        local_model.get_all_expert_state_dicts(
            clone=True,
            to_cpu=True,
        )
    )

    shared_delta = _compute_state_delta(
        initial_state=initial_shared_state,
        final_state=final_shared_state,
    )

    expert_deltas = [
        _compute_state_delta(
            initial_state=initial_expert_states[expert_idx],
            final_state=final_expert_states[expert_idx],
        )
        for expert_idx in range(local_model.num_experts)
    ]

    _assert_state_finite(
        shared_delta,
        name=f"client {client_id} shared delta",
    )

    for expert_idx, expert_delta in enumerate(expert_deltas):
        _assert_state_finite(
            expert_delta,
            name=(
                f"client {client_id} "
                f"expert {expert_idx} delta"
            ),
        )

    expert_update_norms = torch.tensor(
        [
            _state_l2_norm(expert_delta)
            for expert_delta in expert_deltas
        ],
        dtype=torch.float64,
    )

    update = ClientUpdate(
        client_id=int(client_id),
        num_examples=resolved_num_examples,
        num_processed_examples=total_processed,
        shared_delta=shared_delta,
        expert_deltas=expert_deltas,
        route_counts=route_counts.detach().cpu(),
        route_weight_sums=(
            route_weight_sums.detach().cpu()
        ),
        expert_update_norms=expert_update_norms,
        train_loss=total_loss_sum / total_processed,
        classification_loss=(
            classification_loss_sum / total_processed
        ),
        balance_loss=(
            balance_loss_sum / total_processed
        ),
        accuracy=total_correct / total_processed,
    )

    del local_model
    del optimizer

    return update


def aggregate_shared_parameters_uniform(
    *,
    global_model: nn.Module,
    client_updates: Sequence[ClientUpdate],
) -> None:
    """
    对共享参数增量执行客户端等权平均。

    不使用客户端数据量和处理样本数作为权重:

        shared_delta =
            (1 / M) * sum_i shared_delta_i

        shared_state_next =
            shared_state_current + shared_delta

    所有有效客户端权重完全相同。
    """
    if not client_updates:
        raise ValueError(
            "Cannot aggregate shared parameters without client updates."
        )

    _validate_moe_model_interface(global_model)

    global_shared_state = global_model.get_shared_state_dict(
        clone=True,
        to_cpu=False,
    )

    expected_keys = set(global_shared_state.keys())

    for update in client_updates:
        provided_keys = set(update.shared_delta.keys())

        if provided_keys != expected_keys:
            missing = expected_keys - provided_keys
            extra = provided_keys - expected_keys

            raise KeyError(
                f"Client {update.client_id} shared delta keys mismatch. "
                f"Missing: {sorted(missing)}; "
                f"Extra: {sorted(extra)}."
            )

    num_updates = len(client_updates)
    aggregated_shared_state: StateDict = {}

    for name, global_tensor in global_shared_state.items():
        # 非浮点 buffer 不参与平均，保持服务器当前值。
        if not (
            torch.is_floating_point(global_tensor)
            or torch.is_complex(global_tensor)
        ):
            aggregated_shared_state[name] = (
                global_tensor.detach().clone()
            )
            continue

        average_delta = torch.zeros_like(global_tensor)

        for update in client_updates:
            client_delta = update.shared_delta[name].to(
                device=global_tensor.device,
                dtype=global_tensor.dtype,
            )
            average_delta.add_(client_delta)

        average_delta.div_(float(num_updates))

        aggregated_shared_state[name] = (
            global_tensor.detach().clone()
            + average_delta
        )

    global_model.load_shared_state_dict(
        aggregated_shared_state,
        strict=True,
    )


def aggregate_expert_parameters(
    *,
    global_model: nn.Module,
    client_updates: Sequence[ClientUpdate],
    expert_aggregator: ExpertAggregator,
    round_idx: int,
) -> Tensor:
    """
    逐专家调用 aggregation/ 中的专家聚合算法。

    只有 route_count > 0 的客户端才被视为对该专家产生了
    有效更新。

    Args:
        expert_aggregator:
            假定遵循以下接口:

            new_state = expert_aggregator.aggregate(
                global_state=global_expert_state,
                updates=expert_client_updates,
                expert_idx=expert_idx,
                round_idx=round_idx,
            )

            返回新的完整专家 state_dict。

    Returns:
        expert_participant_counts，形状 [E]。
    """
    _validate_moe_model_interface(global_model)

    if round_idx < 0:
        raise ValueError("round_idx must be non-negative.")

    participant_counts = torch.zeros(
        global_model.num_experts,
        dtype=torch.long,
    )

    for expert_idx in range(global_model.num_experts):
        valid_updates: list[ExpertClientUpdate] = []

        for client_update in client_updates:
            route_count = int(
                client_update.route_counts[expert_idx].item()
            )

            if route_count <= 0:
                continue

            valid_updates.append(
                ExpertClientUpdate(
                    client_id=client_update.client_id,
                    delta=client_update.expert_deltas[
                        expert_idx
                    ],
                    route_count=route_count,
                    route_weight_sum=float(
                        client_update.route_weight_sums[
                            expert_idx
                        ].item()
                    ),
                    num_examples=client_update.num_examples,
                    metadata={
                        "num_processed_examples": (
                            client_update.num_processed_examples
                        ),
                        "expert_update_norm": float(
                            client_update.expert_update_norms[
                                expert_idx
                            ].item()
                        ),
                        "client_train_loss": (
                            client_update.train_loss
                        ),
                        "client_accuracy": (
                            client_update.accuracy
                        ),
                    },
                )
            )

        participant_counts[expert_idx] = len(valid_updates)

        # 本轮无人使用该专家时，保持服务器专家不变。
        if not valid_updates:
            continue

        global_expert_state = (
            global_model.get_expert_state_dict(
                expert_idx=expert_idx,
                clone=True,
                to_cpu=False,
            )
        )

        aggregated_expert_state = (
            expert_aggregator.aggregate(
                global_state=global_expert_state,
                updates=valid_updates,
                expert_idx=expert_idx,
                round_idx=round_idx,
            )
        )

        if not isinstance(aggregated_expert_state, Mapping):
            raise TypeError(
                "Expert aggregator must return a mapping "
                "representing the complete expert state_dict."
            )

        _assert_state_finite(
            aggregated_expert_state,
            name=(
                f"aggregated expert {expert_idx} state"
            ),
        )

        global_model.load_expert_state_dict(
            expert_idx=expert_idx,
            expert_state=aggregated_expert_state,
            strict=True,
        )

    return participant_counts


def aggregate_global_model(
    *,
    global_model: nn.Module,
    client_updates: Sequence[ClientUpdate],
    expert_aggregator: ExpertAggregator,
    round_idx: int,
) -> AggregationSummary:
    """
    完成一轮服务器聚合。

    顺序:
        1. 共享参数使用 uniform aggregation；
        2. 每个专家调用指定专家聚合算法。

    专家聚合器接收到的是客户端相对于本轮旧全局专家的增量。
    """
    if not client_updates:
        raise ValueError(
            "Cannot aggregate a global model without client updates."
        )

    aggregate_shared_parameters_uniform(
        global_model=global_model,
        client_updates=client_updates,
    )

    expert_participant_counts = (
        aggregate_expert_parameters(
            global_model=global_model,
            client_updates=client_updates,
            expert_aggregator=expert_aggregator,
            round_idx=round_idx,
        )
    )

    return AggregationSummary(
        num_client_updates=len(client_updates),
        expert_participant_counts=expert_participant_counts,
    )


@torch.no_grad()
def evaluate_model(
    *,
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | str,
    balance_loss_weight: float = 0.01,
    use_amp: bool = False,
) -> EvaluationMetrics:
    """
    使用与训练阶段相同的 Top-k 路由和概率加权方式评估模型。
    """
    device = torch.device(device)

    if balance_loss_weight < 0.0:
        raise ValueError(
            "balance_loss_weight must be non-negative."
        )

    _validate_moe_model_interface(model)

    was_training = model.training
    model.to(device)
    model.eval()

    amp_enabled = bool(use_amp and device.type == "cuda")

    total_loss_sum = 0.0
    classification_loss_sum = 0.0
    balance_loss_sum = 0.0
    total_correct = 0
    total_examples = 0

    route_counts = torch.zeros(
        model.num_experts,
        dtype=torch.long,
        device=device,
    )
    route_weight_sums = torch.zeros(
        model.num_experts,
        dtype=torch.float64,
        device=device,
    )

    try:
        for batch in data_loader:
            images, labels = _unpack_batch(batch)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            batch_size = int(labels.shape[0])

            if batch_size == 0:
                continue

            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                output = model(images)

                classification_loss = F.cross_entropy(
                    output.logits,
                    labels,
                )

                total_loss = (
                    classification_loss
                    + balance_loss_weight * output.balance_loss
                )

            _assert_finite_scalar(
                total_loss,
                name="evaluation total loss",
            )

            predictions = output.logits.argmax(dim=1)

            total_correct += int(
                predictions.eq(labels).sum().item()
            )
            total_examples += batch_size

            total_loss_sum += (
                float(total_loss.item()) * batch_size
            )
            classification_loss_sum += (
                float(classification_loss.item())
                * batch_size
            )
            balance_loss_sum += (
                float(output.balance_loss.item())
                * batch_size
            )

            route_counts += (
                output.route_counts.to(
                    device=device,
                    dtype=torch.long,
                )
            )
            route_weight_sums += (
                output.route_weight_sums.to(
                    device=device,
                    dtype=torch.float64,
                )
            )

    finally:
        model.train(was_training)

    if total_examples == 0:
        raise ValueError(
            "Evaluation data loader produced no valid examples."
        )

    total_assignments = int(route_counts.sum().item())

    if total_assignments > 0:
        route_distribution = (
            route_counts.to(torch.float64)
            / float(total_assignments)
        )
    else:
        route_distribution = torch.zeros(
            model.num_experts,
            dtype=torch.float64,
            device=device,
        )

    return EvaluationMetrics(
        loss=total_loss_sum / total_examples,
        classification_loss=(
            classification_loss_sum / total_examples
        ),
        balance_loss=(
            balance_loss_sum / total_examples
        ),
        accuracy=total_correct / total_examples,
        num_examples=total_examples,
        route_counts=route_counts.cpu(),
        route_weight_sums=route_weight_sums.cpu(),
        route_distribution=route_distribution.cpu(),
    )


def _resolve_num_examples(
    *,
    train_loader: DataLoader,
    explicit_num_examples: Optional[int],
) -> int:
    if explicit_num_examples is not None:
        if explicit_num_examples < 0:
            raise ValueError(
                "num_examples must be non-negative."
            )
        return int(explicit_num_examples)

    dataset = getattr(train_loader, "dataset", None)

    if dataset is None:
        raise ValueError(
            "Unable to infer client dataset size. "
            "Pass num_examples explicitly."
        )

    try:
        dataset_size = len(dataset)
    except TypeError as error:
        raise ValueError(
            "Client dataset does not implement __len__. "
            "Pass num_examples explicitly."
        ) from error

    if dataset_size < 0:
        raise ValueError(
            "Client dataset size must be non-negative."
        )

    return int(dataset_size)


def _unpack_batch(batch) -> tuple[Tensor, Tensor]:
    """
    当前版本约定 DataLoader 返回 (images, labels)。

    对额外附加字段的 tuple/list，仅读取前两个元素。
    """
    if not isinstance(batch, (tuple, list)):
        raise TypeError(
            "Each data loader batch must be a tuple or list "
            "whose first two elements are images and labels."
        )

    if len(batch) < 2:
        raise ValueError(
            "Each batch must contain at least images and labels."
        )

    images, labels = batch[0], batch[1]

    if not isinstance(images, Tensor):
        raise TypeError("Batch images must be a torch.Tensor.")

    if not isinstance(labels, Tensor):
        raise TypeError("Batch labels must be a torch.Tensor.")

    return images, labels


def _compute_state_delta(
    *,
    initial_state: Mapping[str, Tensor],
    final_state: Mapping[str, Tensor],
) -> StateDict:
    """
    计算 final - initial。

    对非浮点 buffer 返回零增量，使服务器保持原值。
    当前 ResNet18-GN 不包含 BatchNorm running statistics，
    但此处理可以避免未来 backbone 的整数 buffer 被错误平均。
    """
    initial_keys = set(initial_state.keys())
    final_keys = set(final_state.keys())

    if initial_keys != final_keys:
        missing = initial_keys - final_keys
        extra = final_keys - initial_keys

        raise KeyError(
            "State keys mismatch while computing delta. "
            f"Missing: {sorted(missing)}; "
            f"Extra: {sorted(extra)}."
        )

    delta: StateDict = {}

    for name, initial_tensor in initial_state.items():
        final_tensor = final_state[name]

        if initial_tensor.shape != final_tensor.shape:
            raise ValueError(
                f"State shape mismatch for {name}: "
                f"{tuple(initial_tensor.shape)} vs "
                f"{tuple(final_tensor.shape)}."
            )

        if (
            torch.is_floating_point(initial_tensor)
            or torch.is_complex(initial_tensor)
        ):
            delta[name] = (
                final_tensor.to(initial_tensor.dtype)
                - initial_tensor
            )
        else:
            delta[name] = torch.zeros_like(initial_tensor)

    return delta


def _state_l2_norm(state: Mapping[str, Tensor]) -> float:
    squared_sum = 0.0

    for tensor in state.values():
        if not (
            torch.is_floating_point(tensor)
            or torch.is_complex(tensor)
        ):
            continue

        tensor_float = tensor.detach().to(
            dtype=torch.float64,
            device="cpu",
        )
        squared_sum += float(
            torch.sum(tensor_float * tensor_float).item()
        )

    return math.sqrt(squared_sum)


def _assert_finite_scalar(
    value: Tensor,
    *,
    name: str,
) -> None:
    if value.numel() != 1:
        raise ValueError(
            f"{name} must be a scalar tensor."
        )

    if not bool(torch.isfinite(value.detach()).item()):
        raise FloatingPointError(
            f"Detected NaN or Inf in {name}."
        )


def _assert_state_finite(
    state: Mapping[str, Tensor],
    *,
    name: str,
) -> None:
    for parameter_name, tensor in state.items():
        if not (
            torch.is_floating_point(tensor)
            or torch.is_complex(tensor)
        ):
            continue

        if not bool(torch.isfinite(tensor).all().item()):
            raise FloatingPointError(
                f"Detected NaN or Inf in {name}: "
                f"{parameter_name}."
            )


def _validate_moe_model_interface(
    model: nn.Module,
) -> None:
    required_attributes = [
        "num_experts",
        "get_shared_state_dict",
        "get_all_expert_state_dicts",
        "get_expert_state_dict",
        "load_shared_state_dict",
        "load_expert_state_dict",
    ]

    missing = [
        name
        for name in required_attributes
        if not hasattr(model, name)
    ]

    if missing:
        raise TypeError(
            "The model does not implement the required Sparse MoE "
            f"interface. Missing: {missing}."
        )

    if int(model.num_experts) <= 1:
        raise ValueError(
            "model.num_experts must be greater than 1."
        )

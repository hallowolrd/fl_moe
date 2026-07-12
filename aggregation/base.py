from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import torch
from torch import Tensor


StateDict = Dict[str, Tensor]


__all__ = [
    "ExpertAggregator",
    "ExpertClientUpdate",
    "StateDict",
    "clone_state_dict",
    "validate_expert_updates",
    "validate_state_dict_compatibility",
]


@dataclass
class ExpertClientUpdate:
    """
    某个客户端对某一个专家产生的本地参数更新。
    """

    client_id: int
    delta: StateDict
    route_count: int
    route_weight_sum: float
    num_examples: int
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.client_id, bool) or not isinstance(
            self.client_id,
            int,
        ):
            raise TypeError("client_id must be an integer.")

        if self.client_id < 0:
            raise ValueError("client_id must be non-negative.")

        if not isinstance(self.delta, dict):
            raise TypeError("delta must be a dict[str, Tensor].")

        if not self.delta:
            raise ValueError("delta must not be empty.")

        for name, tensor in self.delta.items():
            if not isinstance(name, str):
                raise TypeError(
                    "Every delta key must be a string."
                )

            if not name:
                raise ValueError(
                    "Delta parameter names must not be empty."
                )

            if not isinstance(tensor, Tensor):
                raise TypeError(
                    f"Delta value for {name!r} must be a Tensor."
                )

        if isinstance(self.route_count, bool) or not isinstance(
            self.route_count,
            int,
        ):
            raise TypeError("route_count must be an integer.")

        if self.route_count <= 0:
            raise ValueError(
                "route_count must be greater than 0."
            )

        if isinstance(self.route_weight_sum, bool) or not isinstance(
            self.route_weight_sum,
            (int, float),
        ):
            raise TypeError(
                "route_weight_sum must be a real number."
            )

        self.route_weight_sum = float(self.route_weight_sum)

        if not math.isfinite(self.route_weight_sum):
            raise ValueError(
                "route_weight_sum must be finite."
            )

        if self.route_weight_sum < 0.0:
            raise ValueError(
                "route_weight_sum must be non-negative."
            )

        if isinstance(self.num_examples, bool) or not isinstance(
            self.num_examples,
            int,
        ):
            raise TypeError("num_examples must be an integer.")

        if self.num_examples < 0:
            raise ValueError(
                "num_examples must be non-negative."
            )

        if (
            self.metadata is not None
            and not isinstance(self.metadata, dict)
        ):
            raise TypeError(
                "metadata must be dict[str, Any] or None."
            )


class ExpertAggregator(ABC):
    """
    所有专家参数聚合算法的统一抽象基类。
    """

    @abstractmethod
    def aggregate(
        self,
        *,
        global_state: Mapping[str, Tensor],
        updates: Sequence[ExpertClientUpdate],
        expert_idx: int,
        round_idx: int,
    ) -> StateDict:
        """
        聚合某一个专家的客户端更新，并返回新的完整专家参数。
        """
        raise NotImplementedError


def clone_state_dict(
    state: Mapping[str, Tensor],
    *,
    to_cpu: bool = False,
) -> StateDict:
    """
    对 state_dict 执行 detach + clone。
    """
    _validate_state_mapping(
        state,
        state_name="state",
        allow_empty=False,
    )

    cloned: StateDict = {}

    for name, tensor in state.items():
        copied = tensor.detach().clone()

        if to_cpu:
            copied = copied.cpu()

        cloned[name] = copied

    return cloned


def validate_state_dict_compatibility(
    reference_state: Mapping[str, Tensor],
    candidate_state: Mapping[str, Tensor],
    *,
    reference_name: str = "reference_state",
    candidate_name: str = "candidate_state",
) -> None:
    """
    严格检查两个 state_dict 的 key、shape 和 dtype 是否一致。
    """
    _validate_state_mapping(
        reference_state,
        state_name=reference_name,
        allow_empty=False,
    )
    _validate_state_mapping(
        candidate_state,
        state_name=candidate_name,
        allow_empty=False,
    )

    reference_keys = set(reference_state.keys())
    candidate_keys = set(candidate_state.keys())

    if reference_keys != candidate_keys:
        missing_keys = reference_keys - candidate_keys
        extra_keys = candidate_keys - reference_keys

        raise KeyError(
            f"{candidate_name} keys are incompatible with "
            f"{reference_name}. "
            f"Missing: {sorted(missing_keys)}; "
            f"Extra: {sorted(extra_keys)}."
        )

    for name in reference_state:
        reference_tensor = reference_state[name]
        candidate_tensor = candidate_state[name]

        if reference_tensor.shape != candidate_tensor.shape:
            raise ValueError(
                f"Shape mismatch for parameter {name!r}: "
                f"{reference_name} has "
                f"{tuple(reference_tensor.shape)}, "
                f"but {candidate_name} has "
                f"{tuple(candidate_tensor.shape)}."
            )

        if reference_tensor.dtype != candidate_tensor.dtype:
            raise TypeError(
                f"Dtype mismatch for parameter {name!r}: "
                f"{reference_name} has {reference_tensor.dtype}, "
                f"but {candidate_name} has "
                f"{candidate_tensor.dtype}."
            )


def validate_expert_updates(
    global_state: Mapping[str, Tensor],
    updates: Sequence[ExpertClientUpdate],
    *,
    expert_idx: int,
    round_idx: int,
    allow_empty: bool = True,
) -> None:
    """
    验证专家聚合器输入。
    """
    if isinstance(expert_idx, bool) or not isinstance(
        expert_idx,
        int,
    ):
        raise TypeError("expert_idx must be an integer.")

    if expert_idx < 0:
        raise ValueError("expert_idx must be non-negative.")

    if isinstance(round_idx, bool) or not isinstance(
        round_idx,
        int,
    ):
        raise TypeError("round_idx must be an integer.")

    if round_idx < 0:
        raise ValueError("round_idx must be non-negative.")

    _validate_state_mapping(
        global_state,
        state_name="global_state",
        allow_empty=False,
    )

    if isinstance(updates, (str, bytes)):
        raise TypeError(
            "updates must be a sequence of ExpertClientUpdate."
        )

    if not isinstance(updates, Sequence):
        raise TypeError(
            "updates must be a sequence of ExpertClientUpdate."
        )

    if not updates and not allow_empty:
        raise ValueError("updates must not be empty.")

    seen_client_ids: set[int] = set()

    for position, update in enumerate(updates):
        if not isinstance(update, ExpertClientUpdate):
            raise TypeError(
                f"updates[{position}] must be an "
                "ExpertClientUpdate."
            )

        if update.client_id in seen_client_ids:
            raise ValueError(
                "Duplicate client update detected for "
                f"client_id={update.client_id}."
            )

        seen_client_ids.add(update.client_id)

        validate_state_dict_compatibility(
            reference_state=global_state,
            candidate_state=update.delta,
            reference_name="global_state",
            candidate_name=f"updates[{position}].delta",
        )


def _validate_state_mapping(
    state: Mapping[str, Tensor],
    *,
    state_name: str,
    allow_empty: bool,
) -> None:
    """
    检查 Mapping、键类型、Tensor 类型以及有限性。
    """
    if not isinstance(state, Mapping):
        raise TypeError(
            f"{state_name} must be a Mapping[str, Tensor]."
        )

    if not state and not allow_empty:
        raise ValueError(f"{state_name} must not be empty.")

    for name, tensor in state.items():
        if not isinstance(name, str):
            raise TypeError(
                f"Every key in {state_name} must be a string."
            )

        if not name:
            raise ValueError(
                f"Keys in {state_name} must not be empty."
            )

        if not isinstance(tensor, Tensor):
            raise TypeError(
                f"{state_name}[{name!r}] must be a Tensor."
            )

        if (
            torch.is_floating_point(tensor)
            or torch.is_complex(tensor)
        ):
            if not bool(torch.isfinite(tensor).all().item()):
                raise FloatingPointError(
                    f"Detected NaN or Inf in "
                    f"{state_name}[{name!r}]."
                )

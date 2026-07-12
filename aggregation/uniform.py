from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor

from .base import (
    ExpertAggregator,
    ExpertClientUpdate,
    StateDict,
    clone_state_dict,
    validate_expert_updates,
)


__all__ = [
    "UniformExpertAggregator",
]


class UniformExpertAggregator(ExpertAggregator):
    """
    专家参数等权平均聚合器。

    对所有实际更新了当前专家的客户端参数增量执行直接平均：

        average_delta =
            (1 / M) * sum_i delta_i

        new_global_state =
            global_state + average_delta

    其中：
        M 为当前专家的有效客户端数量；
        delta_i 为客户端 i 相对于本轮全局专家参数的增量。

    本聚合器不使用：
        - route_count；
        - route_weight_sum；
        - num_examples；
        - metadata。

    这些字段仍保留在 ExpertClientUpdate 中，供其他专家聚合算法使用。

    当 updates 为空时，返回 global_state 的独立副本。
    """

    def aggregate(
        self,
        *,
        global_state: Mapping[str, Tensor],
        updates: Sequence[ExpertClientUpdate],
        expert_idx: int,
        round_idx: int,
    ) -> StateDict:
        """
        对某个专家的客户端更新进行等权平均。

        Args:
            global_state:
                当前通信轮开始时的全局专家完整参数。

            updates:
                实际更新了该专家的客户端更新序列。

            expert_idx:
                当前专家编号。

            round_idx:
                当前通信轮编号，从 0 开始。

        Returns:
            聚合后的完整专家 state_dict。
        """
        validate_expert_updates(
            global_state=global_state,
            updates=updates,
            expert_idx=expert_idx,
            round_idx=round_idx,
            allow_empty=True,
        )

        if not updates:
            return clone_state_dict(global_state)

        num_updates = len(updates)
        aggregated_state: StateDict = {}

        for parameter_name, global_tensor in global_state.items():
            # 当前专家仅包含浮点参数。
            # 为兼容未来可能出现的整数 buffer，非浮点状态保持不变。
            if not (
                torch.is_floating_point(global_tensor)
                or torch.is_complex(global_tensor)
            ):
                aggregated_state[parameter_name] = (
                    global_tensor.detach().clone()
                )
                continue

            average_delta = torch.zeros_like(global_tensor)

            for update in updates:
                client_delta = update.delta[
                    parameter_name
                ].to(
                    device=global_tensor.device,
                    dtype=global_tensor.dtype,
                )

                average_delta.add_(client_delta)

            average_delta.div_(float(num_updates))

            aggregated_state[parameter_name] = (
                global_tensor.detach().clone()
                + average_delta
            )

        return aggregated_state

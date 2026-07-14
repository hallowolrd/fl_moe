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
    专家参数零更新等权聚合器。

    对实际更新了当前专家的客户端参数增量求和，
    但使用本轮全部有效客户端数量作为分母：

        average_delta =
            (1 / num_round_clients) * sum_i delta_i

        new_global_state =
            global_state + average_delta

    其中：
        num_round_clients 为本轮全部有效客户端数量；
        delta_i 为客户端 i 相对于本轮全局专家参数的增量；
        未激活当前专家的客户端隐式贡献零增量。

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
        num_round_clients: int,
        expert_idx: int,
        round_idx: int,
    ) -> StateDict:
        """
        对某个专家的客户端更新执行零更新等权平均。

        Args:
            global_state:
                当前通信轮开始时的全局专家完整参数。
            updates:
                实际更新了该专家的客户端更新序列。
            num_round_clients:
                本轮实际完成本地训练并返回 ClientUpdate
                的有效客户端总数。未激活当前专家的客户端
                不出现在 updates 中，但仍计入聚合分母。
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

        if (
            isinstance(num_round_clients, bool)
            or not isinstance(num_round_clients, int)
        ):
            raise TypeError(
                "num_round_clients must be an integer."
            )

        if num_round_clients <= 0:
            raise ValueError(
                "num_round_clients must be greater than 0."
            )

        if len(updates) > num_round_clients:
            raise ValueError(
                "The number of active expert updates cannot "
                "exceed num_round_clients."
            )

        if not updates:
            return clone_state_dict(global_state)

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

            # 分子只包含实际激活当前专家的客户端增量；
            # 分母使用本轮全部有效客户端数量。
            # 未激活客户端因此等价于贡献零增量。
            average_delta.div_(float(num_round_clients))

            aggregated_state[parameter_name] = (
                global_tensor.detach().clone()
                + average_delta
            )

        return aggregated_state

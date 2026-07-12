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


__all__ = ["ActivationWeightedExpertAggregator"]


class ActivationWeightedExpertAggregator(ExpertAggregator):
    """
    按专家激活次数对客户端专家参数增量进行加权聚合。

    对当前专家 e，客户端 i 的聚合权重为：

        weight_i = route_count_i / sum_j(route_count_j)

    聚合后的完整专家参数为：

        new_global_state
            = global_state
            + sum_i(weight_i * delta_i)

    其中：
        - route_count_i 表示客户端 i 在本地训练期间，
          当前专家被 Top-k Router 选中的累计次数；
        - delta_i 表示客户端 i 相对于本轮全局专家参数的增量。

    本聚合器：
        - 只使用 route_count 计算权重；
        - 不使用 route_weight_sum；
        - 不使用 num_examples；
        - 不使用 metadata；
        - 不维护任何内部状态；
        - 空 updates 时返回 global_state 的独立副本。
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
        聚合某一个专家的客户端更新。

        Args:
            global_state:
                当前通信轮开始时的全局专家完整状态。

            updates:
                当前专家的有效客户端参数增量。

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

        total_route_count = sum(
            update.route_count
            for update in updates
        )

        # ExpertClientUpdate 已要求 route_count > 0。
        # 这里继续保留防御性检查。
        if total_route_count <= 0:
            raise ValueError(
                "The total route count must be greater than 0."
            )

        aggregated_state: StateDict = {}

        for parameter_name, global_tensor in global_state.items():
            # 整数、布尔等非浮点状态不参与参数增量聚合，
            # 直接保持服务器当前全局值。
            if not (
                torch.is_floating_point(global_tensor)
                or torch.is_complex(global_tensor)
            ):
                aggregated_state[parameter_name] = (
                    global_tensor.detach().clone()
                )
                continue

            weighted_delta = torch.zeros_like(global_tensor)

            for update in updates:
                weight = (
                    update.route_count
                    / total_route_count
                )

                client_delta = update.delta[
                    parameter_name
                ].to(
                    device=global_tensor.device,
                    dtype=global_tensor.dtype,
                )

                weighted_delta.add_(
                    client_delta,
                    alpha=float(weight),
                )

            aggregated_state[parameter_name] = (
                global_tensor.detach().clone()
                + weighted_delta
            )

        return aggregated_state

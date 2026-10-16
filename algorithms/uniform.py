from __future__ import annotations

"""Uniform expert aggregation experiment.

All shared experiment behavior is owned by base.py.  This file contains only
the expert-parameter aggregation policy and its executable entrypoint.
"""

# Import base before importing torch so base.py can prepare deterministic CUDA
# environment variables before PyTorch initializes CUDA/cuBLAS.
import base
from typing import Any, Dict, Sequence


ALGORITHM_NAME = "uniform"
Aggregator = base.Aggregator
ClientUpdate = base.ClientUpdate
build_uniform_weights = base.build_uniform_weights

EMBEDDED_METHOD_CONFIG = {
    "agg": {
        "non_expert": {"method": "uniform"},
        "expert": {"method": ALGORITHM_NAME},
    },
}


class UniformExpertAggregator(Aggregator):
    """
    直接平均聚合器。

    权重规则：
        每个参与客户端权重相同。

    公式：
        w_i = 1 / K

    其中：
        K 是本轮参与聚合的客户端数量。

    聚合公式：
        theta_new = theta_global + sum_i w_i * delta_i
    """

    @property
    def method_name(self) -> str:
        """返回当前聚合方法名称。"""
        return ALGORITHM_NAME

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """
        计算直接平均权重。

        输入：
            client_updates:
                本轮参与训练的客户端更新。

        输出：
            {
                client_id: weight
            }

        示例：
            如果本轮有 4 个客户端参与：
                client 0 -> 0.25
                client 1 -> 0.25
                client 2 -> 0.25
                client 3 -> 0.25
        """
        return build_uniform_weights(client_updates)


def build_expert_aggregator(cfg: Any) -> base.Aggregator:
    """Build the expert-only uniform aggregator injected into base.py."""
    return UniformExpertAggregator(
        cfg=cfg,
        param_group_name="expert",
    )


def main() -> int:
    return base.main(
        expert_aggregator_builder=build_expert_aggregator,
        embedded_method_config=EMBEDDED_METHOD_CONFIG,
        expert_method_name=ALGORITHM_NAME,
    )


if __name__ == "__main__":
    raise SystemExit(main())

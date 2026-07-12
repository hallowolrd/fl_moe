from __future__ import annotations

from typing import Any, Dict, Type

from .base import ExpertAggregator, ExpertClientUpdate
from .uniform import UniformExpertAggregator
from .activation_weighted import ActivationWeightedExpertAggregator

EXPERT_AGGREGATOR_REGISTRY: Dict[
    str,
    Type[ExpertAggregator],
] = {
    "uniform": UniformExpertAggregator,
    "activation_weighted": ActivationWeightedExpertAggregator,
}


def build_expert_aggregator(
    name: str,
    **kwargs: Any,
) -> ExpertAggregator:
    """
    根据名称创建专家聚合器。

    当前支持:
        uniform
        activation_weighted

    注意:
        聚合器名称严格匹配，不自动转换大小写，
        也不移除首尾空格。

    Args:
        name:
            聚合器注册名称。
        **kwargs:
            传递给聚合器构造函数的额外参数。

    Returns:
        ExpertAggregator 实例。

    Raises:
        TypeError:
            name 不是字符串。
        ValueError:
            name 未注册。
    """
    if not isinstance(name, str):
        raise TypeError(
            "Expert aggregator name must be a string."
        )

    if name not in EXPERT_AGGREGATOR_REGISTRY:
        available = ", ".join(
            sorted(EXPERT_AGGREGATOR_REGISTRY.keys())
        )

        raise ValueError(
            f"Unknown expert aggregator: {name!r}. "
            f"Available aggregators: {available}."
        )

    aggregator_class = EXPERT_AGGREGATOR_REGISTRY[name]

    return aggregator_class(**kwargs)


__all__ = [
    "EXPERT_AGGREGATOR_REGISTRY",
    "ExpertAggregator",
    "ExpertClientUpdate",
    "UniformExpertAggregator",
    "build_expert_aggregator",
    "ActivationWeightedExpertAggregator"
]

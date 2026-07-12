from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


__all__ = [
    "ClassificationExpert",
    "MoEOutput",
    "RouterOutput",
    "SparseMoEClassifier",
    "TopKRouter",
    "build_sparse_moe",
]


@dataclass
class RouterOutput:
    """
    Top-k Router 输出。

    Attributes:
        probabilities:
            所有专家的完整 softmax 概率，形状 [B, E]。
        topk_probabilities:
            每个样本被选中的 Top-k 专家概率，形状 [B, k]。
        topk_indices:
            每个样本被选中的 Top-k 专家编号，形状 [B, k]。
    """

    probabilities: Tensor
    topk_probabilities: Tensor
    topk_indices: Tensor


@dataclass
class MoEOutput:
    """
    Sparse MoE 模型输出。

    Attributes:
        logits:
            最终分类 logits，形状 [B, C]。
        router_probabilities:
            所有专家的完整路由概率，形状 [B, E]。
        topk_probabilities:
            被选中的 Top-k 专家概率，形状 [B, k]。
        topk_indices:
            被选中的 Top-k 专家编号，形状 [B, k]。
        route_counts:
            当前 batch 中每个专家被选中的次数，形状 [E]。
            Top-k 时总和为 B * k。
        route_weight_sums:
            当前 batch 中每个专家获得的路由概率总和，形状 [E]。
        balance_loss:
            Switch-style 负载均衡辅助损失，标量。
    """

    logits: Tensor
    router_probabilities: Tensor
    topk_probabilities: Tensor
    topk_indices: Tensor
    route_counts: Tensor
    route_weight_sums: Tensor
    balance_loss: Tensor


class TopKRouter(nn.Module):
    """
    样本级 Top-k Router。

    路由过程:
        features -> Linear -> Softmax -> Top-k

    注意:
        本实现保留原始 softmax 概率，不对 Top-k 概率重新归一化。
        因此 top_k=1 时，选中专家概率仍会缩放专家输出，
        分类损失可以通过该概率向 Router 反向传播。
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 1,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be greater than 0.")

        if num_experts <= 1:
            raise ValueError("num_experts must be greater than 1.")

        self.input_dim = int(input_dim)
        self.num_experts = int(num_experts)
        self.top_k = self._validate_top_k(top_k)

        self.gate = nn.Linear(
            in_features=self.input_dim,
            out_features=self.num_experts,
            bias=True,
        )

    def _validate_top_k(self, top_k: int) -> int:
        top_k = int(top_k)

        if not 1 <= top_k <= self.num_experts:
            raise ValueError(
                "top_k must satisfy "
                f"1 <= top_k <= {self.num_experts}, "
                f"but received {top_k}."
            )

        return top_k

    def set_top_k(self, top_k: int) -> None:
        """
        修改 Top-k。

        建议同一组完整实验从训练开始到结束固定 top_k。
        """
        self.top_k = self._validate_top_k(top_k)

    def forward(self, features: Tensor) -> RouterOutput:
        if features.ndim != 2:
            raise ValueError(
                "Router input must have shape [B, D], "
                f"but received {tuple(features.shape)}."
            )

        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Router expected feature dimension {self.input_dim}, "
                f"but received {features.shape[1]}."
            )

        router_logits = self.gate(features)
        probabilities = F.softmax(router_logits, dim=-1)

        topk_probabilities, topk_indices = torch.topk(
            probabilities,
            k=self.top_k,
            dim=-1,
            largest=True,
            sorted=True,
        )

        return RouterOutput(
            probabilities=probabilities,
            topk_probabilities=topk_probabilities,
            topk_indices=topk_indices,
        )


class ClassificationExpert(nn.Module):
    """
    完整分类专家。

    结构:
        Linear(input_dim, hidden_dim)
        ReLU
        Linear(hidden_dim, num_classes)

    第二个 Linear 是该专家自己的分类头，因此属于专家参数。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be greater than 0.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than 0.")

        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)

        self.fc1 = nn.Linear(
            in_features=self.input_dim,
            out_features=self.hidden_dim,
            bias=True,
        )
        self.activation = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.num_classes,
            bias=True,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        为 ReLU MLP 使用 Kaiming 初始化。
        """
        nn.init.kaiming_normal_(
            self.fc1.weight,
            mode="fan_in",
            nonlinearity="relu",
        )
        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2:
            raise ValueError(
                "Expert input must have shape [N, D], "
                f"but received {tuple(features.shape)}."
            )

        hidden = self.fc1(features)
        hidden = self.activation(hidden)
        logits = self.fc2(hidden)

        return logits


class SparseMoEClassifier(nn.Module):
    """
    可替换 Backbone 的样本级 Top-k Sparse MoE 分类模型。

    Backbone 约束:
        1. 必须是 nn.Module。
        2. 必须提供整数属性 `out_dim`。
        3. forward(images) 必须返回 [B, out_dim]。

    模型结构:
        backbone
        -> feature_adapter
        -> feature_norm
        -> Top-k Router
        -> selected Experts
        -> 按原始 Router 概率加权求和 logits

    参数边界:
        共享参数:
            backbone.*
            feature_adapter.*
            feature_norm.*
            router.*

        专家参数:
            experts.0.*
            experts.1.*
            ...

    其中每个专家的 fc2 是专家独立分类头。
    """

    EXPERT_PREFIX = "experts."

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        num_experts: int = 4,
        top_k: int = 1,
        moe_dim: int = 512,
        expert_hidden_dim: int = 1024,
    ) -> None:
        super().__init__()

        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be an nn.Module.")

        if not hasattr(backbone, "out_dim"):
            raise ValueError(
                "backbone must provide an integer `out_dim` attribute."
            )

        backbone_out_dim = int(getattr(backbone, "out_dim"))

        if backbone_out_dim <= 0:
            raise ValueError("backbone.out_dim must be greater than 0.")

        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1.")

        if num_experts <= 1:
            raise ValueError("num_experts must be greater than 1.")

        if moe_dim <= 0:
            raise ValueError("moe_dim must be greater than 0.")

        if expert_hidden_dim <= 0:
            raise ValueError(
                "expert_hidden_dim must be greater than 0."
            )

        self.num_classes = int(num_classes)
        self.num_experts = int(num_experts)
        self.moe_dim = int(moe_dim)
        self.expert_hidden_dim = int(expert_hidden_dim)
        self.backbone_out_dim = backbone_out_dim

        # ====================================================
        # 共享模块
        # ====================================================

        self.backbone = backbone

        if self.backbone_out_dim == self.moe_dim:
            self.feature_adapter = nn.Identity()
        else:
            self.feature_adapter = nn.Linear(
                in_features=self.backbone_out_dim,
                out_features=self.moe_dim,
                bias=True,
            )

        self.feature_norm = nn.LayerNorm(self.moe_dim)

        self.router = TopKRouter(
            input_dim=self.moe_dim,
            num_experts=self.num_experts,
            top_k=top_k,
        )

        # ====================================================
        # 专家模块
        # ====================================================

        self.experts = nn.ModuleList(
            [
                ClassificationExpert(
                    input_dim=self.moe_dim,
                    hidden_dim=self.expert_hidden_dim,
                    num_classes=self.num_classes,
                )
                for _ in range(self.num_experts)
            ]
        )

        self._initialize_shared_layers()

    def _initialize_shared_layers(self) -> None:
        """
        初始化 backbone 之外的共享层。
        Backbone 由其自身负责初始化。
        """
        if isinstance(self.feature_adapter, nn.Linear):
            nn.init.xavier_uniform_(self.feature_adapter.weight)
            nn.init.zeros_(self.feature_adapter.bias)

        nn.init.ones_(self.feature_norm.weight)
        nn.init.zeros_(self.feature_norm.bias)

        nn.init.xavier_uniform_(self.router.gate.weight)
        nn.init.zeros_(self.router.gate.bias)

    @property
    def top_k(self) -> int:
        return self.router.top_k

    def set_top_k(self, top_k: int) -> None:
        self.router.set_top_k(top_k)

    def extract_features(self, images: Tensor) -> Tensor:
        """
        提取并统一 backbone 特征。

        Returns:
            [B, moe_dim]
        """
        features = self.backbone(images)

        if features.ndim != 2:
            raise RuntimeError(
                "Backbone must return a 2D tensor [B, D], "
                f"but received {tuple(features.shape)}."
            )

        if features.shape[1] != self.backbone_out_dim:
            raise RuntimeError(
                "Backbone output dimension does not match backbone.out_dim. "
                f"Expected {self.backbone_out_dim}, "
                f"but received {features.shape[1]}."
            )

        features = self.feature_adapter(features)
        features = self.feature_norm(features)

        return features

    def _dispatch_to_experts(
        self,
        features: Tensor,
        topk_probabilities: Tensor,
        topk_indices: Tensor,
    ) -> Tensor:
        """
        只执行被 Top-k 选中的专家，并将加权 logits 累加回原 batch。

        最终形式:
            logits(x) = sum_{e in TopK(x)} p_e(x) * E_e(x)
        """
        batch_size = features.shape[0]

        final_logits = torch.zeros(
            batch_size,
            self.num_classes,
            device=features.device,
            dtype=features.dtype,
        )

        for expert_idx, expert in enumerate(self.experts):
            selected_mask = topk_indices.eq(expert_idx)

            # 每一行是 [sample_index, topk_rank_index]。
            selected_positions = torch.nonzero(
                selected_mask,
                as_tuple=False,
            )

            if selected_positions.numel() == 0:
                continue

            sample_indices = selected_positions[:, 0]
            rank_indices = selected_positions[:, 1]

            selected_features = features.index_select(
                dim=0,
                index=sample_indices,
            )

            expert_logits = expert(selected_features)

            selected_weights = topk_probabilities[
                sample_indices,
                rank_indices,
            ].unsqueeze(dim=-1)

            weighted_logits = selected_weights * expert_logits

            # 使用非原地 index_add，避免复杂计算图中的原地修改问题。
            final_logits = final_logits.index_add(
                dim=0,
                index=sample_indices,
                source=weighted_logits,
            )

        return final_logits

    def _compute_route_statistics(
        self,
        router_probabilities: Tensor,
        topk_probabilities: Tensor,
        topk_indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        计算专家使用统计和负载均衡损失。

        Top-k 硬分配比例:
            f_e = count(e) / (B * k)

        平均软概率:
            P_e = mean_batch p_e(x)

        负载均衡损失:
            L_balance = E * sum_e f_e * P_e
        """
        batch_size = router_probabilities.shape[0]
        total_assignments = batch_size * self.top_k

        flattened_indices = topk_indices.reshape(-1)
        flattened_probabilities = topk_probabilities.reshape(-1)

        route_counts = torch.bincount(
            flattened_indices,
            minlength=self.num_experts,
        )

        route_weight_sums = torch.zeros(
            self.num_experts,
            device=router_probabilities.device,
            dtype=router_probabilities.dtype,
        ).index_add(
            dim=0,
            index=flattened_indices,
            source=flattened_probabilities,
        )

        hard_fraction = (
            route_counts.to(router_probabilities.dtype)
            / float(total_assignments)
        ).detach()

        mean_router_probability = router_probabilities.mean(dim=0)

        balance_loss = self.num_experts * torch.sum(
            hard_fraction * mean_router_probability
        )

        return route_counts, route_weight_sums, balance_loss

    def forward(self, images: Tensor) -> MoEOutput:
        if images.ndim != 4:
            raise ValueError(
                "Input images must have shape [B, C, H, W], "
                f"but received {tuple(images.shape)}."
            )

        if images.shape[0] == 0:
            raise ValueError("Input batch must not be empty.")

        features = self.extract_features(images)
        router_output = self.router(features)

        final_logits = self._dispatch_to_experts(
            features=features,
            topk_probabilities=router_output.topk_probabilities,
            topk_indices=router_output.topk_indices,
        )

        (
            route_counts,
            route_weight_sums,
            balance_loss,
        ) = self._compute_route_statistics(
            router_probabilities=router_output.probabilities,
            topk_probabilities=router_output.topk_probabilities,
            topk_indices=router_output.topk_indices,
        )

        return MoEOutput(
            logits=final_logits,
            router_probabilities=router_output.probabilities,
            topk_probabilities=router_output.topk_probabilities,
            topk_indices=router_output.topk_indices,
            route_counts=route_counts,
            route_weight_sums=route_weight_sums,
            balance_loss=balance_loss,
        )

    # ========================================================
    # 参数分组
    # ========================================================

    @classmethod
    def is_expert_key(cls, name: str) -> bool:
        """
        判断参数或 state_dict key 是否属于专家模块。
        """
        return name.startswith(cls.EXPERT_PREFIX)

    def shared_named_parameters(
        self,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """
        遍历全部共享可训练参数。
        """
        for name, parameter in self.named_parameters():
            if not self.is_expert_key(name):
                yield name, parameter

    def shared_parameters(self) -> Iterator[nn.Parameter]:
        for _, parameter in self.shared_named_parameters():
            yield parameter

    def expert_named_parameters(
        self,
        expert_idx: Optional[int] = None,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """
        Args:
            expert_idx:
                None 时返回全部专家参数；
                指定编号时只返回该专家参数。
        """
        if expert_idx is None:
            for name, parameter in self.named_parameters():
                if self.is_expert_key(name):
                    yield name, parameter
            return

        self._check_expert_index(expert_idx)
        prefix = f"experts.{expert_idx}."

        for name, parameter in self.named_parameters():
            if name.startswith(prefix):
                yield name, parameter

    def expert_parameters(
        self,
        expert_idx: Optional[int] = None,
    ) -> Iterator[nn.Parameter]:
        for _, parameter in self.expert_named_parameters(expert_idx):
            yield parameter

    def parameter_groups(self) -> Dict[str, list[nn.Parameter]]:
        """
        返回适合优化器使用的参数分组。
        """
        return {
            "shared": list(self.shared_parameters()),
            "experts": list(self.expert_parameters()),
        }

    def validate_parameter_partition(self) -> None:
        """
        检查共享参数和专家参数是否完整且互斥。

        建议在 train.py 创建模型后调用一次。
        """
        all_parameter_ids = {
            id(parameter)
            for parameter in self.parameters()
        }
        shared_parameter_ids = {
            id(parameter)
            for parameter in self.shared_parameters()
        }
        expert_parameter_ids = {
            id(parameter)
            for parameter in self.expert_parameters()
        }

        overlap = shared_parameter_ids & expert_parameter_ids
        covered = shared_parameter_ids | expert_parameter_ids

        if overlap:
            raise RuntimeError(
                "Shared and expert parameter groups overlap."
            )

        if covered != all_parameter_ids:
            raise RuntimeError(
                "Shared and expert parameter groups do not cover "
                "all model parameters."
            )

    # ========================================================
    # 面向联邦学习的 state_dict 接口
    # ========================================================

    def get_shared_state_dict(
        self,
        clone: bool = True,
        to_cpu: bool = False,
    ) -> Dict[str, Tensor]:
        """
        获取全部共享参数和共享 buffer。

        返回 key 保留完整模型路径，例如:
            backbone.layer1.0.conv1.weight
            feature_norm.weight
            router.gate.weight
        """
        result: Dict[str, Tensor] = {}

        for name, value in self.state_dict().items():
            if self.is_expert_key(name):
                continue

            tensor = value.detach()

            if clone:
                tensor = tensor.clone()

            if to_cpu:
                tensor = tensor.cpu()

            result[name] = tensor

        return result

    def get_expert_state_dict(
        self,
        expert_idx: int,
        clone: bool = True,
        to_cpu: bool = False,
    ) -> Dict[str, Tensor]:
        """
        获取指定专家的 state_dict。

        返回 key 为专家内部相对名称:
            fc1.weight
            fc1.bias
            fc2.weight
            fc2.bias
        """
        self._check_expert_index(expert_idx)

        result: Dict[str, Tensor] = {}

        for name, value in self.experts[expert_idx].state_dict().items():
            tensor = value.detach()

            if clone:
                tensor = tensor.clone()

            if to_cpu:
                tensor = tensor.cpu()

            result[name] = tensor

        return result

    def get_all_expert_state_dicts(
        self,
        clone: bool = True,
        to_cpu: bool = False,
    ) -> list[Dict[str, Tensor]]:
        return [
            self.get_expert_state_dict(
                expert_idx=expert_idx,
                clone=clone,
                to_cpu=to_cpu,
            )
            for expert_idx in range(self.num_experts)
        ]

    def load_shared_state_dict(
        self,
        shared_state: Mapping[str, Tensor],
        strict: bool = True,
    ) -> None:
        """
        只加载共享状态，保留当前专家状态不变。
        """
        current_state = self.state_dict()
        expected_shared_keys = {
            name
            for name in current_state
            if not self.is_expert_key(name)
        }
        provided_keys = set(shared_state.keys())

        invalid_expert_keys = {
            name
            for name in provided_keys
            if self.is_expert_key(name)
        }

        if invalid_expert_keys:
            raise ValueError(
                "Shared state contains expert keys: "
                f"{sorted(invalid_expert_keys)}"
            )

        unknown_keys = provided_keys - expected_shared_keys

        if unknown_keys:
            raise KeyError(
                "Unknown shared state keys: "
                f"{sorted(unknown_keys)}"
            )

        if strict:
            missing_keys = expected_shared_keys - provided_keys

            if missing_keys:
                raise KeyError(
                    "Missing shared state keys: "
                    f"{sorted(missing_keys)}"
                )

        merged_state = dict(current_state)
        merged_state.update(shared_state)

        self.load_state_dict(
            merged_state,
            strict=True,
        )

    def load_expert_state_dict(
        self,
        expert_idx: int,
        expert_state: Mapping[str, Tensor],
        strict: bool = True,
    ) -> None:
        """
        只加载指定专家状态。
        """
        self._check_expert_index(expert_idx)

        self.experts[expert_idx].load_state_dict(
            expert_state,
            strict=strict,
        )

    # ========================================================
    # 参数统计
    # ========================================================

    def count_shared_parameters(
        self,
        trainable_only: bool = True,
    ) -> int:
        return sum(
            parameter.numel()
            for parameter in self.shared_parameters()
            if (parameter.requires_grad or not trainable_only)
        )

    def count_expert_parameters(
        self,
        expert_idx: Optional[int] = None,
        trainable_only: bool = True,
    ) -> int:
        return sum(
            parameter.numel()
            for parameter in self.expert_parameters(expert_idx)
            if (parameter.requires_grad or not trainable_only)
        )

    def count_total_parameters(
        self,
        trainable_only: bool = True,
    ) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if (parameter.requires_grad or not trainable_only)
        )

    def _check_expert_index(self, expert_idx: int) -> None:
        if not 0 <= expert_idx < self.num_experts:
            raise IndexError(
                f"expert_idx must be in [0, {self.num_experts - 1}], "
                f"but received {expert_idx}."
            )


def build_sparse_moe(
    backbone: nn.Module,
    num_classes: int,
    num_experts: int = 4,
    top_k: int = 1,
    moe_dim: int = 512,
    expert_hidden_dim: int = 1024,
) -> SparseMoEClassifier:
    """
    构建 Sparse MoE 分类模型。

    配置建议由 train.py 统一读取并传入。
    """
    return SparseMoEClassifier(
        backbone=backbone,
        num_classes=num_classes,
        num_experts=num_experts,
        top_k=top_k,
        moe_dim=moe_dim,
        expert_hidden_dim=expert_hidden_dim,
    )

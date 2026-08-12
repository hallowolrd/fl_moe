from __future__ import annotations

"""
Shared experiment base for federated Sparse-MoE experiments.

This file intentionally centralizes the parts that should remain identical
across comparison methods:
- model definitions (ResNet18-GN + Sparse MoE),
- dataset / fixed Dirichlet partition,
- reproducibility and seed derivation,
- default client training,
- evaluation,
- client sampling,
- logging / output,
- the federated round loop.

Method-specific server aggregation is injected by each xxx.py file.
The current refactor is designed to preserve the execution and RNG order of
the original expert_equal_activation_weighted experiment.
"""

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
from statistics import mean
from typing import Callable, Dict, Iterator, Mapping, Optional, Sequence

# CUDA strict determinism requirement must be set before torch initializes CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARTITION_FORMAT_VERSION = 1
StateDict = dict[str, Tensor]

# =============================================================================
# ResNet18-GN model
# =============================================================================

def make_group_norm(
    num_channels: int,
    max_groups: int = 32,
) -> nn.GroupNorm:
    """
    创建 GroupNorm。

    从 min(max_groups, num_channels) 开始向下寻找能够整除
    num_channels 的最大分组数。

    Args:
        num_channels: 输入通道数。
        max_groups: 最大分组数，默认 32。

    Returns:
        nn.GroupNorm。
    """
    if num_channels <= 0:
        raise ValueError("num_channels must be greater than 0.")

    if max_groups <= 0:
        raise ValueError("max_groups must be greater than 0.")

    num_groups = min(max_groups, num_channels)

    while num_channels % num_groups != 0:
        num_groups -= 1

    return nn.GroupNorm(
        num_groups=num_groups,
        num_channels=num_channels,
    )


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    """
    3×3 卷积，padding=1，不使用 bias。

    GroupNorm 自带可学习偏置，因此卷积层不需要 bias。
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    """
    1×1 卷积，主要用于残差分支的维度或步幅匹配。
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    """
    ResNet18/34 使用的基本残差块。

    主分支:
        Conv3×3 -> GroupNorm -> ReLU
        Conv3×3 -> GroupNorm

    残差分支:
        Identity，或 Conv1×1 -> GroupNorm

    输出:
        ReLU(main + identity)
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Callable[[int], nn.Module] = make_group_norm,
    ) -> None:
        super().__init__()

        if stride not in (1, 2):
            raise ValueError("BasicBlock stride must be 1 or 2.")

        self.conv1 = conv3x3(
            in_channels=in_channels,
            out_channels=channels,
            stride=stride,
        )
        self.norm1 = norm_layer(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(
            in_channels=channels,
            out_channels=channels,
            stride=1,
        )
        self.norm2 = norm_layer(channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet18GN(nn.Module):
    """
    从零实现的 ResNet18-GroupNorm backbone。

    该模块只负责特征提取，不包含分类头。

    标准大图像 stem:
        Conv7×7(stride=2) -> GN -> ReLU -> MaxPool

    CIFAR 等小图像 stem:
        Conv3×3(stride=1) -> GN -> ReLU
        不使用 MaxPool

    网络阶段:
        layer1: 64  channels, 2 blocks
        layer2: 128 channels, 2 blocks
        layer3: 256 channels, 2 blocks
        layer4: 512 channels, 2 blocks

    最终输出:
        AdaptiveAvgPool2d(1) -> Flatten
        shape = [batch_size, 512]

    Attributes:
        out_dim: backbone 输出维度，固定为 512。
    """

    def __init__(
        self,
        in_channels: int = 3,
        small_image_stem: bool = False,
        max_gn_groups: int = 32,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be greater than 0.")

        if max_gn_groups <= 0:
            raise ValueError("max_gn_groups must be greater than 0.")

        self.in_channels = 64
        self.out_dim = 512
        self.small_image_stem = small_image_stem
        self.max_gn_groups = max_gn_groups

        def norm_layer(num_channels: int) -> nn.GroupNorm:
            return make_group_norm(
                num_channels=num_channels,
                max_groups=self.max_gn_groups,
            )

        self._norm_layer = norm_layer

        if small_image_stem:
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            self.maxpool = nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
            )

        self.norm1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(
            channels=64,
            blocks=2,
            stride=1,
        )
        self.layer2 = self._make_layer(
            channels=128,
            blocks=2,
            stride=2,
        )
        self.layer3 = self._make_layer(
            channels=256,
            blocks=2,
            stride=2,
        )
        self.layer4 = self._make_layer(
            channels=512,
            blocks=2,
            stride=2,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights(
            zero_init_residual=zero_init_residual,
        )

    def _make_layer(
        self,
        channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """
        构建一个 ResNet stage。
        """
        if blocks <= 0:
            raise ValueError("blocks must be greater than 0.")

        out_channels = channels * BasicBlock.expansion

        downsample: Optional[nn.Module] = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride,
                ),
                self._norm_layer(out_channels),
            )

        layers = [
            BasicBlock(
                in_channels=self.in_channels,
                channels=channels,
                stride=stride,
                downsample=downsample,
                norm_layer=self._norm_layer,
            )
        ]

        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    in_channels=self.in_channels,
                    channels=channels,
                    stride=1,
                    downsample=None,
                    norm_layer=self._norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(
        self,
        zero_init_residual: bool,
    ) -> None:
        """
        使用适合 ReLU 的 Kaiming 初始化。

        zero_init_residual=True 时，将每个残差块最后一个
        GroupNorm 的缩放参数初始化为 0，使残差分支初始更接近
        恒等映射。
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.GroupNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    if module.norm2.weight is not None:
                        nn.init.zeros_(module.norm2.weight)

    def forward_features(self, x: Tensor) -> Tensor:
        """
        提取全局池化前的二维特征图。

        Returns:
            标准 stem、224×224 输入时通常为 [B, 512, 7, 7]；
            小图像 stem、32×32 输入时通常为 [B, 512, 4, 4]。
        """
        if x.ndim != 4:
            raise ValueError(
                "Expected input shape [B, C, H, W], "
                f"but received {tuple(x.shape)}."
            )

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        提取全局图像特征。

        Args:
            x: [B, C, H, W]

        Returns:
            features: [B, 512]
        """
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if x.ndim != 2 or x.shape[1] != self.out_dim:
            raise RuntimeError(
                "Unexpected backbone output shape. "
                f"Expected [B, {self.out_dim}], "
                f"but received {tuple(x.shape)}."
            )

        return x


def build_resnet18_gn(
    in_channels: int = 3,
    small_image_stem: bool = False,
    max_gn_groups: int = 32,
    zero_init_residual: bool = False,
) -> ResNet18GN:
    """
    构建 ResNet18-GN backbone。

    该工厂函数主要用于 train.py 根据配置创建 backbone。
    """
    return ResNet18GN(
        in_channels=in_channels,
        small_image_stem=small_image_stem,
        max_gn_groups=max_gn_groups,
        zero_init_residual=zero_init_residual,
    )

# =============================================================================
# Sparse-MoE model
# =============================================================================

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

        if num_experts <= 0:
            raise ValueError("num_experts must be greater than 0.")

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

        if num_experts <= 0:
            raise ValueError("num_experts must be greater than 0.")

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

# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class ExperimentConfig:
    # 实验与复现
    seed: int = 0
    deterministic: bool = True
    device: str = "auto"

    # 相对路径均相对于项目根目录解析
    data_dir: str = "data"
    partition_root: str = "partitions"
    output_root: str = "outputs"
    dataset_name: str = "cifar10"
    backbone_name: str = "resnet18_gn"

    # 联邦学习
    num_clients: int = 10
    participation_rate: float = 1.0
    num_rounds: int = 200
    local_epochs: int = 1
    dirichlet_alpha: float = 0.1

    # Dataset split / DataLoader
    dataset_test_fraction: float = 0.2
    client_batch_size: int = 64
    test_batch_size: int = 256
    drop_last: bool = False

    # 模型
    num_experts: int = 4
    top_k: int = 1
    moe_dim: int = 512
    expert_hidden_dim: int = 1024
    small_image_stem: bool = True
    max_gn_groups: int = 32
    zero_init_residual: bool = False
    balance_loss_weight: float = 0.0

    # 本地优化
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 5e-4
    use_amp: bool = False
    max_grad_norm: float | None = None

    # 输出
    summary_window: int = 10

def parse_config(
    *,
    description: str = "Federated Sparse-MoE experiment.",
    method_validator: Callable[[ExperimentConfig], None] | None = None,
) -> ExperimentConfig:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    def add(*names: str, **kwargs) -> None:
        kwargs.setdefault("default", argparse.SUPPRESS)
        parser.add_argument(*names, **kwargs)

    def add_bool(name: str, help_text: str) -> None:
        add(
            f"--{name.replace('_', '-')}",
            dest=name,
            action=argparse.BooleanOptionalAction,
            help=help_text,
        )

    add("--seed", type=int)
    add_bool("deterministic", "启用严格确定性算法。")
    add("--device", type=str, help="auto、cpu、cuda、cuda:0 等。")

    add("--data-dir", type=str)
    add("--partition-root", type=str)
    add("--output-root", type=str)
    add("--dataset-name", type=str)
    add("--backbone-name", type=str)

    add("--num-clients", type=int)
    add("--participation-rate", type=float)
    add("--num-rounds", type=int)
    add("--local-epochs", type=int)
    add("--dirichlet-alpha", type=float)

    add("--dataset-test-fraction", type=float)
    add("--client-batch-size", type=int)
    add("--test-batch-size", type=int)
    add_bool("drop_last", "客户端训练是否丢弃最后一个不完整 batch。")

    add("--num-experts", type=int)
    add("--top-k", type=int)
    add("--moe-dim", type=int)
    add("--expert-hidden-dim", type=int)
    add_bool("small_image_stem", "ResNet18 使用 CIFAR 小图像 stem。")
    add("--max-gn-groups", type=int)
    add_bool("zero_init_residual", "残差块最后一个 GN 权重初始化为 0。")
    add("--balance-loss-weight", type=float)

    add("--learning-rate", type=float)
    add("--momentum", type=float)
    add("--weight-decay", type=float)
    add_bool("use_amp", "启用 CUDA AMP。")
    add("--max-grad-norm", type=float)

    add("--summary-window", type=int)

    config = replace(defaults, **vars(parser.parse_args()))
    validate_config(config)
    if method_validator is not None:
        method_validator(config)
    return config


DATASET_ALIASES = {
    "cifar10": "cifar10",
    "cifar-10": "cifar10",
    "cifar100": "cifar100",
    "cifar-100": "cifar100",
    "fashionmnist": "fashion_mnist",
    "fashion-mnist": "fashion_mnist",
    "fashion_mnist": "fashion_mnist",
    "svhn": "svhn",
    "cinic10": "cinic10",
    "cinic-10": "cinic10",
    "tinyimagenet": "tiny_imagenet",
    "tiny-imagenet": "tiny_imagenet",
    "tiny_imagenet": "tiny_imagenet",
    "tiny-imagenet-200": "tiny_imagenet",
    "gtsrb": "gtsrb",
    "stl10": "stl10",
    "stl-10": "stl10",
    "usps": "usps",
    "pacs": "pacs",
    "terra_incognita": "terra_incognita",
    "terra-incognita": "terra_incognita",
    "terraincognita": "terra_incognita",
    "ham10000": "ham10000",
    "femnist": "femnist",
}

SUPPORTED_DATASETS = (
    "cifar10",
    "cifar100",
    "fashion_mnist",
    "svhn",
    "cinic10",
    "tiny_imagenet",
    "gtsrb",
    "stl10",
    "usps",
    "pacs",
    "terra_incognita",
    "ham10000",
    "femnist",
)


def canonical_dataset_name(name: str) -> str:
    key = name.strip().lower()
    try:
        return DATASET_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset_name={name!r}. Supported datasets: "
            + ", ".join(SUPPORTED_DATASETS)
        ) from exc


def validate_config(config: ExperimentConfig) -> None:
    canonical_dataset_name(config.dataset_name)
    if config.backbone_name.lower() != "resnet18_gn":
        raise ValueError("This experiment currently supports ResNet18-GN only.")
    if config.seed < 0:
        raise ValueError("seed must be non-negative.")
    if config.num_clients <= 0:
        raise ValueError("num_clients must be greater than 0.")
    if not 0.0 < config.participation_rate <= 1.0:
        raise ValueError("participation_rate must be in (0, 1].")
    if config.num_rounds <= 0 or config.local_epochs <= 0:
        raise ValueError("num_rounds and local_epochs must be greater than 0.")
    if config.dirichlet_alpha <= 0.0:
        raise ValueError("dirichlet_alpha must be greater than 0.")
    if not 0.0 < config.dataset_test_fraction < 1.0:
        raise ValueError("dataset_test_fraction must be in (0, 1).")
    if config.client_batch_size <= 0 or config.test_batch_size <= 0:
        raise ValueError("batch sizes must be greater than 0.")
    if config.num_experts <= 0:
        raise ValueError("num_experts must be greater than 0.")
    if not 1 <= config.top_k <= config.num_experts:
        raise ValueError("top_k must satisfy 1 <= top_k <= num_experts.")
    if config.moe_dim <= 0 or config.expert_hidden_dim <= 0:
        raise ValueError("model dimensions must be greater than 0.")
    if config.max_gn_groups <= 0:
        raise ValueError("max_gn_groups must be greater than 0.")
    if config.balance_loss_weight < 0.0:
        raise ValueError("balance_loss_weight must be non-negative.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be greater than 0.")
    if config.momentum < 0.0 or config.weight_decay < 0.0:
        raise ValueError("momentum and weight_decay must be non-negative.")
    if config.max_grad_norm is not None and config.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive or omitted.")
    if config.summary_window <= 0:
        raise ValueError("summary_window must be greater than 0.")

# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ClientUpdate:
    client_id: int
    num_examples: int
    num_processed_examples: int
    shared_delta: StateDict
    expert_deltas: list[StateDict]
    route_counts: Tensor
    train_loss: float
    standard_classification_loss: float
    balance_loss: float
    accuracy: float
    # Method-specific client-side information, e.g. K-FAC/Fisher statistics,
    # SCAFFOLD control-variate deltas, FedNova normalization metadata, etc.
    # The default empty payload preserves the behavior of existing methods.
    method_payload: dict[str, object] = field(default_factory=dict)


@dataclass
class AggregationResult:
    # Common MoE diagnostics. Methods that do not use them may leave them empty.
    expert_participants: list[int] = field(default_factory=list)
    expert_client_weights: list[dict[int, float]] = field(default_factory=list)
    # Optional method-specific round diagnostics. Values should be JSON-serializable
    # if the method wants them written into metrics.csv / summary.json.
    method_metrics: dict[str, object] = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    total_loss: float
    classification_loss: float
    balance_loss: float
    accuracy: float
    route_counts: Tensor
    route_distribution: Tensor

# =============================================================================
# Reproducibility, paths, and basic output
# =============================================================================

def seed_all(seed: int) -> None:
    seed = int(seed) % (2**63 - 1)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_reproducibility(config: ExperimentConfig) -> None:
    seed_all(config.seed)

    # TF32 始终关闭，不作为命令行参数。
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False

    if config.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def derive_seed(base_seed: int, namespace: str, *values: int) -> int:
    text = "|".join([str(base_seed), namespace, *(str(v) for v in values)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def resolve_device(name: str) -> torch.device:
    if name.strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device

def create_output_dir(
    config: ExperimentConfig,
    algorithm_name: str,
) -> Path:
    # 微秒级时间戳避免连续启动时覆盖已有实验目录。
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = (
        project_path(config.output_root)
        / canonical_dataset_name(config.dataset_name)
        / config.backbone_name.lower()
        / algorithm_name
        / f"seed_{config.seed}"
        / stamp
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


class _ConsoleLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not bool(getattr(record, "file_only", False))


class _FileLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not bool(getattr(record, "console_only", False))


def create_logger(
    path: Path,
    algorithm_name: str,
) -> logging.Logger:
    logger = logging.getLogger(f"{algorithm_name}.{path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_FileLogFilter())

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(_ConsoleLogFilter())

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def log_round(
    logger: logging.Logger,
    *,
    round_number: int,
    total_rounds: int,
    selected_ids: list[int],
    valid_ids: list[int],
    client_loss: float,
    client_accuracy: float,
    test_loss: float,
    test_accuracy: float,
    expert_participants: list[int],
    round_seconds: float,
) -> None:
    console_message = (
        f"Round {round_number}/{total_rounds} | "
        f"client_loss={client_loss:.6f} | "
        f"client_acc={client_accuracy * 100.0:.2f}% | "
        f"test_loss={test_loss:.6f} | "
        f"test_acc={test_accuracy * 100.0:.2f}% | "
        f"experts={expert_participants} | "
        f"time={round_seconds:.2f}s"
    )
    file_message = (
        f"Round {round_number}/{total_rounds} | "
        f"selected={selected_ids} | valid={valid_ids} | "
        f"client_loss={client_loss:.6f} | "
        f"client_acc={client_accuracy * 100.0:.2f}% | "
        f"test_loss={test_loss:.6f} | "
        f"test_acc={test_accuracy * 100.0:.2f}% | "
        f"experts={expert_participants} | "
        f"time={round_seconds:.2f}s"
    )
    logger.info(console_message, extra={"console_only": True})
    logger.info(file_message, extra={"file_only": True})


def save_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)

# =============================================================================
# Image datasets and fixed Dirichlet partition
# =============================================================================

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".tiff", ".webp"}


class PathImageDataset(Dataset):
    """Simple image-classification dataset backed by explicit (path, label) pairs."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        classes: list[str],
        transform: Callable | None = None,
        image_mode: str = "RGB",
    ) -> None:
        self.samples = [(Path(path), int(label)) for path, label in samples]
        self.targets = [int(label) for _, label in self.samples]
        self.classes = list(classes)
        self.transform = transform
        self.image_mode = image_mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert(self.image_mode)
            if self.transform is not None:
                image = self.transform(image)
        return image, target


class LEAFFEMNISTDataset(Dataset):
    """
    True FEMNIST reader for the JSON files produced by the TalwalkarLab/LEAF
    preprocessing pipeline. Writer identities are intentionally ignored by the
    federated partitioner; samples are flattened and later repartitioned by label.
    """

    def __init__(
        self,
        json_dir: Path,
        transform: Callable | None = None,
        cache_size: int = 2,
    ) -> None:
        self.json_dir = Path(json_dir)
        self.transform = transform
        self.cache_size = max(1, int(cache_size))
        self.classes = [str(i) for i in range(62)]
        self.records: list[tuple[Path, str, int, int]] = []
        self.targets: list[int] = []
        self._cache: OrderedDict[Path, dict] = OrderedDict()

        json_files = sorted(self.json_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No LEAF FEMNIST JSON files were found in {self.json_dir}."
            )

        for json_path in json_files:
            with json_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            users = payload.get("users", [])
            user_data = payload.get("user_data", {})
            if not isinstance(users, list) or not isinstance(user_data, dict):
                raise RuntimeError(f"Invalid LEAF FEMNIST file: {json_path}")
            for user in users:
                data = user_data.get(user)
                if not isinstance(data, dict):
                    raise RuntimeError(
                        f"LEAF FEMNIST file {json_path} is missing user {user!r}."
                    )
                labels = data.get("y", [])
                images = data.get("x", [])
                if len(labels) != len(images):
                    raise RuntimeError(
                        f"LEAF FEMNIST user {user!r} has inconsistent x/y lengths."
                    )
                for sample_idx, label in enumerate(labels):
                    label = int(label)
                    if not 0 <= label < 62:
                        raise RuntimeError(
                            f"Invalid FEMNIST label {label} in {json_path}."
                        )
                    self.records.append((json_path, str(user), sample_idx, label))
                    self.targets.append(label)

        if not self.records:
            raise RuntimeError(f"LEAF FEMNIST split {self.json_dir} is empty.")

    def __len__(self) -> int:
        return len(self.records)

    def _load_payload(self, path: Path) -> dict:
        payload = self._cache.get(path)
        if payload is not None:
            self._cache.move_to_end(path)
            return payload
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        self._cache[path] = payload
        self._cache.move_to_end(path)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return payload

    def __getitem__(self, index: int):
        json_path, user, sample_idx, target = self.records[index]
        payload = self._load_payload(json_path)
        raw = payload["user_data"][user]["x"][sample_idx]
        array = np.asarray(raw, dtype=np.float32)
        if array.size != 28 * 28:
            raise RuntimeError(
                f"Unexpected FEMNIST sample size {array.size}; expected 784."
            )
        array = array.reshape(28, 28)
        if float(array.max(initial=0.0)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
        image = Image.fromarray(array, mode="L")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def dataset_input_channels(dataset_name: str) -> int:
    dataset_name = canonical_dataset_name(dataset_name)
    return 1 if dataset_name in {"fashion_mnist", "usps", "femnist"} else 3


def _dataset_transforms(dataset_name: str) -> tuple[Callable, Callable]:
    dataset_name = canonical_dataset_name(dataset_name)

    if dataset_name == "cifar10":
        # Keep the original CIFAR-10 transform path exactly unchanged.
        mean_values = (0.4914, 0.4822, 0.4465)
        std_values = (0.2470, 0.2435, 0.2616)
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean_values, std_values),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean_values, std_values)]
        )
        return train_transform, test_transform

    if dataset_name == "cifar100":
        mean_values = (0.5071, 0.4867, 0.4408)
        std_values = (0.2675, 0.2565, 0.2761)
        return (
            transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_values, std_values),
                ]
            ),
            transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean_values, std_values)]
            ),
        )

    if dataset_name == "fashion_mnist":
        mean_values = (0.2860,)
        std_values = (0.3530,)
        return (
            transforms.Compose(
                [
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_values, std_values),
                ]
            ),
            transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean_values, std_values)]
            ),
        )

    if dataset_name == "svhn":
        mean_values = (0.4377, 0.4438, 0.4728)
        std_values = (0.1980, 0.2010, 0.1970)
        return (
            transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_values, std_values),
                ]
            ),
            transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean_values, std_values)]
            ),
        )

    if dataset_name == "cinic10":
        mean_values = (0.4789, 0.4723, 0.4305)
        std_values = (0.2421, 0.2383, 0.2587)
        return (
            transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_values, std_values),
                ]
            ),
            transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean_values, std_values)]
            ),
        )

    if dataset_name == "tiny_imagenet":
        mean_values = (0.4802, 0.4481, 0.3975)
        std_values = (0.2302, 0.2265, 0.2262)
        return (
            transforms.Compose(
                [
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_values, std_values),
                ]
            ),
            transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean_values, std_values)]
            ),
        )

    if dataset_name == "gtsrb":
        mean_values = (0.5, 0.5, 0.5)
        std_values = (0.5, 0.5, 0.5)
        return (
            transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_values, std_values),
                ]
            ),
            transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_values, std_values),
                ]
            ),
        )

    if dataset_name == "stl10":
        mean_values = (0.4467, 0.4398, 0.4066)
        std_values = (0.2603, 0.2566, 0.2713)
        return (
            transforms.Compose(
                [
                    transforms.RandomCrop(96, padding=12),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_values, std_values),
                ]
            ),
            transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean_values, std_values)]
            ),
        )

    if dataset_name in {"usps", "femnist"}:
        mean_values = (0.5,)
        std_values = (0.5,)
        return (
            transforms.Compose(
                [
                    transforms.Resize((28, 28)),
                    transforms.RandomCrop(28, padding=2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_values, std_values),
                ]
            ),
            transforms.Compose(
                [
                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_values, std_values),
                ]
            ),
        )

    if dataset_name in {"pacs", "terra_incognita", "ham10000"}:
        mean_values = (0.485, 0.456, 0.406)
        std_values = (0.229, 0.224, 0.225)
        return (
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_values, std_values),
                ]
            ),
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_values, std_values),
                ]
            ),
        )

    raise AssertionError(f"No transforms configured for {dataset_name}.")


def _find_existing_dir(data_dir: Path, candidates: Sequence[str], dataset_name: str) -> Path:
    for candidate in candidates:
        path = data_dir / candidate
        if path.is_dir():
            return path
    expected = "\n  - ".join(str(data_dir / candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"{dataset_name} was not found. Expected one of:\n  - {expected}"
    )


def _all_image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _scan_domain_class_samples(root: Path) -> tuple[list[tuple[Path, int]], list[str]]:
    domain_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not domain_dirs:
        raise RuntimeError(f"No domain directories were found in {root}.")

    class_names = sorted(
        {
            class_dir.name
            for domain_dir in domain_dirs
            for class_dir in domain_dir.iterdir()
            if class_dir.is_dir()
        }
    )
    if len(class_names) <= 1:
        raise RuntimeError(
            f"Could not infer multiple class directories from domain dataset {root}."
        )
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    samples: list[tuple[Path, int]] = []
    for domain_dir in domain_dirs:
        for class_name in class_names:
            class_dir = domain_dir / class_name
            if not class_dir.is_dir():
                continue
            for image_path in _all_image_files(class_dir):
                samples.append((image_path, class_to_idx[class_name]))
    if not samples:
        raise RuntimeError(f"No images were found in domain dataset {root}.")
    return samples, class_names


def format_float(value: float) -> str:
    return format(value, ".12g").replace("-", "m").replace(".", "p")


def _dataset_split_path(config: ExperimentConfig, dataset_name: str) -> Path:
    filename = (
        f"stratified_test{format_float(config.dataset_test_fraction)}"
        f"_seed{config.seed}.json"
    )
    return project_path(config.partition_root) / dataset_name / filename


def _make_stratified_train_test_split(
    labels: np.ndarray,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        if len(class_indices) <= 1:
            train_indices.extend(class_indices.tolist())
            continue
        test_count = int(round(len(class_indices) * test_fraction))
        test_count = min(len(class_indices) - 1, max(1, test_count))
        test_indices.extend(class_indices[:test_count].tolist())
        train_indices.extend(class_indices[test_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    if not train_indices or not test_indices:
        raise RuntimeError("Stratified dataset split produced an empty train or test set.")
    return train_indices, test_indices


def _load_or_create_dataset_split(
    config: ExperimentConfig,
    dataset_name: str,
    labels: np.ndarray,
) -> tuple[list[int], list[int], Path, bool]:
    path = _dataset_split_path(config, dataset_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype="<i8")
    labels_hash = hashlib.sha256(labels.tobytes(order="C")).hexdigest()
    created = False

    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        expected = {
            "format_version": PARTITION_FORMAT_VERSION,
            "dataset": dataset_name,
            "split_method": "fixed_stratified_train_test",
            "seed": config.seed,
            "num_total_samples": len(labels),
            "labels_sha256": labels_hash,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(
                    f"Existing dataset split metadata mismatch for {key!r}: "
                    f"expected {value!r}, got {payload.get(key)!r}. "
                    "The file is not overwritten automatically."
                )
        if not math.isclose(
            float(payload.get("test_fraction", float("nan"))),
            config.dataset_test_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("Existing dataset split test_fraction does not match config.")
        train_indices = [int(index) for index in payload.get("train_indices", [])]
        test_indices = [int(index) for index in payload.get("test_indices", [])]
    else:
        train_indices, test_indices = _make_stratified_train_test_split(
            labels,
            config.dataset_test_fraction,
            config.seed,
        )
        save_json(
            path,
            {
                "format_version": PARTITION_FORMAT_VERSION,
                "dataset": dataset_name,
                "split_method": "fixed_stratified_train_test",
                "test_fraction": config.dataset_test_fraction,
                "seed": config.seed,
                "num_total_samples": len(labels),
                "labels_sha256": labels_hash,
                "train_indices": train_indices,
                "test_indices": test_indices,
            },
        )
        created = True

    train_set = set(train_indices)
    test_set = set(test_indices)
    if train_set & test_set:
        raise RuntimeError("Saved dataset train/test split overlaps.")
    if train_set | test_set != set(range(len(labels))):
        raise RuntimeError("Saved dataset train/test split does not cover the dataset.")
    return train_indices, test_indices, path, created


def _build_cinic10(
    data_dir: Path,
    train_transform: Callable,
    test_transform: Callable,
) -> tuple[Dataset, Dataset]:
    root = _find_existing_dir(
        data_dir,
        ("cinic10", "CINIC-10", "cinic-10"),
        "CINIC-10",
    )
    train_root = root / "train"
    test_root = root / "test"
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError(
            f"CINIC-10 expects {train_root} and {test_root}."
        )
    train_dataset = datasets.ImageFolder(str(train_root), transform=train_transform)
    test_dataset = datasets.ImageFolder(str(test_root), transform=test_transform)
    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise RuntimeError("CINIC-10 train/test class mappings differ.")
    return train_dataset, test_dataset


def _build_tiny_imagenet(
    data_dir: Path,
    train_transform: Callable,
    test_transform: Callable,
) -> tuple[Dataset, Dataset]:
    root = _find_existing_dir(
        data_dir,
        ("tiny-imagenet-200", "tiny_imagenet_200", "tiny_imagenet"),
        "Tiny-ImageNet",
    )
    train_root = root / "train"
    val_root = root / "val"
    if not train_root.is_dir() or not val_root.is_dir():
        raise FileNotFoundError(
            f"Tiny-ImageNet expects {train_root} and {val_root}."
        )

    train_dataset = datasets.ImageFolder(str(train_root), transform=train_transform)

    # Support both the original validation layout and the commonly reorganized
    # ImageFolder-compatible validation layout.
    val_class_dirs = [path for path in val_root.iterdir() if path.is_dir() and path.name != "images"]
    if val_class_dirs:
        test_dataset = datasets.ImageFolder(str(val_root), transform=test_transform)
        if train_dataset.class_to_idx != test_dataset.class_to_idx:
            raise RuntimeError("Tiny-ImageNet train/val class mappings differ.")
        return train_dataset, test_dataset

    annotations_path = val_root / "val_annotations.txt"
    images_root = val_root / "images"
    if not annotations_path.is_file() or not images_root.is_dir():
        raise FileNotFoundError(
            "Tiny-ImageNet validation split requires val/val_annotations.txt "
            "and val/images/, or an ImageFolder-reorganized val directory."
        )
    samples: list[tuple[Path, int]] = []
    with annotations_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            filename, wnid = parts[0], parts[1]
            if wnid not in train_dataset.class_to_idx:
                raise RuntimeError(f"Unknown Tiny-ImageNet validation class {wnid!r}.")
            image_path = images_root / filename
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            samples.append((image_path, train_dataset.class_to_idx[wnid]))
    test_dataset = PathImageDataset(
        samples=samples,
        classes=list(train_dataset.classes),
        transform=test_transform,
        image_mode="RGB",
    )
    return train_dataset, test_dataset


def _build_domain_dataset(
    config: ExperimentConfig,
    data_dir: Path,
    dataset_name: str,
    candidates: Sequence[str],
    train_transform: Callable,
    test_transform: Callable,
) -> tuple[Dataset, Dataset]:
    root = _find_existing_dir(data_dir, candidates, dataset_name)
    samples, classes = _scan_domain_class_samples(root)
    labels = np.asarray([label for _, label in samples], dtype="<i8")
    train_indices, test_indices, _, _ = _load_or_create_dataset_split(
        config,
        dataset_name,
        labels,
    )
    train_samples = [samples[index] for index in train_indices]
    test_samples = [samples[index] for index in test_indices]
    return (
        PathImageDataset(train_samples, classes, train_transform, "RGB"),
        PathImageDataset(test_samples, classes, test_transform, "RGB"),
    )


def _build_ham10000(
    config: ExperimentConfig,
    data_dir: Path,
    train_transform: Callable,
    test_transform: Callable,
) -> tuple[Dataset, Dataset]:
    root = _find_existing_dir(
        data_dir,
        ("ham10000", "HAM10000", "HAM10000_dataset"),
        "HAM10000",
    )
    metadata_path = root / "HAM10000_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"HAM10000 metadata file was not found: {metadata_path}"
        )

    image_by_id: dict[str, Path] = {}
    for image_path in _all_image_files(root):
        image_by_id.setdefault(image_path.stem, image_path)

    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    samples: list[tuple[Path, int]] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if "image_id" not in (reader.fieldnames or []) or "dx" not in (reader.fieldnames or []):
            raise RuntimeError("HAM10000 metadata must contain image_id and dx columns.")
        for row in reader:
            image_id = row["image_id"].strip()
            diagnosis = row["dx"].strip()
            if diagnosis not in class_to_idx:
                raise RuntimeError(f"Unknown HAM10000 diagnosis {diagnosis!r}.")
            image_path = image_by_id.get(image_id)
            if image_path is None:
                raise FileNotFoundError(
                    f"HAM10000 image {image_id!r} referenced by metadata was not found."
                )
            samples.append((image_path, class_to_idx[diagnosis]))

    labels = np.asarray([label for _, label in samples], dtype="<i8")
    train_indices, test_indices, _, _ = _load_or_create_dataset_split(
        config,
        "ham10000",
        labels,
    )
    return (
        PathImageDataset(
            [samples[index] for index in train_indices],
            classes,
            train_transform,
            "RGB",
        ),
        PathImageDataset(
            [samples[index] for index in test_indices],
            classes,
            test_transform,
            "RGB",
        ),
    )


def _find_femnist_split_dirs(data_dir: Path) -> tuple[Path, Path]:
    roots = [
        data_dir / "femnist",
        data_dir / "FEMNIST",
    ]
    for root in roots:
        candidates = [
            (root / "data" / "train", root / "data" / "test"),
            (root / "train", root / "test"),
        ]
        for train_dir, test_dir in candidates:
            if train_dir.is_dir() and test_dir.is_dir():
                return train_dir, test_dir
    raise FileNotFoundError(
        "FEMNIST expects true LEAF JSON splits under either "
        f"{data_dir / 'femnist' / 'data' / 'train'} and .../test, "
        "or femnist/train and femnist/test."
    )


def build_datasets(config: ExperimentConfig) -> tuple[Dataset, Dataset]:
    data_dir = project_path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = canonical_dataset_name(config.dataset_name)
    train_transform, test_transform = _dataset_transforms(dataset_name)

    if dataset_name == "cifar10":
        # Preserve the original CIFAR-10 root and constructor calls exactly.
        train_dataset = datasets.CIFAR10(
            root=str(data_dir),
            train=True,
            transform=train_transform,
            download=True,
        )
        test_dataset = datasets.CIFAR10(
            root=str(data_dir),
            train=False,
            transform=test_transform,
            download=True,
        )
        if len(train_dataset) != 50_000 or len(test_dataset) != 10_000:
            raise RuntimeError("Unexpected CIFAR-10 dataset size.")
        return train_dataset, test_dataset

    if dataset_name == "cifar100":
        return (
            datasets.CIFAR100(
                root=str(data_dir), train=True, transform=train_transform, download=True
            ),
            datasets.CIFAR100(
                root=str(data_dir), train=False, transform=test_transform, download=True
            ),
        )

    if dataset_name == "fashion_mnist":
        return (
            datasets.FashionMNIST(
                root=str(data_dir), train=True, transform=train_transform, download=True
            ),
            datasets.FashionMNIST(
                root=str(data_dir), train=False, transform=test_transform, download=True
            ),
        )

    if dataset_name == "svhn":
        return (
            datasets.SVHN(
                root=str(data_dir), split="train", transform=train_transform, download=True
            ),
            datasets.SVHN(
                root=str(data_dir), split="test", transform=test_transform, download=True
            ),
        )

    if dataset_name == "cinic10":
        return _build_cinic10(data_dir, train_transform, test_transform)

    if dataset_name == "tiny_imagenet":
        return _build_tiny_imagenet(data_dir, train_transform, test_transform)

    if dataset_name == "gtsrb":
        return (
            datasets.GTSRB(
                root=str(data_dir), split="train", transform=train_transform, download=True
            ),
            datasets.GTSRB(
                root=str(data_dir), split="test", transform=test_transform, download=True
            ),
        )

    if dataset_name == "stl10":
        return (
            datasets.STL10(
                root=str(data_dir), split="train", transform=train_transform, download=True
            ),
            datasets.STL10(
                root=str(data_dir), split="test", transform=test_transform, download=True
            ),
        )

    if dataset_name == "usps":
        return (
            datasets.USPS(
                root=str(data_dir), train=True, transform=train_transform, download=True
            ),
            datasets.USPS(
                root=str(data_dir), train=False, transform=test_transform, download=True
            ),
        )

    if dataset_name == "pacs":
        return _build_domain_dataset(
            config,
            data_dir,
            "pacs",
            ("pacs", "PACS/kfold", "PACS"),
            train_transform,
            test_transform,
        )

    if dataset_name == "terra_incognita":
        return _build_domain_dataset(
            config,
            data_dir,
            "terra_incognita",
            ("terra_incognita", "TerraIncognita", "terra-incognita"),
            train_transform,
            test_transform,
        )

    if dataset_name == "ham10000":
        return _build_ham10000(config, data_dir, train_transform, test_transform)

    if dataset_name == "femnist":
        train_dir, test_dir = _find_femnist_split_dirs(data_dir)
        return (
            LEAFFEMNISTDataset(train_dir, transform=train_transform),
            LEAFFEMNISTDataset(test_dir, transform=test_transform),
        )

    raise AssertionError(f"Unhandled dataset {dataset_name}.")


def get_dataset_targets(dataset: Dataset) -> np.ndarray:
    for attribute in ("targets", "labels", "_labels"):
        values = getattr(dataset, attribute, None)
        if values is not None:
            if isinstance(values, Tensor):
                values = values.detach().cpu().numpy()
            return np.asarray(values, dtype="<i8")

    for attribute in ("samples", "_samples"):
        samples = getattr(dataset, attribute, None)
        if samples is not None:
            return np.asarray([int(sample[1]) for sample in samples], dtype="<i8")

    raise RuntimeError(
        f"Dataset {type(dataset).__name__} does not expose targets/labels/samples."
    )


def detect_num_classes(dataset: Dataset) -> int:
    classes = getattr(dataset, "classes", None)
    if classes is not None:
        num_classes = len(classes)
    else:
        targets = get_dataset_targets(dataset)
        num_classes = int(np.unique(targets).size)

    if num_classes <= 1:
        raise RuntimeError(f"Invalid detected class count: {num_classes}.")
    return num_classes


def partition_path(config: ExperimentConfig) -> Path:
    dataset_name = canonical_dataset_name(config.dataset_name)
    filename = (
        f"feddyn_balanced_dirichlet_label_clients{config.num_clients}"
        f"_alpha{format_float(config.dirichlet_alpha)}"
        f"_seed{config.seed}.json"
    )
    return project_path(config.partition_root) / dataset_name / filename


def make_dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
) -> list[list[int]]:
    """
    构造 FedDyn 风格的等样本数 label-Dirichlet 划分。

    每个客户端先独立采样一个跨类别的对称 Dirichlet prior：
        q_i ~ Dirichlet(alpha, ..., alpha)

    然后为每个客户端分配完全相同数量的样本。分配某个样本时，
    先从尚未填满的客户端中均匀选择一个客户端，再按照该客户端
    的类别 prior 从仍有剩余样本的类别中采样。若某类别已经耗尽，
    则等价地在剩余类别上对该客户端 prior 重新条件化。

    该过程保持：
    - 每个客户端样本数严格一致；
    - 所有训练样本恰好使用一次；
    - 客户端之间的标签分布由各自的 Dirichlet prior 控制。
    """
    labels = np.asarray(labels, dtype="<i8")
    dataset_size = len(labels)

    if dataset_size % num_clients != 0:
        raise ValueError(
            "FedDyn-style balanced Dirichlet partition requires the "
            "training-set size to be divisible by num_clients so every "
            "client has exactly the same number of samples. "
            f"Got dataset_size={dataset_size}, num_clients={num_clients}."
        )

    rng = np.random.default_rng(seed)
    samples_per_client = dataset_size // num_clients

    class_ids = np.unique(labels)
    num_classes = len(class_ids)
    if num_classes <= 1:
        raise ValueError(
            "FedDyn-style Dirichlet partition requires at least two classes."
        )

    # FedDyn-style client-wise label priors:
    # one Dirichlet distribution over classes for each client.
    client_class_priors = rng.dirichlet(
        np.full(num_classes, alpha, dtype=np.float64),
        size=num_clients,
    )

    # Build one shuffled sample pool per class.
    class_pools: list[list[int]] = []
    class_remaining = np.zeros(num_classes, dtype=np.int64)

    for class_position, class_id in enumerate(class_ids):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        class_pools.append([int(index) for index in class_indices])
        class_remaining[class_position] = len(class_indices)

    result: list[list[int]] = [[] for _ in range(num_clients)]
    client_remaining = np.full(
        num_clients,
        samples_per_client,
        dtype=np.int64,
    )

    while int(client_remaining.sum()) > 0:
        active_clients = np.flatnonzero(client_remaining > 0)
        if active_clients.size == 0:
            raise RuntimeError(
                "FedDyn-style Dirichlet partition has unassigned samples "
                "but no client has remaining capacity."
            )

        # FedDyn repeatedly samples a client and rejects already-full clients.
        # Sampling uniformly from the active clients is the equivalent
        # conditional distribution without rejection.
        client_id = int(rng.choice(active_clients))

        available_classes = class_remaining > 0
        if not np.any(available_classes):
            raise RuntimeError(
                "FedDyn-style Dirichlet partition exhausted all class pools "
                "before filling every client."
            )

        # FedDyn redraws the class whenever the sampled class is exhausted.
        # Masking exhausted classes and renormalizing is the equivalent
        # conditional distribution and avoids potentially long rejection loops.
        probabilities = (
            client_class_priors[client_id]
            * available_classes.astype(np.float64)
        )
        probability_sum = float(probabilities.sum())

        if probability_sum <= 0.0:
            probabilities = available_classes.astype(np.float64)
            probability_sum = float(probabilities.sum())

        probabilities = probabilities / probability_sum
        class_position = int(
            rng.choice(num_classes, p=probabilities)
        )

        sample_index = class_pools[class_position].pop()
        result[client_id].append(sample_index)

        client_remaining[client_id] -= 1
        class_remaining[class_position] -= 1

    if np.any(client_remaining != 0):
        raise RuntimeError(
            "FedDyn-style Dirichlet partition did not fill all client "
            "capacities."
        )

    if np.any(class_remaining != 0):
        raise RuntimeError(
            "FedDyn-style Dirichlet partition did not consume all training "
            "samples."
        )

    for indices in result:
        rng.shuffle(indices)

    return result

def validate_partition(indices: list[list[int]], dataset_size: int, num_clients: int) -> None:
    if len(indices) != num_clients:
        raise RuntimeError("Partition client count does not match configuration.")
    if dataset_size % num_clients != 0:
        raise RuntimeError(
            "Balanced partition requires dataset_size to be divisible by num_clients."
        )
    expected_samples_per_client = dataset_size // num_clients
    invalid_client_sizes = {
        client_id: len(client_indices)
        for client_id, client_indices in enumerate(indices)
        if len(client_indices) != expected_samples_per_client
    }
    if invalid_client_sizes:
        raise RuntimeError(
            "Balanced partition requires every client to contain exactly "
            f"{expected_samples_per_client} samples, but got "
            f"{invalid_client_sizes}."
        )
    flat = [int(index) for client in indices for index in client]
    if len(flat) != dataset_size:
        raise RuntimeError("Partition does not cover the full training set.")
    if len(set(flat)) != dataset_size:
        raise RuntimeError("Partition contains duplicated training indices.")
    if flat and (min(flat) < 0 or max(flat) >= dataset_size):
        raise RuntimeError("Partition contains out-of-range indices.")


def load_or_create_partition(
    config: ExperimentConfig,
    train_dataset: Dataset,
    num_classes: int,
) -> tuple[list[list[int]], Path, bool]:
    """Read a fixed FedDyn-style balanced label-Dirichlet partition; existing files are never overwritten."""
    dataset_name = canonical_dataset_name(config.dataset_name)
    path = partition_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = get_dataset_targets(train_dataset)
    labels_hash = hashlib.sha256(labels.tobytes(order="C")).hexdigest()
    created = False

    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        expected = {
            "format_version": PARTITION_FORMAT_VERSION,
            "dataset": dataset_name,
            "split": "train",
            "partition_method": "feddyn_balanced_dirichlet_label",
            "seed": config.seed,
            "num_clients": config.num_clients,
            "num_classes": num_classes,
            "num_total_samples": len(train_dataset),
            "labels_sha256": labels_hash,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(
                    f"Existing partition metadata mismatch for {key!r}: "
                    f"expected {value!r}, got {payload.get(key)!r}. "
                    "The file is not overwritten automatically."
                )

        if not math.isclose(
            float(payload.get("alpha", float("nan"))),
            config.dirichlet_alpha,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("Existing partition alpha does not match config.")

        clients = payload.get("clients")
        if not isinstance(clients, dict):
            raise RuntimeError("Existing partition has an invalid clients section.")
        expected_client_keys = {str(i) for i in range(config.num_clients)}
        if set(clients) != expected_client_keys:
            raise RuntimeError("Existing partition has unexpected client keys.")

        client_indices: list[list[int]] = []
        for client_id in range(config.num_clients):
            client_payload = clients.get(str(client_id))
            if not isinstance(client_payload, dict):
                raise RuntimeError(f"Partition is missing client {client_id}.")

            indices = [int(index) for index in client_payload.get("indices", [])]
            if client_payload.get("num_samples") != len(indices):
                raise RuntimeError(
                    f"Partition client {client_id} num_samples is inconsistent."
                )

            expected_counts = {str(class_id): 0 for class_id in range(num_classes)}
            for sample_index in indices:
                if not 0 <= sample_index < len(train_dataset):
                    raise RuntimeError(
                        f"Partition client {client_id} contains invalid index {sample_index}."
                    )
                expected_counts[str(int(labels[sample_index]))] += 1

            if client_payload.get("class_counts") != expected_counts:
                raise RuntimeError(
                    f"Partition client {client_id} class_counts are inconsistent."
                )
            client_indices.append(indices)
    else:
        client_indices = make_dirichlet_partition(
            labels=labels,
            num_clients=config.num_clients,
            alpha=config.dirichlet_alpha,
            seed=config.seed,
        )
        clients = {}
        for client_id, indices in enumerate(client_indices):
            class_counts = {str(class_id): 0 for class_id in range(num_classes)}
            for sample_index in indices:
                class_counts[str(int(labels[sample_index]))] += 1
            clients[str(client_id)] = {
                "num_samples": len(indices),
                "class_counts": class_counts,
                "indices": indices,
            }

        payload = {
            "format_version": PARTITION_FORMAT_VERSION,
            "dataset": dataset_name,
            "split": "train",
            "partition_method": "feddyn_balanced_dirichlet_label",
            "alpha": config.dirichlet_alpha,
            "seed": config.seed,
            "num_clients": config.num_clients,
            "num_classes": num_classes,
            "num_total_samples": len(train_dataset),
            "labels_sha256": labels_hash,
            "clients": clients,
        }
        save_json(path, payload)
        created = True

    validate_partition(client_indices, len(train_dataset), config.num_clients)
    return client_indices, path, created


def build_test_loader(
    config: ExperimentConfig,
    test_dataset: Dataset,
) -> DataLoader:
    return DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        drop_last=False,
    )

# =============================================================================
# Model construction, default client training, and shared aggregation
# =============================================================================

def build_model(config: ExperimentConfig, num_classes: int) -> nn.Module:
    backbone = build_resnet18_gn(
        in_channels=dataset_input_channels(config.dataset_name),
        small_image_stem=config.small_image_stem,
        max_gn_groups=config.max_gn_groups,
        zero_init_residual=config.zero_init_residual,
    )
    model = build_sparse_moe(
        backbone=backbone,
        num_classes=num_classes,
        num_experts=config.num_experts,
        top_k=config.top_k,
        moe_dim=config.moe_dim,
        expert_hidden_dim=config.expert_hidden_dim,
    )
    model.validate_parameter_partition()
    return model


def state_delta(local: Mapping[str, Tensor], global_: Mapping[str, Tensor]) -> StateDict:
    result: StateDict = {}
    for key, local_value in local.items():
        global_value = global_[key]
        if torch.is_floating_point(local_value):
            result[key] = local_value.detach().cpu() - global_value.detach().cpu()
        else:
            # 非浮点 buffer 不参与加法聚合。
            result[key] = torch.zeros_like(local_value, device="cpu")
    return result


def make_client_loader(
    config: ExperimentConfig,
    train_dataset: Dataset,
    indices: list[int],
) -> DataLoader:
    return DataLoader(
        Subset(train_dataset, indices),
        batch_size=config.client_batch_size,
        shuffle=True,
        drop_last=config.drop_last,
    )


def train_client(
    *,
    config: ExperimentConfig,
    global_model: nn.Module,
    train_dataset: Dataset,
    client_indices: list[int],
    client_id: int,
    round_idx: int,
    device: torch.device,
    method_state: object | None = None,
    post_local_train_statistics_fn: Callable[..., Mapping[str, object] | None] | None = None,
) -> ClientUpdate | None:
    if not client_indices:
        return None

    client_seed = derive_seed(config.seed, "client_round", round_idx, client_id)
    seed_all(client_seed)
    loader = make_client_loader(config, train_dataset, client_indices)

    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    global_shared = global_model.get_shared_state_dict(to_cpu=True)
    global_experts = global_model.get_all_expert_state_dicts(to_cpu=True)

    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    total_loss = 0.0
    total_standard_classification_loss = 0.0
    total_balance_loss = 0.0
    total_correct = 0
    total_processed = 0
    route_counts = torch.zeros(
        config.num_experts, dtype=torch.long, device=device
    )

    for _ in range(config.local_epochs):
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = local_model(images)
                standard_classification_loss = F.cross_entropy(
                    output.logits,
                    targets,
                )
                loss = (
                    standard_classification_loss
                    + config.balance_loss_weight * output.balance_loss
                )

            scaler.scale(loss).backward()
            if config.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    local_model.parameters(), config.max_grad_norm
                )
            scaler.step(optimizer)
            scaler.update()

            batch_size = targets.size(0)
            total_processed += batch_size
            total_loss += float(loss.detach().item()) * batch_size
            total_standard_classification_loss += (
                float(standard_classification_loss.detach().item()) * batch_size
            )
            total_balance_loss += float(output.balance_loss.detach().item()) * batch_size
            total_correct += int(output.logits.argmax(dim=1).eq(targets).sum().item())
            route_counts += output.route_counts.detach().to(
                device=device, dtype=torch.long
            )

    if total_processed == 0:
        return None

    local_shared = local_model.get_shared_state_dict(to_cpu=True)
    local_experts = local_model.get_all_expert_state_dicts(to_cpu=True)

    # Freeze the actual client update before any optional post-training statistics
    # pass. A Fisher/K-FAC statistics collector therefore cannot accidentally
    # change the model delta uploaded for this round.
    shared_delta = state_delta(local_shared, global_shared)
    expert_deltas = [
        state_delta(local_experts[e], global_experts[e])
        for e in range(config.num_experts)
    ]

    method_payload: dict[str, object] = {}
    if post_local_train_statistics_fn is not None:
        payload = post_local_train_statistics_fn(
            local_model=local_model,
            global_model=global_model,
            loader=loader,
            train_dataset=train_dataset,
            client_indices=client_indices,
            config=config,
            client_id=client_id,
            round_idx=round_idx,
            device=device,
            method_state=method_state,
        )
        if payload is not None:
            method_payload = dict(payload)

    update = ClientUpdate(
        client_id=client_id,
        num_examples=len(client_indices),
        num_processed_examples=total_processed,
        shared_delta=shared_delta,
        expert_deltas=expert_deltas,
        route_counts=route_counts.cpu(),
        train_loss=total_loss / total_processed,
        standard_classification_loss=(
            total_standard_classification_loss / total_processed
        ),
        balance_loss=total_balance_loss / total_processed,
        accuracy=total_correct / total_processed,
        method_payload=method_payload,
    )
    del local_model, optimizer, scaler
    return update


def aggregate_shared_uniform(model: nn.Module, updates: list[ClientUpdate]) -> None:
    old_state = model.get_shared_state_dict(to_cpu=True)
    count = float(len(updates))
    new_state: StateDict = {}

    for key, old_value in old_state.items():
        if torch.is_floating_point(old_value):
            accumulated = torch.zeros_like(old_value)
            for update in updates:
                accumulated.add_(update.shared_delta[key].to(old_value.dtype))
            new_state[key] = old_value + accumulated / count
        else:
            new_state[key] = old_value

    model.load_shared_state_dict(new_state, strict=True)

# =============================================================================
# Server evaluation
# =============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    balance_loss_weight: float,
    num_experts: int,
    use_amp: bool = False,
) -> EvaluationMetrics:
    was_training = model.training
    model.to(device)
    model.eval()
    amp_enabled = bool(use_amp and device.type == "cuda")

    total_loss = 0.0
    total_classification_loss = 0.0
    total_balance_loss = 0.0
    total_correct = 0
    total_examples = 0
    route_counts = torch.zeros(num_experts, dtype=torch.long, device=device)

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(images)
            classification_loss = F.cross_entropy(output.logits, targets)
            loss = classification_loss + balance_loss_weight * output.balance_loss

        batch_size = targets.size(0)
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_classification_loss += float(classification_loss.item()) * batch_size
        total_balance_loss += float(output.balance_loss.item()) * batch_size
        total_correct += int(output.logits.argmax(dim=1).eq(targets).sum().item())
        route_counts += output.route_counts.detach().to(
            device=device, dtype=torch.long
        )

    if total_examples == 0:
        raise RuntimeError("Server test loader is empty.")

    route_counts_cpu = route_counts.cpu()
    route_total = int(route_counts_cpu.sum())
    route_distribution = (
        route_counts_cpu.to(torch.float64) / route_total
        if route_total > 0
        else torch.zeros(num_experts, dtype=torch.float64)
    )

    model.train(was_training)
    model.cpu()
    return EvaluationMetrics(
        total_loss=total_loss / total_examples,
        classification_loss=total_classification_loss / total_examples,
        balance_loss=total_balance_loss / total_examples,
        accuracy=total_correct / total_examples,
        route_counts=route_counts_cpu,
        route_distribution=route_distribution,
    )


# =============================================================================
# 主实验流程
# =============================================================================

# =============================================================================
# Federated experiment loop
# =============================================================================

def sample_clients(
    generator: torch.Generator,
    num_clients: int,
    participation_rate: float,
) -> list[int]:
    sample_size = min(num_clients, max(1, math.ceil(num_clients * participation_rate)))
    selected = torch.randperm(num_clients, generator=generator)[:sample_size].tolist()
    selected.sort()
    return selected


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"

BuildModelFn = Callable[[ExperimentConfig, int], nn.Module]
SelectClientsFn = Callable[..., list[int]]
ClientTrainFn = Callable[..., ClientUpdate | None]
InitMethodStateFn = Callable[[nn.Module, ExperimentConfig], object]
ServerAggregateFn = Callable[..., AggregationResult]
EvaluateFn = Callable[..., EvaluationMetrics]


def default_select_clients(
    *,
    generator: torch.Generator,
    config: ExperimentConfig,
    round_idx: int,
    global_model: nn.Module,
    client_indices: list[list[int]],
    method_state: object | None,
) -> list[int]:
    del round_idx, global_model, client_indices, method_state
    return sample_clients(
        generator,
        config.num_clients,
        config.participation_rate,
    )


def default_evaluate(
    *,
    global_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    round_idx: int,
    method_state: object | None,
    train_dataset: Dataset,
    test_dataset: Dataset,
    client_indices: list[list[int]],
) -> EvaluationMetrics:
    del round_idx, method_state, train_dataset, test_dataset, client_indices
    return evaluate(
        global_model,
        test_loader,
        device,
        config.balance_loss_weight,
        config.num_experts,
        use_amp=config.use_amp,
    )


def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
    logger: logging.Logger,
    *,
    algorithm_name: str,
    server_aggregate_fn: ServerAggregateFn,
    build_model_fn: BuildModelFn = build_model,
    select_clients_fn: SelectClientsFn = default_select_clients,
    local_train_fn: ClientTrainFn = train_client,
    init_method_state_fn: InitMethodStateFn | None = None,
    evaluate_fn: EvaluateFn = default_evaluate,
    local_objective_description: str = "",
    aggregation_description: str = "",
) -> dict:
    device = resolve_device(config.device)

    train_dataset, test_dataset = build_datasets(config)
    num_classes = detect_num_classes(train_dataset)
    test_num_classes = detect_num_classes(test_dataset)
    if test_num_classes != num_classes:
        raise RuntimeError(
            "Train/test datasets report different class counts: "
            f"{num_classes} vs {test_num_classes}."
        )

    client_indices, used_partition_path, partition_created = (
        load_or_create_partition(config, train_dataset, num_classes)
    )
    test_loader = build_test_loader(config, test_dataset)

    # 数据集与划分过程可能消耗随机数，模型初始化前重新设置全局 seed。
    seed_all(config.seed)
    global_model = build_model_fn(config, num_classes).cpu()
    method_state = (
        init_method_state_fn(global_model, config)
        if init_method_state_fn is not None
        else None
    )
    sampling_generator = torch.Generator().manual_seed(config.seed)

    client_sample_counts = [len(indices) for indices in client_indices]
    empty_clients = [
        client_id
        for client_id, count in enumerate(client_sample_counts)
        if count == 0
    ]

    save_json(
        output_dir / "config.json",
        {
            **asdict(config),
            "algorithm_name": algorithm_name,
            "project_root": str(PROJECT_ROOT),
            "resolved_device": str(device),
            "detected_num_classes": num_classes,
            "partition_file": str(used_partition_path),
            "partition_created_this_run": partition_created,
        },
    )

    logger.info("Algorithm: %s", algorithm_name)
    if local_objective_description:
        logger.info("%s", local_objective_description)
    if aggregation_description:
        logger.info("%s", aggregation_description)
    logger.info(
        "Runtime: device=%s | seed=%d | deterministic=%s",
        device,
        config.seed,
        config.deterministic,
    )
    logger.info(
        "Dataset: name=%s | train_samples=%d | test_samples=%d | "
        "detected_num_classes=%d",
        config.dataset_name,
        len(train_dataset),
        len(test_dataset),
        num_classes,
    )
    logger.info(
        "Partition: file=%s | created=%s | alpha=%s | num_clients=%d",
        used_partition_path,
        partition_created,
        config.dirichlet_alpha,
        config.num_clients,
    )
    logger.info("Client sample counts: %s", client_sample_counts)
    logger.info("Empty clients: %s", empty_clients)
    logger.info(
        "Training: rounds=%d | participation_rate=%s | local_epochs=%d | "
        "client_batch_size=%d | test_batch_size=%d | learning_rate=%s",
        config.num_rounds,
        config.participation_rate,
        config.local_epochs,
        config.client_batch_size,
        config.test_batch_size,
        config.learning_rate,
    )
    logger.info(
        "Model: backbone=%s | num_experts=%d | top_k=%d | "
        "total_parameters=%d | shared_parameters=%d | expert_parameters=%d",
        config.backbone_name,
        config.num_experts,
        config.top_k,
        global_model.count_total_parameters(),
        global_model.count_shared_parameters(),
        global_model.count_expert_parameters(),
    )

    metrics_path = output_dir / "metrics.csv"
    fieldnames = [
        "round",
        "selected_client_ids",
        "valid_client_ids",
        "num_valid_clients",
        "mean_client_loss",
        "mean_client_standard_classification_loss",
        "mean_client_balance_loss",
        "mean_client_accuracy",
        "test_total_loss",
        "test_classification_loss",
        "test_balance_loss",
        "test_accuracy",
        "test_route_counts",
        "test_route_distribution",
        "expert_participant_counts",
        "expert_client_weights",
        "aggregation_method_metrics",
        "round_seconds",
    ]

    accuracy_history: list[float] = []
    loss_history: list[float] = []
    final_round: dict | None = None
    experiment_start = time.perf_counter()

    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for round_idx in range(config.num_rounds):
            round_number = round_idx + 1
            round_start = time.perf_counter()
            selected_ids = select_clients_fn(
                generator=sampling_generator,
                config=config,
                round_idx=round_idx,
                global_model=global_model,
                client_indices=client_indices,
                method_state=method_state,
            )
            selected_ids = [int(client_id) for client_id in selected_ids]
            if len(selected_ids) != len(set(selected_ids)):
                raise RuntimeError(
                    f"Round {round_number} client selection contains duplicates: "
                    f"{selected_ids}."
                )
            if any(
                client_id < 0 or client_id >= config.num_clients
                for client_id in selected_ids
            ):
                raise RuntimeError(
                    f"Round {round_number} client selection contains an "
                    f"out-of-range client id: {selected_ids}."
                )
            selected_ids.sort()

            updates: list[ClientUpdate] = []
            for client_id in selected_ids:
                update = local_train_fn(
                    config=config,
                    global_model=global_model,
                    train_dataset=train_dataset,
                    client_indices=client_indices[client_id],
                    client_id=client_id,
                    round_idx=round_idx,
                    device=device,
                    method_state=method_state,
                )
                if update is not None:
                    updates.append(update)

            if not updates:
                raise RuntimeError(
                    f"Round {round_number} produced no valid client updates. "
                    "All sampled clients may be empty or drop_last removed all batches."
                )

            aggregation_result = server_aggregate_fn(
                global_model=global_model,
                updates=updates,
                config=config,
                method_state=method_state,
                round_idx=round_idx,
            )
            if not isinstance(aggregation_result, AggregationResult):
                raise TypeError(
                    "server_aggregate_fn must return AggregationResult, "
                    f"got {type(aggregation_result).__name__}."
                )
            expert_participants = aggregation_result.expert_participants
            expert_client_weights = aggregation_result.expert_client_weights

            test_metrics = evaluate_fn(
                global_model=global_model,
                test_loader=test_loader,
                device=device,
                config=config,
                round_idx=round_idx,
                method_state=method_state,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                client_indices=client_indices,
            )
            if not isinstance(test_metrics, EvaluationMetrics):
                raise TypeError(
                    "evaluate_fn must return EvaluationMetrics, "
                    f"got {type(test_metrics).__name__}."
                )

            valid_ids = [update.client_id for update in updates]
            mean_client_loss = mean(update.train_loss for update in updates)
            mean_client_standard_classification_loss = mean(
                update.standard_classification_loss for update in updates
            )
            mean_client_balance_loss = mean(
                update.balance_loss for update in updates
            )
            mean_client_accuracy = mean(update.accuracy for update in updates)
            round_seconds = time.perf_counter() - round_start

            row = {
                "round": round_number,
                "selected_client_ids": json.dumps(selected_ids),
                "valid_client_ids": json.dumps(valid_ids),
                "num_valid_clients": len(updates),
                "mean_client_loss": f"{mean_client_loss:.8f}",
                "mean_client_standard_classification_loss": (
                    f"{mean_client_standard_classification_loss:.8f}"
                ),
                "mean_client_balance_loss": f"{mean_client_balance_loss:.8f}",
                "mean_client_accuracy": f"{mean_client_accuracy:.8f}",
                "test_total_loss": f"{test_metrics.total_loss:.8f}",
                "test_classification_loss": (
                    f"{test_metrics.classification_loss:.8f}"
                ),
                "test_balance_loss": f"{test_metrics.balance_loss:.8f}",
                "test_accuracy": f"{test_metrics.accuracy:.8f}",
                "test_route_counts": json.dumps(test_metrics.route_counts.tolist()),
                "test_route_distribution": json.dumps(
                    [round(float(value), 8) for value in test_metrics.route_distribution]
                ),
                "expert_participant_counts": json.dumps(expert_participants),
                "expert_client_weights": json.dumps(
                    expert_client_weights,
                    sort_keys=True,
                ),
                "aggregation_method_metrics": json.dumps(
                    aggregation_result.method_metrics,
                    sort_keys=True,
                ),
                "round_seconds": f"{round_seconds:.6f}",
            }
            writer.writerow(row)
            file.flush()

            log_round(
                logger,
                round_number=round_number,
                total_rounds=config.num_rounds,
                selected_ids=selected_ids,
                valid_ids=valid_ids,
                client_loss=mean_client_loss,
                client_accuracy=mean_client_accuracy,
                test_loss=test_metrics.total_loss,
                test_accuracy=test_metrics.accuracy,
                expert_participants=expert_participants,
                round_seconds=round_seconds,
            )

            accuracy_history.append(test_metrics.accuracy)
            loss_history.append(test_metrics.total_loss)
            final_round = {
                "round": round_number,
                "selected_client_ids": selected_ids,
                "valid_client_ids": valid_ids,
                "mean_client_loss": mean_client_loss,
                "mean_client_accuracy": mean_client_accuracy,
                "test_accuracy": test_metrics.accuracy,
                "test_total_loss": test_metrics.total_loss,
                "test_route_counts": test_metrics.route_counts.tolist(),
                "expert_participant_counts": expert_participants,
                "expert_client_weights": expert_client_weights,
                "aggregation_method_metrics": aggregation_result.method_metrics,
            }

    if final_round is None:
        raise RuntimeError("No training round was completed.")

    elapsed_seconds = time.perf_counter() - experiment_start
    summary_count = min(config.summary_window, len(accuracy_history))
    last_accuracy = accuracy_history[-summary_count:]
    last_loss = loss_history[-summary_count:]
    final_accuracy = accuracy_history[-1]
    best_accuracy = max(accuracy_history)
    mean_accuracy = float(np.mean(last_accuracy))
    std_accuracy = float(np.std(last_accuracy))

    summary = {
        "algorithm_name": algorithm_name,
        "seed": config.seed,
        "detected_num_classes": num_classes,
        "partition_file": str(used_partition_path),
        "final_round": final_round,
        "best_test_accuracy": best_accuracy,
        "last_rounds_summary": {
            "num_rounds": summary_count,
            "mean_test_accuracy": mean_accuracy,
            "std_test_accuracy": std_accuracy,
            "mean_test_total_loss": float(np.mean(last_loss)),
            "std_test_total_loss": float(np.std(last_loss)),
        },
        "elapsed_seconds": elapsed_seconds,
    }
    save_json(output_dir / "summary.json", summary)

    logger.info(
        "Finished | final_test_acc=%.2f%% | best_test_acc=%.2f%% | "
        "last_%d_mean=%.2f%% | last_%d_std=%.2f%% | elapsed=%s",
        final_accuracy * 100.0,
        best_accuracy * 100.0,
        summary_count,
        mean_accuracy * 100.0,
        summary_count,
        std_accuracy * 100.0,
        format_duration(elapsed_seconds),
    )
    return summary

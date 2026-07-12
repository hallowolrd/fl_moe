from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor


__all__ = [
    "BasicBlock",
    "ResNet18GN",
    "build_resnet18_gn",
]


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


if __name__ == "__main__":
    # 最小自检，不参与正式训练。
    model = ResNet18GN(
        in_channels=3,
        small_image_stem=True,
    )

    sample = torch.randn(4, 3, 32, 32)
    output = model(sample)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Output shape: {tuple(output.shape)}")
    print(f"Trainable parameters: {trainable_parameters:,}")

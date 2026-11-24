"""Switch Transformer 专家同质化指标、模型诊断与初始化诊断入口。

提供的指标：
- 专家权重余弦相似度
- Top-k 主奇异子空间重叠
- Linear CKA
- 归一化专家输出 MSE

模型诊断时，先通过正常 SwitchViT 前向传播捕获各 MoE 层输入，
再将同一批 Token 输入该层的全部专家，计算未经过 Router Gate 的专家表示。
本模块不修改模型参数，不参与训练和参数聚合，也不保存模型 Checkpoint。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

# 直接执行本文件时，必须先导入 base.py 完成确定性环境初始化，
# 再导入 torch；作为普通模块导入时不引入训练框架依赖。
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    algorithms_dir = project_root / "algorithms"
    if str(algorithms_dir) not in sys.path:
        sys.path.insert(0, str(algorithms_dir))
    import base  # noqa: F401

import torch
from torch import nn


SubspaceSide = Literal["left", "right"]


def _require_floating_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} 必须是 torch.Tensor。")
    if not value.is_floating_point():
        raise TypeError(f"{name} 必须是浮点 Tensor，当前 dtype={value.dtype}。")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} 包含 NaN 或 Inf。")


def _to_float64(value: torch.Tensor) -> torch.Tensor:
    """以 float64 执行诊断指标计算，避免低精度累积误差。"""
    return value.detach().to(dtype=torch.float64)


def compute_weight_cosine(
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
) -> float:
    """计算两个同形状权重张量展平后的余弦相似度。

    不取余弦绝对值：正方向相似与反方向相似应保留区别。
    零范数权重没有定义良好的余弦相似度，因此显式报错。
    """
    _require_floating_tensor(weight_a, "weight_a")
    _require_floating_tensor(weight_b, "weight_b")
    if weight_a.shape != weight_b.shape:
        raise ValueError(
            "权重 Tensor Shape 不一致："
            f"{tuple(weight_a.shape)} vs {tuple(weight_b.shape)}"
        )
    if weight_a.device != weight_b.device:
        raise ValueError("两个权重 Tensor 必须位于同一设备。")

    a = _to_float64(weight_a).reshape(-1)
    b = _to_float64(weight_b).reshape(-1)
    norm_a = torch.linalg.vector_norm(a)
    norm_b = torch.linalg.vector_norm(b)
    if norm_a.item() == 0.0 or norm_b.item() == 0.0:
        raise ValueError("零范数权重的余弦相似度未定义。")

    similarity = torch.dot(a, b) / (norm_a * norm_b)
    return float(similarity.clamp(min=-1.0, max=1.0).item())


def _effective_rank(
    singular_values: torch.Tensor,
    source_dtype: torch.dtype,
    matrix_shape: tuple[int, int],
) -> int:
    """按输入 Tensor 精度估计数值秩，避免把数值零空间当作有效子空间。"""
    if singular_values.numel() == 0:
        return 0
    source_eps = torch.finfo(source_dtype).eps
    largest = singular_values[0]
    if largest.item() == 0.0:
        return 0
    tolerance = source_eps * max(matrix_shape) * largest
    return int((singular_values > tolerance).sum().item())


def extract_topk_subspace(
    weight: torch.Tensor,
    max_k: int,
    side: SubspaceSide,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """一次 SVD 提取可复用的 Top-k 子空间基、奇异值和有效数值秩。

    返回的 basis Shape 为 [ambient_dim, max_k]。同一专家在多个 k 值和
    多个专家对之间比较时，应只分解一次，并复用该 basis。
    """
    if isinstance(max_k, bool) or not isinstance(max_k, int) or max_k <= 0:
        raise ValueError(f"max_k 必须是正整数，当前值：{max_k!r}")
    if side not in ("left", "right"):
        raise ValueError(f"side 只支持 'left' 或 'right'，当前值：{side!r}")

    _require_floating_tensor(weight, "weight")
    if weight.ndim != 2:
        raise ValueError("子空间分解要求输入为二维权重矩阵。")
    if max_k > min(weight.shape):
        raise ValueError(
            f"max_k={max_k} 超过矩阵可分解的最大维度 {min(weight.shape)}。"
        )

    matrix = _to_float64(weight)
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    rank = _effective_rank(
        singular_values,
        weight.dtype,
        tuple(weight.shape),
    )
    if rank < max_k:
        raise ValueError(
            "Top-k 子空间超出有效数值秩："
            f"max_k={max_k}, rank={rank}。"
        )

    if side == "right":
        basis = vh[:max_k, :].transpose(0, 1)
    else:
        basis = u[:, :max_k]
    return basis, singular_values, rank


def compute_subspace_overlap_from_bases(
    basis_a: torch.Tensor,
    basis_b: torch.Tensor,
    k: int,
) -> float:
    """根据预先提取的正交基计算 Top-k 平均平方余弦重叠。"""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k 必须是正整数，当前值：{k!r}")
    _require_floating_tensor(basis_a, "basis_a")
    _require_floating_tensor(basis_b, "basis_b")
    if basis_a.ndim != 2 or basis_b.ndim != 2:
        raise ValueError("子空间基必须是二维 Tensor。")
    if basis_a.shape != basis_b.shape:
        raise ValueError(
            "子空间基 Shape 不一致："
            f"{tuple(basis_a.shape)} vs {tuple(basis_b.shape)}"
        )
    if basis_a.device != basis_b.device:
        raise ValueError("两个子空间基必须位于同一设备。")
    if k > basis_a.shape[1]:
        raise ValueError(
            f"k={k} 超过已提取的子空间维度 {basis_a.shape[1]}。"
        )

    a = _to_float64(basis_a[:, :k])
    b = _to_float64(basis_b[:, :k])
    identity = torch.eye(k, dtype=torch.float64, device=a.device)
    if not torch.allclose(
        a.transpose(0, 1) @ a,
        identity,
        atol=1e-5,
        rtol=1e-5,
    ) or not torch.allclose(
        b.transpose(0, 1) @ b,
        identity,
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("输入必须是列正交归一的子空间基。")
    cross = a.transpose(0, 1) @ b
    overlap = cross.square().sum() / k
    return float(overlap.clamp(min=0.0, max=1.0).item())


def compute_subspace_overlap(
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
    k: int,
    side: SubspaceSide,
) -> float:
    """计算两个权重矩阵 Top-k 主奇异子空间的平均平方余弦重叠。

    Args:
        weight_a: 第一个二维权重矩阵。
        weight_b: 第二个二维权重矩阵，Shape 必须与 weight_a 相同。
        k: 主子空间维度。
        side:
            "right" 使用右奇异向量，适合比较 FC1 的输入方向；
            "left" 使用左奇异向量，适合比较 FC2 的输出方向。

    Returns:
        1/k * ||Q_a.T @ Q_b||_F^2，取值范围为 [0, 1]。

    注意：
        只比较 Top-k 主子空间，不比较可能天然满秩且缺乏辨别力的完整行/列空间。
        如果数值秩小于 k，则拒绝计算，避免把任意零空间方向纳入结果。
    """
    _require_floating_tensor(weight_a, "weight_a")
    _require_floating_tensor(weight_b, "weight_b")
    if weight_a.shape != weight_b.shape:
        raise ValueError(
            "权重矩阵 Shape 不一致："
            f"{tuple(weight_a.shape)} vs {tuple(weight_b.shape)}"
        )
    if weight_a.device != weight_b.device:
        raise ValueError("两个权重矩阵必须位于同一设备。")
    basis_a, _, _ = extract_topk_subspace(weight_a, k, side)
    basis_b, _, _ = extract_topk_subspace(weight_b, k, side)
    return compute_subspace_overlap_from_bases(basis_a, basis_b, k)

def compute_linear_cka(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
) -> float:
    """计算按样本对齐的两个特征矩阵之间的 Linear CKA。

    输入 Shape 分别为 [num_samples, feature_dim]。两个矩阵必须具有相同的
    样本数和样本顺序，但特征维度可以不同。函数会沿样本维度中心化。

    使用特征矩阵形式：
        ||X_c.T @ Y_c||_F^2
        -----------------------------------------------
        ||X_c.T @ X_c||_F * ||Y_c.T @ Y_c||_F

    当表示没有有效变化（分母为零或数值退化）时显式报错，不将其伪装成 CKA=0。
    """
    _require_floating_tensor(features_a, "features_a")
    _require_floating_tensor(features_b, "features_b")
    if features_a.ndim != 2 or features_b.ndim != 2:
        raise ValueError("Linear CKA 要求输入 Shape 为 [num_samples, feature_dim]。")
    if features_a.shape[0] != features_b.shape[0]:
        raise ValueError(
            "CKA 输入样本数不一致："
            f"{features_a.shape[0]} vs {features_b.shape[0]}"
        )
    if features_a.device != features_b.device:
        raise ValueError("CKA 两个特征 Tensor 必须位于同一设备。")
    if features_a.shape[0] < 2:
        raise ValueError("Linear CKA 至少需要两个对齐样本。")
    if features_a.shape[1] == 0 or features_b.shape[1] == 0:
        raise ValueError("Linear CKA 的特征维度不能为空。")

    x = _to_float64(features_a)
    y = _to_float64(features_b)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    cross = x.transpose(0, 1) @ y
    gram_x = x.transpose(0, 1) @ x
    gram_y = y.transpose(0, 1) @ y

    numerator = torch.linalg.matrix_norm(cross, ord="fro").square()
    norm_x = torch.linalg.matrix_norm(gram_x, ord="fro")
    norm_y = torch.linalg.matrix_norm(gram_y, ord="fro")
    denominator = norm_x * norm_y
    if not torch.isfinite(denominator).item() or denominator.item() <= 0.0:
        raise ValueError("Linear CKA 未定义：至少一个输入表示没有有效方差。")

    cka = numerator / denominator
    return float(cka.clamp(min=0.0, max=1.0).item())


def compute_normalized_output_mse(
    output_a: torch.Tensor,
    output_b: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """计算同一输入下两个专家原始输出的归一化平方差。

    D = ||Y_a - Y_b||_F^2 /
        (0.5 * (||Y_a||_F^2 + ||Y_b||_F^2) + eps)

    不进行中心化或正交对齐，因为该指标用于比较专家在共同输出坐标系中的实际数值输出。
    """
    _require_floating_tensor(output_a, "output_a")
    _require_floating_tensor(output_b, "output_b")
    if output_a.shape != output_b.shape:
        raise ValueError(
            "专家输出 Shape 不一致："
            f"{tuple(output_a.shape)} vs {tuple(output_b.shape)}"
        )
    if output_a.device != output_b.device:
        raise ValueError("两个专家输出必须位于同一设备。")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps 必须是有限正数，当前值：{eps!r}")

    a = _to_float64(output_a)
    b = _to_float64(output_b)
    numerator = (a - b).square().sum()
    energy = 0.5 * (a.square().sum() + b.square().sum())
    if energy.item() <= eps:
        raise ValueError("归一化输出 MSE 未定义：两个输出的总能量过低。")

    return float((numerator / (energy + eps)).item())



def compute_token_output_cosine_stats(
    output_a: torch.Tensor,
    output_b: torch.Tensor,
) -> dict[str, float | int]:
    """同一批对齐 Token 的逐 Token 输出余弦相似度分布。

    与 CKA 不同，此指标比较同一个 Transformer 残差坐标系中的输出方向。
    零范数 Token 被排除，并显式报告 valid_tokens / total_tokens；
    如果所有 Token 都退化，则报错而不伪造相似度。
    """
    _require_floating_tensor(output_a, "output_a")
    _require_floating_tensor(output_b, "output_b")
    if output_a.ndim != 2 or output_b.ndim != 2:
        raise ValueError("逐 Token 余弦要求输入为 [tokens, hidden_dim]。")
    if output_a.shape != output_b.shape:
        raise ValueError(
            "逐 Token 输出 Shape 不一致："
            f"{tuple(output_a.shape)} vs {tuple(output_b.shape)}"
        )
    if output_a.device != output_b.device:
        raise ValueError("逐 Token 输出必须位于同一设备。")
    if output_a.shape[0] == 0 or output_a.shape[1] == 0:
        raise ValueError("逐 Token 输出不能为空。")

    a = _to_float64(output_a)
    b = _to_float64(output_b)
    norms_a = torch.linalg.vector_norm(a, dim=1)
    norms_b = torch.linalg.vector_norm(b, dim=1)
    valid_mask = (norms_a > 0) & (norms_b > 0)
    valid_tokens = int(valid_mask.sum().item())
    if valid_tokens == 0:
        raise ValueError("所有 Token 的输出余弦均未定义：存在零范数输出。")

    cosine = (
        (a[valid_mask] * b[valid_mask]).sum(dim=1)
        / (norms_a[valid_mask] * norms_b[valid_mask])
    ).clamp(-1.0, 1.0)
    quantiles = torch.quantile(
        cosine,
        torch.tensor(
            [0.05, 0.50, 0.95],
            dtype=torch.float64,
            device=cosine.device,
        ),
    )
    return {
        "mean": float(cosine.mean().item()),
        "std": float(cosine.std(unbiased=False).item()),
        "median": float(quantiles[1].item()),
        "p05": float(quantiles[0].item()),
        "p95": float(quantiles[2].item()),
        "min": float(cosine.min().item()),
        "max": float(cosine.max().item()),
        "valid_tokens": valid_tokens,
        "total_tokens": int(output_a.shape[0]),
    }


def compute_shuffled_cka_controls(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    *,
    repeats: int,
    seed: int,
    paired_cka: float | None = None,
) -> dict[str, Any]:
    """把第二个专家的样本对应关系打乱，建立 Linear CKA 负对照。

    每次置换使用局部 torch.Generator，不消耗全局训练 RNG。
    若采样结果恰好是恒等置换，则循环平移一个位置。
    返回每次置换的 CKA、均值、总体标准差及原始配对 CKA。
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats 必须是正整数。")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed 必须是非负整数。")
    _require_floating_tensor(features_a, "features_a")
    _require_floating_tensor(features_b, "features_b")
    if features_a.ndim != 2 or features_b.ndim != 2:
        raise ValueError("CKA 负对照要求二维特征矩阵。")
    if features_a.shape[0] != features_b.shape[0]:
        raise ValueError("CKA 负对照两组特征的样本数必须一致。")
    if features_a.shape[0] < 3:
        raise ValueError("CKA 负对照至少需要三个 Token。")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    n = int(features_a.shape[0])
    identity = torch.arange(n)
    shuffled_values: list[float] = []
    for _ in range(repeats):
        permutation = torch.randperm(n, generator=generator)
        if torch.equal(permutation, identity):
            permutation = permutation.roll(1)
        reordered = features_b.index_select(
            0,
            permutation.to(device=features_b.device),
        )
        shuffled_values.append(compute_linear_cka(features_a, reordered))

    mean = sum(shuffled_values) / repeats
    std = math.sqrt(
        sum((value - mean) ** 2 for value in shuffled_values) / repeats
    )
    paired = (
        compute_linear_cka(features_a, features_b)
        if paired_cka is None
        else float(paired_cka)
    )
    return {
        "paired_cka": paired,
        "shuffled_values": shuffled_values,
        "shuffled_mean": mean,
        "shuffled_std": std,
        "paired_minus_shuffled_mean": paired - mean,
        "repeats": repeats,
        "seed": seed,
    }

def _discover_moe_layers(
    model: nn.Module,
) -> dict[str, list[tuple[str, nn.Module]]]:
    """发现具有 ModuleDict / ModuleList experts 容器的 MoE 模块。"""
    layers: dict[str, list[tuple[str, nn.Module]]] = {}
    for module_name, module in model.named_modules():
        experts = getattr(module, "experts", None)
        if not isinstance(experts, (nn.ModuleDict, nn.ModuleList)):
            continue
        if len(experts) == 0:
            raise ValueError(
                f"MoE 层 {module_name!r} 的 experts 容器不能为空。"
            )
        if len(experts) < 2:
            raise ValueError(
                f"MoE 层 {module_name!r} 至少需要两个专家才能计算同质化指标。"
            )

        if isinstance(experts, nn.ModuleDict):
            expert_items = list(experts.items())
        else:
            expert_items = [
                (str(expert_id), expert)
                for expert_id, expert in enumerate(experts)
            ]

        for expert_name, expert in expert_items:
            required_modules = ("fc1", "activation", "dropout", "fc2")
            missing = [
                name
                for name in required_modules
                if not isinstance(getattr(expert, name, None), nn.Module)
            ]
            if missing:
                raise TypeError(
                    f"MoE 层 {module_name!r} 的专家 {expert_name!r} "
                    f"缺少兼容的专家模块：{missing}"
                )
            if not isinstance(expert.fc1, nn.Linear) or not isinstance(
                expert.fc2, nn.Linear
            ):
                raise TypeError(
                    f"MoE 层 {module_name!r} 的专家 {expert_name!r} "
                    "要求 fc1 和 fc2 均为 nn.Linear。"
                )

        layer_name = module_name or "<root>"
        if layer_name in layers:
            raise ValueError(f"发现重复的 MoE 层名称：{layer_name}")
        layers[layer_name] = expert_items

    if not layers:
        raise ValueError(
            "模型中没有发现包含 ModuleDict / ModuleList experts 的 MoE 层。"
        )
    return layers


def _extract_images_from_batch(batch: Any) -> torch.Tensor:
    """从常见 DataLoader batch 格式中提取图像 Tensor。"""
    if isinstance(batch, Mapping):
        for key in ("image", "images", "x"):
            if key in batch:
                images = batch[key]
                break
        else:
            raise KeyError(
                "诊断 DataLoader 的 batch dict 必须包含 image、images 或 x。"
            )
    elif isinstance(batch, (tuple, list)) and batch:
        images = batch[0]
    else:
        raise TypeError(
            "诊断 DataLoader 的 batch 必须是包含图像的 tuple/list 或 dict。"
        )

    if not torch.is_tensor(images):
        raise TypeError(
            f"batch 图像必须是 torch.Tensor，当前类型：{type(images).__name__}。"
        )
    if images.ndim != 4:
        raise ValueError(
            "SwitchViT 诊断图像必须是 [batch, channels, height, width]，"
            f"当前 shape={tuple(images.shape)}。"
        )
    if not images.is_floating_point():
        raise TypeError("诊断图像 Tensor 必须是浮点类型。")
    return images


def _validate_optional_positive_int(value: int | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数或 None，当前值：{value!r}。")


@torch.inference_mode()
def _collect_expert_representations(
    expert: nn.Module,
    tokens: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """复用专家自身 forward，并通过 activation hook 提取隐藏表示。"""
    hidden_chunks: list[torch.Tensor] = []
    output_chunks: list[torch.Tensor] = []
    captured_hidden: list[torch.Tensor] = []
    parameter = next(expert.parameters(), None)
    if parameter is None:
        raise ValueError("专家模块没有参数，无法确定计算设备和数据类型。")

    def capture_activation(
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        if not torch.is_tensor(output):
            raise TypeError("专家 activation 必须返回 Tensor。")
        captured_hidden.append(
            output.detach().to(device="cpu").clone()
        )

    handle = expert.activation.register_forward_hook(capture_activation)
    try:
        for start in range(0, tokens.shape[0], batch_size):
            token_batch = tokens[start : start + batch_size].to(
                device=device,
                dtype=parameter.dtype,
            )
            captured_hidden.clear()
            expert_output = expert(token_batch)
            if len(captured_hidden) != 1:
                raise RuntimeError(
                    "专家 forward 必须且只能执行一次 activation；"
                    f"实际执行次数：{len(captured_hidden)}。"
                )

            hidden_chunks.append(captured_hidden[0])
            output_chunks.append(
                expert_output.detach().to(device="cpu").clone()
            )
    finally:
        handle.remove()

    return torch.cat(hidden_chunks, dim=0), torch.cat(output_chunks, dim=0)


def _summarize_pairwise_metrics(
    pairwise: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    """汇总所有专家对的指标均值、最小值和最大值。"""
    metric_names = (
        "fc1_weight_cosine",
        "fc2_weight_cosine",
        "fc1_right_subspace_overlap",
        "fc2_left_subspace_overlap",
        "hidden_cka",
        "output_cka",
        "output_normalized_mse",
    )
    optional_metrics = (
        "hidden_cka_shuffled_mean",
        "output_cka_shuffled_mean",
        "hidden_cka_vs_shuffled_gap",
        "output_cka_vs_shuffled_gap",
        "token_output_cosine_mean",
    )
    metric_names += tuple(
        name for name in optional_metrics
        if any(name in pair for pair in pairwise.values())
    )
    summary: dict[str, dict[str, float | int | None]] = {}
    for metric_name in metric_names:
        values = [
            float(pair[metric_name])
            for pair in pairwise.values()
            if pair.get(metric_name) is not None
        ]
        summary[metric_name] = {
            "mean": sum(values) / len(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "valid_pairs": len(values),
            "total_pairs": len(pairwise),
        }
    return summary


def diagnose_model_experts(
    model: nn.Module,
    data_loader: Any,
    *,
    device: torch.device | str | None = None,
    max_batches: int | None = 4,
    max_tokens_per_layer: int | None = 4096,
    subspace_k: int = 8,
    expert_batch_size: int = 1024,
    shuffled_cka_repeats: int = 0,
    shuffle_seed: int = 0,
    token_output_cosine: bool = False,
) -> dict[str, Any]:
    """诊断 SwitchViT 各 MoE 层的专家参数、隐藏表示和输出相似性。

    正常模型前向传播负责生成各 MoE 层的真实输入。随后对每一层：
    1. 将完全相同的输入 Token 分别送入该层的全部专家；
    2. 提取 GELU 后的隐藏表示和未经过 Router Gate 的专家输出；
    3. 计算所有专家对的权重、子空间、CKA 和输出差异指标。

    模型在诊断期间切换到 eval 模式并禁用梯度；结束后恢复原有设备和
    每个子模块各自的 training 状态。通过 fork_rng 隔离 DataLoader 迭代
    和模型推理的随机数消耗，不改变调用方的 CPU/CUDA 全局 RNG 状态。
    """
    _validate_optional_positive_int(max_batches, "max_batches")
    _validate_optional_positive_int(
        max_tokens_per_layer,
        "max_tokens_per_layer",
    )
    _validate_optional_positive_int(expert_batch_size, "expert_batch_size")
    if isinstance(subspace_k, bool) or not isinstance(subspace_k, int):
        raise TypeError("subspace_k 必须是严格正整数。")
    if subspace_k <= 0:
        raise ValueError(f"subspace_k 必须大于 0，当前值：{subspace_k}。")
    if (
        isinstance(shuffled_cka_repeats, bool)
        or not isinstance(shuffled_cka_repeats, int)
        or shuffled_cka_repeats < 0
    ):
        raise ValueError("shuffled_cka_repeats 必须是非负整数。")
    if isinstance(shuffle_seed, bool) or not isinstance(shuffle_seed, int):
        raise ValueError("shuffle_seed 必须是非负整数。")
    if shuffle_seed < 0:
        raise ValueError("shuffle_seed 必须是非负整数。")
    if not isinstance(token_output_cosine, bool):
        raise TypeError("token_output_cosine 必须是 bool。")

    moe_layers = _discover_moe_layers(model)
    parameters = list(model.parameters())
    if not parameters:
        raise ValueError("待诊断模型没有参数。")

    model_devices = {parameter.device for parameter in parameters}
    model_devices.update(buffer.device for buffer in model.buffers())
    if len(model_devices) != 1:
        raise ValueError(
            f"诊断暂不支持跨设备模型，当前设备集合：{sorted(map(str, model_devices))}。"
        )
    original_device = next(iter(model_devices))
    target_device = (
        torch.device(device)
        if device is not None
        else original_device
    )
    input_dtype = parameters[0].dtype

    training_states = {
        module: module.training
        for module in model.modules()
    }
    captured_batches: dict[str, list[torch.Tensor]] = {
        layer_name: []
        for layer_name in moe_layers
    }
    active_capture: dict[str, torch.Tensor] = {}
    hook_handles = []
    num_batches = 0
    num_images = 0

    def make_pre_hook(layer_name: str):
        def capture_input(
            _module: nn.Module,
            inputs: tuple[Any, ...],
        ) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                raise TypeError(
                    f"MoE 层 {layer_name} 的第一个输入必须是 Tensor。"
                )
            if layer_name in active_capture:
                raise RuntimeError(
                    f"同一次模型前向中 MoE 层 {layer_name} 被重复调用。"
                )
            layer_input = inputs[0]
            if layer_input.ndim != 3:
                raise ValueError(
                    f"MoE 层 {layer_name} 输入必须是 [batch, tokens, hidden_dim]，"
                    f"当前 shape={tuple(layer_input.shape)}。"
                )
            active_capture[layer_name] = (
                layer_input.detach().to(device="cpu").clone()
            )

        return capture_input

    try:
        model.to(target_device)
        model.eval()
        named_modules = dict(model.named_modules())
        for layer_name in moe_layers:
            module_path = "" if layer_name == "<root>" else layer_name
            hook_handles.append(
                named_modules[module_path].register_forward_pre_hook(
                    make_pre_hook(layer_name)
                )
            )

        # DataLoader iterator creation may draw a base seed from the global
        # CPU RNG even when shuffle=False and num_workers=0. Preserve the
        # caller's CPU/CUDA RNG states around model inference and data loading.
        rng_devices = (
            [
                target_device.index
                if target_device.index is not None
                else torch.cuda.current_device()
            ]
            if target_device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=rng_devices):
            with torch.inference_mode():
                for batch_index, batch in enumerate(data_loader):
                    if max_batches is not None and batch_index >= max_batches:
                        break

                    images = _extract_images_from_batch(batch).to(
                        device=target_device,
                        dtype=input_dtype,
                        non_blocking=target_device.type == "cuda",
                    )
                    active_capture.clear()
                    model(images)

                    missing_layers = set(moe_layers) - set(active_capture)
                    if missing_layers:
                        raise RuntimeError(
                            "模型前向没有触发全部 MoE 层输入采集："
                            f"{sorted(missing_layers)}。"
                        )

                    for layer_name in moe_layers:
                        captured_batches[layer_name].append(
                            active_capture[layer_name]
                        )
                    num_batches += 1
                    num_images += int(images.shape[0])

        if num_batches == 0:
            raise ValueError("诊断 DataLoader 没有提供任何 batch。")

        layer_reports: dict[str, Any] = {}
        for layer_index, (layer_name, expert_items) in enumerate(
            moe_layers.items()
        ):
            layer_inputs = torch.cat(
                captured_batches[layer_name],
                dim=0,
            )
            if layer_inputs.ndim != 3:
                raise ValueError(
                    f"MoE 层 {layer_name} 累积输入维度异常："
                    f"{tuple(layer_inputs.shape)}。"
                )

            tokens = layer_inputs.reshape(-1, layer_inputs.shape[-1])
            total_tokens = int(tokens.shape[0])
            if (
                max_tokens_per_layer is not None
                and total_tokens > max_tokens_per_layer
            ):
                selected_indices = torch.linspace(
                    0,
                    total_tokens - 1,
                    steps=max_tokens_per_layer,
                ).round().to(dtype=torch.long)
                tokens = tokens.index_select(0, selected_indices)

            if tokens.shape[0] < 2:
                raise ValueError(
                    f"MoE 层 {layer_name} 采集到的 Token 少于 2 个，"
                    "无法计算 CKA。"
                )

            hidden_by_expert: dict[str, torch.Tensor] = {}
            output_by_expert: dict[str, torch.Tensor] = {}
            for expert_name, expert in expert_items:
                hidden, expert_output = _collect_expert_representations(
                    expert=expert,
                    tokens=tokens,
                    device=target_device,
                    batch_size=expert_batch_size,
                )
                hidden_by_expert[expert_name] = hidden
                output_by_expert[expert_name] = expert_output

            subspace_bases: dict[str, dict[str, torch.Tensor]] = {
                "fc1_right": {},
                "fc2_left": {},
            }
            subspace_errors: dict[str, dict[str, str]] = {
                "fc1_right": {},
                "fc2_left": {},
            }
            for expert_name, expert in expert_items:
                for cache_name, weight, side in (
                    ("fc1_right", expert.fc1.weight, "right"),
                    ("fc2_left", expert.fc2.weight, "left"),
                ):
                    weight_cpu = weight.detach().to(device="cpu")
                    if subspace_k > min(weight_cpu.shape):
                        raise ValueError(
                            f"MoE 层 {layer_name} 专家 {expert_name} 的 "
                            f"{cache_name} 权重维度为 {tuple(weight_cpu.shape)}，"
                            f"无法提取 k={subspace_k} 的子空间。"
                        )
                    try:
                        basis, _, _ = extract_topk_subspace(
                            weight_cpu,
                            max_k=subspace_k,
                            side=side,
                        )
                        subspace_bases[cache_name][expert_name] = basis
                    except (ValueError, RuntimeError) as error:
                        subspace_errors[cache_name][expert_name] = str(error)

            pairwise: dict[str, Any] = {}
            for pair_index, (
                (expert_a_name, expert_a),
                (expert_b_name, expert_b),
            ) in enumerate(itertools.combinations(expert_items, 2)):
                pair_key = f"{expert_a_name}|{expert_b_name}"
                metrics: dict[str, Any] = {}
                errors: dict[str, str] = {}

                metric_functions = (
                    (
                        "fc1_weight_cosine",
                        lambda: compute_weight_cosine(
                            expert_a.fc1.weight.detach().cpu(),
                            expert_b.fc1.weight.detach().cpu(),
                        ),
                    ),
                    (
                        "fc2_weight_cosine",
                        lambda: compute_weight_cosine(
                            expert_a.fc2.weight.detach().cpu(),
                            expert_b.fc2.weight.detach().cpu(),
                        ),
                    ),
                    (
                        "hidden_cka",
                        lambda: compute_linear_cka(
                            hidden_by_expert[expert_a_name],
                            hidden_by_expert[expert_b_name],
                        ),
                    ),
                    (
                        "output_cka",
                        lambda: compute_linear_cka(
                            output_by_expert[expert_a_name],
                            output_by_expert[expert_b_name],
                        ),
                    ),
                    (
                        "output_normalized_mse",
                        lambda: compute_normalized_output_mse(
                            output_by_expert[expert_a_name],
                            output_by_expert[expert_b_name],
                        ),
                    ),
                )
                for metric_name, metric_function in metric_functions:
                    try:
                        metrics[metric_name] = float(metric_function())
                    except (ValueError, RuntimeError) as error:
                        metrics[metric_name] = None
                        errors[metric_name] = str(error)

                for metric_name, cache_name in (
                    ("fc1_right_subspace_overlap", "fc1_right"),
                    ("fc2_left_subspace_overlap", "fc2_left"),
                ):
                    basis_a = subspace_bases[cache_name].get(expert_a_name)
                    basis_b = subspace_bases[cache_name].get(expert_b_name)
                    if basis_a is None or basis_b is None:
                        metrics[metric_name] = None
                        errors[metric_name] = (
                            f"子空间无法计算："
                            f"{expert_a_name}={subspace_errors[cache_name].get(expert_a_name)}, "
                            f"{expert_b_name}={subspace_errors[cache_name].get(expert_b_name)}"
                        )
                        continue
                    try:
                        metrics[metric_name] = (
                            compute_subspace_overlap_from_bases(
                                basis_a,
                                basis_b,
                                k=subspace_k,
                            )
                        )
                    except (ValueError, RuntimeError) as error:
                        metrics[metric_name] = None
                        errors[metric_name] = str(error)

                if shuffled_cka_repeats:
                    # 层与专家对的种子不依赖 Python hash()；
                    # 隐藏层、输出层使用相同的置换来构造负对照。
                    control_seed = (
                        shuffle_seed + layer_index * 100_003 + pair_index * 997
                    )
                    for prefix, features_x, features_y in (
                        (
                            "hidden_cka",
                            hidden_by_expert[expert_a_name],
                            hidden_by_expert[expert_b_name],
                        ),
                        (
                            "output_cka",
                            output_by_expert[expert_a_name],
                            output_by_expert[expert_b_name],
                        ),
                    ):
                        try:
                            control = compute_shuffled_cka_controls(
                                features_x,
                                features_y,
                                repeats=shuffled_cka_repeats,
                                seed=control_seed,
                                paired_cka=metrics[prefix],
                            )
                            metrics[f"{prefix}_shuffled_mean"] = (
                                control["shuffled_mean"]
                            )
                            metrics[f"{prefix}_vs_shuffled_gap"] = (
                                control["paired_minus_shuffled_mean"]
                            )
                            metrics[f"{prefix}_shuffled_std"] = (
                                control["shuffled_std"]
                            )
                            metrics[f"{prefix}_shuffled_values"] = (
                                control["shuffled_values"]
                            )
                        except (ValueError, RuntimeError) as error:
                            metrics[f"{prefix}_shuffled_mean"] = None
                            metrics[f"{prefix}_vs_shuffled_gap"] = None
                            errors[f"{prefix}_shuffled"] = str(error)

                if token_output_cosine:
                    try:
                        token_stats = compute_token_output_cosine_stats(
                            output_by_expert[expert_a_name],
                            output_by_expert[expert_b_name],
                        )
                        metrics["token_output_cosine"] = token_stats
                        metrics["token_output_cosine_mean"] = token_stats["mean"]
                    except (ValueError, RuntimeError) as error:
                        metrics["token_output_cosine"] = None
                        metrics["token_output_cosine_mean"] = None
                        errors["token_output_cosine"] = str(error)

                pairwise[pair_key] = {
                    **metrics,
                    "errors": errors,
                }

            layer_reports[layer_name] = {
                "num_experts": len(expert_items),
                "num_tokens_before_sampling": total_tokens,
                "num_tokens_used": int(tokens.shape[0]),
                "input_hidden_dim": int(tokens.shape[-1]),
                "subspace_k": subspace_k,
                "router_gate_applied_to_expert_outputs": False,
                "pairwise": pairwise,
                "summary": _summarize_pairwise_metrics(pairwise),
            }

        return {
            "schema_version": 1,
            "diagnostic": "switch_transformer_expert_homogeneity",
            "evaluation_mode": "eval_inference_mode",
            "expert_comparison": "same_layer_input_all_experts_ungated",
            "num_batches": num_batches,
            "num_images": num_images,
            "max_batches": max_batches,
            "max_tokens_per_layer": max_tokens_per_layer,
            "subspace_k": subspace_k,
            "controls": {
                "shuffled_cka_repeats": shuffled_cka_repeats,
                "shuffle_seed": shuffle_seed,
                "token_output_cosine": token_output_cosine,
                "shuffling_unit": "token",
            },
            "layers": layer_reports,
        }
    finally:
        for handle in hook_handles:
            handle.remove()
        model.to(original_device)
        for module, was_training in training_states.items():
            module.training = was_training



def summarize_initialization_reports(
    reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """汇总随机初始化种子间的各层指标；每个种子权重相同。

    先对该种子的六个专家对求均值，再对不同种子均匀求均值。
    这不是独立 Token 样本的置信区间，不能用种子标准差代替统计检验。
    """
    if not reports:
        raise ValueError("随机初始化诊断报告不能为空。")
    seed_ids = [report["seed"] for report in reports]
    if len(set(seed_ids)) != len(seed_ids):
        raise ValueError("初始化诊断报告存在重复 seed。")

    first_layer_names = set(reports[0]["layers"])
    if any(set(report["layers"]) != first_layer_names for report in reports):
        raise ValueError("不同随机种子的 MoE 层集合不一致。")

    summary_rows: list[dict[str, Any]] = []
    for layer_name in sorted(first_layer_names):
        metric_names = set(reports[0]["layers"][layer_name]["summary"])
        if any(
            set(report["layers"][layer_name]["summary"]) != metric_names
            for report in reports
        ):
            raise ValueError(
                f"MoE 层 {layer_name} 各随机种子的诊断指标集合不一致。"
            )
        for metric_name in sorted(metric_names):
            values = [
                float(summary["mean"])
                for report in reports
                for summary in [
                    report["layers"][layer_name]["summary"][metric_name]
                ]
                if summary["mean"] is not None
            ]
            mean = sum(values) / len(values) if values else None
            std = (
                math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
                if values
                else None
            )
            summary_rows.append({
                "moe_layer": layer_name,
                "metric": metric_name,
                "seed_mean": mean,
                "seed_std": std,
                "seed_min": min(values) if values else None,
                "seed_max": max(values) if values else None,
                "valid_seeds": len(values),
                "total_seeds": len(reports),
            })
    return summary_rows


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, ensure_ascii=False, allow_nan=False)


def _write_csv(
    path: Path,
    columns: list[str],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _create_unique_diagnosis_directory(base_dir: Path) -> Path:
    """采用原子性 mkdir 分配实验目录，避免覆盖已有诊断结果。"""
    for version in range(1, 10_000):
        candidate = (
            base_dir
            if version == 1
            else base_dir.with_name(f"{base_dir.name}_v{version}")
        )
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("无法为专家同质化诊断分配不重复的输出目录。")


def save_initialization_controls(
    reports: list[dict[str, Any]],
    output_dir: Path,
    *,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    """保存逐种子 JSON、逐专家对 CSV、逐种子 CSV 和跨种子汇总。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "diagnosis_config.json", metadata)
    pair_rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    for report in reports:
        seed = report["seed"]
        _write_json(output_dir / f"seed_{seed}.json", report)
        for layer_name, layer_report in report["layers"].items():
            for metric_name, summary in layer_report["summary"].items():
                per_seed_rows.append({
                    "seed": seed,
                    "moe_layer": layer_name,
                    "metric": metric_name,
                    "mean": summary["mean"],
                    "min": summary["min"],
                    "max": summary["max"],
                    "valid_pairs": summary["valid_pairs"],
                    "total_pairs": summary["total_pairs"],
                })
            for pair_name, metric_values in layer_report["pairwise"].items():
                for metric_name, value in metric_values.items():
                    if metric_name == "errors" or metric_name.endswith(
                        "_shuffled_values"
                    ):
                        continue
                    if metric_name == "token_output_cosine":
                        if value is None:
                            continue
                        # 分位数、有效 Token 数等详细信息单独写为长表行。
                        for stat_name, stat_value in value.items():
                            if stat_name == "mean":
                                # 已经通过 token_output_cosine_mean 输出，避免重复行。
                                continue
                            pair_rows.append({
                                "seed": seed,
                                "moe_layer": layer_name,
                                "expert_pair": pair_name,
                                "metric": f"token_output_cosine_{stat_name}",
                                "value": stat_value,
                            })
                        continue
                    pair_rows.append({
                        "seed": seed,
                        "moe_layer": layer_name,
                        "expert_pair": pair_name,
                        "metric": metric_name,
                        "value": value,
                    })

    summary_rows = summarize_initialization_reports(reports)
    paths = {
        "summary": output_dir / "initialization_summary.csv",
        "per_seed": output_dir / "seed_summary.csv",
        "pairwise": output_dir / "pairwise_metrics.csv",
    }
    _write_csv(
        paths["summary"],
        [
            "moe_layer", "metric", "seed_mean", "seed_std",
            "seed_min", "seed_max", "valid_seeds", "total_seeds",
        ],
        summary_rows,
    )
    _write_csv(
        paths["per_seed"],
        [
            "seed", "moe_layer", "metric", "mean", "min", "max",
            "valid_pairs", "total_pairs",
        ],
        per_seed_rows,
    )
    _write_csv(
        paths["pairwise"],
        ["seed", "moe_layer", "expert_pair", "metric", "value"],
        pair_rows,
    )
    return paths

def main(argv: list[str] | None = None) -> int:
    """在固定 CIFAR 诊断图像上运行随机初始化专家对照实验 A/B/C。"""
    parser = argparse.ArgumentParser(
        description="SwitchViT random-initialization expert homogeneity controls."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="独立初始化随机种子，例如 --seeds 0 1 2 3 4 5 6 7 8 9。",
    )
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--max-tokens-per-layer", type=int, default=4096)
    parser.add_argument("--subspace-k", type=int, default=8)
    parser.add_argument("--expert-batch-size", type=int, default=1024)
    parser.add_argument(
        "--shuffle-repeats",
        type=int,
        default=1,
        help="每个专家对负对照的随机 Token 置换次数；0 表示关闭。",
    )
    parser.add_argument(
        "--no-token-cosine",
        action="store_true",
        help="关闭逐 Token 输出余弦统计。",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("expert_diagnosis/initialization_controls"),
        help="相对项目根目录的输出路径；自动分配 _v2 等目录以避免覆盖。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="兼容旧用法：仅一个 seed 时额外保存一份单独 JSON。",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    algorithms_dir = project_root / "algorithms"
    if str(algorithms_dir) not in sys.path:
        sys.path.insert(0, str(algorithms_dir))

    import base

    cfg = (
        base.load_config(args.config)
        if args.config is not None
        else base.load_embedded_config()
    )
    base.set_seed(
        seed=int(cfg.seed),
        deterministic=bool(cfg.deterministic),
    )
    device = base.resolve_device(cfg)
    seeds = list(args.seeds) if args.seeds is not None else [int(cfg.seed)]
    if len(set(seeds)) != len(seeds):
        parser.error("--seeds 不允许包含重复的种子。")
    if any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
        parser.error("--seeds 的每个随机种子必须位于 [0, 2**32-1]。")
    if args.output is not None and len(seeds) != 1:
        parser.error("--output 仅支持单个 seed；多个 seed 请使用 --output-dir。")
    if args.max_batches <= 0:
        parser.error("--max-batches 必须是正整数。")
    if args.shuffle_repeats < 0:
        parser.error("--shuffle-repeats 必须是非负整数。")
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(cfg.test_batch_size)
    )
    num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(cfg.num_workers)
    )
    if batch_size <= 0 or num_workers < 0:
        parser.error("--batch-size 必须为正数，--num-workers 不能为负数。")

    dataset_bundle = base.build_datasets(cfg)
    diagnostic_loader = base._build_dataloader(
        dataset=dataset_bundle.train_evidence_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
            and base._infer_pin_memory(cfg)
        ),
        drop_last=False,
        seed=int(cfg.seed),
    )
    # 数据集在这里只加载一次；各随机初始化模型使用完全相同的图像、
    # 图像顺序和关闭随机增强的 Transform。
    fixed_batches = [
        (_extract_images_from_batch(batch).detach().cpu().clone(),)
        for batch in itertools.islice(diagnostic_loader, args.max_batches)
    ]
    if not fixed_batches:
        raise ValueError("诊断数据集为空，没有可用于初始化基线的图像。")

    output_base = args.output_dir.expanduser()
    if not output_base.is_absolute():
        output_base = project_root / output_base
    output_dir = _create_unique_diagnosis_directory(output_base)

    model_cfg = (
        cfg.model_cfg.to_dict()
        if hasattr(cfg.model_cfg, "to_dict")
        else dict(cfg.model_cfg)
    )
    reports: list[dict[str, Any]] = []
    for seed in seeds:
        print(
            f"[Expert Homogeneity] diagnosing initialization seed={seed} "
            f"({len(reports) + 1}/{len(seeds)})",
            flush=True,
        )
        base.set_seed(
            seed=seed,
            deterministic=bool(cfg.deterministic),
        )
        model = base.build_switch_vit_model(cfg).to(device)
        report = diagnose_model_experts(
            model=model,
            data_loader=fixed_batches,
            device=device,
            max_batches=args.max_batches,
            max_tokens_per_layer=args.max_tokens_per_layer,
            subspace_k=args.subspace_k,
            expert_batch_size=args.expert_batch_size,
            shuffled_cka_repeats=args.shuffle_repeats,
            shuffle_seed=seed,
            token_output_cosine=not args.no_token_cosine,
        )
        report["diagnostic_stage"] = "initialization"
        report["dataset"] = str(cfg.dataset)
        report["data_split"] = "train_evidence_no_augmentation"
        report["sampling_policy"] = "fixed_first_batches_then_evenly_spaced_tokens"
        report["model"] = "SwitchViT"
        report["model_state"] = "random_initialization"
        report["model_config"] = {
            "num_experts": int(cfg.num_experts),
            **model_cfg,
        }
        report["input_shape"] = list(dataset_bundle.input_shape)
        report["seed"] = seed
        report["device"] = str(device)
        report["metric_directions"] = {
            "weight_cosine": "near 1 means aligned parameter vectors",
            "subspace_overlap": "1 means identical Top-k subspaces",
            "cka": "high values mean similar sample geometry, not identical outputs",
            "shuffled_cka": "negative control with broken token correspondence",
            "cka_gap": "paired CKA minus shuffled CKA",
            "output_normalized_mse": "0 means identical outputs",
            "token_output_cosine_mean": "mean direction agreement per aligned token",
        }
        reports.append(report)
        del model

    if args.output is not None:
        output_path = args.output.expanduser()
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output_path, reports[0])

    metadata = {
        "schema_version": 1,
        "purpose": "random_initialization_baseline_and_negative_controls",
        "seeds": seeds,
        "dataset": str(cfg.dataset),
        "input_shape": list(dataset_bundle.input_shape),
        "num_fixed_images": sum(int(batch[0].shape[0]) for batch in fixed_batches),
        "data_split": "train_evidence_no_augmentation",
        "diagnosis_data_used_for_training": False,
        "num_batches": len(fixed_batches),
        "batch_size": batch_size,
        "max_tokens_per_layer": args.max_tokens_per_layer,
        "subspace_k": args.subspace_k,
        "expert_batch_size": args.expert_batch_size,
        "shuffle_repeats": args.shuffle_repeats,
        "token_cosine_enabled": not args.no_token_cosine,
        "device": str(device),
        "deterministic": bool(cfg.deterministic),
        "model_config": {"num_experts": int(cfg.num_experts), **model_cfg},
        "checkpoint_saved": False,
        "interpretation": (
            "Compare initialization distributions across seeds; shuffled CKA "
            "breaks token pairing. High CKA alone does not prove functional redundancy."
        ),
    }
    files = save_initialization_controls(
        reports,
        output_dir,
        metadata=metadata,
    )
    print(f"[Expert Homogeneity] saved to: {output_dir}")
    for label, path in files.items():
        print(f"  {label}: {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""独立的 Switch Transformer 专家同质化诊断工具。

本模块首先提供可复用的数学指标：
- 专家权重余弦相似度
- Top-k 主奇异子空间重叠
- Linear CKA
- 归一化专家输出 MSE

联邦训练接入、模型特征采集和结果输出在这些基础指标通过测试后接入。
本文件不修改模型参数，也不负责保存模型 Checkpoint。
"""

from __future__ import annotations

import math
from typing import Literal

import torch


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

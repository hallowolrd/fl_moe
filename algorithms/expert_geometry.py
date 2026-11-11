from __future__ import annotations

"""Algorithm-side parameter geometry helpers for expert Stiefel constraints.

The model only exposes ordinary named parameters. The selected aggregation
algorithm defines which expert parameter tensors form each Stiefel variable.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Dict, Iterable, Tuple

import torch


@dataclass(frozen=True)
class StiefelParameterBlock:
    """A Stiefel variable formed by stacking row-oriented Parameters.

    Each source Parameter has shape [rows_i, columns]. The block coordinate is
    X = cat(parameters, dim=0).T, so X has shape [columns, sum(rows_i)] and
    satisfies X.T @ X = I.
    """

    name: str
    parameter_names: Tuple[str, ...]
    shape: Tuple[int, int]

    def get_point(
        self,
        named_parameters: Mapping[str, torch.nn.Parameter],
    ) -> torch.Tensor:
        weights = [named_parameters[name] for name in self.parameter_names]
        return torch.cat([weight.detach() for weight in weights], dim=0).transpose(0, 1).clone()

    def get_gradient(
        self,
        named_parameters: Mapping[str, torch.nn.Parameter],
    ) -> torch.Tensor:
        gradients = []
        for name in self.parameter_names:
            parameter = named_parameters[name]
            gradient = parameter.grad
            if gradient is None:
                gradient = torch.zeros_like(parameter)
            elif gradient.is_sparse:
                raise NotImplementedError(
                    f"Stiefel block {self.name!r} does not support sparse gradient: {name}"
                )
            gradients.append(gradient.detach())
        return torch.cat(gradients, dim=0).transpose(0, 1).clone()

    @torch.no_grad()
    def set_point_(
        self,
        named_parameters: Mapping[str, torch.nn.Parameter],
        value: torch.Tensor,
    ) -> None:
        if not torch.is_tensor(value) or value.ndim != 2:
            raise TypeError(f"Stiefel block {self.name!r} must be a 2-D Tensor.")
        if tuple(value.shape) != self.shape:
            raise ValueError(
                f"Stiefel block {self.name!r} shape mismatch: "
                f"expected={self.shape}, actual={tuple(value.shape)}"
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"Stiefel block {self.name!r} contains NaN or Inf.")

        parameters = [named_parameters[name] for name in self.parameter_names]
        rows = value.detach().transpose(0, 1)
        offset = 0
        for parameter in parameters:
            count = int(parameter.shape[0])
            parameter.copy_(
                rows[offset:offset + count].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )
            offset += count

    @torch.no_grad()
    def retract_(
        self,
        named_parameters: Mapping[str, torch.nn.Parameter],
    ) -> torch.Tensor:
        point = self.get_point(named_parameters)
        compute_dtype = (
            torch.float64
            if point.dtype == torch.float64
            else torch.float32
            if point.dtype in (torch.float16, torch.bfloat16)
            else point.dtype
        )
        q, r = torch.linalg.qr(point.to(dtype=compute_dtype), mode="reduced")
        diagonal = torch.diagonal(r)
        signs = torch.where(
            diagonal < 0,
            -torch.ones_like(diagonal),
            torch.ones_like(diagonal),
        )
        retracted = (q * signs.unsqueeze(0)).to(dtype=point.dtype)
        self.set_point_(named_parameters, retracted)
        return retracted


def build_stiefel_parameter_blocks(
    model: torch.nn.Module,
    expert_parameter_names: Iterable[str],
    block_config: Mapping[str, object] | None,
) -> Dict[str, StiefelParameterBlock]:
    """Resolve algorithm-selected expert parameter patterns into Stiefel blocks.

    Configuration format:
        {
            "expert_fc1": {
                "parameter_patterns": ["moe_head.experts.*.fc1.weight"]
            }
        }

    A block may contain one matrix or multiple matrices. All selected matrices
    are stacked along rows and must share the same input-column dimension.
    """
    if block_config is None:
        block_config = {}
    if not isinstance(block_config, Mapping):
        raise TypeError("orthogonal.stiefel_blocks 必须是 mapping。")

    named_parameters = dict(model.named_parameters())
    expert_names = set(expert_parameter_names)
    result: Dict[str, StiefelParameterBlock] = {}
    assigned_names: set[str] = set()

    for raw_name, raw_spec in block_config.items():
        block_name = str(raw_name).strip()
        if not block_name:
            raise ValueError("Stiefel block 名称不能为空。")

        if isinstance(raw_spec, Mapping):
            patterns = raw_spec.get("parameter_patterns", raw_spec.get("patterns", ()))
        else:
            patterns = raw_spec

        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, Sequence) or not patterns:
            raise ValueError(
                f"Stiefel block {block_name!r} 必须提供非空 parameter_patterns。"
            )
        normalized_patterns = [str(pattern).strip() for pattern in patterns]
        if any(not pattern for pattern in normalized_patterns):
            raise ValueError(
                f"Stiefel block {block_name!r} 包含空参数匹配模式。"
            )

        matched_names = [
            name
            for name in named_parameters
            if name in expert_names
            and any(fnmatchcase(name, pattern) for pattern in normalized_patterns)
        ]
        if not matched_names:
            raise ValueError(
                f"Stiefel block {block_name!r} 未匹配到任何可训练专家参数；"
                f"patterns={normalized_patterns}"
            )

        overlap = assigned_names.intersection(matched_names)
        if overlap:
            raise ValueError(
                f"专家参数被多个 Stiefel block 重复选择：{sorted(overlap)}"
            )

        parameters = [named_parameters[name] for name in matched_names]
        for name, parameter in zip(matched_names, parameters):
            if not parameter.requires_grad:
                raise ValueError(
                    f"Stiefel block {block_name!r} 选择了被冻结的参数：{name}"
                )
            if parameter.ndim != 2:
                raise ValueError(
                    f"Stiefel block {block_name!r} 只支持二维参数矩阵；"
                    f"{name} shape={tuple(parameter.shape)}"
                )
            if not parameter.is_floating_point():
                raise TypeError(
                    f"Stiefel block {block_name!r} 要求浮点参数：{name}"
                )

        column_dims = {int(parameter.shape[1]) for parameter in parameters}
        if len(column_dims) != 1:
            raise ValueError(
                f"Stiefel block {block_name!r} 中的矩阵列数必须一致；"
                f"shapes={[tuple(parameter.shape) for parameter in parameters]}"
            )

        total_rows = sum(int(parameter.shape[0]) for parameter in parameters)
        column_dim = next(iter(column_dims))
        if total_rows > column_dim:
            raise ValueError(
                f"Stiefel block {block_name!r} 要求总行数不超过列数；"
                f"total_rows={total_rows}, columns={column_dim}。"
            )

        shape = (column_dim, total_rows)
        result[block_name] = StiefelParameterBlock(
            name=block_name,
            parameter_names=tuple(matched_names),
            shape=shape,
        )
        assigned_names.update(matched_names)

    return result


def stiefel_project_tangent(
    point: torch.Tensor,
    ambient_vector: torch.Tensor,
) -> torch.Tensor:
    """Orthogonally project an ambient matrix onto T_point St(n, p)."""
    if point.ndim != 2 or ambient_vector.shape != point.shape:
        raise ValueError(
            "Stiefel tangent projection requires equally shaped 2-D tensors; "
            f"got point={tuple(point.shape)}, vector={tuple(ambient_vector.shape)}."
        )
    if point.shape[0] < point.shape[1]:
        raise ValueError(
            f"Stiefel matrix must have n >= p, got {tuple(point.shape)}."
        )

    compute_dtype = (
        torch.float64
        if point.dtype == torch.float64 or ambient_vector.dtype == torch.float64
        else torch.float32
        if point.dtype in (torch.float16, torch.bfloat16)
        or ambient_vector.dtype in (torch.float16, torch.bfloat16)
        else point.dtype
    )
    x = point.to(dtype=compute_dtype)
    g = ambient_vector.to(device=point.device, dtype=compute_dtype)
    xtg = x.transpose(0, 1) @ g
    sym_xtg = 0.5 * (xtg + xtg.transpose(0, 1))
    return (g - x @ sym_xtg).to(dtype=point.dtype)


def stiefel_orthogonality_error(point: torch.Tensor) -> float:
    """Return ||X.T X - I||_F in a numerically safe accumulation dtype."""
    work = (
        point.float()
        if point.dtype in (torch.float16, torch.bfloat16)
        else point
    )
    identity = torch.eye(
        work.shape[1],
        device=work.device,
        dtype=work.dtype,
    )
    return float(
        torch.linalg.norm(
            work.transpose(0, 1) @ work - identity,
            ord="fro",
        ).item()
    )

from __future__ import annotations

"""Shared diagnostics for algorithm-selected Stiefel parameter blocks."""

from typing import Any, Dict, Mapping

import torch


def stiefel_orthogonality_error(point: torch.Tensor) -> float:
    """Return ||X^T X - I||_F using a safe accumulation dtype."""
    if not torch.is_tensor(point) or point.ndim != 2:
        raise TypeError("Stiefel point 必须是二维 Tensor。")

    work = point.detach()
    if work.dtype in (torch.float16, torch.bfloat16):
        work = work.float()
    elif not work.is_floating_point():
        raise TypeError(
            f"Stiefel point 必须是浮点 Tensor，当前 dtype={work.dtype}。"
        )

    if not bool(torch.isfinite(work).all().item()):
        raise ValueError("Stiefel point 包含 NaN 或 Inf。")

    identity = torch.eye(
        work.shape[1],
        device=work.device,
        dtype=work.dtype,
    )
    residual = work.transpose(0, 1) @ work - identity
    return float(torch.linalg.norm(residual, ord="fro").item())


@torch.no_grad()
def build_stiefel_orthogonality_diagnostics(
    model: torch.nn.Module,
    blocks: Mapping[str, Any],
) -> Dict[str, Any]:
    """Measure every configured Stiefel block on the final server model.

    mean_error is the unweighted arithmetic mean across configured blocks.
    An empty block mapping is represented with null aggregate metrics rather
    than zero, which would incorrectly imply a measured zero constraint error.
    """
    named_parameters = dict(model.named_parameters())
    block_diagnostics: Dict[str, Any] = {}
    errors = []

    for raw_name, block in blocks.items():
        name = str(raw_name)
        point = block.get_point(named_parameters)
        error = stiefel_orthogonality_error(point)
        errors.append(error)

        block_diagnostics[name] = {
            "orthogonality_error": error,
            "shape": list(point.shape),
            "parameter_names": list(block.parameter_names),
        }

    return {
        "metric": "frobenius_norm_XtX_minus_I",
        "measurement_stage": "after_server_aggregation_and_retraction",
        "max_error": max(errors) if errors else None,
        "mean_error": sum(errors) / len(errors) if errors else None,
        "blocks": block_diagnostics,
    }

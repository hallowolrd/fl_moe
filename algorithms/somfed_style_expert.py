from __future__ import annotations

"""SOMFed-style similarity-threshold expert aggregation for shared Sparse-MoE.

This module is a controlled adaptation of SOMFed's DFL-for-Experts (DE)
mechanism to the common global-expert architecture used by this project.
All model structure, local training, router training, non-expert aggregation,
data partitioning, and evaluation remain unchanged.

Paper-preserved core used here:
    1. Treat each post-local-training expert as a candidate expert model.
    2. Measure expert-model similarity with cosine similarity.
    3. For each target local expert, keep itself and every source expert whose
       similarity reaches the configured threshold (paper experiment: 0.6).
    4. Aggregate only the selected similar experts.

Controlled choices required by this project's comparison protocol:
    - SOMFed's full MASS / personalized MoE construction is not used.
    - The public paper text available to this implementation establishes
      similarity-threshold selection but does not provide a reliably exposed
      similarity-weighting equation for DE. Therefore selected experts are
      uniformly averaged; cosine similarity is used only for selection.
    - SOMFed does not collapse all personalized expert results into one shared
      global MoE. To keep every comparison method on the exact same global
      Sparse-MoE lifecycle, Scheme A is used:

          B[e, j] = mean_k A[(client_k, expert_e), j]

      where A is the personalized threshold-aggregation matrix and B is the
      global collapse matrix. Global expert e is then aggregated from the
      current round's local source experts using B[e, :].

The router remains a non-expert parameter and is still aggregated by base.py's
fixed uniform non-expert aggregation. No extra evidence pass or client state is
introduced.
"""

import argparse
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

import base


ALGORITHM_NAME = "somfed_style_expert"

Aggregator = base.Aggregator
AggregationResult = base.AggregationResult
ClientUpdate = base.ClientUpdate
clone_state_dict = base.clone_state_dict
check_finite_state_dict = base.check_finite_state_dict
build_uniform_weights = base.build_uniform_weights
ConfigError = getattr(base, "ConfigError", ValueError)


EMBEDDED_METHOD_CONFIG = {
    "agg": {
        "non_expert": {"method": "uniform"},
        "expert": {"method": ALGORITHM_NAME},
    },
    "somfed": {
        "similarity_threshold": 0.6,
        "similarity_source": "expert_params",
        "log_matrix": False,
        "log_top_sources": 8,
        "log_detail": True,
    },
}


METHOD_CONFIG_DEFAULTS = {
    "somfed": {
        "similarity_threshold": 0.6,
        "similarity_source": "expert_params",
        "log_matrix": False,
        "log_top_sources": 8,
        "log_detail": True,
    },
}


_EXPERT_ID_PATTERN = re.compile(r"(?P<prefix>(?:^|\.)experts\.)(?P<id>\d+)(?=\.|$)")


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """Read dotted configuration paths from ConfigNode/dict/object values."""
    if cfg is None:
        return default

    if hasattr(cfg, "get"):
        try:
            value = cfg.get(path, None)
        except TypeError:
            value = None
        if value is not None:
            return value

    current = cfg
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return default

    return current


def validate_method_config(cfg: Mapping[str, Any]) -> None:
    """Validate SOMFed-style method-owned configuration."""
    method_cfg = cfg.get("somfed", {})
    if not isinstance(method_cfg, Mapping):
        raise ConfigError("somfed 必须是 dict。")

    threshold = float(method_cfg.get("similarity_threshold", 0.6))
    if not math.isfinite(threshold) or not (-1.0 <= threshold <= 1.0):
        raise ConfigError(
            "somfed.similarity_threshold 必须是 [-1, 1] 内的有限数，"
            f"当前值：{threshold}"
        )

    similarity_source = str(
        method_cfg.get("similarity_source", "expert_params")
    ).lower().strip()
    if similarity_source != "expert_params":
        raise ConfigError(
            "当前 SOMFed-style 主版本只支持 "
            "somfed.similarity_source=expert_params。"
            "delta cosine 不属于本 baseline 的主聚合规则。"
        )

    log_top_sources = int(method_cfg.get("log_top_sources", 8))
    if log_top_sources < 0:
        raise ConfigError(
            "somfed.log_top_sources 不能小于 0，"
            f"当前值：{log_top_sources}"
        )


def register_method_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Register SOMFed-only command-line overrides."""
    parser.add_argument("--somfed-threshold", type=float, default=None)


def build_method_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Map explicit SOMFed CLI values into the nested configuration schema."""
    overrides: Dict[str, Any] = {
        "agg": {
            "non_expert": {"method": "uniform"},
            "expert": {"method": ALGORITHM_NAME},
        }
    }

    if getattr(args, "somfed_threshold", None) is not None:
        overrides["somfed"] = {
            "similarity_threshold": float(args.somfed_threshold),
        }

    return overrides


class SOMFedStyleExpertAggregator(Aggregator):
    """Expert-parameter cosine thresholding plus Scheme-A global collapse."""

    @property
    def method_name(self) -> str:
        return ALGORITHM_NAME

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """Interface fallback; actual aggregation uses a cross-expert matrix."""
        return build_uniform_weights(client_updates)

    def aggregate(
        self,
        global_state: Mapping[str, torch.Tensor],
        client_updates: Sequence[ClientUpdate],
        param_names: Optional[Iterable[str]] = None,
        base_state: Optional[Mapping[str, torch.Tensor]] = None,
        strict: bool = True,
    ) -> AggregationResult:
        self._validate_client_updates(client_updates)

        if self.param_group_name != "expert":
            raise ValueError("somfed_style_expert 只能用于 expert 参数聚合。")

        target_param_names = _resolve_param_names(global_state, param_names)
        expert_param_names = _group_expert_param_names(target_param_names)
        expert_ids = sorted(expert_param_names)

        if not expert_ids:
            raise ValueError("没有找到 experts.<id> 参数，无法执行 SOMFed-style 聚合。")

        expected_expert_ids = list(range(len(expert_ids)))
        if expert_ids != expected_expert_ids:
            raise ValueError(
                "SOMFed-style 当前要求 expert id 连续从 0 开始："
                f"found={expert_ids}, expected={expected_expert_ids}"
            )

        client_updates_sorted = sorted(
            client_updates,
            key=lambda update: int(update.client_id),
        )
        client_ids = [int(update.client_id) for update in client_updates_sorted]
        pairs = [
            (client_id, expert_id)
            for client_id in client_ids
            for expert_id in expert_ids
        ]

        local_expert_vectors = _reconstruct_local_expert_vectors(
            global_state=global_state,
            client_updates=client_updates_sorted,
            expert_param_names=expert_param_names,
            expert_ids=expert_ids,
            strict=strict,
        )
        similarity_matrix = _build_cosine_similarity_matrix(local_expert_vectors)

        threshold = float(
            _cfg_get(self.cfg, "somfed.similarity_threshold", 0.6)
        )
        aggregation_matrix = _build_threshold_uniform_matrix(
            similarity_matrix=similarity_matrix,
            threshold=threshold,
        )

        collapse_matrix = _build_uniform_collapse_matrix(
            aggregation_matrix=aggregation_matrix,
            pairs=pairs,
            client_ids=client_ids,
            expert_ids=expert_ids,
        )

        if base_state is None:
            new_state_dict = clone_state_dict(global_state)
        else:
            new_state_dict = clone_state_dict(base_state)

        _apply_global_expert_collapse(
            new_state_dict=new_state_dict,
            global_state=global_state,
            client_updates=client_updates_sorted,
            target_param_names=target_param_names,
            expert_ids=expert_ids,
            pairs=pairs,
            collapse_matrix=collapse_matrix,
            strict=strict,
        )

        check_finite_state_dict(
            state_dict=new_state_dict,
            param_names=target_param_names,
        )

        result_weights = _collapse_source_weights_to_clients(
            collapse_matrix=collapse_matrix,
            pairs=pairs,
            client_ids=client_ids,
        )
        round_id = _resolve_round_id(client_updates_sorted)

        diagnostics = _build_diagnostics(
            method_name=self.method_name,
            param_group_name=self.param_group_name,
            target_param_names=target_param_names,
            client_ids=client_ids,
            expert_ids=expert_ids,
            pairs=pairs,
            result_weights=result_weights,
            similarity_matrix=similarity_matrix,
            aggregation_matrix=aggregation_matrix,
            collapse_matrix=collapse_matrix,
            threshold=threshold,
            round_id=round_id,
            cfg=self.cfg,
        )

        if bool(_cfg_get(self.cfg, "somfed.log_detail", True)):
            print(
                "[SOMFed-style] "
                f"round={round_id} "
                f"threshold={threshold:.6g} "
                f"local_experts={diagnostics['num_local_experts']} "
                f"global_experts={diagnostics['num_global_experts']} "
                f"mean_similarity={diagnostics['mean_offdiag_similarity']:.6f} "
                f"fraction_above_threshold={diagnostics['fraction_offdiag_above_threshold']:.6f} "
                f"mean_selected={diagnostics['mean_selected_experts_per_personalized_row']:.2f} "
                f"self_only={diagnostics['self_only_targets']} "
                f"all_selected={diagnostics['all_selected_targets']}",
                flush=True,
            )

        return AggregationResult(
            new_state_dict=new_state_dict,
            weights=result_weights,
            diagnostics=diagnostics,
        )


def _resolve_param_names(
    global_state: Mapping[str, torch.Tensor],
    param_names: Optional[Iterable[str]],
) -> List[str]:
    if param_names is None:
        names = [
            name
            for name in global_state.keys()
            if _get_expert_id(name) is not None
        ]
    else:
        names = list(param_names)

    for name in names:
        if name not in global_state:
            raise KeyError(f"global_state 中不存在参数：{name}")
        if _get_expert_id(name) is None:
            raise ValueError(
                "SOMFed-style 的 param_names 只能包含 expert 参数，"
                f"发现：{name}"
            )

    return names


def _get_expert_id(name: str) -> Optional[int]:
    match = _EXPERT_ID_PATTERN.search(name)
    if match is None:
        return None
    return int(match.group("id"))


def _replace_expert_id(name: str, expert_id: int) -> str:
    match = _EXPERT_ID_PATTERN.search(name)
    if match is None:
        raise ValueError(f"参数名中没有 experts.<id>：{name}")

    start, end = match.span("id")
    return f"{name[:start]}{int(expert_id)}{name[end:]}"


def _group_expert_param_names(
    param_names: Sequence[str],
) -> Dict[int, List[str]]:
    result: Dict[int, List[str]] = {}
    for name in param_names:
        expert_id = _get_expert_id(name)
        if expert_id is None:
            continue
        result.setdefault(expert_id, []).append(name)

    return {
        expert_id: sorted(names)
        for expert_id, names in sorted(result.items())
    }


def _resolve_round_id(client_updates: Sequence[ClientUpdate]) -> int:
    round_ids = {int(update.round_id) for update in client_updates}
    if len(round_ids) != 1:
        raise ValueError(f"同一聚合批次出现多个 round_id：{sorted(round_ids)}")

    round_id = next(iter(round_ids))
    if round_id <= 0:
        raise ValueError(f"round_id 必须大于 0，当前值：{round_id}")
    return round_id


def _reconstruct_local_expert_vectors(
    global_state: Mapping[str, torch.Tensor],
    client_updates: Sequence[ClientUpdate],
    expert_param_names: Mapping[int, Sequence[str]],
    expert_ids: Sequence[int],
    strict: bool,
) -> torch.Tensor:
    """Flatten every post-local-training full expert into one cosine vector."""
    vectors: List[torch.Tensor] = []

    # All slots must have the same relative parameter structure so cross-slot
    # parameter aggregation is well-defined.
    reference_id = int(expert_ids[0])
    reference_names = list(expert_param_names[reference_id])
    reference_relative = [
        _replace_expert_id(name, 0)
        for name in reference_names
    ]

    for expert_id in expert_ids[1:]:
        names = list(expert_param_names[int(expert_id)])
        relative = [
            _replace_expert_id(name, 0)
            for name in names
        ]
        if relative != reference_relative:
            raise ValueError(
                "不同 expert slot 的参数结构不一致，无法执行跨 expert SOMFed 聚合："
                f"expert0={reference_relative}, expert{expert_id}={relative}"
            )

    for update in client_updates:
        for expert_id in expert_ids:
            parts: List[torch.Tensor] = []
            for target_name in reference_names:
                source_name = _replace_expert_id(target_name, int(expert_id))
                if source_name not in global_state:
                    raise KeyError(f"global_state 缺少 expert 参数：{source_name}")

                source_global = global_state[source_name]
                if not torch.is_tensor(source_global):
                    continue
                if not torch.is_floating_point(source_global):
                    continue

                if source_name not in update.model_delta:
                    if strict:
                        raise KeyError(
                            f"客户端 {update.client_id} 的 model_delta 缺少 expert 参数："
                            f"{source_name}"
                        )
                    source_local = source_global.detach().cpu().float()
                else:
                    source_local = (
                        source_global.detach().cpu().float()
                        + update.model_delta[source_name].detach().cpu().float()
                    )

                if not torch.isfinite(source_local).all():
                    raise ValueError(
                        f"客户端 {update.client_id} expert {expert_id} 参数 "
                        f"{source_name} 包含 NaN/Inf。"
                    )
                parts.append(source_local.reshape(-1))

            if not parts:
                raise ValueError(
                    f"客户端 {update.client_id} expert {expert_id} 没有可用于 cosine 的浮点参数。"
                )

            vector = torch.cat(parts, dim=0)
            norm = float(vector.norm().item())
            if norm <= 1.0e-12:
                raise ValueError(
                    f"客户端 {update.client_id} expert {expert_id} 参数向量范数为 0，"
                    "无法计算 cosine similarity。"
                )
            vectors.append(vector)

    return torch.stack(vectors, dim=0)


def _build_cosine_similarity_matrix(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.dim() != 2:
        raise ValueError(f"vectors 必须为 [M,D]，当前 shape={tuple(vectors.shape)}")

    normalized = F.normalize(vectors.float(), p=2, dim=1, eps=1.0e-12)
    similarity = normalized.matmul(normalized.transpose(0, 1))
    similarity = similarity.clamp(min=-1.0, max=1.0)

    if not torch.isfinite(similarity).all():
        raise ValueError("SOMFed cosine similarity matrix 出现 NaN/Inf。")

    return similarity


def _build_threshold_uniform_matrix(
    similarity_matrix: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Select similar experts by threshold and uniformly average them."""
    if similarity_matrix.dim() != 2:
        raise ValueError("similarity_matrix 必须是二维矩阵。")
    if similarity_matrix.size(0) != similarity_matrix.size(1):
        raise ValueError("similarity_matrix 必须是方阵。")
    if not math.isfinite(float(threshold)) or not (-1.0 <= threshold <= 1.0):
        raise ValueError("threshold 必须是 [-1,1] 内的有限数。")

    num_local_experts = int(similarity_matrix.size(0))
    if num_local_experts <= 0:
        raise ValueError("没有 local experts。")

    matrix = torch.zeros_like(similarity_matrix, dtype=torch.float32)

    for row_id in range(num_local_experts):
        row = similarity_matrix[row_id].float()
        mask = row >= float(threshold)

        # The local target expert always keeps itself, even under an extreme
        # threshold or tiny floating-point diagonal deviation.
        mask[row_id] = True

        selected_count = int(mask.sum().item())
        if selected_count <= 0:
            raise RuntimeError("SOMFed threshold selection 产生了空集合。")

        matrix[row_id, mask] = 1.0 / float(selected_count)

    row_sums = matrix.sum(dim=1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise RuntimeError("SOMFed personalized aggregation matrix A 的行权重和不为 1。")

    return matrix


def _build_uniform_collapse_matrix(
    aggregation_matrix: torch.Tensor,
    pairs: Sequence[Tuple[int, int]],
    client_ids: Sequence[int],
    expert_ids: Sequence[int],
) -> torch.Tensor:
    """Scheme A: B[e,:] = mean_k A[(client,e),:]."""
    if aggregation_matrix.size(0) != len(pairs):
        raise ValueError("aggregation_matrix 与 pairs 尺寸不一致。")

    pair_to_index = {
        (int(client_id), int(expert_id)): index
        for index, (client_id, expert_id) in enumerate(pairs)
    }

    rows: List[torch.Tensor] = []
    for expert_id in expert_ids:
        target_rows = [
            aggregation_matrix[pair_to_index[(int(client_id), int(expert_id))]]
            for client_id in client_ids
        ]
        rows.append(torch.stack(target_rows, dim=0).mean(dim=0))

    collapse = torch.stack(rows, dim=0)
    row_sums = collapse.sum(dim=1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise RuntimeError("SOMFed Scheme-A collapse matrix B 的行权重和不为 1。")

    return collapse


def _apply_global_expert_collapse(
    new_state_dict: Dict[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
    client_updates: Sequence[ClientUpdate],
    target_param_names: Sequence[str],
    expert_ids: Sequence[int],
    pairs: Sequence[Tuple[int, int]],
    collapse_matrix: torch.Tensor,
    strict: bool,
) -> None:
    """Aggregate current local source experts into shared global expert slots."""
    update_by_client = {
        int(update.client_id): update
        for update in client_updates
    }
    expert_row = {
        int(expert_id): row_id
        for row_id, expert_id in enumerate(expert_ids)
    }

    for target_name in target_param_names:
        target_expert_id = _get_expert_id(target_name)
        if target_expert_id is None:
            continue

        target_global = global_state[target_name]
        if not torch.is_tensor(target_global):
            continue
        if not torch.is_floating_point(target_global):
            continue

        row = collapse_matrix[expert_row[int(target_expert_id)]]
        aggregated = torch.zeros_like(target_global)

        for source_index, ((client_id, source_expert_id), coefficient) in enumerate(
            zip(pairs, row)
        ):
            weight = float(coefficient.item())
            if weight == 0.0:
                continue

            source_name = _replace_expert_id(target_name, source_expert_id)
            if source_name not in global_state:
                if strict:
                    raise KeyError(
                        f"source expert 参数不存在：{source_name} "
                        f"(target={target_name}, source_index={source_index})"
                    )
                continue

            source_global = global_state[source_name]
            if tuple(source_global.shape) != tuple(target_global.shape):
                raise ValueError(
                    "跨 expert 聚合要求对应参数 shape 一致："
                    f"target={target_name}{tuple(target_global.shape)}, "
                    f"source={source_name}{tuple(source_global.shape)}"
                )

            update = update_by_client[int(client_id)]
            if source_name not in update.model_delta:
                if strict:
                    raise KeyError(
                        f"客户端 {client_id} 的 model_delta 缺少 source expert 参数："
                        f"{source_name}"
                    )
                source_local = source_global
            else:
                source_local = (
                    source_global
                    + update.model_delta[source_name].to(
                        device=source_global.device,
                        dtype=source_global.dtype,
                    )
                )

            aggregated = aggregated + weight * source_local.to(
                device=target_global.device,
                dtype=target_global.dtype,
            )

        new_state_dict[target_name] = aggregated


def _collapse_source_weights_to_clients(
    collapse_matrix: torch.Tensor,
    pairs: Sequence[Tuple[int, int]],
    client_ids: Sequence[int],
) -> Dict[int, float]:
    """Create a client-level diagnostic weight from the global B matrix."""
    if collapse_matrix.numel() == 0:
        return {int(client_id): 0.0 for client_id in client_ids}

    source_weight = collapse_matrix.mean(dim=0)
    result = {int(client_id): 0.0 for client_id in client_ids}

    for index, (client_id, _) in enumerate(pairs):
        result[int(client_id)] += float(source_weight[index].item())

    total = float(sum(result.values()))
    if total > 0:
        result = {
            client_id: float(value) / total
            for client_id, value in result.items()
        }

    return result


def _offdiag_values(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.dim() != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError("matrix 必须是方阵。")

    size = int(matrix.size(0))
    if size <= 1:
        return matrix.new_empty((0,))

    mask = ~torch.eye(size, dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def _build_diagnostics(
    *,
    method_name: str,
    param_group_name: str,
    target_param_names: Sequence[str],
    client_ids: Sequence[int],
    expert_ids: Sequence[int],
    pairs: Sequence[Tuple[int, int]],
    result_weights: Mapping[int, float],
    similarity_matrix: torch.Tensor,
    aggregation_matrix: torch.Tensor,
    collapse_matrix: torch.Tensor,
    threshold: float,
    round_id: int,
    cfg: Any,
) -> Dict[str, Any]:
    selected_per_row = (aggregation_matrix > 0).sum(dim=1).tolist()
    num_local_experts = len(pairs)
    offdiag = _offdiag_values(similarity_matrix)

    if int(offdiag.numel()) > 0:
        mean_offdiag = float(offdiag.mean().item())
        min_offdiag = float(offdiag.min().item())
        max_offdiag = float(offdiag.max().item())
        fraction_above = float((offdiag >= float(threshold)).float().mean().item())
    else:
        mean_offdiag = 1.0
        min_offdiag = 1.0
        max_offdiag = 1.0
        fraction_above = 0.0

    self_only_targets = sum(int(value) == 1 for value in selected_per_row)
    all_selected_targets = sum(
        int(value) == num_local_experts
        for value in selected_per_row
    )

    # How often a target row chooses experts from another client / another slot.
    pair_to_index = {
        (int(client_id), int(expert_id)): index
        for index, (client_id, expert_id) in enumerate(pairs)
    }
    cross_client_selected = 0
    cross_slot_selected = 0
    nonself_selected = 0

    for target_client, target_expert in pairs:
        row_id = pair_to_index[(int(target_client), int(target_expert))]
        selected_indices = torch.nonzero(
            aggregation_matrix[row_id] > 0,
            as_tuple=False,
        ).reshape(-1).tolist()

        for source_index in selected_indices:
            source_client, source_expert = pairs[int(source_index)]
            if (
                int(source_client) == int(target_client)
                and int(source_expert) == int(target_expert)
            ):
                continue

            nonself_selected += 1
            if int(source_client) != int(target_client):
                cross_client_selected += 1
            if int(source_expert) != int(target_expert):
                cross_slot_selected += 1

    denom = max(nonself_selected, 1)

    top_sources_limit = int(_cfg_get(cfg, "somfed.log_top_sources", 8))
    top_sources: Dict[int, List[Dict[str, Any]]] = {}
    if top_sources_limit > 0:
        for row_id, expert_id in enumerate(expert_ids):
            row = collapse_matrix[row_id]
            count = min(top_sources_limit, int(row.numel()))
            values, indices = torch.topk(row, k=count, largest=True, sorted=True)
            top_sources[int(expert_id)] = [
                {
                    "client_id": int(pairs[int(index)][0]),
                    "source_expert_id": int(pairs[int(index)][1]),
                    "weight": float(value),
                }
                for value, index in zip(values.tolist(), indices.tolist())
                if float(value) > 0.0
            ]

    diagnostics: Dict[str, Any] = {
        "method": method_name,
        "param_group": param_group_name,
        "num_clients": len(client_ids),
        "param_count": len(target_param_names),
        "weights": {
            int(client_id): float(weight)
            for client_id, weight in result_weights.items()
        },
        "adaptation": "somfed_de_threshold_uniform_plus_scheme_a_global_collapse",
        "similarity_source": "expert_params",
        "similarity_threshold": float(threshold),
        "round_id": int(round_id),
        "num_global_experts": len(expert_ids),
        "num_local_experts": num_local_experts,
        "mean_offdiag_similarity": mean_offdiag,
        "min_offdiag_similarity": min_offdiag,
        "max_offdiag_similarity": max_offdiag,
        "fraction_offdiag_above_threshold": fraction_above,
        "mean_selected_experts_per_personalized_row": float(
            sum(int(value) for value in selected_per_row)
            / max(len(selected_per_row), 1)
        ),
        "min_selected_experts_per_personalized_row": int(min(selected_per_row)),
        "max_selected_experts_per_personalized_row": int(max(selected_per_row)),
        "self_only_targets": int(self_only_targets),
        "all_selected_targets": int(all_selected_targets),
        "cross_client_selection_ratio": float(cross_client_selected / denom),
        "cross_slot_selection_ratio": float(cross_slot_selected / denom),
        "top_sources_by_global_expert": top_sources,
    }

    if bool(_cfg_get(cfg, "somfed.log_matrix", False)):
        diagnostics["expert_pairs"] = [
            {
                "index": index,
                "client_id": int(client_id),
                "expert_id": int(expert_id),
            }
            for index, (client_id, expert_id) in enumerate(pairs)
        ]
        diagnostics["similarity_matrix_R"] = similarity_matrix.tolist()
        diagnostics["personalized_aggregation_matrix_A"] = aggregation_matrix.tolist()
        diagnostics["global_collapse_matrix_B"] = collapse_matrix.tolist()

    return diagnostics


def build_expert_aggregator(cfg: Any) -> base.Aggregator:
    return SOMFedStyleExpertAggregator(
        cfg=cfg,
        param_group_name="expert",
    )


def main() -> int:
    kwargs: Dict[str, Any] = {
        "expert_aggregator_builder": build_expert_aggregator,
        "embedded_method_config": EMBEDDED_METHOD_CONFIG,
        "expert_method_name": ALGORITHM_NAME,
        "method_config_defaults": METHOD_CONFIG_DEFAULTS,
        "method_config_validator": validate_method_config,
        "method_cli_argument_registrar": register_method_cli_arguments,
        "method_cli_overrides_builder": build_method_cli_overrides,
    }
    return base.main(**kwargs)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

"""FedMoE-DA-style domain-aware expert aggregation for the shared Sparse-MoE.

This module is a controlled adaptation of FedMoE-DA's expert aggregation
(Eq. 8-11) to the common global-expert architecture used by this project.
All model structure, local training, router training, non-expert aggregation,
data partitioning, and evaluation remain unchanged.

Paper-preserved core:
    1. Reconstruct each client's post-local-training gate proxies.
    2. Compute cosine similarity among all client-expert proxy vectors.
    3. For each target local expert, keep itself plus its P most relevant
       experts using the paper's top-value threshold rule.
    4. Apply softmax(similarity / tau) to obtain the sparse aggregation
       matrix A.
    5. Aggregate local experts with A (FedMoE-DA Eq. 11).
    6. Reuse a stale A and refresh it every I rounds.

Controlled adaptation required by this project's global-model protocol:
    FedMoE-DA keeps a personalized expert set for every client. This project
    must end each communication round with one shared set of global experts.
    Therefore, for global expert slot e, we uniformly average the FedMoE-DA
    personalized rows whose target slot is e. Equivalently, we collapse A to
    a global matrix B:

        B[e, j] = mean_k A[(client_k, expert_e), j]

    and then aggregate the current round's local source experts with B.

The router itself remains a non-expert parameter and is still aggregated by
base.py's fixed uniform non-expert aggregation. No extra client evidence pass
is introduced.
"""

import argparse
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

import base


ALGORITHM_NAME = "fedmoe_da_style_expert"

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
    "fedmoe_da": {
        "requested_experts": 5,
        "temperature": 1.0,
        "matrix_update_interval": 5,
        "proxy_param_name": "moe_head.gating.gate.weight",
        "require_full_participation": True,
        "log_matrix": False,
        "log_top_sources": 8,
    },
}


# Method defaults are supplied even when an external YAML file is used.
METHOD_CONFIG_DEFAULTS = {
    "fedmoe_da": {
        "requested_experts": 5,
        "temperature": 1.0,
        "matrix_update_interval": 5,
        "proxy_param_name": "moe_head.gating.gate.weight",
        "require_full_participation": True,
        "log_matrix": False,
        "log_top_sources": 8,
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
    """Validate FedMoE-DA-style method-owned configuration."""
    method_cfg = cfg.get("fedmoe_da", {})
    if not isinstance(method_cfg, Mapping):
        raise ConfigError("fedmoe_da 必须是 dict。")

    requested_experts = int(method_cfg.get("requested_experts", 5))
    if requested_experts < 0:
        raise ConfigError(
            "fedmoe_da.requested_experts 不能小于 0，"
            f"当前值：{requested_experts}"
        )

    temperature = float(method_cfg.get("temperature", 1.0))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ConfigError(
            "fedmoe_da.temperature 必须是有限正数，"
            f"当前值：{temperature}"
        )

    interval = int(method_cfg.get("matrix_update_interval", 5))
    if interval <= 0:
        raise ConfigError(
            "fedmoe_da.matrix_update_interval 必须是正整数，"
            f"当前值：{interval}"
        )

    proxy_param_name = str(
        method_cfg.get("proxy_param_name", "moe_head.gating.gate.weight")
    ).strip()
    if not proxy_param_name:
        raise ConfigError("fedmoe_da.proxy_param_name 不能为空。")

    log_top_sources = int(method_cfg.get("log_top_sources", 8))
    if log_top_sources < 0:
        raise ConfigError(
            "fedmoe_da.log_top_sources 不能小于 0，"
            f"当前值：{log_top_sources}"
        )

    if bool(method_cfg.get("require_full_participation", True)):
        frac = float(cfg.get("frac", 1.0))
        if abs(frac - 1.0) > 1.0e-12:
            raise ConfigError(
                "FedMoE-DA-style 默认要求 frac=1.0。论文 Algorithm 1 每轮使用所有客户端，"
                "且历史 aggregation matrix 的行/列对应固定 client-expert pool。"
                f"当前 frac={frac}。"
            )


def register_method_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Register method-specific CLI overrides when supported by base.py."""
    parser.add_argument("--fedmoe-da-p", type=int, default=None)
    parser.add_argument("--fedmoe-da-tau", type=float, default=None)
    parser.add_argument("--fedmoe-da-interval", type=int, default=None)


def build_method_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Map method identity and explicit FedMoE-DA CLI values into config."""
    # Keep the entrypoint authoritative even when --config is used. This makes
    # sure run naming / saved config cannot accidentally retain the base
    # default expert method (uniform).
    overrides: Dict[str, Any] = {
        "agg": {
            "non_expert": {"method": "uniform"},
            "expert": {"method": ALGORITHM_NAME},
        }
    }
    values: Dict[str, Any] = {}

    if getattr(args, "fedmoe_da_p", None) is not None:
        values["requested_experts"] = int(args.fedmoe_da_p)
    if getattr(args, "fedmoe_da_tau", None) is not None:
        values["temperature"] = float(args.fedmoe_da_tau)
    if getattr(args, "fedmoe_da_interval", None) is not None:
        values["matrix_update_interval"] = int(args.fedmoe_da_interval)

    if values:
        overrides["fedmoe_da"] = values

    return overrides


class FedMoEDAStyleExpertAggregator(Aggregator):
    """FedMoE-DA Eq. (8)-(11) plus uniform global-expert collapse."""

    def __init__(self, cfg: Any, param_group_name: str) -> None:
        super().__init__(cfg=cfg, param_group_name=param_group_name)
        self._aggregation_matrix: Optional[torch.Tensor] = None
        self._matrix_pairs: List[Tuple[int, int]] = []
        self._matrix_round: Optional[int] = None
        self._matrix_proxy_param_name: Optional[str] = None

    @property
    def method_name(self) -> str:
        return ALGORITHM_NAME

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """Interface fallback; the actual aggregation uses cross-expert A/B."""
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
            raise ValueError("fedmoe_da_style_expert 只能用于 expert 参数聚合。")

        target_param_names = _resolve_param_names(global_state, param_names)
        expert_param_names = _group_expert_param_names(target_param_names)
        expert_ids = sorted(expert_param_names)

        if not expert_ids:
            raise ValueError("没有找到 experts.<id> 参数，无法执行 FedMoE-DA-style 聚合。")

        client_updates_sorted = sorted(
            client_updates,
            key=lambda update: int(update.client_id),
        )
        client_ids = [int(update.client_id) for update in client_updates_sorted]

        expected_num_clients = int(_cfg_get(self.cfg, "num_clients", len(client_ids)))
        require_full = bool(
            _cfg_get(self.cfg, "fedmoe_da.require_full_participation", True)
        )
        if require_full and len(client_ids) != expected_num_clients:
            raise ValueError(
                "FedMoE-DA-style 要求本轮所有客户端参与："
                f"expected={expected_num_clients}, actual={len(client_ids)}。"
            )

        proxy_param_name = _resolve_proxy_param_name(
            global_state=global_state,
            configured_name=str(
                _cfg_get(
                    self.cfg,
                    "fedmoe_da.proxy_param_name",
                    "moe_head.gating.gate.weight",
                )
            ),
            num_experts=len(expert_ids),
        )

        global_proxy = global_state[proxy_param_name]
        if global_proxy.dim() != 2:
            raise ValueError(
                f"router proxy 参数必须是二维矩阵，当前 {proxy_param_name} "
                f"shape={tuple(global_proxy.shape)}"
            )
        if int(global_proxy.size(0)) != len(expert_ids):
            raise ValueError(
                "router proxy 数量与 expert 数量不一致："
                f"proxy_rows={int(global_proxy.size(0))}, expert_ids={expert_ids}"
            )

        expected_expert_ids = list(range(len(expert_ids)))
        if expert_ids != expected_expert_ids:
            raise ValueError(
                "FedMoE-DA-style 当前要求 expert id 连续从 0 开始："
                f"found={expert_ids}, expected={expected_expert_ids}"
            )

        pairs = [
            (client_id, expert_id)
            for client_id in client_ids
            for expert_id in expert_ids
        ]
        round_id = _resolve_round_id(client_updates_sorted)
        interval = int(
            _cfg_get(self.cfg, "fedmoe_da.matrix_update_interval", 5)
        )

        scheduled_refresh = (
            self._aggregation_matrix is None
            or self._matrix_round is None
            or ((round_id - 1) % interval == 0)
        )
        pair_change = pairs != self._matrix_pairs
        proxy_name_change = proxy_param_name != self._matrix_proxy_param_name
        matrix_refreshed = bool(
            scheduled_refresh or pair_change or proxy_name_change
        )
        forced_refresh = bool(
            matrix_refreshed and not scheduled_refresh
        )

        similarity_matrix: Optional[torch.Tensor] = None
        if matrix_refreshed:
            proxies = _reconstruct_local_proxies(
                global_state=global_state,
                client_updates=client_updates_sorted,
                proxy_param_name=proxy_param_name,
                expert_ids=expert_ids,
                strict=strict,
            )
            similarity_matrix = _build_cosine_similarity_matrix(proxies)
            self._aggregation_matrix = _build_fedmoe_da_matrix(
                similarity_matrix=similarity_matrix,
                requested_experts=int(
                    _cfg_get(self.cfg, "fedmoe_da.requested_experts", 5)
                ),
                temperature=float(
                    _cfg_get(self.cfg, "fedmoe_da.temperature", 1.0)
                ),
            )
            self._matrix_pairs = list(pairs)
            self._matrix_round = int(round_id)
            self._matrix_proxy_param_name = proxy_param_name

        if self._aggregation_matrix is None:
            raise RuntimeError("FedMoE-DA aggregation matrix 未初始化。")

        aggregation_matrix = self._aggregation_matrix
        if aggregation_matrix.size(0) != len(pairs):
            raise RuntimeError(
                "stale aggregation matrix 尺寸与当前 client-expert pool 不一致。"
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

        diagnostics = _build_diagnostics(
            method_name=self.method_name,
            param_group_name=self.param_group_name,
            target_param_names=target_param_names,
            client_ids=client_ids,
            expert_ids=expert_ids,
            pairs=pairs,
            result_weights=result_weights,
            aggregation_matrix=aggregation_matrix,
            collapse_matrix=collapse_matrix,
            similarity_matrix=similarity_matrix,
            matrix_refreshed=matrix_refreshed,
            forced_refresh=forced_refresh,
            matrix_round=int(self._matrix_round or round_id),
            round_id=round_id,
            proxy_param_name=proxy_param_name,
            cfg=self.cfg,
        )

        if bool(_cfg_get(self.cfg, "fedmoe_da.log_detail", True)):
            print(
                "[FedMoE-DA-style] "
                f"round={round_id} "
                f"P={diagnostics['requested_experts']} "
                f"effective_P={diagnostics['effective_requested_experts']} "
                f"tau={diagnostics['temperature']:.6g} "
                f"I={diagnostics['matrix_update_interval']} "
                f"matrix_refreshed={diagnostics['matrix_refreshed']} "
                f"matrix_round={diagnostics['matrix_round']} "
                f"local_experts={diagnostics['num_local_experts']} "
                f"global_experts={diagnostics['num_global_experts']} "
                f"mean_sources_per_row={diagnostics['mean_sources_per_personalized_row']:.2f}",
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
                "FedMoE-DA-style 的 param_names 只能包含 expert 参数，"
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
        expert_id: names
        for expert_id, names in sorted(result.items())
    }


def _resolve_proxy_param_name(
    global_state: Mapping[str, torch.Tensor],
    configured_name: str,
    num_experts: int,
) -> str:
    configured_name = configured_name.strip()
    if configured_name in global_state:
        return configured_name

    candidates: List[str] = []
    for name, tensor in global_state.items():
        if _get_expert_id(name) is not None:
            continue
        if not torch.is_tensor(tensor) or tensor.dim() != 2:
            continue
        if int(tensor.size(0)) != int(num_experts):
            continue
        lowered = name.lower()
        if lowered.endswith("gating.gate.weight") or lowered.endswith("gate.weight"):
            candidates.append(name)

    if len(candidates) == 1:
        return candidates[0]

    raise KeyError(
        "无法定位 FedMoE-DA proxy/gating 参数。"
        f"配置名={configured_name!r}, 自动候选={candidates}。"
    )


def _resolve_round_id(client_updates: Sequence[ClientUpdate]) -> int:
    round_ids = {int(update.round_id) for update in client_updates}
    if len(round_ids) != 1:
        raise ValueError(f"同一聚合批次出现多个 round_id：{sorted(round_ids)}")
    round_id = next(iter(round_ids))
    if round_id <= 0:
        raise ValueError(f"round_id 必须大于 0，当前值：{round_id}")
    return round_id


def _reconstruct_local_proxies(
    global_state: Mapping[str, torch.Tensor],
    client_updates: Sequence[ClientUpdate],
    proxy_param_name: str,
    expert_ids: Sequence[int],
    strict: bool,
) -> torch.Tensor:
    global_proxy = global_state[proxy_param_name].detach().cpu().float()
    proxies: List[torch.Tensor] = []

    for update in client_updates:
        if proxy_param_name not in update.model_delta:
            if strict:
                raise KeyError(
                    f"客户端 {update.client_id} 的 model_delta 缺少 router 参数："
                    f"{proxy_param_name}"
                )
            local_proxy = global_proxy
        else:
            local_proxy = (
                global_proxy
                + update.model_delta[proxy_param_name].detach().cpu().float()
            )

        if local_proxy.shape != global_proxy.shape:
            raise ValueError(
                f"客户端 {update.client_id} 的 router shape 不一致："
                f"local={tuple(local_proxy.shape)}, global={tuple(global_proxy.shape)}"
            )

        for expert_id in expert_ids:
            proxy = local_proxy[int(expert_id)].reshape(-1)
            if not torch.isfinite(proxy).all():
                raise ValueError(
                    f"客户端 {update.client_id} expert {expert_id} proxy 包含 NaN/Inf。"
                )
            norm = float(proxy.norm().item())
            if norm <= 1.0e-12:
                raise ValueError(
                    f"客户端 {update.client_id} expert {expert_id} proxy 范数为 0，"
                    "无法计算 cosine similarity。"
                )
            proxies.append(proxy)

    return torch.stack(proxies, dim=0)


def _build_cosine_similarity_matrix(proxies: torch.Tensor) -> torch.Tensor:
    if proxies.dim() != 2:
        raise ValueError(f"proxies 必须为 [M,D]，当前 shape={tuple(proxies.shape)}")

    normalized = F.normalize(proxies.float(), p=2, dim=1, eps=1.0e-12)
    similarity = normalized.matmul(normalized.transpose(0, 1))
    similarity = similarity.clamp(min=-1.0, max=1.0)

    if not torch.isfinite(similarity).all():
        raise ValueError("FedMoE-DA cosine similarity matrix 出现 NaN/Inf。")

    return similarity


def _build_fedmoe_da_matrix(
    similarity_matrix: torch.Tensor,
    requested_experts: int,
    temperature: float,
) -> torch.Tensor:
    """Build paper Eq. (9)-(10) sparse aggregation matrix A."""
    if similarity_matrix.dim() != 2:
        raise ValueError("similarity_matrix 必须是二维矩阵。")
    if similarity_matrix.size(0) != similarity_matrix.size(1):
        raise ValueError("similarity_matrix 必须是方阵。")
    if requested_experts < 0:
        raise ValueError("requested_experts 不能小于 0。")
    if temperature <= 0 or not math.isfinite(float(temperature)):
        raise ValueError("temperature 必须是有限正数。")

    num_local_experts = int(similarity_matrix.size(0))
    if num_local_experts <= 0:
        raise ValueError("没有 local experts。")

    matrix = torch.zeros_like(similarity_matrix, dtype=torch.float32)

    # The paper uses the (P+1)-th largest value because self similarity is
    # included. P=0 is explicitly interpreted by the paper's sensitivity
    # study as "experts are not aggregated", so keep self only in that case.
    if requested_experts == 0:
        matrix.fill_diagonal_(1.0)
        return matrix

    effective_p = min(int(requested_experts), num_local_experts - 1)
    keep_count = min(effective_p + 1, num_local_experts)

    for row_id in range(num_local_experts):
        row = similarity_matrix[row_id].float()

        if keep_count >= num_local_experts:
            mask = torch.ones_like(row, dtype=torch.bool)
        else:
            sorted_values, _ = torch.sort(row, descending=True)
            threshold = sorted_values[keep_count - 1]
            # Eq. (9): r_ij >= top_value(..., P+1). Ties are intentionally kept.
            mask = row >= threshold

        # Self must always participate in Eq. (11).
        mask[row_id] = True

        selected_scores = row[mask] / float(temperature)
        selected_weights = torch.softmax(selected_scores, dim=0)
        matrix[row_id, mask] = selected_weights

    row_sums = matrix.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=1e-6):
        raise RuntimeError("FedMoE-DA aggregation matrix A 的行权重和不为 1。")

    return matrix


def _build_uniform_collapse_matrix(
    aggregation_matrix: torch.Tensor,
    pairs: Sequence[Tuple[int, int]],
    client_ids: Sequence[int],
    expert_ids: Sequence[int],
) -> torch.Tensor:
    """Scheme A: B[e,:] = uniform mean of A[(client,e),:]."""
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
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=1e-6):
        raise RuntimeError("Uniform collapse matrix B 的行权重和不为 1。")

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
    """Aggregate source local experts into shared global expert slots."""
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
            # Current expert tensors are floating. Defensive fallback keeps the
            # previous/base value for any future non-floating expert buffer.
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

    # Average across target global expert slots, then sum all source experts
    # belonging to the same client. The result is normalized and sums to one.
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


def _build_diagnostics(
    *,
    method_name: str,
    param_group_name: str,
    target_param_names: Sequence[str],
    client_ids: Sequence[int],
    expert_ids: Sequence[int],
    pairs: Sequence[Tuple[int, int]],
    result_weights: Mapping[int, float],
    aggregation_matrix: torch.Tensor,
    collapse_matrix: torch.Tensor,
    similarity_matrix: Optional[torch.Tensor],
    matrix_refreshed: bool,
    forced_refresh: bool,
    matrix_round: int,
    round_id: int,
    proxy_param_name: str,
    cfg: Any,
) -> Dict[str, Any]:
    requested_experts = int(_cfg_get(cfg, "fedmoe_da.requested_experts", 5))
    temperature = float(_cfg_get(cfg, "fedmoe_da.temperature", 1.0))
    interval = int(_cfg_get(cfg, "fedmoe_da.matrix_update_interval", 5))
    effective_p = min(requested_experts, max(len(pairs) - 1, 0))

    nonzero_per_row = (aggregation_matrix > 0).sum(dim=1).tolist()
    top_sources_limit = int(_cfg_get(cfg, "fedmoe_da.log_top_sources", 8))

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
        "adaptation": "fedmoe_da_eq8_11_plus_uniform_global_collapse",
        "requested_experts": requested_experts,
        "effective_requested_experts": int(effective_p),
        "temperature": temperature,
        "matrix_update_interval": interval,
        "matrix_refreshed": bool(matrix_refreshed),
        "forced_refresh_due_to_pool_change": bool(forced_refresh),
        "matrix_round": int(matrix_round),
        "round_id": int(round_id),
        "proxy_param_name": proxy_param_name,
        "num_global_experts": len(expert_ids),
        "num_local_experts": len(pairs),
        "mean_sources_per_personalized_row": float(
            sum(int(value) for value in nonzero_per_row)
            / max(len(nonzero_per_row), 1)
        ),
        "min_sources_per_personalized_row": int(min(nonzero_per_row)),
        "max_sources_per_personalized_row": int(max(nonzero_per_row)),
        "top_sources_by_global_expert": top_sources,
    }

    if bool(_cfg_get(cfg, "fedmoe_da.log_matrix", False)):
        diagnostics["expert_pairs"] = [
            {
                "index": index,
                "client_id": int(client_id),
                "expert_id": int(expert_id),
            }
            for index, (client_id, expert_id) in enumerate(pairs)
        ]
        diagnostics["aggregation_matrix_A"] = aggregation_matrix.tolist()
        diagnostics["global_collapse_matrix_B"] = collapse_matrix.tolist()
        if similarity_matrix is not None:
            diagnostics["similarity_matrix_R"] = similarity_matrix.tolist()

    return diagnostics


def build_expert_aggregator(cfg: Any) -> base.Aggregator:
    return FedMoEDAStyleExpertAggregator(
        cfg=cfg,
        param_group_name="expert",
    )


def main() -> int:
    kwargs: Dict[str, Any] = {
        "expert_aggregator_builder": build_expert_aggregator,
        "embedded_method_config": EMBEDDED_METHOD_CONFIG,
        "expert_method_name": ALGORITHM_NAME,
    }

    # Current pluginized base.py supports these hooks. Keep guarded assignment
    # out of the numerical path so the method remains easy to inspect.
    kwargs["method_config_defaults"] = METHOD_CONFIG_DEFAULTS
    kwargs["method_config_validator"] = validate_method_config
    kwargs["method_cli_argument_registrar"] = register_method_cli_arguments
    kwargs["method_cli_overrides_builder"] = build_method_cli_overrides

    return base.main(**kwargs)


if __name__ == "__main__":
    raise SystemExit(main())

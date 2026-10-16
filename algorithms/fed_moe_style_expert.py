from __future__ import annotations

"""Controlled Fed-MoE-style expert aggregation for the shared Sparse-MoE FL setup.

This plugin adapts the relevance-weighted routed-expert update from
"Heterogeneous Federated Learning with Scalable Server Mixture-of-Experts"
(IJCAI 2025) while keeping the common experiment model and local training
unchanged.

Preserved Fed-MoE idea:
    server evidence labeled data -> server gate response Q
    x local source-expert true-label confidence P_y
    -> relevance -> row softmax -> moving expert fusion.

Controlled adaptations:
    * every (client, local expert slot) is treated as a source expert;
    * the existing shared SparseMoEClassifier is kept unchanged;
    * no main expert is added;
    * the server router is NOT additionally trained on reserved data;
    * Stage-C personalized client synchronization is NOT used;
    * one moving expert update is performed per FL round.

This plugin uses the generic MethodContext and server_evidence lifecycle
provided by base.py; no Fed-MoE-specific branch is required in base.py.
"""

# Import base first so its deterministic pre-PyTorch bootstrap runs before
# PyTorch is initialized in this method file.
import base

import argparse
import math
import random
import re
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


Aggregator = base.Aggregator
AggregationResult = base.AggregationResult
ClientUpdate = base.ClientUpdate
check_finite_state_dict = base.check_finite_state_dict
clone_state_dict = base.clone_state_dict

ALGORITHM_NAME = "fed_moe_style_expert"

# The paper states lambda in (0, 1), but the IJCAI paper does not report a
# concrete experimental value that could be verified from the public text.
# 0.5 is therefore an explicit adaptation hyperparameter, NOT a paper default.
ADAPTATION_DEFAULT_MOVING_RATE = 0.5

METHOD_CONFIG_DEFAULTS = {
    "server_evidence": {
        "size": 250,
        "class_balanced": False,
        "batch_size": 256,
    },
    "fed_moe": {
        "moving_rate": ADAPTATION_DEFAULT_MOVING_RATE,
        "log_detail": True,
    },
}

EMBEDDED_METHOD_CONFIG = {
    "agg": {
        "non_expert": {"method": "uniform"},
        "expert": {"method": ALGORITHM_NAME},
    },
    **METHOD_CONFIG_DEFAULTS,
}

_EXPERT_ID_PATTERN = re.compile(r"(?P<prefix>(?:^|\.)experts\.)(?P<id>\d+)(?=\.|$)")


class FedMoEStyleExpertAggregator(Aggregator):
    """Reserved-response relevance weighted expert aggregation."""

    @property
    def method_name(self) -> str:
        return ALGORITHM_NAME

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        # The real aggregation has target-expert-specific source weights.
        # This method exists only to satisfy the shared Aggregator interface.
        if len(client_updates) == 0:
            raise ValueError("client_updates 不能为空。")
        weight = 1.0 / float(len(client_updates))
        return {int(update.client_id): weight for update in client_updates}

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
            raise ValueError("fed_moe_style_expert 只能用于 expert 参数聚合。")

        target_param_names = _resolve_param_names(global_state, param_names)
        expert_param_names = _group_expert_param_names(target_param_names)
        expert_ids = sorted(expert_param_names)
        if not expert_ids:
            raise ValueError("没有找到 experts.<id> 参数，无法执行 Fed-MoE-style 聚合。")

        expected_num_experts = int(_cfg_get(self.cfg, "num_experts", len(expert_ids)))
        if expert_ids != list(range(expected_num_experts)):
            raise ValueError(
                "Fed-MoE-style 要求 expert slot 连续且与 num_experts 一致："
                f"expert_ids={expert_ids}, num_experts={expected_num_experts}"
            )
        _validate_cross_slot_structure(expert_param_names, expert_ids)

        reserved_loader, device, reserved_summary, model_builder = _get_method_context_resources(self)
        moving_rate = float(_cfg_get(self.cfg, "fed_moe.moving_rate", ADAPTATION_DEFAULT_MOVING_RATE))
        if not (0.0 < moving_rate < 1.0):
            raise ValueError(
                "fed_moe.moving_rate 必须在 (0,1) 内，"
                f"当前值：{moving_rate}"
            )

        client_ids = [int(update.client_id) for update in client_updates]
        if len(set(client_ids)) != len(client_ids):
            raise ValueError(f"client_updates 中存在重复 client_id：{client_ids}")

        # Source ordering is fixed as (client order, expert slot order).
        source_pairs: List[Tuple[int, int]] = [
            (int(update.client_id), int(expert_id))
            for update in client_updates
            for expert_id in expert_ids
        ]

        relevance = _compute_reserved_relevance(
            cfg=self.cfg,
            global_state=global_state,
            non_expert_state=(base_state if base_state is not None else global_state),
            client_updates=client_updates,
            expert_ids=expert_ids,
            reserved_loader=reserved_loader,
            device=device,
            model_builder=model_builder,
            strict=strict,
        )
        expected_shape = (len(expert_ids), len(source_pairs))
        if tuple(relevance.shape) != expected_shape:
            raise RuntimeError(
                "Fed-MoE relevance matrix shape 错误："
                f"expected={expected_shape}, actual={tuple(relevance.shape)}"
            )

        source_weights = F.softmax(relevance.float(), dim=1)
        if not torch.isfinite(source_weights).all():
            raise ValueError("Fed-MoE source weight matrix 出现 NaN/Inf。")
        row_sums = source_weights.sum(dim=1)
        if not torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            atol=1.0e-6,
            rtol=1.0e-6,
        ):
            raise RuntimeError("Fed-MoE source weight matrix 的行权重和不为 1。")

        if base_state is None:
            new_state_dict = clone_state_dict(global_state)
        else:
            new_state_dict = clone_state_dict(base_state)

        _apply_moving_expert_fusion(
            new_state_dict=new_state_dict,
            global_state=global_state,
            client_updates=client_updates,
            target_param_names=target_param_names,
            expert_ids=expert_ids,
            source_pairs=source_pairs,
            source_weights=source_weights,
            moving_rate=moving_rate,
            strict=strict,
        )

        check_finite_state_dict(
            state_dict=new_state_dict,
            param_names=target_param_names,
        )

        client_weights = _collapse_source_weights_to_clients(
            source_weights=source_weights,
            source_pairs=source_pairs,
            client_ids=client_ids,
        )
        diagnostics = _build_diagnostics(
            relevance=relevance,
            source_weights=source_weights,
            source_pairs=source_pairs,
            expert_ids=expert_ids,
            client_ids=client_ids,
            moving_rate=moving_rate,
            reserved_summary=reserved_summary,
            target_param_count=len(target_param_names),
        )

        if bool(_cfg_get(self.cfg, "fed_moe.log_detail", True)):
            print(
                "[FedMoEStyle] "
                f"reserved_size={diagnostics['reserved_size']} "
                f"moving_rate={moving_rate:.6f} "
                f"relevance_mean={diagnostics['mean_relevance']:.6e} "
                f"mean_weight_entropy={diagnostics['mean_weight_entropy']:.6f} "
                f"mean_max_source_weight={diagnostics['mean_max_source_weight']:.6f} "
                f"top_sources={diagnostics['top_sources_by_global_expert']}",
                flush=True,
            )

        return AggregationResult(
            new_state_dict=new_state_dict,
            weights=client_weights,
            diagnostics=diagnostics,
        )


def validate_method_config(cfg: Mapping[str, Any]) -> None:
    fed_cfg = cfg.get("fed_moe", {})
    if not isinstance(fed_cfg, Mapping):
        raise base.ConfigError("fed_moe 必须是 dict。")

    moving_rate = float(fed_cfg.get("moving_rate", ADAPTATION_DEFAULT_MOVING_RATE))
    if not math.isfinite(moving_rate) or not (0.0 < moving_rate < 1.0):
        raise base.ConfigError(
            "fed_moe.moving_rate 必须是 (0,1) 内的有限数，"
            f"当前值：{moving_rate}"
        )

    evidence_cfg = cfg.get("server_evidence", {})
    if not isinstance(evidence_cfg, Mapping):
        raise base.ConfigError("server_evidence 必须是 dict。")
    evidence_size = int(evidence_cfg.get("size", 0))
    if evidence_size <= 0:
        raise base.ConfigError(
            "Fed-MoE-style 需要独立的有标签 server evidence set，"
            "请设置 server_evidence.size > 0。"
        )


def register_method_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fed-moe-moving-rate",
        type=float,
        default=None,
        help=(
            "Fed-MoE moving expert fusion rate lambda. "
            "The paper states lambda in (0,1) but does not report a verified "
            "experimental value; this is an explicit adaptation hyperparameter."
        ),
    )


def build_method_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    moving_rate = getattr(args, "fed_moe_moving_rate", None)
    if moving_rate is None:
        return {}
    return {"fed_moe": {"moving_rate": float(moving_rate)}}


def build_expert_aggregator(cfg: Any) -> base.Aggregator:
    return FedMoEStyleExpertAggregator(cfg=cfg, param_group_name="expert")


def main() -> int:
    return base.main(
        expert_aggregator_builder=build_expert_aggregator,
        embedded_method_config=EMBEDDED_METHOD_CONFIG,
        expert_method_name=ALGORITHM_NAME,
        method_config_defaults=METHOD_CONFIG_DEFAULTS,
        method_config_validator=validate_method_config,
        method_cli_argument_registrar=register_method_cli_arguments,
        method_cli_overrides_builder=build_method_cli_overrides,
    )


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        value = cfg.get(key, None)
        if value is not None:
            return value
    current = cfg
    for part in key.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return default
    return current


def _resolve_param_names(
    global_state: Mapping[str, torch.Tensor],
    param_names: Optional[Iterable[str]],
) -> List[str]:
    names = list(global_state.keys()) if param_names is None else list(param_names)
    for name in names:
        if name not in global_state:
            raise KeyError(f"global_state 中不存在参数：{name}")
    return names


def _get_expert_id(name: str) -> Optional[int]:
    match = _EXPERT_ID_PATTERN.search(name)
    return None if match is None else int(match.group("id"))


def _replace_expert_id(name: str, expert_id: int) -> str:
    match = _EXPERT_ID_PATTERN.search(name)
    if match is None:
        raise ValueError(f"参数名中没有 experts.<id>：{name}")
    start, end = match.span("id")
    return name[:start] + str(int(expert_id)) + name[end:]


def _group_expert_param_names(param_names: Sequence[str]) -> Dict[int, List[str]]:
    result: Dict[int, List[str]] = {}
    for name in param_names:
        expert_id = _get_expert_id(name)
        if expert_id is not None:
            result.setdefault(expert_id, []).append(name)
    return {
        expert_id: sorted(names)
        for expert_id, names in sorted(result.items())
    }


def _validate_cross_slot_structure(
    expert_param_names: Mapping[int, Sequence[str]],
    expert_ids: Sequence[int],
) -> None:
    reference_id = int(expert_ids[0])
    reference = [
        _replace_expert_id(name, 0)
        for name in expert_param_names[reference_id]
    ]
    for expert_id in expert_ids[1:]:
        current = [
            _replace_expert_id(name, 0)
            for name in expert_param_names[int(expert_id)]
        ]
        if current != reference:
            raise ValueError(
                "不同 expert slot 参数结构不一致，无法执行 Fed-MoE 跨 expert 聚合："
                f"reference={reference}, expert{expert_id}={current}"
            )


def _get_method_context_resources(
    aggregator: FedMoEStyleExpertAggregator,
) -> Tuple[Any, torch.device, Mapping[str, Any], Any]:
    """Read generic server-side resources injected by base.MethodContext."""
    context = getattr(aggregator, "method_context", None)
    if context is None:
        raise RuntimeError(
            "Fed-MoE-style 没有收到 MethodContext。"
            "请使用支持 MethodContext 的新版 base.py。"
        )

    method_context_type = getattr(base, "MethodContext", None)
    if method_context_type is not None and not isinstance(context, method_context_type):
        raise TypeError(
            "aggregator.method_context 类型错误："
            f"expected=base.MethodContext, actual={type(context).__name__}"
        )

    reserved_loader = getattr(context, "server_evidence_loader", None)
    if reserved_loader is None:
        raise RuntimeError(
            "Fed-MoE-style 没有收到 server_evidence_loader。"
            "请确认 server_evidence.size > 0。"
        )

    model_builder = getattr(context, "model_builder", None)
    if model_builder is None or not callable(model_builder):
        raise RuntimeError(
            "Fed-MoE-style 需要 MethodContext.model_builder 来重建 server/local model。"
        )

    device = torch.device(getattr(context, "device", "cpu"))
    summary = _summarize_server_evidence_loader(reserved_loader)
    return reserved_loader, device, summary, model_builder


def _summarize_server_evidence_loader(
    reserved_loader: Any,
) -> Dict[str, Any]:
    """Build lightweight diagnostics without changing the evidence iteration order."""
    dataset = getattr(reserved_loader, "dataset", None)
    if dataset is None:
        return {"size": 0, "class_counts": {}, "class_balanced": False}

    size = int(len(dataset))
    class_counts: Dict[int, int] = {}
    try:
        targets = base.get_dataset_targets(dataset)
        for target in targets:
            class_id = int(target)
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
    except (AttributeError, TypeError, ValueError):
        class_counts = {}

    class_balanced = False
    if class_counts:
        counts = list(class_counts.values())
        class_balanced = max(counts) - min(counts) <= 1

    return {
        "size": size,
        "class_counts": class_counts,
        "class_balanced": class_balanced,
    }


@contextmanager
def _preserve_rng_state() -> Any:
    """Prevent temporary model reconstruction from perturbing FL randomness."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _reconstruct_local_state(
    global_state: Mapping[str, torch.Tensor],
    update: ClientUpdate,
    strict: bool,
) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    for name, global_value in global_state.items():
        if not torch.is_tensor(global_value):
            result[name] = global_value
            continue
        value = global_value.detach().cpu().clone()
        if torch.is_floating_point(value):
            delta = update.model_delta.get(name)
            if delta is None:
                if strict:
                    raise KeyError(
                        f"客户端 {update.client_id} 的 model_delta 缺少参数：{name}"
                    )
            else:
                value = value + delta.detach().cpu().to(dtype=value.dtype)
        result[name] = value
    return result


def _build_model_without_rng_side_effect(
    cfg: Any,
    model_builder: Any,
) -> torch.nn.Module:
    if model_builder is None or not callable(model_builder):
        raise RuntimeError("MethodContext.model_builder 必须是可调用对象。")
    with _preserve_rng_state():
        return model_builder(cfg)


def _forward_moe_features(
    model: torch.nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    """Follow SparseMoEClassifier's backbone -> adapter feature path exactly."""
    if not hasattr(model, "backbone"):
        raise AttributeError("Fed-MoE-style 要求模型包含 backbone。")
    if not hasattr(model, "backbone_adapter"):
        raise AttributeError("Fed-MoE-style 要求模型包含 backbone_adapter。")

    features = model.backbone(images)
    features = model.backbone_adapter(features)
    return features


def _compute_reserved_relevance(
    cfg: Any,
    global_state: Mapping[str, torch.Tensor],
    non_expert_state: Mapping[str, torch.Tensor],
    client_updates: Sequence[ClientUpdate],
    expert_ids: Sequence[int],
    reserved_loader: Any,
    device: torch.device,
    model_builder: Any,
    strict: bool,
) -> torch.Tensor:
    """Compute E x (C*E) relevance from labeled reserved functional responses."""
    num_experts = len(expert_ids)
    if num_experts <= 0:
        raise ValueError("num_experts 必须大于 0。")

    # Server target routing responses use the already-aggregated non-expert
    # backbone/router from base_state; expert values in this temporary model are
    # irrelevant to Q.
    server_model = _build_model_without_rng_side_effect(
        cfg=cfg,
        model_builder=model_builder,
    )
    server_model.load_state_dict(dict(non_expert_state), strict=True)
    server_model.to(device)
    server_model.eval()

    q_batches: List[torch.Tensor] = []
    label_batches: List[torch.Tensor] = []
    sample_count = 0

    with torch.inference_mode():
        for batch in reserved_loader:
            images, targets = base.unpack_batch(batch)
            images = images.to(device)
            targets = targets.to(device=device, dtype=torch.long)

            features = _forward_moe_features(server_model, images)
            _, _, router_probs, _ = server_model.moe_head.gating(features)
            q_batches.append(router_probs.detach().cpu().float())
            label_batches.append(targets.detach().cpu())
            sample_count += int(targets.numel())

    server_model.to("cpu")
    del server_model

    if sample_count <= 0 or not q_batches:
        raise ValueError("server evidence loader 为空，无法计算 Fed-MoE relevance。")

    relevance_blocks: List[torch.Tensor] = []

    for update in client_updates:
        local_state = _reconstruct_local_state(
            global_state=global_state,
            update=update,
            strict=strict,
        )
        local_model = _build_model_without_rng_side_effect(
            cfg=cfg,
            model_builder=model_builder,
        )
        local_model.load_state_dict(local_state, strict=True)
        local_model.to(device)
        local_model.eval()

        block = torch.zeros(num_experts, num_experts, dtype=torch.float64)
        seen = 0

        with torch.inference_mode():
            for batch_id, batch in enumerate(reserved_loader):
                if batch_id >= len(q_batches):
                    raise RuntimeError("reserved loader 两次迭代的 batch 数不一致。")
                images, targets = base.unpack_batch(batch)
                images = images.to(device)
                targets = targets.to(device=device, dtype=torch.long)

                expected_targets = label_batches[batch_id]
                if not torch.equal(targets.detach().cpu(), expected_targets):
                    raise RuntimeError(
                        "reserved loader 在重复迭代时样本顺序发生变化；"
                        "Fed-MoE 需要 shuffle=False 的 deterministic reserved loader。"
                    )

                features = _forward_moe_features(local_model, images)
                confidence_columns: List[torch.Tensor] = []
                for expert_id in expert_ids:
                    logits = local_model.moe_head.experts[int(expert_id)](features)
                    true_prob = F.softmax(logits.float(), dim=-1).gather(
                        dim=1,
                        index=targets.reshape(-1, 1),
                    ).squeeze(1)
                    confidence_columns.append(true_prob)

                p = torch.stack(confidence_columns, dim=1)  # [B,E]
                q = q_batches[batch_id].to(device=device, dtype=torch.float32)
                if tuple(q.shape) != tuple(p.shape):
                    raise RuntimeError(
                        "server gate 与 local expert confidence shape 不一致："
                        f"Q={tuple(q.shape)}, P={tuple(p.shape)}"
                    )

                block += q.transpose(0, 1).double().matmul(p.double()).cpu()
                seen += int(targets.numel())

        local_model.to("cpu")
        del local_model, local_state

        if seen != sample_count:
            raise RuntimeError(
                "reserved loader 重复迭代样本数不一致："
                f"server={sample_count}, client={seen}"
            )
        relevance_blocks.append((block / float(sample_count)).float())

    # Concatenate source-expert columns in (client order, expert order).
    relevance = torch.cat(relevance_blocks, dim=1)
    if not torch.isfinite(relevance).all():
        raise ValueError("Fed-MoE relevance matrix 出现 NaN/Inf。")
    return relevance


def _apply_moving_expert_fusion(
    new_state_dict: Dict[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
    client_updates: Sequence[ClientUpdate],
    target_param_names: Sequence[str],
    expert_ids: Sequence[int],
    source_pairs: Sequence[Tuple[int, int]],
    source_weights: torch.Tensor,
    moving_rate: float,
    strict: bool,
) -> None:
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
        if not torch.is_tensor(target_global) or not torch.is_floating_point(target_global):
            continue

        row = source_weights[expert_row[int(target_expert_id)]]
        fused = torch.zeros_like(target_global)

        for source_index, ((client_id, source_expert_id), coefficient) in enumerate(
            zip(source_pairs, row)
        ):
            weight = float(coefficient.item())
            source_name = _replace_expert_id(target_name, int(source_expert_id))
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
                    "Fed-MoE 跨 expert 聚合要求对应参数 shape 一致："
                    f"target={target_name}{tuple(target_global.shape)}, "
                    f"source={source_name}{tuple(source_global.shape)}"
                )

            update = update_by_client[int(client_id)]
            delta = update.model_delta.get(source_name)
            if delta is None:
                if strict:
                    raise KeyError(
                        f"客户端 {client_id} model_delta 缺少 source expert 参数："
                        f"{source_name}"
                    )
                source_local = source_global
            else:
                source_local = source_global + delta.to(
                    device=source_global.device,
                    dtype=source_global.dtype,
                )

            fused = fused + weight * source_local.to(
                device=target_global.device,
                dtype=target_global.dtype,
            )

        new_state_dict[target_name] = (
            (1.0 - float(moving_rate)) * target_global
            + float(moving_rate) * fused
        )


def _collapse_source_weights_to_clients(
    source_weights: torch.Tensor,
    source_pairs: Sequence[Tuple[int, int]],
    client_ids: Sequence[int],
) -> Dict[int, float]:
    """Diagnostic client weight: mean over target experts, sum source slots."""
    mean_source_weight = source_weights.mean(dim=0)
    result = {int(client_id): 0.0 for client_id in client_ids}
    for index, (client_id, _) in enumerate(source_pairs):
        result[int(client_id)] += float(mean_source_weight[index].item())

    total = float(sum(result.values()))
    if total > 0:
        result = {key: value / total for key, value in result.items()}
    return result


def _build_diagnostics(
    relevance: torch.Tensor,
    source_weights: torch.Tensor,
    source_pairs: Sequence[Tuple[int, int]],
    expert_ids: Sequence[int],
    client_ids: Sequence[int],
    moving_rate: float,
    reserved_summary: Mapping[str, Any],
    target_param_count: int,
) -> Dict[str, Any]:
    relevance_flat = relevance.reshape(-1).float()
    entropy = -torch.sum(
        source_weights * torch.log(source_weights.clamp_min(1.0e-12)),
        dim=1,
    )
    max_weights = source_weights.max(dim=1).values

    top_sources: Dict[int, List[Dict[str, Any]]] = {}
    topn = min(5, source_weights.size(1))
    for row_id, expert_id in enumerate(expert_ids):
        values, indices = torch.topk(source_weights[row_id], k=topn)
        rows: List[Dict[str, Any]] = []
        for value, index in zip(values.tolist(), indices.tolist()):
            client_id, source_expert_id = source_pairs[int(index)]
            rows.append(
                {
                    "client_id": int(client_id),
                    "source_expert_id": int(source_expert_id),
                    "weight": float(value),
                    "relevance": float(relevance[row_id, int(index)].item()),
                }
            )
        top_sources[int(expert_id)] = rows

    return {
        "method": ALGORITHM_NAME,
        "param_group": "expert",
        "num_clients": int(len(client_ids)),
        "param_count": int(target_param_count),
        "num_target_experts": int(len(expert_ids)),
        "num_source_experts": int(len(source_pairs)),
        "moving_rate": float(moving_rate),
        "moving_rate_source": "adaptation_hyperparameter_not_reported_in_public_paper",
        "reserved_size": int(reserved_summary.get("size", 0)),
        "reserved_class_counts": dict(reserved_summary.get("class_counts", {})),
        "reserved_class_balanced": bool(reserved_summary.get("class_balanced", False)),
        "min_relevance": float(relevance_flat.min().item()),
        "mean_relevance": float(relevance_flat.mean().item()),
        "max_relevance": float(relevance_flat.max().item()),
        "mean_weight_entropy": float(entropy.mean().item()),
        "min_weight_entropy": float(entropy.min().item()),
        "max_weight_entropy": float(entropy.max().item()),
        "mean_max_source_weight": float(max_weights.mean().item()),
        "min_max_source_weight": float(max_weights.min().item()),
        "max_max_source_weight": float(max_weights.max().item()),
        "top_sources_by_global_expert": top_sources,
    }


if __name__ == "__main__":
    raise SystemExit(main())

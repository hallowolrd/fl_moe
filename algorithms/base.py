from __future__ import annotations

"""Shared standalone base for federated MoE experiments.

This file owns configuration, data, partitioning, models, client training,
optional method evidence hooks, fixed uniform non-expert aggregation, server
rounds, evaluation, logging, and output. Expert-method behavior is injected by
a method script and is intentionally not implemented here.
"""

# ============================================================
# 注意：
# 下面这一段必须放在 import torch 之前。
#
# 原因：
# 1. CUBLAS_WORKSPACE_CONFIG 必须在 CUDA / cuBLAS 初始化前设置。
# 2. PYTHONHASHSEED 必须在 Python 解释器启动前生效。
#
# 所以如果当前进程的 PYTHONHASHSEED 和配置文件里的 seed 不一致，
# 这里会自动重新执行一次当前 Python 命令。
# ============================================================

import os
import sys
from pathlib import Path


# 与原 configs/base.yaml 保持一致。该值必须在 import torch 之前可用，
# 以便没有外部 --config 文件时仍能正确设置 PYTHONHASHSEED。
EMBEDDED_DEFAULT_SEED = 0


def _get_cli_arg_value(name: str) -> str | None:
    """
    从命令行参数中读取指定参数值。

    支持两种写法：
        1. --config configs/uniform.yaml
        2. --config=configs/uniform.yaml
    """
    prefix = name + "="
    argv = sys.argv

    for idx, arg in enumerate(argv):
        if arg == name and idx + 1 < len(argv):
            return argv[idx + 1]

        if arg.startswith(prefix):
            return arg[len(prefix):]

    return None


def _clean_simple_yaml_value(value: str) -> str:
    """
    清理简单 YAML 标量值。

    这里只服务于启动早期读取 seed / include，
    不替代项目里的正式 load_config。
    """
    value = value.strip()

    # 去掉简单引号
    if len(value) >= 2:
        if (value[0] == value[-1]) and value[0] in {"'", '"'}:
            value = value[1:-1]

    return value.strip()


def _read_top_level_scalar_from_yaml_like_file(
    path: Path,
    key: str,
    visited: set[Path] | None = None,
) -> str | None:
    """
    在 import torch 之前，轻量读取 YAML 文件里的顶层简单标量。

    目的：
        - 提前读取 seed，让 PYTHONHASHSEED 可以跟随训练 seed。
        - 支持你的配置风格：xxx.yaml 里 include: base.yaml。

    注意：
        - 这不是完整 YAML 解析器；
        - 只用于启动阶段读取 seed / include；
        - 正式配置仍然由 utils.config.load_config() 读取。
    """
    if visited is None:
        visited = set()

    path = path.resolve()

    if path in visited:
        return None

    visited.add(path)

    if not path.exists():
        return None

    include_path: Path | None = None
    local_value: str | None = None

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            # 去掉行内注释。这里足够处理当前配置里的 seed / include。
            line = raw_line.split("#", 1)[0].strip()

            if not line:
                continue

            if ":" not in line:
                continue

            left, right = line.split(":", 1)
            name = left.strip()
            value = _clean_simple_yaml_value(right)

            if name == "include":
                include_path = (path.parent / value).resolve()
                continue

            if name == key:
                local_value = value
                continue

    # 当前配置里的 seed 优先级高于 include 里的 seed。
    if local_value is not None:
        return local_value

    if include_path is not None:
        return _read_top_level_scalar_from_yaml_like_file(
            path=include_path,
            key=key,
            visited=visited,
        )

    return None


def _prepare_deterministic_env_before_torch() -> None:
    """
    在 torch / CUDA 初始化前准备确定性相关环境变量。

    CUBLAS_WORKSPACE_CONFIG:
        控制 cuBLAS 矩阵乘法的确定性。
        如果用户已经手动设置了 :16:8 或其他合法值，这里不覆盖。

    PYTHONHASHSEED:
        让 Python hash 随机种子跟随配置文件里的 seed。
        该变量必须在解释器启动前生效，所以必要时自动 re-exec 一次。
    """
    # cuBLAS 确定性配置。显存不紧张时优先使用 :4096:8。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    config_arg = _get_cli_arg_value("--config")
    cli_seed = _get_cli_arg_value("--seed")
    config_seed = None

    if config_arg is not None:
        config_seed = _read_top_level_scalar_from_yaml_like_file(path=Path(config_arg), key="seed")

    target_hash_seed = str(
        cli_seed if cli_seed is not None else (
            config_seed if config_seed is not None else EMBEDDED_DEFAULT_SEED
        )
    )

    current_hash_seed = os.environ.get("PYTHONHASHSEED")
    already_reexec = os.environ.get("CLEAN_REEXEC_FOR_PYTHONHASHSEED") == "1"

    if current_hash_seed != target_hash_seed:
        os.environ["PYTHONHASHSEED"] = target_hash_seed

        # PYTHONHASHSEED 必须在 Python 解释器启动前生效。
        # 当前进程已经启动了，所以这里自动重启一次当前命令。
        if not already_reexec:
            os.environ["CLEAN_REEXEC_FOR_PYTHONHASHSEED"] = "1"
            os.execvpe(
                sys.executable,
                [sys.executable] + sys.argv,
                os.environ,
            )


_prepare_deterministic_env_before_torch()


# ============================================================================
# Consolidated imports; keep this block after the deterministic bootstrap.
# ============================================================================

import argparse
import copy
import csv
import gc
import json
import math
import random
import re
import threading
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    TextIO,
    Tuple,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm


# ============================================================================
# Bundled from utils/config.py
# ============================================================================


# =========================
# 可扩展的合法取值注册区
# =========================
# 后续新增数据集、模型时，优先改这里。

SUPPORTED_DATASETS = {
    "cifar10",
    "cifar100",
    "cinic10",
    "fashionmnist",
    "stl10",
    "tiny-imagenet-200",
    "femnist",
}

DATASET_ALIASES = {
    "cifar10": "cifar10",
    "cifar-10": "cifar10",
    "cifar100": "cifar100",
    "cifar-100": "cifar100",
    "cinic10": "cinic10",
    "cinic-10": "cinic10",
    "fashionmnist": "fashionmnist",
    "fashion-mnist": "fashionmnist",
    "stl10": "stl10",
    "stl-10": "stl10",
    "tiny-imagenet-200": "tiny-imagenet-200",
    "tinyimagenet200": "tiny-imagenet-200",
    "tiny_imagenet_200": "tiny-imagenet-200",
    "femnist": "femnist",
}


def normalize_dataset_name(dataset_name: str) -> str:
    """Normalize common dataset spelling variants to one canonical name."""
    key = str(dataset_name).strip().lower()
    return DATASET_ALIASES.get(key, key)

SUPPORTED_MODELS = {
    "sparse_moe_classifier",
}


class ConfigError(Exception):
    """配置相关错误。"""

    pass


class ConfigNode:
    """
    轻量级配置对象。

    支持两种读取方式：
        cfg.dataset
        cfg.agg.non_expert.method
        cfg.agg.expert.method

    也支持路径读取：
        cfg.get("agg.non_expert.method", "uniform")
        cfg.get("agg.expert.method", "uniform")
    """

    def __init__(self, data: Mapping[str, Any]):
        for key, value in data.items():
            setattr(self, key, self._wrap(value))

    @staticmethod
    def _wrap(value: Any) -> Any:
        """把嵌套 dict 自动转成 ConfigNode。"""
        if isinstance(value, Mapping):
            return ConfigNode(value)

        if isinstance(value, list):
            return [
                ConfigNode._wrap(item)
                for item in value
            ]

        return value

    def get(self, path: str, default: Any = None) -> Any:
        """
        按路径读取配置。

        示例：
            cfg.get("agg.non_expert.method", "uniform")
            cfg.get("agg.expert.method", "uniform")
        """
        current: Any = self

        for part in path.split("."):
            if isinstance(current, ConfigNode) and hasattr(current, part):
                current = getattr(current, part)
            else:
                return default

        return current

    def to_dict(self) -> Dict[str, Any]:
        """把 ConfigNode 递归转换回普通 dict。"""
        result = {}

        for key, value in self.__dict__.items():
            result[key] = self._unwrap(value)

        return result

    @staticmethod
    def _unwrap(value: Any) -> Any:
        """把 ConfigNode 递归转换成普通 Python 对象。"""
        if isinstance(value, ConfigNode):
            return value.to_dict()

        if isinstance(value, list):
            return [
                ConfigNode._unwrap(item)
                for item in value
            ]

        return value

    def __getitem__(self, key: str) -> Any:
        """支持 cfg["dataset"] 形式读取。"""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __repr__(self) -> str:
        return repr(self.to_dict())


EMBEDDED_BASE_CONFIG: Dict[str, Any] = {
    "dataset": "cifar10",
    "data_root": "./datasets",
    "num_clients": 5,
    "alpha": 0.1,
    "frac": 1.0,
    "rounds": 50,
    "local_epochs": 5,
    "batch_size": 64,
    "test_batch_size": 64,
    "num_workers": 2,
    "model": "sparse_moe_classifier",
    "model_cfg": {
        "backbone": "resnet_cifar",
    },
    "num_experts": 4,
    "topk": 2,
    "optimizer": {
        "type": "sgd",
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
    },
    "agg": {
        "non_expert": {"method": "uniform"},
        "expert": {"method": "uniform"},
    },
    "seed": EMBEDDED_DEFAULT_SEED,
    "deterministic": True,
    "device": "auto",
    "output_dir": "outputs",
    "run_name": "auto",
    "run": {
        "unique_name": True,
        "overwrite": False,
    },
    "server_evidence": {
        "size": 0,
        "batch_size": 256,
        "class_balanced": True,
    },
    "logging": {
        "log_every": 1,
        "save_config": True,
        "save_results_csv": True,
        "progress_bar": True,
        "progress_in_non_tty": False,
        "console_round_summary": True,
        "file_round_detail": True,
        "log_round_clients": True,
        "log_client_table": True,
        "log_client_metrics": True,
        "log_agg_weights": True,
        "compact_uniform_weights": True,
        "collect_expert_usage": True,
        "expert_usage_max_batches": 0,
    },
}


def load_embedded_config(
    method_overrides: Mapping[str, Any] | None = None,
    method_defaults: Mapping[str, Any] | None = None,
    method_validator: Optional[Callable[[Mapping[str, Any]], None]] = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> ConfigNode:
    """Build the effective configuration without requiring a YAML file."""
    raw_cfg: Dict[str, Any] = {}

    if method_overrides is not None:
        raw_cfg = copy.deepcopy(dict(method_overrides))

    raw_cfg = _apply_defaults(raw_cfg)
    raw_cfg = _apply_method_defaults(raw_cfg, method_defaults)
    raw_cfg = _apply_config_overrides(raw_cfg, config_overrides)
    raw_cfg = _finalize_run_info(raw_cfg)
    _validate_config(raw_cfg)
    if method_validator is not None:
        method_validator(raw_cfg)
    return ConfigNode(raw_cfg)


def load_config(
    config_path: str | Path,
    method_defaults: Mapping[str, Any] | None = None,
    method_validator: Optional[Callable[[Mapping[str, Any]], None]] = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> ConfigNode:
    """
    读取配置文件，并返回 ConfigNode。

    主要流程：
    1. 读取 yaml
    2. 处理 include
    3. 合并默认值
    4. 自动生成 run_name / run_dir
    5. 做基础合法性检查
    6. 转成 ConfigNode
    """
    config_path = Path(config_path).expanduser().resolve()

    raw_cfg = _load_yaml_with_include(config_path)
    raw_cfg = _apply_defaults(raw_cfg)
    raw_cfg = _apply_method_defaults(raw_cfg, method_defaults)
    raw_cfg = _apply_config_overrides(raw_cfg, config_overrides)
    raw_cfg = _finalize_run_info(raw_cfg)
    _validate_config(raw_cfg)
    if method_validator is not None:
        method_validator(raw_cfg)
    return ConfigNode(raw_cfg)


def save_config(
    cfg: ConfigNode | Mapping[str, Any],
    output_path: str | Path,
) -> None:
    """
    保存最终配置。

    一般用于保存：
        outputs/<run_name>/config_used.yaml
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(cfg, ConfigNode):
        cfg_dict = cfg.to_dict()
    else:
        cfg_dict = dict(cfg)

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            cfg_dict,
            f,
            allow_unicode=True,
            sort_keys=False,
        )


def ensure_run_dir(cfg: ConfigNode | Mapping[str, Any]) -> Path:
    """
    创建实验输出目录，并返回 Path。

    注意：
    load_config 只负责生成 run_dir 字段；
    真正创建目录放在这里，避免读取配置时产生太多副作用。
    """
    if isinstance(cfg, ConfigNode):
        run_dir = Path(cfg.run_dir)
    else:
        run_dir = Path(cfg["run_dir"])

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_yaml_with_include(
    config_path: Path,
    stack: Optional[list[Path]] = None,
) -> Dict[str, Any]:
    """
    读取 yaml，并处理 include。

    支持：
        include: base.yaml

    也支持：
        include:
          - base.yaml
          - model/resnet.yaml

    子配置会覆盖 base 配置。
    """
    if stack is None:
        stack = []

    if config_path in stack:
        chain = " -> ".join(str(path) for path in stack + [config_path])
        raise ConfigError(f"检测到循环 include：{chain}")

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, Mapping):
        raise ConfigError(f"配置文件顶层必须是 dict：{config_path}")

    cfg = dict(cfg)
    include = cfg.pop("include", None)

    if include is None:
        return cfg

    if isinstance(include, str):
        include_files = [include]
    elif isinstance(include, list):
        include_files = include
    else:
        raise ConfigError("include 必须是字符串或字符串列表。")

    merged_cfg: Dict[str, Any] = {}

    for include_file in include_files:
        include_path = (config_path.parent / include_file).resolve()
        base_cfg = _load_yaml_with_include(
            include_path,
            stack=stack + [config_path],
        )
        merged_cfg = _deep_merge(merged_cfg, base_cfg)

    # 当前配置覆盖 include 进来的配置。
    merged_cfg = _deep_merge(merged_cfg, cfg)

    return merged_cfg


def _deep_merge(
    base: MutableMapping[str, Any],
    override: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    递归合并配置。

    规则：
    1. override 里的普通字段覆盖 base
    2. override 里的 dict 会递归覆盖 base 里的 dict
    3. list 不做递归合并，直接整体覆盖
    """
    result = copy.deepcopy(dict(base))

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def _apply_method_defaults(
    cfg: Dict[str, Any],
    method_defaults: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Apply method-owned defaults while keeping explicit config values."""
    if method_defaults is None:
        return copy.deepcopy(cfg)

    return _deep_merge(
        base=copy.deepcopy(dict(method_defaults)),
        override=cfg,
    )


def _apply_config_overrides(
    cfg: Dict[str, Any],
    config_overrides: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Apply highest-priority runtime overrides such as command-line values."""
    if config_overrides is None:
        return copy.deepcopy(cfg)
    return _deep_merge(base=cfg, override=config_overrides)


def _apply_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Use EMBEDDED_BASE_CONFIG as the single shared default source."""
    cfg = _deep_merge(base=EMBEDDED_BASE_CONFIG, override=cfg)

    dataset_name = normalize_dataset_name(cfg["dataset"])
    cfg["dataset"] = dataset_name

    cfg.setdefault("num_classes", _infer_num_classes(dataset_name))

    dataset_info = DATASET_INFO.get(dataset_name)
    if dataset_info is not None:
        cfg.setdefault("input_shape", tuple(dataset_info["input_shape"]))

    return cfg


def _finalize_run_info(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成 run_name 和 run_dir。

    规则：
    1. run_name 缺失 / 为空 / auto / null 时，自动根据实验设置生成
    2. 如果输出目录已存在，默认自动追加 _v2 / _v3
    3. 如果 run.overwrite=True，则允许使用已有目录，不自动追加版本号
    """
    cfg = copy.deepcopy(cfg)

    raw_run_name = cfg.get("run_name", "auto")
    should_auto_name = _is_auto_run_name(raw_run_name)

    if should_auto_name:
        run_name = _build_auto_run_name(cfg)
    else:
        run_name = _safe_name(raw_run_name)

    output_dir = Path(cfg.get("output_dir", "outputs_bias"))
    unique_name = bool(cfg.get("run", {}).get("unique_name", True))
    overwrite = bool(cfg.get("run", {}).get("overwrite", False))

    if unique_name and not overwrite:
        run_name = _make_unique_run_name(run_name, output_dir)

    cfg["run_name"] = run_name
    cfg["run_dir"] = str(output_dir / run_name)

    return cfg


def _is_auto_run_name(value: Any) -> bool:
    """判断 run_name 是否需要自动生成。"""
    if value is None:
        return True

    text = str(value).strip().lower()

    return text in {
        "",
        "auto",
        "none",
        "null",
    }


def _build_auto_run_name(cfg: Mapping[str, Any]) -> str:
    """
    根据关键实验设置自动生成实验名。

    文件夹名只放关键字段，详细参数会保存到 config_used.yaml。
    """
    dataset = _safe_name(cfg.get("dataset", "dataset"))
    num_clients = _safe_name(cfg.get("num_clients", "c"))
    alpha = _safe_name(cfg.get("alpha", "iid"))
    model = _safe_name(cfg.get("model", "model"))
    model_cfg = cfg.get("model_cfg", {})
    backbone = _safe_name(
        model_cfg.get("backbone", "backbone")
        if isinstance(model_cfg, Mapping)
        else "backbone"
    )
    num_experts = _safe_name(cfg.get("num_experts", "e"))
    topk = _safe_name(cfg.get("topk", "topk"))
    rounds = _safe_name(cfg.get("rounds", "r"))
    local_epochs = _safe_name(cfg.get("local_epochs", "ep"))
    seed = _safe_name(cfg.get("seed", "seed"))

    agg_cfg = cfg.get("agg", {})
    non_expert_method = _safe_name(
        agg_cfg.get("non_expert", {}).get("method", "non_expert")
    )
    expert_method = _safe_name(
        agg_cfg.get("expert", {}).get("method", "expert")
    )

    return (
        f"{dataset}"
        f"_c{num_clients}"
        f"_a{alpha}"
        f"_{model}"
        f"_bb{backbone}"
        f"_e{num_experts}"
        f"_top{topk}"
        f"_r{rounds}"
        f"_ep{local_epochs}"
        f"_ne{non_expert_method}"
        f"_ex{expert_method}"
        f"_s{seed}"
    )


def _safe_name(value: Any) -> str:
    """
    把任意值转换成适合做文件夹名的字符串。

    示例：
        0.1 -> 0p1
        cuda:0 -> cuda_0
    """
    text = str(value).strip()
    text = text.replace(".", "p")
    text = text.replace("-", "m")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("_")


def _make_unique_run_name(run_name: str, output_dir: Path) -> str:
    """
    如果输出目录已存在，自动追加版本号。

    示例：
        exp
        exp_v2
        exp_v3
    """
    candidate = run_name
    index = 2

    while (output_dir / candidate).exists():
        candidate = f"{run_name}_v{index}"
        index += 1

    return candidate


def _infer_num_classes(dataset: str) -> int:
    """Infer class count from the dataset registry."""
    dataset_name = normalize_dataset_name(dataset)
    info = DATASET_INFO.get(dataset_name)
    if info is None:
        # Validation will report the unsupported dataset with a clearer message.
        return 10
    return int(info["num_classes"])


def _validate_config(cfg: Mapping[str, Any]) -> None:
    """
    基础合法性检查。

    这个函数只检查通用配置。
    后面新增 方法插件 / history / Bayes 时，可以继续拆出新的 validate 函数。
    """
    dataset = normalize_dataset_name(str(cfg.get("dataset")))
    if dataset not in SUPPORTED_DATASETS:
        raise ConfigError(
            f"不支持的数据集：{dataset}。"
            f"当前支持：{sorted(SUPPORTED_DATASETS)}"
        )

    model = cfg.get("model")
    if model not in SUPPORTED_MODELS:
        raise ConfigError(
            f"不支持的模型：{model}。"
            f"当前支持：{sorted(SUPPORTED_MODELS)}"
        )

    model_cfg = cfg.get("model_cfg", {})
    if not isinstance(model_cfg, Mapping):
        raise ConfigError("model_cfg 必须是 dict。")
    backbone = str(model_cfg.get("backbone", "resnet_cifar")).lower().strip()
    if backbone not in BACKBONE_BUILDERS:
        raise ConfigError(
            f"不支持的 backbone：{backbone}。"
            f"当前支持：{sorted(BACKBONE_BUILDERS.keys())}"
        )

    agg_cfg = cfg.get("agg", {})
    non_expert_method = agg_cfg.get("non_expert", {}).get("method")
    expert_method = agg_cfg.get("expert", {}).get("method")

    if str(non_expert_method).lower().strip() != "uniform":
        raise ConfigError(
            "base.py 已固定 non_expert 使用 uniform 聚合，"
            f"当前配置却是 {non_expert_method!r}。"
        )

    if not isinstance(expert_method, str) or not expert_method.strip():
        raise ConfigError(
            "agg.expert.method 必须是非空字符串。"
            "具体专家聚合方法由启动脚本注入，base.py 不维护方法白名单。"
        )

    _require_positive_int(cfg, "num_classes")
    _require_positive_int(cfg, "num_clients")
    _require_positive_int(cfg, "rounds")
    _require_positive_int(cfg, "local_epochs")
    _require_positive_int(cfg, "batch_size")
    _require_positive_int(cfg, "test_batch_size")
    _require_non_negative_int(cfg, "num_workers")
    _require_positive_int(cfg, "num_experts")
    _require_positive_int(cfg, "topk")

    server_evidence_cfg = cfg.get("server_evidence", {})
    if not isinstance(server_evidence_cfg, Mapping):
        raise ConfigError("server_evidence 必须是 dict。")

    server_evidence_size = server_evidence_cfg.get("size", 0)
    if not isinstance(server_evidence_size, int) or server_evidence_size < 0:
        raise ConfigError(
            "server_evidence.size 必须是非负整数，"
            f"当前值：{server_evidence_size}"
        )

    server_evidence_batch_size = server_evidence_cfg.get("batch_size", 256)
    if not isinstance(server_evidence_batch_size, int) or server_evidence_batch_size <= 0:
        raise ConfigError(
            "server_evidence.batch_size 必须是正整数，"
            f"当前值：{server_evidence_batch_size}"
        )

    if int(cfg["topk"]) > int(cfg["num_experts"]):
        raise ConfigError(
            f"topk 不能大于 num_experts："
            f"topk={cfg['topk']}, num_experts={cfg['num_experts']}"
        )

    frac = float(cfg.get("frac"))
    if not (0.0 < frac <= 1.0):
        raise ConfigError(f"frac 必须在 (0, 1] 范围内，当前值：{frac}")

    alpha = float(cfg.get("alpha"))
    if alpha <= 0:
        raise ConfigError(f"alpha 必须大于 0，当前值：{alpha}")

    # 优化器参数必须统一写在 optimizer 下，避免同时存在两套配置入口。
    forbidden_top_level_optimizer_keys = {
        "lr",
        "momentum",
        "weight_decay",
    }

    for key in forbidden_top_level_optimizer_keys:
        if key in cfg:
            raise ConfigError(
                f"请不要在顶层配置 {key}。"
                f"请统一写到 optimizer.{key} 下面。"
            )

    optimizer_cfg = cfg.get("optimizer", {})
    optimizer_type = optimizer_cfg.get("type")

    if optimizer_type not in {"sgd", "adam", "adamw"}:
        raise ConfigError(
            f"不支持的优化器：{optimizer_type}。"
            f"当前支持：sgd, adam, adamw"
        )

    lr = float(optimizer_cfg.get("lr"))
    if lr <= 0:
        raise ConfigError(f"optimizer.lr 必须大于 0，当前值：{lr}")

    weight_decay = float(optimizer_cfg.get("weight_decay"))
    if weight_decay < 0:
        raise ConfigError(
            f"optimizer.weight_decay 不能小于 0，当前值：{weight_decay}"
        )


def _require_positive_int(cfg: Mapping[str, Any], key: str) -> None:
    """检查某个字段是否为正整数。"""
    value = cfg.get(key)

    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{key} 必须是正整数，当前值：{value}")


def _require_non_negative_int(cfg: Mapping[str, Any], key: str) -> None:
    """检查某个字段是否为非负整数。"""
    value = cfg.get(key)

    if not isinstance(value, int) or value < 0:
        raise ConfigError(f"{key} 必须是非负整数，当前值：{value}")


# Shared configuration accessor used by all bundled sections.
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a plain or dotted key from ConfigNode, dict, or a regular object."""
    if cfg is None:
        return default

    if hasattr(cfg, "get"):
        value = cfg.get(key, None)
        if value is not None:
            return value

    current = cfg
    for part in key.split("."):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
            continue
        if hasattr(current, part):
            current = getattr(current, part)
            continue
        return default
    return current


# ============================================================================
# Bundled from utils/seed.py
# ============================================================================


def disable_tf32() -> None:
    """
    关闭 TF32，减少 Ampere 及以后 NVIDIA GPU 上的数值差异。

    说明：
        1. torch.backends.cuda.matmul.allow_tf32
           控制 Linear / matmul / bmm 等矩阵乘法是否允许 TF32。

        2. torch.backends.cudnn.allow_tf32
           控制 cuDNN 卷积是否允许 TF32。

        3. torch.set_float32_matmul_precision("highest")
           是 PyTorch 2.x 的补充设置，表示 float32 矩阵乘法尽量使用更高精度路径。

    注意：
        关闭 TF32 会让训练稍慢一点，但更适合做可复现实验。
    """
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def set_seed(
    seed: int,
    deterministic: bool = True,
    benchmark: Optional[bool] = None,
) -> None:
    """
    设置全局随机种子。

    作用范围：
        1. Python random
        2. NumPy
        3. PyTorch CPU
        4. PyTorch CUDA
        5. Python hash seed

    参数：
        seed:
            随机种子。

        deterministic:
            是否开启 PyTorch 确定性模式。
            如果为 True，实验更容易复现，但训练速度可能变慢。

        benchmark:
            是否开启 cudnn.benchmark。
            如果为 None，则根据 deterministic 自动决定。
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed 必须是 int，当前类型：{type(seed)}")

    if seed < 0:
        raise ValueError(f"seed 必须是非负整数，当前值：{seed}")

    # 固定 Python hash 随机性。
    #
    # 注意：
    #     PYTHONHASHSEED 严格来说需要在 Python 解释器启动前设置才完全生效。
    #     train.py 顶部已经在 import torch 前根据配置文件 seed 做了提前设置和必要重启。
    #     这里保留是为了记录当前运行环境，并作为兜底。
    os.environ["PYTHONHASHSEED"] = str(seed)

    # cuBLAS 确定性配置兜底。
    #
    # 注意：
    #     CUBLAS_WORKSPACE_CONFIG 也应该尽量在 import torch 前设置。
    #     train.py 顶部已经提前设置，这里只是兜底，避免其他入口漏掉。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # 固定 Python / NumPy 随机性
    random.seed(seed)
    np.random.seed(seed)

    # 固定 PyTorch 随机性
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 关闭 TF32，避免 Ampere 及以后 GPU 上使用低精度 Tensor Core 路径。
    # 这会让训练稍慢一点，但数值更稳定，更适合做可重复实验。
    disable_tf32()

    # 控制 cudnn 行为
    torch.backends.cudnn.deterministic = deterministic

    if benchmark is None:
        torch.backends.cudnn.benchmark = not deterministic
    else:
        torch.backends.cudnn.benchmark = benchmark

    # PyTorch 确定性算法开关。
    # warn_only=True 可以避免部分算子不支持确定性时直接崩掉。
    # 如果后面想做更严格的 bitwise 复现，可以再改成 warn_only=False。
    torch.use_deterministic_algorithms(
        deterministic,
        warn_only=True,
    )


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker 的随机种子初始化函数。

    用法：
        DataLoader(
            dataset,
            worker_init_fn=seed_worker,
            generator=build_torch_generator(seed),
        )
    """
    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_torch_generator(seed: int) -> torch.Generator:
    """
    创建带固定随机种子的 torch.Generator。

    主要用于 DataLoader，保证 shuffle 更可复现。
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed 必须是 int，当前类型：{type(seed)}")

    if seed < 0:
        raise ValueError(f"seed 必须是非负整数，当前值：{seed}")

    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator


# ============================================================================
# Bundled from utils/logging.py
# ============================================================================


class TeeStream:
    """
    双写输出流。

    作用：
    把写入 stdout / stderr 的内容同时写到：
    1. 原始控制台
    2. 日志文件

    额外能力：
    可以过滤 tqdm 这类动态进度条，避免 train.log 被进度条污染。
    """

    def __init__(
        self,
        console_stream: TextIO,
        log_file: TextIO,
        *,
        filter_progress: bool = False,
    ) -> None:
        self.console_stream = console_stream
        self.log_file = log_file
        self.filter_progress = bool(filter_progress)
        self.lock = threading.Lock()

    def _should_write_to_log(self, text: str) -> bool:
        """
        判断当前输出是否应该写入日志文件。

        tqdm / rich / 部分终端动态刷新通常会包含：
        1. \\r：回到行首刷新进度条
        2. ANSI 控制符：例如 \\x1b[...m
        3. tqdm 常见片段：%|、it/s、s/it 等

        这些内容适合显示在控制台，但不适合写入 train.log。
        """
        if not self.filter_progress:
            return True

        if not text:
            return False

        # tqdm 动态刷新最常见特征：使用 \r 回到行首重绘。
        if "\r" in text:
            return False

        # 过滤 ANSI 控制符，避免颜色、清屏、光标移动等控制字符进入日志。
        if "\x1b[" in text:
            return False

        # 过滤常见 tqdm 进度条文本。
        progress_patterns = [
            r"\d+%\|",      # 例如： 34%|
            r"\|\s*\d+/",   # 例如： | 34/100
            r"it/s",        # 例如： 2.81it/s
            r"s/it",        # 例如： 1.23s/it
            r"\[\d{2}:\d{2}",  # 例如： [00:12<00:23
        ]

        for pattern in progress_patterns:
            if re.search(pattern, text):
                return False

        return True

    def write(self, text: str) -> int:
        """
        写入控制台，并在必要时写入日志文件。

        注意：
        控制台永远保留原始输出；
        只有日志文件会过滤进度条。
        """
        with self.lock:
            self.console_stream.write(text)
            self.console_stream.flush()

            if self._should_write_to_log(text):
                self.log_file.write(text)
                self.log_file.flush()

        return len(text)

    def flush(self) -> None:
        """
        同时刷新控制台和日志文件。
        """
        with self.lock:
            self.console_stream.flush()
            self.log_file.flush()

    def isatty(self) -> bool:
        """
        保留控制台 TTY 判断能力。

        这对 tqdm / 某些终端输出工具有用。
        """
        return self.console_stream.isatty()

    @property
    def encoding(self) -> str:
        """
        返回原始控制台编码。
        """
        return getattr(self.console_stream, "encoding", "utf-8")


@contextmanager
def tee_output_to_file(
    log_path: str | Path,
    *,
    filter_stderr_progress: bool = True,
) -> Iterator[None]:
    """
    把 stdout 和 stderr 同时写入日志文件。

    用法：
        with tee_output_to_file("outputs/run/train.log"):
            print("hello")

    效果：
    1. 控制台能看到 hello
    2. train.log 里也会保存 hello

    额外说明：
    - stdout 默认不过滤，普通 print 会进入 train.log。
    - stderr 默认过滤进度条，因为 tqdm 通常写 stderr。
    - traceback / 报错信息一般不包含 tqdm 特征，所以仍会进入 train.log。
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    with log_path.open(
        "a",
        encoding="utf-8",
        buffering=1,
    ) as log_file:
        sys.stdout = TeeStream(
            console_stream=old_stdout,
            log_file=log_file,
            filter_progress=False,
        )

        sys.stderr = TeeStream(
            console_stream=old_stderr,
            log_file=log_file,
            filter_progress=filter_stderr_progress,
        )

        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# ============================================================================
# Bundled from utils/state_dict_ops.py
# ============================================================================


StateDict = Mapping[str, torch.Tensor]
MutableStateDict = Dict[str, torch.Tensor]


def clone_state_dict(state_dict: StateDict) -> MutableStateDict:
    """
    深拷贝 state_dict。

    用途：
        1. 保存全局模型参数快照
        2. 避免后续原地修改污染原始模型
    """
    return {
        name: tensor.detach().clone()
        for name, tensor in state_dict.items()
    }


def state_dict_to(
    state_dict: StateDict,
    device: torch.device | str,
) -> MutableStateDict:
    """
    把 state_dict 移动到指定设备。

    示例：
        state_dict_to(state, "cpu")
        state_dict_to(state, "cuda")
    """
    return {
        name: tensor.to(device)
        for name, tensor in state_dict.items()
    }


def subtract_state_dict(
    local_state: StateDict,
    global_state: StateDict,
    param_names: Optional[Iterable[str]] = None,
    strict: bool = True,
) -> MutableStateDict:
    """
    计算客户端本地模型相对全局模型的参数变化量。

    公式：
        delta = local_state - global_state

    说明：
        1. 只处理浮点 tensor
        2. 非浮点 tensor 会跳过
        3. 例如 BatchNorm 的 num_batches_tracked 通常是 int64，不适合做加减
    """
    names = _resolve_param_names(
        state_dict=global_state,
        param_names=param_names,
    )

    delta: MutableStateDict = {}

    for name in names:
        if name not in local_state:
            if strict:
                raise KeyError(f"local_state 缺少参数：{name}")
            continue

        if name not in global_state:
            if strict:
                raise KeyError(f"global_state 缺少参数：{name}")
            continue

        local_tensor = local_state[name]
        global_tensor = global_state[name]

        if not _is_float_tensor(local_tensor):
            continue

        if not _is_float_tensor(global_tensor):
            continue

        delta[name] = local_tensor.detach() - global_tensor.detach()

    return delta


def apply_weighted_delta(
    global_state: StateDict,
    client_updates: Sequence[Any],
    weights: Mapping[int, float],
    param_names: Optional[Iterable[str]] = None,
    base_state: Optional[StateDict] = None,
    strict: bool = True,
) -> MutableStateDict:
    """
    对多个客户端 delta 做加权聚合，并更新到全局模型上。

    公式：
        delta_global = sum_i weight_i * delta_i
        new_state = global_state + delta_global

    参数：
        global_state:
            本轮聚合前的全局模型参数。

        client_updates:
            客户端上传结果列表。
            每个 update 需要包含：
                update.client_id
                update.model_delta

            也兼容 dict 形式：
                update["client_id"]
                update["model_delta"]

        weights:
            每个客户端的聚合权重。
            例如：
                {0: 0.2, 1: 0.3, 2: 0.5}

        param_names:
            只聚合指定参数。
            后面 FL + MoE 解耦时会用它区分：
                非专家参数
                专家参数

        base_state:
            聚合结果写入的基础 state_dict。
            如果为 None，就从 global_state clone 一份。
            如果先聚合非专家参数，再聚合专家参数，可以把上一步结果传进来。

        strict:
            如果为 True，缺少权重或缺少 delta 时直接报错。
            如果为 False，则跳过缺失项。
    """
    if len(client_updates) == 0:
        raise ValueError("client_updates 不能为空。")

    _validate_weights(weights)

    if base_state is None:
        new_state = clone_state_dict(global_state)
    else:
        new_state = clone_state_dict(base_state)

    names = _resolve_param_names(
        state_dict=global_state,
        param_names=param_names,
    )

    for name in names:
        global_tensor = global_state[name]

        # 非浮点 tensor 不参与聚合，保留 base_state 中原来的值。
        if not _is_float_tensor(global_tensor):
            continue

        total_delta = torch.zeros_like(global_tensor)

        for update in client_updates:
            client_id = _get_client_id(update)
            model_delta = _get_model_delta(update)

            if client_id not in weights:
                if strict:
                    raise KeyError(f"weights 缺少客户端 {client_id} 的权重")
                continue

            if name not in model_delta:
                if strict:
                    raise KeyError(
                        f"客户端 {client_id} 的 model_delta 缺少参数：{name}"
                    )
                continue

            weight = float(weights[client_id])
            delta_tensor = model_delta[name].to(global_tensor.device)

            total_delta = total_delta + weight * delta_tensor

        new_state[name] = global_tensor + total_delta

    return new_state


def check_finite_state_dict(
    state_dict: StateDict,
    param_names: Optional[Iterable[str]] = None,
) -> None:
    """
    检查 state_dict 中是否存在 NaN 或 Inf。

    如果发现异常，直接抛出 ValueError。
    """
    names = _resolve_param_names(
        state_dict=state_dict,
        param_names=param_names,
    )

    for name in names:
        tensor = state_dict[name]

        if not _is_float_tensor(tensor):
            continue

        if not torch.isfinite(tensor).all():
            raise ValueError(f"参数 {name} 中存在 NaN 或 Inf。")


def normalize_weights(weights: Mapping[int, float]) -> Dict[int, float]:
    """
    把客户端权重归一化到和为 1。

    输入：
        {0: 10, 1: 20, 2: 30}

    输出：
        {0: 1/6, 1: 2/6, 2: 3/6}
    """
    if len(weights) == 0:
        raise ValueError("weights 不能为空。")

    total = 0.0

    for client_id, weight in weights.items():
        weight = float(weight)

        if not math.isfinite(weight):
            raise ValueError(f"客户端 {client_id} 的权重不是有限数：{weight}")

        if weight < 0:
            raise ValueError(f"客户端 {client_id} 的权重小于 0：{weight}")

        total += weight

    if total <= 0:
        raise ValueError(f"weights 总和必须大于 0，当前总和：{total}")

    return {
        int(client_id): float(weight) / total
        for client_id, weight in weights.items()
    }


def _resolve_param_names(
    state_dict: StateDict,
    param_names: Optional[Iterable[str]],
) -> List[str]:
    """
    解析需要处理的参数名列表。

    如果 param_names 为 None，则使用 state_dict 的所有 key。
    """
    if param_names is None:
        return list(state_dict.keys())

    names = list(param_names)

    for name in names:
        if name not in state_dict:
            raise KeyError(f"state_dict 中不存在参数：{name}")

    return names


def _is_float_tensor(tensor: torch.Tensor) -> bool:
    """
    判断 tensor 是否是浮点 tensor。
    """
    return torch.is_tensor(tensor) and torch.is_floating_point(tensor)


def _validate_weights(weights: Mapping[int, float]) -> None:
    """
    检查客户端权重是否合法。

    注意：
        这里只检查非空、有限、非负。
        不强制要求权重和等于 1。
        如果需要归一化，请先调用 normalize_weights。
    """
    if len(weights) == 0:
        raise ValueError("weights 不能为空。")

    for client_id, weight in weights.items():
        weight = float(weight)

        if not math.isfinite(weight):
            raise ValueError(f"客户端 {client_id} 的权重不是有限数：{weight}")

        if weight < 0:
            raise ValueError(f"客户端 {client_id} 的权重小于 0：{weight}")


def _get_client_id(update: Any) -> int:
    """
    从 ClientUpdate 或 dict 中读取 client_id。
    """
    if isinstance(update, Mapping):
        return int(update["client_id"])

    return int(update.client_id)


def _get_model_delta(update: Any) -> Mapping[str, torch.Tensor]:
    """
    从 ClientUpdate 或 dict 中读取 model_delta。
    """
    if isinstance(update, Mapping):
        return update["model_delta"]

    return update.model_delta


# ============================================================================
# Bundled from utils/eval.py
# ============================================================================


@dataclass(frozen=True)
class EvalResult:
    """
    评估结果。

    loss:
        测试集平均 loss。

    acc:
        Top-1 准确率，百分比形式。
        例如 63.25 表示 63.25%。

    correct:
        预测正确的样本数。

    total:
        总样本数。

    extra:
        预留额外评估指标。
        例如后面可以放 top5_acc、router_usage 等。
    """

    loss: float
    acc: float
    correct: int
    total: int
    extra: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """
        转成普通 dict，方便写日志 / csv / json。
        """
        return {
            "loss": float(self.loss),
            "acc": float(self.acc),
            "correct": int(self.correct),
            "total": int(self.total),
            "extra": dict(self.extra),
        }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | str,
    criterion: Optional[nn.Module] = None,
) -> EvalResult:
    """
    在测试集上评估模型。

    这个函数默认只使用分类 logits。
    不会使用 aux_loss。
    不会加入 router balance。
    不会加入 entropy。
    不会加入 diversity。
    不会加入 consistency。

    参数：
        model:
            待评估模型。

        data_loader:
            测试集 DataLoader。

        device:
            评估设备，例如 "cuda" 或 "cpu"。

        criterion:
            loss 函数。
            如果为 None，则默认使用 CrossEntropyLoss。

    返回：
        EvalResult
    """
    device = torch.device(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    model.to(device)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in data_loader:
        images, targets = unpack_batch(batch)

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        logits = extract_logits(outputs)

        loss = criterion(logits, targets)

        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_correct += count_correct(logits, targets)
        total_samples += batch_size

    if total_samples <= 0:
        raise ValueError("评估集为空，无法计算指标。")

    avg_loss = total_loss / total_samples
    acc = 100.0 * total_correct / total_samples

    return EvalResult(
        loss=avg_loss,
        acc=acc,
        correct=total_correct,
        total=total_samples,
        extra={},
    )


@torch.inference_mode()
def evaluate_topk(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | str,
    topk: Sequence[int] = (1, 5),
    criterion: Optional[nn.Module] = None,
) -> EvalResult:
    """
    支持 Top-k 的评估函数。

    第一版主流程可以先用 evaluate()。
    这个函数主要给 CIFAR100 或后续诊断预留。

    返回：
        EvalResult.extra 里会包含：
            top1_acc
            top5_acc
            ...
    """
    device = torch.device(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    model.to(device)

    topk = tuple(sorted(set(int(k) for k in topk)))
    max_k = max(topk)

    total_loss = 0.0
    total_samples = 0
    topk_correct = {
        k: 0
        for k in topk
    }

    for batch in data_loader:
        images, targets = unpack_batch(batch)

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        logits = extract_logits(outputs)

        loss = criterion(logits, targets)

        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        batch_topk_correct = count_topk_correct(
            logits=logits,
            targets=targets,
            topk=topk,
            max_k=max_k,
        )

        for k, value in batch_topk_correct.items():
            topk_correct[k] += value

    if total_samples <= 0:
        raise ValueError("评估集为空，无法计算指标。")

    avg_loss = total_loss / total_samples

    extra = {
        f"top{k}_acc": 100.0 * correct / total_samples
        for k, correct in topk_correct.items()
    }

    top1_acc = extra.get("top1_acc", 0.0)

    return EvalResult(
        loss=avg_loss,
        acc=top1_acc,
        correct=topk_correct.get(1, 0),
        total=total_samples,
        extra=extra,
    )


def unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 DataLoader batch 中取出 images 和 targets。

    支持常见格式：
        1. (images, targets)
        2. [images, targets]
        3. {"image": images, "label": targets}
        4. {"x": images, "y": targets}

    当前 CIFAR 默认是第一种。
    """
    if isinstance(batch, Mapping):
        if "image" in batch and "label" in batch:
            return batch["image"], batch["label"]

        if "images" in batch and "labels" in batch:
            return batch["images"], batch["labels"]

        if "x" in batch and "y" in batch:
            return batch["x"], batch["y"]

        raise KeyError(
            "不支持的 batch dict 格式。"
            "需要包含 image/label、images/labels 或 x/y。"
        )

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]

    raise TypeError(
        f"不支持的 batch 类型：{type(batch)}。"
        "期望 batch 是 tuple/list 或 dict。"
    )


def extract_logits(outputs: Any) -> torch.Tensor:
    """
    从模型输出中提取 logits。

    支持：
        1. 直接返回 logits tensor
        2. 返回对象，且对象有 .logits
        3. 返回 dict，且 dict["logits"]
        4. 返回 tuple/list，默认第一个元素是 logits

    这样可以兼容：
        logits = model(x)

    也可以兼容：
        output = model(x, return_router_info=True)
        logits = output.logits
    """
    if torch.is_tensor(outputs):
        return outputs

    if hasattr(outputs, "logits"):
        logits = outputs.logits
        if not torch.is_tensor(logits):
            raise TypeError("outputs.logits 不是 torch.Tensor。")
        return logits

    if isinstance(outputs, Mapping):
        if "logits" not in outputs:
            raise KeyError("模型输出 dict 中缺少 logits。")

        logits = outputs["logits"]

        if not torch.is_tensor(logits):
            raise TypeError('outputs["logits"] 不是 torch.Tensor。')

        return logits

    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        logits = outputs[0]

        if not torch.is_tensor(logits):
            raise TypeError("模型输出 tuple/list 的第一个元素不是 torch.Tensor。")

        return logits

    raise TypeError(
        f"无法从模型输出中提取 logits，输出类型：{type(outputs)}"
    )


def count_correct(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> int:
    """
    统计 Top-1 预测正确数量。
    """
    preds = logits.argmax(dim=1)
    correct = preds.eq(targets).sum().item()

    return int(correct)


def count_topk_correct(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Iterable[int],
    max_k: Optional[int] = None,
) -> Dict[int, int]:
    """
    统计 Top-k 预测正确数量。

    返回：
        {
            1: top1_correct,
            5: top5_correct,
            ...
        }
    """
    topk = tuple(sorted(set(int(k) for k in topk)))

    if len(topk) == 0:
        raise ValueError("topk 不能为空。")

    if max_k is None:
        max_k = max(topk)

    max_k = int(max_k)

    if max_k <= 0:
        raise ValueError(f"max_k 必须大于 0，当前值：{max_k}")

    if max_k > logits.size(1):
        max_k = int(logits.size(1))

    _, pred = logits.topk(
        k=max_k,
        dim=1,
        largest=True,
        sorted=True,
    )

    pred = pred.t()
    correct = pred.eq(targets.reshape(1, -1).expand_as(pred))

    result: Dict[int, int] = {}

    for k in topk:
        actual_k = min(k, logits.size(1))
        correct_k = correct[:actual_k].reshape(-1).float().sum().item()
        result[k] = int(correct_k)

    return result


# ============================================================================
# Bundled from data/datasets.py
# ============================================================================


# =========================
# 数据集元信息注册区
# =========================
# 后续新增数据集时，优先在这里注册。
DATASET_INFO: Dict[str, Dict[str, Any]] = {
    "cifar10": {
        "num_classes": 10,
        "input_shape": (3, 32, 32),
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "num_classes": 100,
        "input_shape": (3, 32, 32),
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "cinic10": {
        "num_classes": 10,
        "input_shape": (3, 32, 32),
        "mean": (0.47889522, 0.47227842, 0.43047404),
        "std": (0.24205776, 0.23828046, 0.25874835),
    },
    "fashionmnist": {
        "num_classes": 10,
        "input_shape": (1, 28, 28),
        "mean": (0.2860,),
        "std": (0.3530,),
    },
    "stl10": {
        "num_classes": 10,
        "input_shape": (3, 96, 96),
        "mean": (0.4467, 0.4398, 0.4066),
        "std": (0.2603, 0.2566, 0.2713),
    },
    "tiny-imagenet-200": {
        "num_classes": 200,
        "input_shape": (3, 64, 64),
        "mean": (0.4802, 0.4481, 0.3975),
        "std": (0.2302, 0.2265, 0.2262),
    },
    "femnist": {
        "num_classes": 62,
        "input_shape": (1, 28, 28),
        # LEAF FEMNIST stores pixels as [0, 1] floats. Keep that scale.
        "mean": (0.0,),
        "std": (1.0,),
    },
}



class _ArrayImageDataset(Dataset):
    """Image dataset backed by uint8 numpy arrays and integer targets."""

    def __init__(
        self,
        data: np.ndarray,
        targets: Sequence[int],
        transform: Optional[Callable] = None,
        image_mode: str = "L",
    ) -> None:
        if len(data) != len(targets):
            raise ValueError(
                f"data/targets 数量不一致：{len(data)} vs {len(targets)}"
            )
        self.data = data
        self.targets = [int(target) for target in targets]
        self.transform = transform
        self.image_mode = str(image_mode)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        array = np.asarray(self.data[index], dtype=np.uint8)
        image = Image.fromarray(array, mode=self.image_mode)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets[index])


class _TinyImageNetValDataset(Dataset):
    """Read the original Tiny-ImageNet validation layout without reorganizing files."""

    def __init__(
        self,
        val_root: Path,
        class_to_idx: Mapping[str, int],
        transform: Optional[Callable] = None,
    ) -> None:
        annotation_path = val_root / "val_annotations.txt"
        image_root = val_root / "images"
        if not annotation_path.is_file() or not image_root.is_dir():
            raise FileNotFoundError(
                "Tiny-ImageNet validation split 缺少 val_annotations.txt 或 val/images。"
            )

        samples: List[Tuple[Path, int]] = []
        targets: List[int] = []
        with annotation_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                filename, wnid = parts[0], parts[1]
                if wnid not in class_to_idx:
                    raise ValueError(f"Tiny-ImageNet val 出现未知类别：{wnid}")
                target = int(class_to_idx[wnid])
                samples.append((image_root / filename, target))
                targets.append(target)

        if not samples:
            raise ValueError("Tiny-ImageNet validation split 为空。")

        self.samples = samples
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(target)


def _resolve_local_dataset_root(
    data_root: Path,
    candidates: Sequence[str],
    required_entries: Sequence[str],
    dataset_name: str,
) -> Path:
    """Resolve a manually downloaded dataset directory from common layouts."""
    roots = [data_root]
    roots.extend(data_root / candidate for candidate in candidates)

    for root in roots:
        if all((root / entry).exists() for entry in required_entries):
            return root

    expected = ", ".join(str(data_root / name) for name in candidates)
    raise FileNotFoundError(
        f"未找到 {dataset_name}。请把数据放到以下任一路径并保持标准目录结构：{expected}，"
        f"或者让 --data-root 直接指向数据集根目录。"
    )


def _resolve_femnist_root(data_root: Path) -> Path:
    """Resolve LEAF FEMNIST processed train/test JSON directories."""
    candidates = [
        data_root,
        data_root / "femnist",
        data_root / "FEMNIST",
        data_root / "femnist" / "data",
        data_root / "leaf" / "data" / "femnist" / "data",
    ]
    for root in candidates:
        if (root / "train").is_dir() and (root / "test").is_dir():
            return root
    raise FileNotFoundError(
        "未找到 LEAF FEMNIST 的 train/*.json 与 test/*.json。"
        "请让 --data-root 指向 leaf/data/femnist/data 或其上层目录。"
    )


def _load_femnist_split_storage(
    split_dir: Path,
) -> Tuple[np.ndarray, List[int]]:
    """Flatten all LEAF users in one split into centralized uint8 images/targets."""
    json_files = sorted(split_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"FEMNIST split 中没有 JSON 文件：{split_dir}")

    total_samples = 0
    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        user_data = payload.get("user_data", {})
        for user in payload.get("users", list(user_data.keys())):
            entry = user_data.get(user, {})
            total_samples += len(entry.get("y", []))

    if total_samples <= 0:
        raise ValueError(f"FEMNIST split 为空：{split_dir}")

    data = np.empty((total_samples, 28, 28), dtype=np.uint8)
    targets: List[int] = [0] * total_samples
    offset = 0

    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        user_data = payload.get("user_data", {})
        for user in payload.get("users", list(user_data.keys())):
            entry = user_data.get(user, {})
            xs = entry.get("x", [])
            ys = entry.get("y", [])
            if len(xs) != len(ys):
                raise ValueError(
                    f"FEMNIST user {user} 的 x/y 数量不一致：{len(xs)} vs {len(ys)}"
                )
            for x, y in zip(xs, ys):
                array = np.asarray(x, dtype=np.float32)
                if array.size != 28 * 28:
                    raise ValueError(
                        f"FEMNIST 样本尺寸不是 784：user={user}, size={array.size}"
                    )
                array = array.reshape(28, 28)
                if float(array.max()) if array.size else 0.0 <= 1.0:
                    array = array * 255.0
                data[offset] = np.clip(array, 0.0, 255.0).astype(np.uint8)
                targets[offset] = int(y)
                offset += 1

    if offset != total_samples:
        raise RuntimeError(
            f"FEMNIST 样本计数错误：expected={total_samples}, actual={offset}"
        )
    return data, targets


@dataclass(frozen=True)
class DatasetBundle:
    """
    数据集打包结果。

    这里只保存原始 train / evidence / test dataset。
    客户端划分和 DataLoader 构建不要放在这里。

    注意：
        train_dataset:
            给客户端本地训练使用，可以根据配置开启数据增强。

        train_evidence_dataset:
            给 方法级 evidence 统计使用，强制关闭随机数据增强。
            这样可以避免 RandomCrop / RandomHorizontalFlip 给 方法插件 证据引入额外随机扰动。

        test_dataset:
            给服务端测试使用，不使用随机增强。
    """

    name: str
    train_dataset: Any
    train_evidence_dataset: Any
    test_dataset: Any
    num_classes: int
    input_shape: Tuple[int, int, int]
    server_evidence_dataset: Optional[Any] = None


def build_datasets(cfg: Any) -> DatasetBundle:
    """Build centralized train/evidence/test datasets for the configured benchmark."""
    dataset_name = normalize_dataset_name(str(cfg.dataset))
    data_root = Path(cfg.data_root).expanduser()

    if dataset_name not in DATASET_INFO:
        raise ValueError(
            f"不支持的数据集：{dataset_name}。当前支持：{sorted(DATASET_INFO.keys())}"
        )

    info = DATASET_INFO[dataset_name]
    train_transform = build_train_transform(
        dataset_name=dataset_name,
        use_augmentation=bool(_cfg_get(cfg, "data_augmentation", True)),
    )
    train_evidence_transform = build_train_transform(
        dataset_name=dataset_name,
        use_augmentation=False,
    )
    test_transform = build_test_transform(dataset_name=dataset_name)
    download = bool(_cfg_get(cfg, "download_data", True))

    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=str(data_root), train=True, transform=train_transform, download=download
        )
        train_evidence_dataset = datasets.CIFAR10(
            root=str(data_root), train=True, transform=train_evidence_transform, download=download
        )
        test_dataset = datasets.CIFAR10(
            root=str(data_root), train=False, transform=test_transform, download=download
        )

    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=str(data_root), train=True, transform=train_transform, download=download
        )
        train_evidence_dataset = datasets.CIFAR100(
            root=str(data_root), train=True, transform=train_evidence_transform, download=download
        )
        test_dataset = datasets.CIFAR100(
            root=str(data_root), train=False, transform=test_transform, download=download
        )

    elif dataset_name == "fashionmnist":
        train_dataset = datasets.FashionMNIST(
            root=str(data_root), train=True, transform=train_transform, download=download
        )
        train_evidence_dataset = datasets.FashionMNIST(
            root=str(data_root), train=True, transform=train_evidence_transform, download=download
        )
        test_dataset = datasets.FashionMNIST(
            root=str(data_root), train=False, transform=test_transform, download=download
        )

    elif dataset_name == "stl10":
        train_dataset = datasets.STL10(
            root=str(data_root), split="train", transform=train_transform, download=download
        )
        train_evidence_dataset = datasets.STL10(
            root=str(data_root), split="train", transform=train_evidence_transform, download=download
        )
        test_dataset = datasets.STL10(
            root=str(data_root), split="test", transform=test_transform, download=download
        )

    elif dataset_name == "cinic10":
        root = _resolve_local_dataset_root(
            data_root=data_root,
            candidates=("cinic10", "cinic-10", "CINIC-10"),
            required_entries=("train", "test"),
            dataset_name="CINIC-10",
        )
        train_dataset = datasets.ImageFolder(root / "train", transform=train_transform)
        train_evidence_dataset = datasets.ImageFolder(
            root / "train", transform=train_evidence_transform
        )
        test_dataset = datasets.ImageFolder(root / "test", transform=test_transform)

    elif dataset_name == "tiny-imagenet-200":
        root = _resolve_local_dataset_root(
            data_root=data_root,
            candidates=("tiny-imagenet-200", "tiny_imagenet_200", "TinyImageNet"),
            required_entries=("train", "val"),
            dataset_name="Tiny-ImageNet-200",
        )
        train_dataset = datasets.ImageFolder(root / "train", transform=train_transform)
        train_evidence_dataset = datasets.ImageFolder(
            root / "train", transform=train_evidence_transform
        )

        val_root = root / "val"
        if (val_root / "val_annotations.txt").is_file() and (val_root / "images").is_dir():
            test_dataset = _TinyImageNetValDataset(
                val_root=val_root,
                class_to_idx=train_dataset.class_to_idx,
                transform=test_transform,
            )
        else:
            # Also accept a validation split that the user has reorganized into class folders.
            test_dataset = datasets.ImageFolder(val_root, transform=test_transform)

    elif dataset_name == "femnist":
        # Scheme A: ignore writer identity after loading, then reuse the existing
        # centralized IID/Dirichlet partitioning code below this dataset layer.
        root = _resolve_femnist_root(data_root)
        train_data, train_targets = _load_femnist_split_storage(root / "train")
        test_data, test_targets = _load_femnist_split_storage(root / "test")

        train_dataset = _ArrayImageDataset(
            data=train_data,
            targets=train_targets,
            transform=train_transform,
            image_mode="L",
        )
        # Share the same uint8 backing array; only the transform differs.
        train_evidence_dataset = _ArrayImageDataset(
            data=train_data,
            targets=train_targets,
            transform=train_evidence_transform,
            image_mode="L",
        )
        test_dataset = _ArrayImageDataset(
            data=test_data,
            targets=test_targets,
            transform=test_transform,
            image_mode="L",
        )

    else:
        raise ValueError(f"未实现的数据集加载逻辑：{dataset_name}")

    if len(train_dataset) != len(train_evidence_dataset):
        raise RuntimeError(
            "train_dataset 与 train_evidence_dataset 样本数不一致："
            f"{len(train_dataset)} vs {len(train_evidence_dataset)}"
        )

    bundle = DatasetBundle(
        name=dataset_name,
        train_dataset=train_dataset,
        train_evidence_dataset=train_evidence_dataset,
        test_dataset=test_dataset,
        num_classes=int(info["num_classes"]),
        input_shape=tuple(info["input_shape"]),
    )

    return apply_server_evidence_holdout(
        cfg=cfg,
        bundle=bundle,
    )


def apply_server_evidence_holdout(
    cfg: Any,
    bundle: DatasetBundle,
) -> DatasetBundle:
    """
    Optionally reserve a deterministic server-side evidence subset from train data.

    The default server_evidence.size=0 is a strict no-op: the original train and
    train_evidence dataset objects are returned unchanged. When enabled, the same
    indices are removed from both client-training views, while the reserved subset
    is taken from the deterministic train_evidence_dataset view.
    """
    evidence_cfg = _cfg_get(cfg, "server_evidence", {})
    evidence_size = int(_cfg_get(evidence_cfg, "size", 0))

    if evidence_size <= 0:
        return bundle

    train_size = len(bundle.train_dataset)
    if train_size != len(bundle.train_evidence_dataset):
        raise RuntimeError(
            "train_dataset 与 train_evidence_dataset 样本数不一致，"
            "无法构建 server evidence holdout。"
        )

    if evidence_size >= train_size:
        raise ValueError(
            "server_evidence.size 必须小于训练集大小："
            f"size={evidence_size}, train_size={train_size}"
        )

    num_clients = int(_cfg_get(cfg, "num_clients", 1))
    if train_size - evidence_size < num_clients:
        raise ValueError(
            "server evidence holdout 后剩余训练样本少于客户端数量："
            f"remaining={train_size - evidence_size}, num_clients={num_clients}"
        )

    targets = get_dataset_targets(bundle.train_evidence_dataset)
    class_balanced = bool(_cfg_get(evidence_cfg, "class_balanced", True))
    seed = int(_cfg_get(cfg, "seed", 0))

    evidence_indices = select_server_evidence_indices(
        targets=targets,
        size=evidence_size,
        seed=seed,
        class_balanced=class_balanced,
    )
    evidence_index_set = set(evidence_indices)
    client_indices = [
        index
        for index in range(train_size)
        if index not in evidence_index_set
    ]

    return DatasetBundle(
        name=bundle.name,
        train_dataset=Subset(bundle.train_dataset, client_indices),
        train_evidence_dataset=Subset(
            bundle.train_evidence_dataset,
            client_indices,
        ),
        test_dataset=bundle.test_dataset,
        num_classes=bundle.num_classes,
        input_shape=bundle.input_shape,
        server_evidence_dataset=Subset(
            bundle.train_evidence_dataset,
            evidence_indices,
        ),
    )


def select_server_evidence_indices(
    targets: Sequence[int],
    size: int,
    seed: int,
    class_balanced: bool = True,
) -> List[int]:
    """Select deterministic holdout indices without touching global RNG state."""
    size = int(size)
    if size < 0:
        raise ValueError(f"size 不能小于 0，当前值：{size}")
    if size > len(targets):
        raise ValueError(
            f"size 不能大于 targets 数量：size={size}, total={len(targets)}"
        )
    if size == 0:
        return []

    rng = np.random.default_rng(int(seed) + 300000)

    if not class_balanced:
        selected = rng.choice(len(targets), size=size, replace=False)
        return sorted(int(index) for index in selected.tolist())

    targets_array = np.asarray(targets, dtype=np.int64)
    class_ids = sorted(int(value) for value in np.unique(targets_array).tolist())
    if len(class_ids) == 0:
        raise ValueError("targets 为空，无法构建 server evidence holdout。")

    indices_by_class: Dict[int, List[int]] = {}
    for class_id in class_ids:
        class_indices = np.where(targets_array == class_id)[0].astype(np.int64)
        rng.shuffle(class_indices)
        indices_by_class[class_id] = [int(index) for index in class_indices.tolist()]

    class_order = [int(value) for value in rng.permutation(class_ids).tolist()]
    offsets = {class_id: 0 for class_id in class_ids}
    selected_indices: List[int] = []

    while len(selected_indices) < size:
        made_progress = False
        for class_id in class_order:
            offset = offsets[class_id]
            class_indices = indices_by_class[class_id]
            if offset >= len(class_indices):
                continue

            selected_indices.append(class_indices[offset])
            offsets[class_id] = offset + 1
            made_progress = True

            if len(selected_indices) >= size:
                break

        if not made_progress:
            raise RuntimeError(
                "无法从训练集选出请求数量的 server evidence 样本。"
            )

    return sorted(selected_indices)


def build_train_transform(
    dataset_name: str,
    use_augmentation: bool = True,
) -> Callable:
    """Build dataset-aware train transforms while preserving the CIFAR path."""
    dataset_name = normalize_dataset_name(dataset_name)
    mean, std = get_normalization_stats(dataset_name)
    input_shape = tuple(DATASET_INFO[dataset_name]["input_shape"])
    image_size = int(input_shape[1])

    transform_list: List[Callable] = []

    if use_augmentation:
        if dataset_name in {"cifar10", "cifar100", "cinic10"}:
            # Keep the original CIFAR augmentation exactly unchanged.
            transform_list.extend(
                [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            )
        elif dataset_name == "stl10":
            transform_list.extend(
                [transforms.RandomCrop(96, padding=12), transforms.RandomHorizontalFlip()]
            )
        elif dataset_name == "tiny-imagenet-200":
            transform_list.extend(
                [transforms.RandomCrop(64, padding=8), transforms.RandomHorizontalFlip()]
            )
        elif dataset_name in {"fashionmnist", "femnist"}:
            # Do not horizontally flip clothing/character glyphs. Translation crop only.
            transform_list.append(transforms.RandomCrop(image_size, padding=4))

    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    return transforms.Compose(transform_list)


def build_test_transform(dataset_name: str) -> Callable:
    """Build deterministic evaluation/evidence preprocessing."""
    dataset_name = normalize_dataset_name(dataset_name)
    mean, std = get_normalization_stats(dataset_name)
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    获取数据集元信息。

    后续模型构建时可以用：
        num_classes
        input_shape
    """
    dataset_name = normalize_dataset_name(dataset_name)

    if dataset_name not in DATASET_INFO:
        raise ValueError(
            f"不支持的数据集：{dataset_name}。"
            f"当前支持：{sorted(DATASET_INFO.keys())}"
        )

    return dict(DATASET_INFO[dataset_name])


def get_normalization_stats(dataset_name: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    获取数据集归一化均值和标准差。
    """
    info = get_dataset_info(dataset_name)
    return tuple(info["mean"]), tuple(info["std"])


# ============================================================================
# Bundled from data/partition.py
# ============================================================================


@dataclass(frozen=True)
class PartitionResult:
    """
    数据划分结果。

    client_indices:
        每个客户端对应的样本索引列表。

    client_sample_counts:
        每个客户端的样本数量。

    client_class_counts:
        每个客户端的类别分布。
        形状大致是：
            {
                client_id: {
                    class_id: count
                }
            }

    num_classes:
        数据集类别数。

    partition_method:
        数据划分方法，例如 iid / dirichlet。
    """

    client_indices: List[List[int]]
    client_sample_counts: Dict[int, int]
    client_class_counts: Dict[int, Dict[int, int]]
    num_classes: int
    partition_method: str


def partition_dataset(cfg: Any, dataset: Any) -> PartitionResult:
    """
    根据配置划分训练集。

    第一版支持：
        1. iid
        2. dirichlet

    默认使用 dirichlet，因为你的 FL 实验主要是 non-IID 场景。

    需要的配置字段：
        cfg.num_clients
        cfg.alpha
        cfg.seed

    可选配置字段：
        cfg.partition_method，默认 dirichlet
        cfg.min_samples_per_client，默认 1
        cfg.partition_max_retries，默认 100
    """
    num_clients = int(cfg.num_clients)
    seed = int(cfg.seed)

    partition_method = str(_cfg_get(cfg, "partition_method", "dirichlet")).lower()

    targets = get_dataset_targets(dataset)
    num_classes = infer_num_classes(targets)

    if partition_method == "iid":
        client_indices = build_iid_partition(
            targets=targets,
            num_clients=num_clients,
            seed=seed,
        )

    elif partition_method == "dirichlet":
        alpha = float(cfg.alpha)
        min_samples_per_client = int(_cfg_get(cfg, "min_samples_per_client", 1))
        max_retries = int(_cfg_get(cfg, "partition_max_retries", 100))

        client_indices = build_dirichlet_partition(
            targets=targets,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
            min_samples_per_client=min_samples_per_client,
            max_retries=max_retries,
        )

    else:
        raise ValueError(
            f"不支持的数据划分方法：{partition_method}。"
            f"当前支持：iid, dirichlet"
        )

    validate_partition(
        client_indices=client_indices,
        dataset_size=len(dataset),
        num_clients=num_clients,
    )

    client_sample_counts = compute_client_sample_counts(client_indices)
    client_class_counts = compute_client_class_counts(
        client_indices=client_indices,
        targets=targets,
        num_classes=num_classes,
    )

    return PartitionResult(
        client_indices=client_indices,
        client_sample_counts=client_sample_counts,
        client_class_counts=client_class_counts,
        num_classes=num_classes,
        partition_method=partition_method,
    )


def build_iid_partition(
    targets: Sequence[int],
    num_clients: int,
    seed: int,
) -> List[List[int]]:
    """
    IID 划分。

    做法：
        1. 打乱全部样本索引
        2. 均匀切成 num_clients 份

    注意：
        这里不会保证每个客户端类别完全均衡。
        它只是从整体数据集中随机均分。
    """
    _validate_num_clients(num_clients)

    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(targets))
    rng.shuffle(all_indices)

    split_indices = np.array_split(all_indices, num_clients)

    return [
        split.astype(int).tolist()
        for split in split_indices
    ]


def build_dirichlet_partition(
    targets: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples_per_client: int = 1,
    max_retries: int = 100,
) -> List[List[int]]:
    """
    Dirichlet non-IID 划分。

    核心思想：
        对每个类别 c：
            1. 找出所有属于类别 c 的样本
            2. 从 Dirichlet(alpha) 采样一个客户端比例
            3. 按这个比例把该类别样本分给不同客户端

    alpha 越小：
        客户端类别分布越不均衡。

    alpha 越大：
        客户端类别分布越接近 IID。
    """
    _validate_num_clients(num_clients)

    if alpha <= 0:
        raise ValueError(f"alpha 必须大于 0，当前值：{alpha}")

    if min_samples_per_client < 0:
        raise ValueError(
            f"min_samples_per_client 不能小于 0，当前值：{min_samples_per_client}"
        )

    if max_retries <= 0:
        raise ValueError(f"max_retries 必须大于 0，当前值：{max_retries}")

    targets_array = np.asarray(targets, dtype=np.int64)
    num_classes = infer_num_classes(targets_array)

    for retry_id in range(max_retries):
        rng = np.random.default_rng(seed + retry_id)
        client_indices: List[List[int]] = [
            []
            for _ in range(num_clients)
        ]

        for class_id in range(num_classes):
            class_indices = np.where(targets_array == class_id)[0]
            rng.shuffle(class_indices)

            # 为当前类别采样每个客户端的分配比例
            proportions = rng.dirichlet(
                alpha=np.full(num_clients, alpha, dtype=np.float64)
            )

            # 根据比例切分该类别样本
            split_points = (
                np.cumsum(proportions)[:-1] * len(class_indices)
            ).astype(int)

            class_splits = np.split(class_indices, split_points)

            for client_id, split in enumerate(class_splits):
                client_indices[client_id].extend(split.astype(int).tolist())

        # 每个客户端内部再打乱一次，避免类别块顺序过于明显
        for client_id in range(num_clients):
            rng.shuffle(client_indices[client_id])

        sample_counts = [
            len(indices)
            for indices in client_indices
        ]

        if min(sample_counts) >= min_samples_per_client:
            return client_indices

    raise RuntimeError(
        "Dirichlet 数据划分失败："
        f"尝试 {max_retries} 次后，仍存在客户端样本数小于 "
        f"{min_samples_per_client}。"
        "可以尝试增大 alpha，或降低 min_samples_per_client。"
    )


def get_dataset_targets(dataset: Any) -> List[int]:
    """
    从 torchvision dataset 中取出标签。

    CIFAR10 / CIFAR100 通常有 dataset.targets。
    为了更通用，也兼容 dataset.labels。
    """
    if isinstance(dataset, Subset):
        parent_targets = get_dataset_targets(dataset.dataset)
        return [
            int(parent_targets[int(index)])
            for index in dataset.indices
        ]

    if hasattr(dataset, "targets"):
        targets = dataset.targets

    elif hasattr(dataset, "labels"):
        targets = dataset.labels

    else:
        raise AttributeError(
            "无法从 dataset 中读取标签。"
            "当前支持 Subset，或包含 targets / labels 属性的数据集。"
        )

    return [
        int(label)
        for label in targets
    ]


def infer_num_classes(targets: Sequence[int]) -> int:
    """
    根据标签推断类别数。

    假设标签是从 0 开始的整数类别。
    """
    if len(targets) == 0:
        raise ValueError("targets 为空，无法推断类别数。")

    return int(max(targets)) + 1


def compute_client_sample_counts(
    client_indices: Sequence[Sequence[int]],
) -> Dict[int, int]:
    """
    统计每个客户端的样本数量。
    """
    return {
        client_id: len(indices)
        for client_id, indices in enumerate(client_indices)
    }


def compute_client_class_counts(
    client_indices: Sequence[Sequence[int]],
    targets: Sequence[int],
    num_classes: int,
) -> Dict[int, Dict[int, int]]:
    """
    统计每个客户端的类别分布。
    """
    targets_array = np.asarray(targets, dtype=np.int64)

    result: Dict[int, Dict[int, int]] = {}

    for client_id, indices in enumerate(client_indices):
        class_counts = {
            class_id: 0
            for class_id in range(num_classes)
        }

        if len(indices) > 0:
            client_targets = targets_array[np.asarray(indices, dtype=np.int64)]
            unique_classes, counts = np.unique(client_targets, return_counts=True)

            for class_id, count in zip(unique_classes, counts):
                class_counts[int(class_id)] = int(count)

        result[client_id] = class_counts

    return result


def validate_partition(
    client_indices: Sequence[Sequence[int]],
    dataset_size: int,
    num_clients: int,
) -> None:
    """
    检查数据划分是否合法。

    检查内容：
        1. 客户端数量是否正确
        2. 所有样本是否都被分配
        3. 是否存在重复样本
        4. 是否存在越界索引
    """
    if len(client_indices) != num_clients:
        raise ValueError(
            f"客户端数量不匹配：期望 {num_clients}，"
            f"实际 {len(client_indices)}"
        )

    all_indices: List[int] = []

    for client_id, indices in enumerate(client_indices):
        for index in indices:
            if index < 0 or index >= dataset_size:
                raise ValueError(
                    f"客户端 {client_id} 存在越界样本索引：{index}，"
                    f"数据集大小：{dataset_size}"
                )

            all_indices.append(int(index))

    if len(all_indices) != dataset_size:
        raise ValueError(
            f"划分后样本总数不等于数据集大小："
            f"划分后 {len(all_indices)}，数据集 {dataset_size}"
        )

    unique_indices = set(all_indices)

    if len(unique_indices) != dataset_size:
        raise ValueError(
            f"数据划分存在重复或遗漏："
            f"唯一索引数 {len(unique_indices)}，数据集大小 {dataset_size}"
        )


def partition_summary_to_dict(partition: PartitionResult) -> Dict[str, Any]:
    """
    把划分结果中的摘要信息转成普通 dict。

    用于后续写日志、保存 json。
    注意：
        不保存完整 client_indices，避免日志文件太大。
    """
    return {
        "partition_method": partition.partition_method,
        "num_classes": partition.num_classes,
        "num_clients": len(partition.client_indices),
        "client_sample_counts": partition.client_sample_counts,
        "client_class_counts": partition.client_class_counts,
    }


def _validate_num_clients(num_clients: int) -> None:
    """检查客户端数量是否合法。"""
    if not isinstance(num_clients, int) or num_clients <= 0:
        raise ValueError(f"num_clients 必须是正整数，当前值：{num_clients}")


# ============================================================================
# Bundled from data/loaders.py
# ============================================================================


@dataclass(frozen=True)
class DataLoaderBundle:
    """
    DataLoader 打包结果。

    client_loaders:
        每个客户端对应一个训练 DataLoader。
        用于本地训练，可以包含随机数据增强。

    client_evidence_loaders:
        每个客户端对应一个 方法级 evidence DataLoader。
        用于统计 方法插件 证据，应该来自关闭随机增强的 train_evidence_dataset。

    test_loader:
        服务端测试集 DataLoader。

    client_datasets:
        每个客户端对应的训练 Subset 数据集。

    client_evidence_datasets:
        每个客户端对应的 evidence Subset 数据集。

    client_sample_counts:
        每个客户端的样本数量。
        这里仍然按训练集 client_datasets 统计。
    """

    client_loaders: List[DataLoader]
    client_evidence_loaders: List[DataLoader]
    test_loader: DataLoader
    server_evidence_loader: Optional[DataLoader]
    client_datasets: List[Subset]
    client_evidence_datasets: List[Subset]
    client_sample_counts: Dict[int, int]


def build_dataloaders(
    cfg: Any,
    train_dataset: Dataset,
    train_evidence_dataset: Dataset,
    test_dataset: Dataset,
    client_indices: Sequence[Sequence[int]],
    server_evidence_dataset: Optional[Dataset] = None,
) -> DataLoaderBundle:
    """
    根据客户端样本索引构建 DataLoader。

    这个函数只负责：
        1. 把 train_dataset 切成多个客户端 Subset
        2. 把 train_evidence_dataset 按同一份 client_indices 切成多个客户端 evidence Subset
        3. 为每个客户端创建训练 DataLoader
        4. 为每个客户端创建 方法级 evidence DataLoader
        5. 为服务端创建测试 DataLoader

    不负责：
        1. 加载原始数据集
        2. 生成 Dirichlet 划分
        3. 本地训练
        4. 参数聚合

    注意：
        train_dataset:
            用于本地训练，是否开启随机数据增强由 data/datasets.py 和配置控制。

        train_evidence_dataset:
            用于 方法级 evidence 统计，应该强制关闭随机数据增强。
            这里使用和 train_dataset 完全相同的 client_indices，保证 evidence 样本归属不变。
    """
    batch_size = int(cfg.batch_size)
    test_batch_size = int(cfg.test_batch_size)
    num_workers = int(cfg.num_workers)
    seed = int(cfg.seed)
    pin_memory = _infer_pin_memory(cfg)

    client_datasets = build_client_datasets(
        train_dataset=train_dataset,
        client_indices=client_indices,
    )

    # 方法级 evidence 使用同一份客户端划分索引，
    # 但底层 dataset 换成关闭随机数据增强的 train_evidence_dataset。
    client_evidence_datasets = build_client_datasets(
        train_dataset=train_evidence_dataset,
        client_indices=client_indices,
    )

    client_loaders = build_client_train_loaders(
        cfg=cfg,
        client_datasets=client_datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )

    client_evidence_loaders = build_client_evidence_loaders(
        client_evidence_datasets=client_evidence_datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )

    test_loader = build_test_loader(
        test_dataset=test_dataset,
        batch_size=test_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )

    server_evidence_loader = build_server_evidence_loader(
        cfg=cfg,
        server_evidence_dataset=server_evidence_dataset,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )

    client_sample_counts = {
        client_id: len(client_dataset)
        for client_id, client_dataset in enumerate(client_datasets)
    }

    return DataLoaderBundle(
        client_loaders=client_loaders,
        client_evidence_loaders=client_evidence_loaders,
        test_loader=test_loader,
        server_evidence_loader=server_evidence_loader,
        client_datasets=client_datasets,
        client_evidence_datasets=client_evidence_datasets,
        client_sample_counts=client_sample_counts,
    )


def build_client_datasets(
    train_dataset: Dataset,
    client_indices: Sequence[Sequence[int]],
) -> List[Subset]:
    """
    根据 client_indices 创建客户端 Subset。

    每个客户端只看到自己对应的训练样本。

    这个函数同时用于：
        1. 普通训练 train_dataset
        2. 方法级 evidence train_evidence_dataset

    二者使用同一份 client_indices，保证客户端数据划分完全一致。
    """
    client_datasets: List[Subset] = []

    for client_id, indices in enumerate(client_indices):
        if len(indices) == 0:
            raise ValueError(
                f"客户端 {client_id} 没有训练样本，"
                "请检查数据划分结果。"
            )

        client_dataset = Subset(
            train_dataset,
            list(indices),
        )
        client_datasets.append(client_dataset)

    return client_datasets


def build_client_train_loaders(
    cfg: Any,
    client_datasets: Sequence[Dataset],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> List[DataLoader]:
    """
    为每个客户端创建训练 DataLoader。

    训练集默认 shuffle=True。
    每个客户端使用不同的 generator seed，避免所有客户端 shuffle 顺序完全一致。
    """
    client_loaders: List[DataLoader] = []
    drop_last = bool(_cfg_get(cfg, "drop_last", False))
    persistent_workers = num_workers > 0

    for client_id, client_dataset in enumerate(client_datasets):
        generator = build_torch_generator(seed + client_id)

        loader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            worker_init_fn=seed_worker,
            generator=generator,
            persistent_workers=persistent_workers,
        )
        client_loaders.append(loader)

    return client_loaders


def build_client_evidence_loaders(
    client_evidence_datasets: Sequence[Dataset],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> List[DataLoader]:
    """
    为每个客户端创建 方法级 evidence DataLoader。

    和训练 DataLoader 的区别：
        1. shuffle=False，保证 evidence 统计顺序稳定。
        2. drop_last=False，保证客户端所有 evidence 样本都参与 方法插件 统计。
        3. dataset 应该来自 train_evidence_dataset，其 transform 已经关闭随机数据增强。

    注意：
        model.eval() 只能关闭 Dropout / BN 这类模型训练态行为，
        不能关闭 torchvision 的 RandomCrop / RandomHorizontalFlip。
        因此必须在 dataset 层面单独使用无增强 transform。
    """
    client_evidence_loaders: List[DataLoader] = []
    persistent_workers = num_workers > 0

    for client_id, client_evidence_dataset in enumerate(client_evidence_datasets):
        generator = build_torch_generator(seed + 100000 + client_id)

        loader = DataLoader(
            client_evidence_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=generator,
            persistent_workers=persistent_workers,
        )
        client_evidence_loaders.append(loader)

    return client_evidence_loaders


def build_server_evidence_loader(
    cfg: Any,
    server_evidence_dataset: Optional[Dataset],
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> Optional[DataLoader]:
    """Build the optional deterministic server-side method evidence loader."""
    if server_evidence_dataset is None:
        return None

    if len(server_evidence_dataset) <= 0:
        raise ValueError("server_evidence_dataset 不能为空。")

    evidence_cfg = _cfg_get(cfg, "server_evidence", {})
    batch_size = int(_cfg_get(evidence_cfg, "batch_size", 256))
    persistent_workers = num_workers > 0

    return DataLoader(
        server_evidence_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=build_torch_generator(int(seed) + 300000),
        persistent_workers=persistent_workers,
    )


def build_test_loader(
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    """
    创建服务端测试 DataLoader。

    测试集必须 shuffle=False，保证每次评估顺序稳定。
    """
    persistent_workers = num_workers > 0

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=build_torch_generator(seed),
        persistent_workers=persistent_workers,
    )


def _infer_pin_memory(cfg: Any) -> bool:
    """
    判断 DataLoader 是否启用 pin_memory。

    规则：
        1. 如果配置里显式写了 pin_memory，就使用配置值
        2. 如果 device 是 cuda 或 auto，就默认启用
        3. 如果 device 是 cpu，就默认关闭
    """
    explicit_pin_memory = _cfg_get(cfg, "pin_memory", None)
    if explicit_pin_memory is not None:
        return bool(explicit_pin_memory)

    device = str(_cfg_get(cfg, "device", "auto")).lower()
    if device == "cpu":
        return False

    return True


# ============================================================================
# Backbone definitions and registry
# ============================================================================


class BasicBlock(nn.Module):
    """
    CIFAR 风格 ResNet BasicBlock。

    结构：
        Conv3x3 -> ReLU -> Conv3x3 -> Residual -> ReLU

    说明：
        这是 联邦专家方法 对齐版 no-BN block。
        不使用 BatchNorm，避免 non-IID FL 中 BN running_mean /
        running_var 在客户端聚合后产生统计失配。

    这个 block 比 torchvision 默认 ResNet 更适合 CIFAR 小图，
    因为前面的 stem 不会过早大幅下采样。
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        # 联邦专家方法 对齐版：不使用 BatchNorm。
        # 保留 bn1 名字并设为 Identity，避免改动 forward 逻辑。
        self.bn1 = nn.Identity()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # 联邦专家方法 对齐版：不使用 BatchNorm。
        # 保留 bn2 名字并设为 Identity，避免改动 forward 逻辑。
        self.bn2 = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

        # 当通道数或空间尺寸变化时，用 1x1 Conv 对齐残差分支。
        # 联邦专家方法 对齐版：shortcut 中同样不使用 BatchNorm。
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    """
    ResNet 图像特征提取器。

    输入：
        x: [B, C, H, W]

    输出：
        feat: [B, 512]

    说明：
    - 对 CIFAR10 / CIFAR100 这类 32x32 小图，stem 使用 stride=1。
    - 对 TinyImageNet 这类更大图，stem 使用 stride=2。
    - 最后通过 AdaptiveAvgPool2d(1) 得到单个全局特征向量。
    - 联邦专家方法 对齐版不使用 BatchNorm。
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
    ) -> None:
        super().__init__()

        stem_stride = 1 if int(image_size) <= 32 else 2

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=3,
                stride=stem_stride,
                padding=1,
                bias=False,
            ),
            # 联邦专家方法 对齐版：stem 中不使用 BatchNorm。
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 512

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> nn.Sequential:
        """
        每个 stage 使用两个 BasicBlock。
        第一个 block 负责必要的下采样，第二个 block 保持尺寸。
        """
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride=stride),
            BasicBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.flatten(1)
        return x


class VGG11Backbone(nn.Module):
    """
    VGG11 backbone without BatchNorm, adapted for the image sizes
    used by this project.

    Standard VGG11 convolution layout:
        64
        M
        128
        M
        256
        256
        M
        512
        512
        M
        512
        512
        M

    Design choices:
    - BatchNorm is not used.
    - 28/32px inputs skip the final MaxPool to avoid excessive downsampling.
    - 64/96px inputs keep all five MaxPool stages.
    - AdaptiveAvgPool2d(1) gives a fixed 512-dimensional feature vector.
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
    ) -> None:
        super().__init__()

        image_size = int(image_size)

        layers: List[nn.Module] = []
        current_channels = int(in_channels)

        # Standard VGG11 / configuration A.
        cfg: Sequence[int | str] = (
            64,
            "M",
            128,
            "M",
            256,
            256,
            "M",
            512,
            512,
            "M",
            512,
            512,
            "M",
        )

        num_pools = 0

        for item in cfg:
            if item == "M":
                num_pools += 1

                # 28/32px images use four pooling stages.
                # Larger images retain the standard five pooling stages.
                if image_size <= 32 and num_pools == 5:
                    continue

                layers.append(
                    nn.MaxPool2d(
                        kernel_size=2,
                        stride=2,
                    )
                )
                continue

            out_channels = int(item)

            layers.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
            )
            layers.append(
                nn.ReLU(inplace=True)
            )

            current_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 512

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


class VGG11075Backbone(nn.Module):
    """
    VGG11-0.75x backbone without BatchNorm.

    This keeps the existing VGG11 topology unchanged and only scales every
    convolutional channel width by 0.75:
        48
        M
        96
        M
        192
        192
        M
        384
        384
        M
        384
        384
        M

    Design choices are identical to VGG11Backbone:
    - BatchNorm is not used.
    - 28/32px inputs skip the final MaxPool to avoid excessive downsampling.
    - 64/96px inputs keep all five MaxPool stages.
    - AdaptiveAvgPool2d(1) gives a fixed 384-dimensional feature vector.

    SparseMoEClassifier will automatically apply Linear(384, 512) before the
    shared router/expert head, so the MoE parameterization remains unchanged.
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
    ) -> None:
        super().__init__()

        image_size = int(image_size)

        layers: List[nn.Module] = []
        current_channels = int(in_channels)

        # VGG11 / configuration A with all convolution widths scaled by 0.75.
        cfg: Sequence[int | str] = (
            48,
            "M",
            96,
            "M",
            192,
            192,
            "M",
            384,
            384,
            "M",
            384,
            384,
            "M",
        )

        num_pools = 0

        for item in cfg:
            if item == "M":
                num_pools += 1

                # Keep the existing VGG11 small-image pooling rule unchanged.
                if image_size <= 32 and num_pools == 5:
                    continue

                layers.append(
                    nn.MaxPool2d(
                        kernel_size=2,
                        stride=2,
                    )
                )
                continue

            out_channels = int(item)

            layers.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
            )
            layers.append(
                nn.ReLU(inplace=True)
            )

            current_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 384

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


class ViTTinyBlock(nn.Module):
    """Pre-norm Transformer encoder block for the small-image ViT-Tiny."""

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        hidden_dim = int(round(embed_dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTinyBackbone(nn.Module):
    """
    ViT-Tiny adapted for the small image sizes used by this project.

    The Transformer scale follows the common ViT/DeiT-Tiny recipe:
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.

    Patch size is dataset-size aware:
        image_size <= 32 -> patch_size=4
        image_size > 32  -> patch_size=8

    The model is trained from scratch. Positional embeddings are created for
    the fixed input resolution of each experiment, so no runtime interpolation
    is needed for the current dataset pipeline.
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        image_size = int(image_size)
        patch_size = 4 if image_size <= 32 else 8
        if image_size % patch_size != 0:
            raise ValueError(
                "ViT-Tiny 要求 image_size 能被 patch_size 整除："
                f"image_size={image_size}, patch_size={patch_size}"
            )

        self.image_size = image_size
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.feat_dim = int(embed_dim)

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

        grid_size = image_size // patch_size
        self.num_patches = int(grid_size * grid_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )

        self.blocks = nn.ModuleList(
            [
                ViTTinyBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"ViT-Tiny 期望输入为 [B, C, H, W]，当前 shape={tuple(x.shape)}"
            )
        if x.size(-2) != self.image_size or x.size(-1) != self.image_size:
            raise ValueError(
                "ViT-Tiny 当前实验使用固定输入尺寸："
                f"expected={self.image_size}x{self.image_size}, "
                f"actual={x.size(-2)}x{x.size(-1)}"
            )

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]


def _swin_window_partition(
    x: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Partition [B, H, W, C] features into non-overlapping windows."""
    if x.dim() != 4:
        raise ValueError(
            f"Swin window partition expects [B, H, W, C], got {tuple(x.shape)}"
        )

    batch_size, height, width, channels = x.shape
    window_size = int(window_size)

    if height % window_size != 0 or width % window_size != 0:
        raise ValueError(
            "Swin feature resolution must be divisible by window_size: "
            f"H={height}, W={width}, window_size={window_size}"
        )

    x = x.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, channels)


def _swin_window_reverse(
    windows: torch.Tensor,
    window_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Reverse window partition back to [B, H, W, C]."""
    window_size = int(window_size)
    height = int(height)
    width = int(width)

    windows_per_image = (height // window_size) * (width // window_size)
    if windows_per_image <= 0 or windows.size(0) % windows_per_image != 0:
        raise ValueError(
            "Invalid Swin window tensor for reverse operation: "
            f"windows={tuple(windows.shape)}, H={height}, W={width}, "
            f"window_size={window_size}"
        )

    batch_size = windows.size(0) // windows_per_image
    x = windows.view(
        batch_size,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch_size, height, width, -1)


class SwinWindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        self.dim = int(dim)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)

        if self.dim % self.num_heads != 0:
            raise ValueError(
                "Swin attention dim must be divisible by num_heads: "
                f"dim={self.dim}, num_heads={self.num_heads}"
            )

        head_dim = self.dim // self.num_heads
        self.scale = head_dim ** -0.5

        relative_position_count = (2 * self.window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_position_count, self.num_heads)
        )

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)

        # This index is architecture metadata, not trainable FL state.
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=False,
        )

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_windows, num_tokens, channels = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            batch_windows,
            num_tokens,
            3,
            self.num_heads,
            channels // self.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        relative_position_bias = relative_position_bias.view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            self.num_heads,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = int(mask.size(0))
            if batch_windows % num_windows != 0:
                raise ValueError(
                    "Swin attention mask/window count mismatch: "
                    f"batch_windows={batch_windows}, num_windows={num_windows}"
                )

            attn = attn.view(
                batch_windows // num_windows,
                num_windows,
                self.num_heads,
                num_tokens,
                num_tokens,
            )
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(
                -1,
                self.num_heads,
                num_tokens,
                num_tokens,
            )

        attn = F.softmax(attn.float(), dim=-1).to(dtype=q.dtype)
        x = (attn @ v).transpose(1, 2).reshape(
            batch_windows,
            num_tokens,
            channels,
        )
        return self.proj(x)


class SwinMLP(nn.Module):
    """Two-layer MLP used inside a Swin Transformer block."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(round(int(dim) * float(mlp_ratio)))
        self.fc1 = nn.Linear(int(dim), hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, int(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SwinTransformerBlock(nn.Module):
    """Pre-norm Swin block with regular or shifted window attention."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 4,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        self.dim = int(dim)
        self.input_resolution = (
            int(input_resolution[0]),
            int(input_resolution[1]),
        )

        height, width = self.input_resolution
        requested_window_size = int(window_size)
        self.window_size = min(requested_window_size, height, width)

        # When one window already covers the full feature map there is no
        # meaningful shifted-window partition, so disable shifting.
        if min(height, width) <= self.window_size:
            self.shift_size = 0
        else:
            self.shift_size = int(shift_size)

        if self.shift_size >= self.window_size:
            raise ValueError(
                "Swin shift_size must be smaller than window_size: "
                f"shift={self.shift_size}, window={self.window_size}"
            )

        if height % self.window_size != 0 or width % self.window_size != 0:
            raise ValueError(
                "The CIFAR Swin implementation requires every stage resolution "
                "to be divisible by its window size: "
                f"resolution={self.input_resolution}, window={self.window_size}"
            )

        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = SwinWindowAttention(
            dim=self.dim,
            window_size=self.window_size,
            num_heads=int(num_heads),
        )
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = SwinMLP(
            dim=self.dim,
            mlp_ratio=float(mlp_ratio),
        )

        attn_mask = self._build_attention_mask()
        # The mask is deterministic architecture metadata and should not be
        # transmitted or aggregated as part of the federated model state.
        self.register_buffer(
            "attn_mask",
            attn_mask,
            persistent=False,
        )

    def _build_attention_mask(self) -> Optional[torch.Tensor]:
        if self.shift_size <= 0:
            return None

        height, width = self.input_resolution
        image_mask = torch.zeros((1, height, width, 1), dtype=torch.float32)

        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                image_mask[:, h_slice, w_slice, :] = count
                count += 1

        mask_windows = _swin_window_partition(
            image_mask,
            self.window_size,
        )
        mask_windows = mask_windows.view(
            -1,
            self.window_size * self.window_size,
        )
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        batch_size, num_tokens, channels = x.shape

        if num_tokens != height * width:
            raise ValueError(
                "Swin token count does not match the configured resolution: "
                f"tokens={num_tokens}, resolution={height}x{width}"
            )
        if channels != self.dim:
            raise ValueError(
                f"Swin channel mismatch: expected={self.dim}, actual={channels}"
            )

        shortcut = x
        x = self.norm1(x).view(batch_size, height, width, channels)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            shifted_x = x

        x_windows = _swin_window_partition(
            shifted_x,
            self.window_size,
        )
        x_windows = x_windows.view(
            -1,
            self.window_size * self.window_size,
            channels,
        )

        attn_windows = self.attn(
            x_windows,
            mask=self.attn_mask,
        )
        attn_windows = attn_windows.view(
            -1,
            self.window_size,
            self.window_size,
            channels,
        )

        shifted_x = _swin_window_reverse(
            attn_windows,
            self.window_size,
            height,
            width,
        )

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x

        x = shortcut + x.view(batch_size, num_tokens, channels)
        x = x + self.mlp(self.norm2(x))
        return x


class SwinPatchMerging(nn.Module):
    """2x spatial downsampling used between Swin stages."""

    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
    ) -> None:
        super().__init__()
        self.input_resolution = (
            int(input_resolution[0]),
            int(input_resolution[1]),
        )
        self.dim = int(dim)
        self.norm = nn.LayerNorm(4 * self.dim)
        self.reduction = nn.Linear(
            4 * self.dim,
            2 * self.dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        batch_size, num_tokens, channels = x.shape

        if num_tokens != height * width:
            raise ValueError(
                "Swin patch merging token count mismatch: "
                f"tokens={num_tokens}, resolution={height}x{width}"
            )
        if channels != self.dim:
            raise ValueError(
                "Swin patch merging channel mismatch: "
                f"expected={self.dim}, actual={channels}"
            )
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                "Swin patch merging requires an even feature resolution: "
                f"resolution={height}x{width}"
            )

        x = x.view(batch_size, height, width, channels)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(batch_size, -1, 4 * channels)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinStage(nn.Module):
    """One hierarchical Swin stage, optionally followed by patch merging."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        downsample: bool,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=int(dim),
                    input_resolution=input_resolution,
                    num_heads=int(num_heads),
                    window_size=int(window_size),
                    shift_size=0 if block_id % 2 == 0 else int(window_size) // 2,
                    mlp_ratio=float(mlp_ratio),
                )
                for block_id in range(int(depth))
            ]
        )

        if downsample:
            self.downsample: Optional[nn.Module] = SwinPatchMerging(
                input_resolution=input_resolution,
                dim=int(dim),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class SwinTinyBackbone(nn.Module):
    """
    Swin-Tiny adapted for CIFAR-10 / CIFAR-100 32x32 inputs.

    CIFAR Scheme A used in this project:
        embed_dim=48
        depths=(2, 2, 6, 2)
        num_heads=(2, 4, 8, 16)
        mlp_ratio=4

    Small-image adaptation:
        patch_size=2
        window_size=4

    For a 32x32 CIFAR image the spatial hierarchy is:
        32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2

    Channel hierarchy:
        48 -> 96 -> 192 -> 384

    Output:
        [B, 384]

    SparseMoEClassifier will automatically apply:
        Linear(384, 512)
    before the shared router/expert head, so the MoE parameterization remains
    identical to the other backbones.

    This backbone is trained from scratch and uses LayerNorm only; it does not
    introduce BatchNorm running statistics into the federated model.
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 2,
        embed_dim: int = 48,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (2, 4, 8, 16),
        window_size: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        image_size = int(image_size)
        patch_size = int(patch_size)
        embed_dim = int(embed_dim)

        if image_size != 32:
            raise ValueError(
                "The current swin_tiny backbone is intentionally adapted for "
                "CIFAR-10/CIFAR-100 32x32 inputs only: "
                f"image_size={image_size}"
            )
        if int(in_channels) != 3:
            raise ValueError(
                "The current swin_tiny backbone expects 3-channel CIFAR input: "
                f"in_channels={in_channels}"
            )
        if len(depths) != 4 or len(num_heads) != 4:
            raise ValueError("Swin-Tiny requires four stage depths and head counts.")
        if image_size % patch_size != 0:
            raise ValueError(
                "Swin-Tiny image_size must be divisible by patch_size: "
                f"image_size={image_size}, patch_size={patch_size}"
            )

        patch_resolution = image_size // patch_size
        if patch_resolution % 8 != 0:
            raise ValueError(
                "Swin-Tiny needs three 2x patch-merging steps after patch embedding: "
                f"patch_resolution={patch_resolution}"
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = tuple(int(value) for value in depths)
        self.num_heads = tuple(int(value) for value in num_heads)
        self.window_size = int(window_size)

        self.patch_embed = nn.Conv2d(
            int(in_channels),
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )
        self.patch_norm = nn.LayerNorm(embed_dim)

        stages: List[nn.Module] = []
        for stage_id in range(4):
            stage_dim = embed_dim * (2 ** stage_id)
            stage_resolution = patch_resolution // (2 ** stage_id)
            stages.append(
                SwinStage(
                    dim=stage_dim,
                    input_resolution=(stage_resolution, stage_resolution),
                    depth=self.depths[stage_id],
                    num_heads=self.num_heads[stage_id],
                    window_size=self.window_size,
                    mlp_ratio=float(mlp_ratio),
                    downsample=stage_id < 3,
                )
            )

        self.stages = nn.ModuleList(stages)
        self.feat_dim = embed_dim * 8
        self.norm = nn.LayerNorm(self.feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"Swin-Tiny expects [B, C, H, W], got shape={tuple(x.shape)}"
            )
        if x.size(1) != 3:
            raise ValueError(
                f"Swin-Tiny expects 3 input channels, got {x.size(1)}"
            )
        if x.size(-2) != self.image_size or x.size(-1) != self.image_size:
            raise ValueError(
                "Swin-Tiny uses a fixed CIFAR input resolution: "
                f"expected={self.image_size}x{self.image_size}, "
                f"actual={x.size(-2)}x{x.size(-1)}"
            )

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.patch_norm(x)

        for stage in self.stages:
            x = stage(x)

        x = self.norm(x)
        return x.mean(dim=1)


# -------------------------
# Backbone builders / registry
# -------------------------

BackboneBuilder = Callable[..., nn.Module]
DEFAULT_BACKBONE_NAME = "resnet_cifar"
MOE_FEATURE_DIM = 512


def build_resnet_cifar_backbone(
    *,
    in_channels: int = 3,
    image_size: int = 32,
) -> ResNetBackbone:
    """Build the current no-BN CIFAR-style ResNet backbone."""
    return ResNetBackbone(
        in_channels=in_channels,
        image_size=image_size,
    )


def build_vgg11_backbone(
    *,
    in_channels: int = 3,
    image_size: int = 32,
) -> VGG11Backbone:
    """Build the no-BN, small-image-adapted VGG11 backbone."""
    return VGG11Backbone(
        in_channels=in_channels,
        image_size=image_size,
    )


def build_vgg11_075_backbone(
    *,
    in_channels: int = 3,
    image_size: int = 32,
) -> VGG11075Backbone:
    """Build the no-BN VGG11 backbone with 0.75x channel widths."""
    return VGG11075Backbone(
        in_channels=in_channels,
        image_size=image_size,
    )


def build_vit_tiny_backbone(
    *,
    in_channels: int = 3,
    image_size: int = 32,
) -> ViTTinyBackbone:
    """Build the small-image ViT-Tiny backbone from scratch."""
    return ViTTinyBackbone(
        in_channels=in_channels,
        image_size=image_size,
    )


def build_swin_tiny_backbone(
    *,
    in_channels: int = 3,
    image_size: int = 32,
) -> SwinTinyBackbone:
    """Build the CIFAR-adapted Scheme-A Swin-Tiny backbone from scratch."""
    return SwinTinyBackbone(
        in_channels=in_channels,
        image_size=image_size,
    )


BACKBONE_BUILDERS: Dict[str, BackboneBuilder] = {
    DEFAULT_BACKBONE_NAME: build_resnet_cifar_backbone,
    "vgg11": build_vgg11_backbone,
    "vgg11_075": build_vgg11_075_backbone,
    "vit_tiny": build_vit_tiny_backbone,
    "swin_tiny": build_swin_tiny_backbone,
}


def build_backbone(
    backbone_name: str,
    *,
    in_channels: int = 3,
    image_size: int = 32,
) -> nn.Module:
    """Build a registered backbone."""
    name = str(backbone_name).lower().strip()
    if name not in BACKBONE_BUILDERS:
        raise ValueError(
            f"不支持的 backbone：{name}。"
            f"当前支持：{sorted(BACKBONE_BUILDERS.keys())}"
        )
    backbone = BACKBONE_BUILDERS[name](
        in_channels=int(in_channels), image_size=int(image_size)
    )
    if not hasattr(backbone, "feat_dim"):
        raise ValueError(f"backbone {name!r} 缺少 feat_dim 属性。")
    feat_dim=int(getattr(backbone,"feat_dim"))
    if feat_dim<=0: raise ValueError(f"backbone {name!r} 的 feat_dim 必须大于 0。")
    return backbone


def list_supported_backbones() -> List[str]:
    return sorted(BACKBONE_BUILDERS.keys())


# ============================================================================
# Sparse MoE model
# ============================================================================


@dataclass(frozen=True)
class SparseMoEClassifierOutput:
    """
    SparseMoEClassifier 的可选输出结构。

    默认训练时不需要这个结构，直接返回 logits 即可。
    当需要分析 router / expert usage 时，可以设置 return_router_info=True。
    """

    logits: torch.Tensor
    router_info: Dict[str, Any]


class ExpertFFN(nn.Module):
    """
    单个 expert。

    这里 expert 内部直接输出分类 logits 的一部分：
        feature -> Linear -> ReLU -> Linear -> num_classes

    因此这个模型里“分类头”是在 expert 内部的。
    聚合 expert 参数时，会同时聚合每个 expert 的 fc1 和 fc2。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x, inplace=False)
        x = self.fc2(x)
        return x


class TopKGating(nn.Module):
    """
    标准 Top-K 路由器。

    输入：
        x: [B, in_dim]

    输出：
        weights: [B, num_experts]
            只有 top-k expert 位置非零。
            默认保持原始 softmax 概率，不重新归一化。
        topk_indices: [B, topk]
            每个样本选中的 expert id。
        router_probs: [B, num_experts]
            softmax 后的完整路由概率。
        router_logits: [B, num_experts]
            router 原始 logits。

    注意：
    - 不加乘法噪声。
    - 不加负载均衡 loss。
    - 不加 router entropy / diversity / consistency 正则。
    """

    def __init__(
        self,
        in_dim: int,
        num_experts: int,
        topk: int,
        renormalize_topk_probs: bool = False,
    ) -> None:
        super().__init__()

        if num_experts <= 0:
            raise ValueError(f"num_experts 必须大于 0，当前值：{num_experts}")
        if topk <= 0:
            raise ValueError(f"topk 必须大于 0，当前值：{topk}")
        if topk > num_experts:
            raise ValueError(
                f"topk 不能大于 num_experts，当前 topk={topk}, "
                f"num_experts={num_experts}"
            )

        self.in_dim = int(in_dim)
        self.num_experts = int(num_experts)
        self.topk = int(topk)
        self.renormalize_topk_probs = bool(renormalize_topk_probs)

        self.gate = nn.Linear(in_dim, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 2:
            raise ValueError(f"TopKGating 期望输入为 [B, D]，当前 shape={tuple(x.shape)}")

        router_logits = self.gate(x)
        router_probs = F.softmax(router_logits.float(), dim=-1)

        topk_probs, topk_indices = torch.topk(
            router_probs,
            k=self.topk,
            dim=-1,
        )

        if self.renormalize_topk_probs:
            topk_probs = topk_probs / topk_probs.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)

        weights = torch.zeros_like(router_probs)
        weights.scatter_(dim=1, index=topk_indices, src=topk_probs)
        weights = weights.to(dtype=x.dtype)

        return weights, topk_indices, router_probs, router_logits


class SparseMoEHead(nn.Module):
    """
    稀疏 MoE 分类头。

    输入：
        x: [B, feat_dim]

    输出：
        logits: [B, num_classes]

    计算方式：
    1. router 为每个样本选择 top-k 个 expert。
    2. 遍历 expert。
    3. 只把被该 expert 选中的样本送入该 expert。
    4. 用 router 权重加权 expert 输出。
    5. 所有被选中的 expert 输出累加成最终 logits。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_experts: int,
        topk: int,
        renormalize_topk_probs: bool = False,
    ) -> None:
        super().__init__()

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.num_experts = int(num_experts)
        self.topk = int(topk)

        self.gating = TopKGating(
            in_dim=in_dim,
            num_experts=num_experts,
            topk=topk,
            renormalize_topk_probs=renormalize_topk_probs,
        )

        # 这个命名很重要：
        # 参数名会包含 moe_head.experts.<expert_id>....
        # 这样现有 param_groups / 方法证据 逻辑更容易识别 expert 参数。
        self.experts = nn.ModuleList(
            [
                ExpertFFN(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    out_dim=num_classes,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_router_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.dim() != 2:
            raise ValueError(f"SparseMoEHead 期望输入为 [B, D]，当前 shape={tuple(x.shape)}")

        weights, topk_indices, router_probs, router_logits = self.gating(x)

        batch_size = x.size(0)
        logits = torch.zeros(
            batch_size,
            self.num_classes,
            device=x.device,
            dtype=x.dtype,
        )

        # 真稀疏计算：每个 expert 只处理路由到自己的样本。
        for expert_id, expert in enumerate(self.experts):
            selected_mask = topk_indices == expert_id
            token_mask = selected_mask.any(dim=-1)

            if not token_mask.any():
                continue

            expert_input = x[token_mask]
            expert_output = expert(expert_input)

            selected_weights = weights[token_mask, expert_id]
            logits[token_mask] = logits[token_mask] + (
                expert_output * selected_weights.unsqueeze(-1)
            )

        if not return_router_info:
            return logits

        # 以下信息只用于诊断，不参与训练 loss。
        expert_one_hot = F.one_hot(
            topk_indices,
            num_classes=self.num_experts,
        ).to(dtype=torch.float32)

        expert_counts = expert_one_hot.sum(dim=(0, 1))
        sample_expert_counts = expert_one_hot.sum(dim=1)

        density = expert_counts / max(float(batch_size * self.topk), 1.0)
        density_proxy = router_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(
            density.to(router_probs.device) * density_proxy
        )

        router_info = {
            "aux_loss": aux_loss,
            "expert_counts": expert_counts.to(x.device),
            "sample_expert_counts": sample_expert_counts.to(x.device),
            "selected_experts": topk_indices,
            "topk_probs": torch.gather(router_probs, dim=1, index=topk_indices).to(
                dtype=x.dtype
            ),
            "router_probs": router_probs.to(dtype=x.dtype),
            "router_logits": router_logits.to(dtype=x.dtype),
        }

        return logits, router_info


class SparseMoEClassifier(nn.Module):
    """
    Backbone + Sparse MoE Head 分类模型。

    整体结构：
        image
          -> selected backbone
          -> backbone_adapter -> fixed feature [B, 512]
          -> SparseMoEHead
          -> logits [B, num_classes]

    为了让 backbone 对照实验中的 router / expert 参数规模保持一致，
    所有 backbone 都在进入 MoE 前统一成 512 维：
        resnet_cifar: 512 -> Identity
        vgg11: 512 -> Identity
        vit_tiny: 192 -> Linear(192, 512)

    backbone / backbone_adapter / router 都属于 non_expert 参数；
    只有 experts.<id> 属于 expert 参数。
    """

    def __init__(
        self,
        num_classes: int,
        num_experts: int = 4,
        topk: int = 2,
        in_channels: int = 3,
        image_size: int = 32,
        moe_hidden_dim: int = 512,
        renormalize_topk_probs: bool = False,
        backbone_name: str = DEFAULT_BACKBONE_NAME,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes 必须大于 0，当前值：{num_classes}")

        self.num_classes = int(num_classes)
        self.num_experts = int(num_experts)
        self.topk = int(topk)
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.moe_hidden_dim = int(moe_hidden_dim)

        self.backbone_name = str(backbone_name).lower().strip()
        self.backbone = build_backbone(
            backbone_name=self.backbone_name,
            in_channels=in_channels,
            image_size=image_size,
        )

        raw_feat_dim = int(self.backbone.feat_dim)
        if raw_feat_dim == MOE_FEATURE_DIM:
            # Keep the existing ResNet numerical path exactly unchanged.
            self.backbone_adapter = nn.Identity()
        else:
            self.backbone_adapter = nn.Linear(
                raw_feat_dim,
                MOE_FEATURE_DIM,
            )

        self.moe_head = SparseMoEHead(
            in_dim=MOE_FEATURE_DIM,
            hidden_dim=moe_hidden_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            topk=topk,
            renormalize_topk_probs=renormalize_topk_probs,
        )

        self._init_extra_weights()

    def _init_extra_weights(self) -> None:
        """
        初始化 Linear 层。

        no-BN 版本中，卷积层使用 PyTorch 默认初始化；
        这里额外对 Linear 做 Xavier 初始化，让 router 和 expert 更稳定一点。
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_router_info: bool = False,
    ) -> torch.Tensor | SparseMoEClassifierOutput:
        feat = self.backbone(x)
        feat = self.backbone_adapter(feat)

        if not return_router_info:
            logits = self.moe_head(
                feat,
                return_router_info=False,
            )
            return logits

        logits, router_info = self.moe_head(
            feat,
            return_router_info=True,
        )

        return SparseMoEClassifierOutput(
            logits=logits,
            router_info=router_info,
        )


def build_sparse_moe_classifier_from_cfg(cfg: Any) -> SparseMoEClassifier:
    """
    根据 cfg 构建 SparseMoEClassifier。

    推荐配置示例：

    model: sparse_moe_classifier
    num_experts: 4
    topk: 2

    model_cfg:
      in_channels: 3
      image_size: 32
      moe_hidden_dim: 512
      renormalize_topk_probs: false

    说明：
    - num_classes 优先从 cfg.num_classes 读取。
    - in_channels / image_size 会优先从 cfg.input_shape 推断。
    - model_cfg 里的 in_channels / image_size 可以覆盖默认值。
    """

    model_cfg = _cfg_get(cfg, "model_cfg", {})

    input_shape = _cfg_get(cfg, "input_shape", (3, 32, 32))
    if input_shape is None:
        input_shape = (3, 32, 32)

    default_in_channels = int(input_shape[0])
    default_image_size = int(input_shape[1])

    # 兼容两种写法：
    # 1. model_cfg.moe_hidden_dim
    # 2. model_cfg.hidden_dim
    # 如果都没写，默认使用原始 moefedavg.py 里的 512。
    default_moe_hidden_dim = _cfg_get(
        model_cfg,
        "hidden_dim",
        512,
    )

    return SparseMoEClassifier(
        num_classes=int(_cfg_get(cfg, "num_classes")),
        num_experts=int(_cfg_get(cfg, "num_experts", 4)),
        topk=int(_cfg_get(cfg, "topk", 2)),
        in_channels=int(
            _cfg_get(
                model_cfg,
                "in_channels",
                default_in_channels,
            )
        ),
        image_size=int(
            _cfg_get(
                model_cfg,
                "image_size",
                default_image_size,
            )
        ),
        moe_hidden_dim=int(
            _cfg_get(
                model_cfg,
                "moe_hidden_dim",
                default_moe_hidden_dim,
            )
        ),
        renormalize_topk_probs=bool(
            _cfg_get(
                model_cfg,
                "renormalize_topk_probs",
                False,
            )
        ),
        backbone_name=str(
            _cfg_get(
                model_cfg,
                "backbone",
                DEFAULT_BACKBONE_NAME,
            )
        ),
    )


# ============================================================================
# Bundled from models/build.py
# ============================================================================


ModelBuilder = Callable[[Any], nn.Module]


MODEL_BUILDERS: Dict[str, ModelBuilder] = {
    "sparse_moe_classifier": build_sparse_moe_classifier_from_cfg,
}


def build_model(cfg: Any) -> nn.Module:
    """
    根据配置创建模型。

    配置示例：
        model: sparse_moe_classifier

    当前支持：
        sparse_moe_classifier

    后续扩展其他模型时，只需要：
        1. 新增模型文件
        2. 写一个 build_xxx_from_cfg(cfg)
        3. 在 MODEL_BUILDERS 里注册
    """
    model_name = get_model_name(cfg)

    if model_name not in MODEL_BUILDERS:
        supported = ", ".join(list_supported_models())
        raise ValueError(
            f"不支持的模型名称：{model_name}。"
            f"当前支持：{supported}"
        )

    model = MODEL_BUILDERS[model_name](cfg)
    return model


def get_model_name(cfg: Any) -> str:
    """
    从配置中读取模型名称。

    支持：
        cfg.model
        cfg.get("model")
    """
    model_name = _cfg_get(cfg, "model", None)

    if model_name is None:
        raise ValueError("配置中缺少 model 字段。")

    model_name = str(model_name).lower().strip()
    if not model_name:
        raise ValueError("配置中的 model 字段不能为空。")

    return model_name


def list_supported_models() -> List[str]:
    """
    返回当前支持的模型名称列表。
    """
    return sorted(MODEL_BUILDERS.keys())


def count_parameters(
    model: nn.Module,
    trainable_only: bool = False,
) -> int:
    """
    统计模型参数量。

    参数：
        trainable_only:
            如果为 True，只统计 requires_grad=True 的参数。
            如果为 False，统计所有参数。
    """
    total = 0

    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        total += int(param.numel())

    return total


def summarize_model(model: nn.Module) -> Dict[str, int]:
    """
    返回模型参数量摘要。

    输出：
        {
            "total_params": ...,
            "trainable_params": ...
        }
    """
    return {
        "total_params": count_parameters(
            model=model,
            trainable_only=False,
        ),
        "trainable_params": count_parameters(
            model=model,
            trainable_only=True,
        ),
    }


# ============================================================================
# Bundled from models/param_groups.py
# ============================================================================


_EXPERT_ID_PATTERN = re.compile(r"(?:^|\.)experts\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class ParamGroups:
    """
    模型参数分组结果。

    all:
        state_dict 里的全部参数名和 buffer 名。

    non_expert:
        非专家参数名。
        例如 backbone / router / classifier / BatchNorm buffer 等。

    expert:
        所有 expert 参数名。

    expert_by_id:
        每个 expert 单独对应的参数名。
        例如：
            {
                0: ["moe.experts.0.fc.weight", ...],
                1: ["moe.experts.1.fc.weight", ...],
            }
    """

    all: List[str]
    non_expert: List[str]
    expert: List[str]
    expert_by_id: Dict[int, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        转成普通 dict，方便写日志或调试。
        """
        return {
            "all": list(self.all),
            "non_expert": list(self.non_expert),
            "expert": list(self.expert),
            "expert_by_id": {
                int(expert_id): list(names)
                for expert_id, names in self.expert_by_id.items()
            },
        }

    def summary(self) -> Dict[str, Any]:
        """
        返回轻量摘要，不包含完整参数名列表。
        """
        return {
            "num_all": len(self.all),
            "num_non_expert": len(self.non_expert),
            "num_expert": len(self.expert),
            "num_experts_found": len(self.expert_by_id),
            "num_params_by_expert": {
                int(expert_id): len(names)
                for expert_id, names in self.expert_by_id.items()
            },
        }


def build_param_groups(
    model: torch.nn.Module,
    expected_num_experts: int | None = None,
    strict: bool = True,
) -> ParamGroups:
    """
    根据模型 state_dict 构建参数分组。

    注意：
        这里使用 model.state_dict().keys()，
        而不是 model.named_parameters()。

    原因：
        state_dict 里不仅有可训练参数，还有 BatchNorm running_mean /
        running_var 等 buffer。联邦聚合时通常也需要处理这些浮点 buffer。
    """
    state_dict = model.state_dict()
    return build_param_groups_from_state_dict(
        state_dict=state_dict,
        expected_num_experts=expected_num_experts,
        strict=strict,
    )


def build_param_groups_from_state_dict(
    state_dict: StateDict,
    expected_num_experts: int | None = None,
    strict: bool = True,
) -> ParamGroups:
    """
    根据 state_dict 构建参数分组。

    专家参数识别规则：
        参数名中出现 experts.<id> 就认为它属于 expert <id>。

    例如：
        moe.experts.0.fc1.weight -> expert 0
        head.experts.3.bias      -> expert 3

    其他参数都归为 non_expert。
    """
    all_names = list(state_dict.keys())

    non_expert_names: List[str] = []
    expert_names: List[str] = []
    expert_by_id: Dict[int, List[str]] = {}

    for name in all_names:
        expert_id = get_expert_id_from_name(name)

        if expert_id is None:
            non_expert_names.append(name)
        else:
            expert_names.append(name)
            expert_by_id.setdefault(expert_id, []).append(name)

    groups = ParamGroups(
        all=all_names,
        non_expert=non_expert_names,
        expert=expert_names,
        expert_by_id={
            expert_id: names
            for expert_id, names in sorted(expert_by_id.items())
        },
    )

    validate_param_groups(
        groups=groups,
        state_dict=state_dict,
        expected_num_experts=expected_num_experts,
        strict=strict,
    )

    return groups


def get_expert_id_from_name(name: str) -> int | None:
    """
    从参数名中解析 expert id。

    匹配规则：
        experts.<id>

    示例：
        "moe.experts.0.fc.weight" -> 0
        "experts.3.bias"          -> 3
        "backbone.conv.weight"    -> None
    """
    match = _EXPERT_ID_PATTERN.search(name)

    if match is None:
        return None

    return int(match.group(1))


def validate_param_groups(
    groups: ParamGroups,
    state_dict: StateDict,
    expected_num_experts: int | None = None,
    strict: bool = True,
) -> None:
    """
    检查参数分组是否合法。

    检查内容：
        1. all 是否覆盖 state_dict 所有 key
        2. non_expert 和 expert 是否有重叠
        3. non_expert + expert 是否刚好覆盖 all
        4. expert_by_id 是否和 expert 一致
        5. 如果 strict=True，要求至少找到一个 expert 参数
        6. 如果传入 expected_num_experts，则检查 expert id 数量
    """
    state_names = set(state_dict.keys())
    all_names = set(groups.all)
    non_expert_names = set(groups.non_expert)
    expert_names = set(groups.expert)

    if all_names != state_names:
        missing = sorted(state_names - all_names)
        extra = sorted(all_names - state_names)
        raise ValueError(
            "ParamGroups.all 和 state_dict keys 不一致。"
            f" missing={missing[:10]}, extra={extra[:10]}"
        )

    overlap = non_expert_names & expert_names
    if overlap:
        raise ValueError(
            "non_expert 和 expert 参数组存在重叠："
            f"{sorted(overlap)[:10]}"
        )

    merged = non_expert_names | expert_names
    if merged != all_names:
        missing = sorted(all_names - merged)
        extra = sorted(merged - all_names)
        raise ValueError(
            "non_expert + expert 没有刚好覆盖 all。"
            f" missing={missing[:10]}, extra={extra[:10]}"
        )

    expert_by_id_names = set()
    for expert_id, names in groups.expert_by_id.items():
        if expert_id < 0:
            raise ValueError(f"expert_id 不能小于 0，当前值：{expert_id}")

        for name in names:
            parsed_expert_id = get_expert_id_from_name(name)
            if parsed_expert_id != expert_id:
                raise ValueError(
                    f"expert_by_id 分组错误：参数 {name} 被放到 expert {expert_id}，"
                    f"但解析结果是 {parsed_expert_id}。"
                )

            expert_by_id_names.add(name)

    if expert_by_id_names != expert_names:
        missing = sorted(expert_names - expert_by_id_names)
        extra = sorted(expert_by_id_names - expert_names)
        raise ValueError(
            "expert_by_id 和 expert 参数组不一致。"
            f" missing={missing[:10]}, extra={extra[:10]}"
        )

    if strict and len(expert_names) == 0:
        raise ValueError(
            "没有找到任何 expert 参数。"
            "请确认模型中的 expert 参数名是否包含 experts.<id>。"
        )

    if expected_num_experts is not None:
        expected_num_experts = int(expected_num_experts)

        if expected_num_experts <= 0:
            raise ValueError(
                f"expected_num_experts 必须大于 0，当前值：{expected_num_experts}"
            )

        found_expert_ids = sorted(groups.expert_by_id.keys())
        expected_expert_ids = list(range(expected_num_experts))

        if strict and found_expert_ids != expected_expert_ids:
            raise ValueError(
                "模型中找到的 expert id 和期望不一致。"
                f" found={found_expert_ids}, expected={expected_expert_ids}"
            )


def count_tensors_by_group(groups: ParamGroups) -> Dict[str, int]:
    """
    统计每个参数组包含多少个 tensor。
    """
    return {
        "all": len(groups.all),
        "non_expert": len(groups.non_expert),
        "expert": len(groups.expert),
        "expert_by_id": {
            int(expert_id): len(names)
            for expert_id, names in groups.expert_by_id.items()
        },
    }


def count_numel_by_group(
    state_dict: StateDict,
    groups: ParamGroups,
    only_floating: bool = True,
) -> Dict[str, Any]:
    """
    统计每个参数组包含多少个元素。

    参数：
        only_floating:
            如果为 True，只统计浮点 tensor。
            如果为 False，所有 tensor 都统计。
    """
    return {
        "all": _count_numel(
            state_dict=state_dict,
            names=groups.all,
            only_floating=only_floating,
        ),
        "non_expert": _count_numel(
            state_dict=state_dict,
            names=groups.non_expert,
            only_floating=only_floating,
        ),
        "expert": _count_numel(
            state_dict=state_dict,
            names=groups.expert,
            only_floating=only_floating,
        ),
        "expert_by_id": {
            int(expert_id): _count_numel(
                state_dict=state_dict,
                names=names,
                only_floating=only_floating,
            )
            for expert_id, names in groups.expert_by_id.items()
        },
    }


def summarize_param_groups(
    state_dict: StateDict,
    groups: ParamGroups,
) -> Dict[str, Any]:
    """
    汇总参数分组信息，方便打印日志。

    输出包括：
        1. tensor 数量
        2. 浮点元素数量
        3. 所有元素数量
    """
    return {
        "tensor_counts": count_tensors_by_group(groups),
        "floating_numel": count_numel_by_group(
            state_dict=state_dict,
            groups=groups,
            only_floating=True,
        ),
        "all_numel": count_numel_by_group(
            state_dict=state_dict,
            groups=groups,
            only_floating=False,
        ),
    }


def _count_numel(
    state_dict: StateDict,
    names: Iterable[str],
    only_floating: bool,
) -> int:
    """
    统计指定参数名对应 tensor 的元素数量。
    """
    total = 0

    for name in names:
        if name not in state_dict:
            raise KeyError(f"state_dict 中不存在参数：{name}")

        tensor = state_dict[name]

        if only_floating and not torch.is_floating_point(tensor):
            continue

        total += int(tensor.numel())

    return total


# ============================================================================
# Bundled from fl/types.py
# ============================================================================


TensorDict = Dict[str, torch.Tensor]


@dataclass
class ClientUpdate:
    """
    客户端每一轮训练后上传给服务端的结果。

    这个结构是 client 和 server 之间的统一接口。

    后面无论是 FedAvg、ExpertFedAvg、方法插件、history filter、Bayes，
    都尽量往这个结构里扩展，而不是让 client.py 和 server.py 互相强耦合。
    """

    # 客户端编号
    client_id: int

    # 当前联邦训练轮数
    round_id: int

    # 当前客户端本地训练样本数
    num_samples: int

    # 本地模型相对全局模型的参数变化量
    # 公式：
    # model_delta = local_model - global_model
    model_delta: TensorDict

    # 客户端本地训练指标
    # 例如：
    # train_loss
    # train_acc
    metrics: Dict[str, float] = field(default_factory=dict)

    # 预留扩展字段
    # 后面可以放：
    # expert_usage
    # method-specific evidence / statistics
    # method_evidence
    # method_evidence_summary
    # sgld_mean
    # sgld_var
    # router_stats
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """
        返回适合写日志的轻量摘要。

        注意：
        不包含 model_delta，因为 tensor 太大，不适合直接写入日志。
        """
        return {
            "client_id": int(self.client_id),
            "round_id": int(self.round_id),
            "num_samples": int(self.num_samples),
            "metrics": dict(self.metrics),
            "extra_keys": sorted(self.extra.keys()),
        }


@dataclass
class AggregationResult:
    """
    聚合器返回给 server 的结果。

    所有聚合方法都应该返回这个结构，使 server 不依赖具体专家聚合算法。
    """

    # 聚合后的新全局模型参数
    new_state_dict: TensorDict

    # 每个客户端的最终聚合权重
    # 例如：
    # {0: 0.1, 1: 0.2, 2: 0.7}
    weights: Dict[int, float]

    # 聚合诊断信息
    # 例如：
    # method
    # num_clients
    # total_samples
    # param_group
    # param_count
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """
        返回适合写日志的轻量摘要。

        注意：
        不包含 new_state_dict，因为模型参数太大。
        """
        return {
            "weights": dict(self.weights),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class RoundResult:
    """
    每一轮联邦训练结束后的结果摘要。

    这个结构主要用于日志、results.csv、后续实验分析。
    """

    # 当前轮数
    round_id: int

    # 本轮参与训练的客户端编号
    selected_clients: List[int]

    # 测试集 loss
    test_loss: float

    # 测试集准确率
    test_acc: float

    # 当前历史最佳准确率
    best_acc: float

    # 本轮客户端训练指标摘要
    client_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # 本轮聚合信息摘要
    aggregation_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        转成普通 dict，方便写 json / csv。
        """
        return {
            "round_id": int(self.round_id),
            "selected_clients": list(self.selected_clients),
            "test_loss": float(self.test_loss),
            "test_acc": float(self.test_acc),
            "best_acc": float(self.best_acc),
            "client_metrics": self.client_metrics,
            "aggregation_info": self.aggregation_info,
        }


@dataclass
class TrainState:
    """
    服务端训练状态。

    用于记录当前轮次和历史最佳测试指标，并写入最终训练摘要。
    """

    # 当前已经完成的轮数
    round_id: int = 0

    # 当前历史最佳准确率
    best_acc: float = 0.0

    # 最佳模型出现在哪一轮
    best_round: int = 0

    # 额外状态
    # 后面可以放：
    # history filter state
    # bayes prior state
    # method-specific running state
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        转成普通 dict，方便写入训练摘要。
        """
        return {
            "round_id": int(self.round_id),
            "best_acc": float(self.best_acc),
            "best_round": int(self.best_round),
            "extra": self.extra,
        }


def average_client_metric(
    client_updates: List[ClientUpdate],
    metric_name: str,
    weighted: bool = True,
    default: Optional[float] = None,
) -> Optional[float]:
    """
    统计客户端指标平均值。

    参数：
        client_updates: 本轮客户端上传结果。
        metric_name: 指标名，例如 train_loss / train_acc。
        weighted: 是否按客户端样本数加权。
        default: 如果没有任何客户端包含该指标，则返回 default。

    注意：
        weighted=True 时，返回的是按 num_samples 加权平均，
        不是客户端等权平均。
    """
    values = []

    for update in client_updates:
        if metric_name not in update.metrics:
            continue

        value = float(update.metrics[metric_name])
        weight = int(update.num_samples) if weighted else 1

        values.append((value, weight))

    if len(values) == 0:
        return default

    total_weight = sum(weight for _, weight in values)

    if total_weight <= 0:
        return default

    return sum(value * weight for value, weight in values) / total_weight


def collect_client_metrics(
    client_updates: List[ClientUpdate],
) -> Dict[int, Dict[str, float]]:
    """
    把客户端训练指标整理成 dict。

    输出格式：
        {
            client_id: {
                "train_loss": ...,
                "train_acc": ...,
                "num_batches": ...
            }
        }

    注意：
        这里只收集 metrics，不包含 num_samples / expert_usage。
        如果需要更完整的诊断信息，请使用 collect_client_diagnostics()。
    """
    return {
        int(update.client_id): dict(update.metrics)
        for update in client_updates
    }


# ============================================================================
# Bundled from fl/client.py
# ============================================================================


@dataclass(frozen=True)
class ClientTrainStats:
    """
    客户端本地训练统计结果。

    avg_loss:
        本地训练平均 loss。

    train_acc:
        本地训练准确率，百分比形式。

    num_samples:
        本地训练样本数。

    num_batches:
        本地训练 batch 数。
    """

    avg_loss: float
    train_acc: float
    num_samples: int
    num_batches: int

    def to_metrics(self) -> Dict[str, float]:
        """
        转成 ClientUpdate.metrics 使用的普通 dict。
        """
        return {
            "train_loss": float(self.avg_loss),
            "train_acc": float(self.train_acc),
            "num_batches": float(self.num_batches),
        }


ExpertEvidenceCollector = Callable[..., Mapping[str, Any]]
MethodClientDiagnosticsBuilder = Callable[[ClientUpdate], Mapping[str, Any]]


class FLClient:
    """
    联邦学习客户端。

    职责：
        1. 接收 server 下发的 global_model
        2. 在自己的 train_loader 上本地训练
        3. 在自己的 evidence_loader 上统计 方法级 evidence
        4. 计算 local_model 相对 global_model 的参数变化量
        5. 返回 ClientUpdate

    不负责：
        1. 选择客户端
        2. 聚合参数
        3. 测试集评估
    """

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        cfg: Any,
        device: torch.device | str,
        evidence_loader: Optional[DataLoader] = None,
        expert_evidence_collector: Optional[ExpertEvidenceCollector] = None,
    ) -> None:
        self.client_id = int(client_id)
        self.train_loader = train_loader

        # 方法级 evidence 专用 loader。
        # 正常情况下由 data/loaders.py 基于 train_evidence_dataset 构建，
        # 其 transform 已经关闭 RandomCrop / RandomHorizontalFlip。
        #
        # 如果旧代码路径没有传入 evidence_loader，则回退到 train_loader，
        # 这样可以兼容旧配置，但推荐新流程始终显式传入 evidence_loader。
        self.evidence_loader = (
            evidence_loader
            if evidence_loader is not None
            else train_loader
        )

        self.cfg = cfg
        self.device = torch.device(device)
        self.expert_evidence_collector = expert_evidence_collector

        if len(self.train_loader.dataset) <= 0:
            raise ValueError(f"客户端 {self.client_id} 的数据集为空。")

        if len(self.evidence_loader.dataset) <= 0:
            raise ValueError(f"客户端 {self.client_id} 的 evidence 数据集为空。")

    @property
    def num_samples(self) -> int:
        """
        当前客户端本地训练样本数。

        注意：
            聚合时的样本数仍然按训练集 train_loader 统计。
            evidence_loader 只是 方法级 统计用，不改变客户端样本权重定义。
        """
        return int(len(self.train_loader.dataset))

    def train(
        self,
        global_model: nn.Module,
        round_id: int,
    ) -> ClientUpdate:
        """
        执行本地训练，并返回客户端更新。

        参数：
            global_model:
                server 当前轮下发的全局模型。

            round_id:
                当前联邦训练轮数。

        返回：
            ClientUpdate:
                包含 model_delta、num_samples、metrics、extra 等信息。
        """
        global_state_cpu = state_dict_to(
            global_model.state_dict(),
            device="cpu",
        )

        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)
        local_model.train()

        criterion = build_criterion(self.cfg)
        optimizer = build_optimizer(
            model=local_model,
            cfg=self.cfg,
        )

        local_epochs = int(_cfg_get(self.cfg, "local_epochs", 5))
        grad_clip = _get_grad_clip(self.cfg)

        stats = train_local_model(
            model=local_model,
            train_loader=self.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            local_epochs=local_epochs,
            grad_clip=grad_clip,
        )

        # ------------------------------------------------------------
        # 可选：采集当前客户端本地模型的 expert usage。
        #
        # 统计含义：
        #     本地训练结束后，local_model 在该客户端自己的 train_loader 上，
        #     每个 expert 被 top-k router 选中了多少次。
        #
        # 注意：
        #     这里仍然使用 train_loader，保持原有日志诊断语义不变。
        #     方法级 evidence 统计才使用 evidence_loader。
        #
        #     topk=2 时，一个样本会贡献 2 次 expert 激活。
        #     所以 expert_counts 的总和通常约等于 num_samples * topk。
        # ------------------------------------------------------------
        expert_usage = None
        if bool(_cfg_get(self.cfg, "logging.collect_expert_usage", True)):
            expert_usage = collect_expert_usage(
                model=local_model,
                train_loader=self.train_loader,
                device=self.device,
                cfg=self.cfg,
            )

        method_extra: Dict[str, Any] = {}
        if self.expert_evidence_collector is not None:
            collected_extra = self.expert_evidence_collector(
                model=local_model,
                evidence_loader=self.evidence_loader,
                criterion=criterion,
                device=self.device,
                cfg=self.cfg,
            )
            if collected_extra is None:
                collected_extra = {}
            if not isinstance(collected_extra, Mapping):
                raise TypeError("expert_evidence_collector 必须返回 Mapping 或 None。")
            method_extra.update(dict(collected_extra))

        local_state_cpu = state_dict_to(
            local_model.state_dict(),
            device="cpu",
        )

        model_delta = subtract_state_dict(
            local_state=local_state_cpu,
            global_state=global_state_cpu,
            strict=True,
        )

        check_finite_state_dict(model_delta)

        update = ClientUpdate(
            client_id=self.client_id,
            round_id=int(round_id),
            num_samples=self.num_samples,
            model_delta=model_delta,
            metrics=stats.to_metrics(),
            extra={
                "optimizer": get_optimizer_type(self.cfg),
                "local_epochs": int(local_epochs),
                "grad_clip": float(grad_clip) if grad_clip is not None else None,
                "expert_usage": expert_usage,
                **method_extra,
            },
        )

        del local_model
        del optimizer
        del criterion

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return update


def train_local_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    local_epochs: int,
    grad_clip: Optional[float] = None,
) -> ClientTrainStats:
    """
    训练一个客户端本地模型。

    这里的训练 loss 只有 CrossEntropyLoss。

    明确不加入：
        1. aux_loss
        2. router balance
        3. entropy regularization
        4. expert diversity
        5. router consistency
        6. proximal loss
    """
    if local_epochs <= 0:
        raise ValueError(f"local_epochs 必须大于 0，当前值：{local_epochs}")

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = 0

    for _ in range(local_epochs):
        for batch in train_loader:
            images, targets = unpack_batch(batch)

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            logits = extract_logits(outputs)

            loss = criterion(logits, targets)
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(grad_clip),
                )

            optimizer.step()

            batch_size = int(targets.size(0))

            total_loss += float(loss.item()) * batch_size
            total_correct += int(logits.argmax(dim=1).eq(targets).sum().item())
            total_samples += batch_size
            total_batches += 1

    if total_samples <= 0:
        raise ValueError("客户端本地训练没有处理任何样本。")

    avg_loss = total_loss / total_samples
    train_acc = 100.0 * total_correct / total_samples

    return ClientTrainStats(
        avg_loss=avg_loss,
        train_acc=train_acc,
        num_samples=total_samples,
        num_batches=total_batches,
    )


@torch.inference_mode()
def collect_expert_usage(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    cfg: Any,
) -> Dict[str, Any]:
    """
    统计一个客户端本地模型的 expert 使用情况。

    统计时机：
        本地训练结束后。

    统计数据：
        当前客户端自己的 train_loader。

    输出字段：
        num_samples:
            实际用于统计的样本数。

        num_batches:
            实际用于统计的 batch 数。

        num_experts:
            expert 总数。

        topk:
            每个样本激活的 expert 数。

        total_activations:
            expert 总激活次数。
            通常约等于 num_samples * topk。

        expert_counts:
            每个 expert 被选中的次数。

        expert_fraction:
            每个 expert 被选中的比例。

        active_experts:
            至少被选中过一次的 expert 数。

        dead_experts:
            本次统计中完全没有被选中的 expert id。

        supported:
            当前模型是否支持 return_router_info=True。

    注意：
        这个函数只做前向统计，不更新模型参数。
    """
    max_batches = int(_cfg_get(cfg, "logging.expert_usage_max_batches", 0))
    num_experts = int(_cfg_get(cfg, "num_experts", 0))
    topk = int(_cfg_get(cfg, "topk", 2))

    if num_experts <= 0:
        return {
            "supported": False,
            "reason": "num_experts <= 0",
        }

    old_training = bool(model.training)
    model.eval()

    expert_counts = torch.zeros(
        num_experts,
        dtype=torch.float64,
        device="cpu",
    )

    total_samples = 0
    total_batches = 0

    supported = True
    unsupported_reason = ""

    try:
        for batch_index, batch in enumerate(train_loader):
            if max_batches > 0 and batch_index >= max_batches:
                break

            images, targets = unpack_batch(batch)
            images = images.to(device, non_blocking=True)

            try:
                outputs = model(
                    images,
                    return_router_info=True,
                )
            except TypeError as exc:
                supported = False
                unsupported_reason = (
                    "model does not support return_router_info=True: "
                    f"{exc}"
                )
                break

            router_info = extract_router_info(outputs)
            if router_info is None:
                supported = False
                unsupported_reason = "model output does not contain router_info"
                break

            batch_expert_counts = router_info.get("expert_counts", None)
            if batch_expert_counts is None:
                supported = False
                unsupported_reason = "router_info does not contain expert_counts"
                break

            batch_expert_counts = batch_expert_counts.detach().to(
                device="cpu",
                dtype=torch.float64,
            )

            if batch_expert_counts.numel() != num_experts:
                supported = False
                unsupported_reason = (
                    "expert_counts length mismatch: "
                    f"expected={num_experts}, actual={batch_expert_counts.numel()}"
                )
                break

            expert_counts += batch_expert_counts.reshape(-1)

            total_samples += int(images.size(0))
            total_batches += 1

    finally:
        if old_training:
            model.train()
        else:
            model.eval()

    if not supported:
        return {
            "supported": False,
            "reason": unsupported_reason,
        }

    total_activations = float(expert_counts.sum().item())

    if total_activations > 0:
        expert_fraction_tensor = expert_counts / total_activations
    else:
        expert_fraction_tensor = torch.zeros_like(expert_counts)

    expert_counts_dict = {
        int(expert_id): int(expert_counts[expert_id].item())
        for expert_id in range(num_experts)
    }

    expert_fraction_dict = {
        int(expert_id): float(expert_fraction_tensor[expert_id].item())
        for expert_id in range(num_experts)
    }

    dead_experts = [
        int(expert_id)
        for expert_id, count in expert_counts_dict.items()
        if count <= 0
    ]

    active_experts = int(num_experts - len(dead_experts))

    return {
        "supported": True,
        "num_samples": int(total_samples),
        "num_batches": int(total_batches),
        "max_batches": int(max_batches),
        "num_experts": int(num_experts),
        "topk": int(topk),
        "total_activations": int(total_activations),
        "expert_counts": expert_counts_dict,
        "expert_fraction": expert_fraction_dict,
        "active_experts": int(active_experts),
        "dead_experts": dead_experts,
    }


def extract_router_info(outputs: Any) -> Optional[Mapping[str, Any]]:
    """
    从模型输出中提取 router_info。

    兼容几种常见输出：
        1. dataclass / object: outputs.router_info
        2. dict: outputs["router_info"]
        3. tuple/list: outputs[1] 是 router_info

    当前 resnet_sparse_moe_head 在 return_router_info=True 时，
    返回对象里包含 .router_info。
    """
    if hasattr(outputs, "router_info"):
        router_info = outputs.router_info
        if isinstance(router_info, Mapping):
            return router_info
        return None

    if isinstance(outputs, Mapping):
        router_info = outputs.get("router_info", None)
        if isinstance(router_info, Mapping):
            return router_info
        return None

    if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
        router_info = outputs[1]
        if isinstance(router_info, Mapping):
            return router_info
        return None

    return None


def build_criterion(cfg: Any) -> nn.Module:
    """
    构建本地训练 loss 函数。

    第一版只使用 CrossEntropyLoss。
    """
    label_smoothing = float(_cfg_get(cfg, "label_smooth", 0.0))

    return nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
    )


def build_optimizer(
    model: nn.Module,
    cfg: Any,
) -> optim.Optimizer:
    """
    根据 cfg.optimizer 构建优化器。

    当前支持：
        sgd
        adam
        adamw
    """
    optimizer_type = get_optimizer_type(cfg)
    optimizer_cfg = _cfg_get(cfg, "optimizer", {})

    lr = float(_cfg_get(optimizer_cfg, "lr", 0.01))
    weight_decay = float(_cfg_get(optimizer_cfg, "weight_decay", 0.0005))

    params = [
        param
        for param in model.parameters()
        if param.requires_grad
    ]

    if len(params) == 0:
        raise ValueError("模型没有可训练参数。")

    if optimizer_type == "sgd":
        momentum = float(_cfg_get(optimizer_cfg, "momentum", 0.9))
        nesterov = bool(_cfg_get(optimizer_cfg, "nesterov", False))

        return optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    if optimizer_type == "adam":
        betas = _cfg_get(optimizer_cfg, "betas", (0.9, 0.999))
        eps = float(_cfg_get(optimizer_cfg, "eps", 1e-8))

        return optim.Adam(
            params,
            lr=lr,
            betas=tuple(betas),
            eps=eps,
            weight_decay=weight_decay,
        )

    if optimizer_type == "adamw":
        betas = _cfg_get(optimizer_cfg, "betas", (0.9, 0.999))
        eps = float(_cfg_get(optimizer_cfg, "eps", 1e-8))

        return optim.AdamW(
            params,
            lr=lr,
            betas=tuple(betas),
            eps=eps,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"不支持的优化器类型：{optimizer_type}。"
        "当前支持：sgd, adam, adamw"
    )


def get_optimizer_type(cfg: Any) -> str:
    """
    从配置中读取优化器类型。
    """
    optimizer_cfg = _cfg_get(cfg, "optimizer", {})
    optimizer_type = _cfg_get(optimizer_cfg, "type", "sgd")

    return str(optimizer_type).lower().strip()


def build_clients(
    cfg: Any,
    client_loaders: Sequence[DataLoader],
    device: torch.device | str,
    client_evidence_loaders: Optional[Sequence[DataLoader]] = None,
    expert_evidence_collector: Optional[ExpertEvidenceCollector] = None,
) -> List[FLClient]:
    """
    根据客户端 DataLoader 列表创建 FLClient 列表。

    参数：
        cfg:
            全局配置。

        client_loaders:
            每个客户端对应一个训练 DataLoader。

        device:
            本地训练使用的设备。

        client_evidence_loaders:
            每个客户端对应一个 方法级 evidence DataLoader。
            如果为 None，则每个客户端回退使用自己的 train_loader。
    """
    if client_evidence_loaders is not None and len(client_evidence_loaders) != len(client_loaders):
        raise ValueError(
            "client_evidence_loaders 数量必须和 client_loaders 一致。"
            f"当前 client_loaders={len(client_loaders)}, "
            f"client_evidence_loaders={len(client_evidence_loaders)}。"
        )

    clients: List[FLClient] = []

    for client_id, train_loader in enumerate(client_loaders):
        if client_evidence_loaders is None:
            evidence_loader = None
        else:
            evidence_loader = client_evidence_loaders[client_id]

        clients.append(
            FLClient(
                client_id=client_id,
                train_loader=train_loader,
                evidence_loader=evidence_loader,
                cfg=cfg,
                device=device,
                expert_evidence_collector=expert_evidence_collector,
            )
        )

    return clients


def select_clients(
    clients: Sequence[FLClient],
    frac: float,
    round_id: int,
    seed: int,
) -> List[FLClient]:
    """
    按比例选择本轮参与训练的客户端。

    选择逻辑：
        每一轮使用 seed + round_id 生成随机数。
        这样同一个 seed 下实验可复现。
    """
    if len(clients) == 0:
        raise ValueError("clients 不能为空。")

    if frac <= 0:
        raise ValueError(f"frac 必须大于 0，当前值：{frac}")

    num_clients = len(clients)
    num_selected = max(1, int(num_clients * float(frac)))
    num_selected = min(num_selected, num_clients)

    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(round_id))

    perm = torch.randperm(
        num_clients,
        generator=generator,
    ).tolist()

    selected_indices = perm[:num_selected]

    return [
        clients[index]
        for index in selected_indices
    ]


def train_selected_clients(
    clients: Sequence[FLClient],
    global_model: nn.Module,
    round_id: int,
) -> List[ClientUpdate]:
    """
    训练本轮选中的客户端。

    server.py 后面可以直接调用这个函数。
    """
    updates: List[ClientUpdate] = []

    for client in clients:
        update = client.train(
            global_model=global_model,
            round_id=round_id,
        )
        updates.append(update)

    return updates


def _get_grad_clip(cfg: Any) -> Optional[float]:
    """
    读取梯度裁剪配置。

    支持两种写法：
        optimizer:
            grad_clip: 5.0

    或者：
        grad_clip: 5.0

    如果没有配置，则返回 None。
    """
    optimizer_cfg = _cfg_get(cfg, "optimizer", {})

    value = _cfg_get(
        optimizer_cfg,
        "grad_clip",
        None,
    )

    if value is None:
        value = _cfg_get(
            cfg,
            "grad_clip",
            None,
        )

    if value is None:
        return None

    value = float(value)

    if value <= 0:
        return None

    return value


# ============================================================================
# Bundled from aggregation/base.py
# ============================================================================


@dataclass(frozen=True)
class MethodContext:
    """
    Generic server-side runtime context exposed to expert aggregation plugins.

    base.py owns only common resources and lifecycle. A method plugin may read
    the resources it needs without introducing method-name branches into base.py.
    """

    cfg: Any
    device: torch.device
    dataset_name: str
    server_evidence_loader: Optional[DataLoader] = None
    model_builder: Optional[Callable[[Any], nn.Module]] = None
    extra: Mapping[str, Any] = field(default_factory=dict)


class Aggregator(ABC):
    """
    聚合器基类。

    每个专家聚合方法在自己的启动文件中继承这个类并实现接口。
    base.py 不维护专家聚合方法白名单。

    这个类只规定统一接口，不绑定具体聚合算法。
    """

    def __init__(
        self,
        cfg: Any,
        param_group_name: str,
    ) -> None:
        """
        初始化聚合器。

        参数：
            cfg:
                全局配置对象。

            param_group_name:
                当前聚合器负责的参数组名称。
                例如：
                    non_expert
                    expert
        """
        self.cfg = cfg
        self.param_group_name = param_group_name
        self.method_context: Optional[MethodContext] = None

    def set_method_context(
        self,
        method_context: Optional[MethodContext],
    ) -> None:
        """Inject common server-side resources after the plugin is constructed."""
        if method_context is not None and not isinstance(method_context, MethodContext):
            raise TypeError(
                "method_context 必须是 MethodContext 或 None，"
                f"当前类型：{type(method_context).__name__}"
            )
        self.method_context = method_context

    @property
    @abstractmethod
    def method_name(self) -> str:
        """
        当前聚合方法名称。

        子类返回自己启动文件对应的方法名称。
        """
        raise NotImplementedError

    @abstractmethod
    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """
        计算每个客户端的聚合权重。

        子类只需要实现这个函数。

        例如 uniform 专家聚合可以为每个客户端返回相同权重。
        """
        raise NotImplementedError

    def aggregate(
        self,
        global_state: Mapping[str, torch.Tensor],
        client_updates: Sequence[ClientUpdate],
        param_names: Optional[Iterable[str]] = None,
        base_state: Optional[Mapping[str, torch.Tensor]] = None,
        strict: bool = True,
    ) -> AggregationResult:
        """
        执行参数聚合。

        这是所有普通加权 delta 聚合方法的公共流程：

            1. 检查客户端更新
            2. 计算客户端权重
            3. 归一化客户端权重
            4. 对指定参数组执行加权 delta 聚合
            5. 返回 AggregationResult

        参数：
            global_state:
                本轮聚合前的全局模型参数。

            client_updates:
                本轮参与训练的客户端更新。

            param_names:
                当前聚合器负责聚合的参数名。
                例如：
                    非专家参数名列表
                    专家参数名列表

            base_state:
                聚合结果写入的基础 state_dict。
                如果为 None，则基于 global_state 生成新 state_dict。
                如果先聚合 non_expert，再聚合 expert，可以把上一步结果传进来。

            strict:
                如果为 True，缺少参数或权重时直接报错。
        """
        self._validate_client_updates(client_updates)

        raw_weights = self.compute_weights(client_updates)
        weights = normalize_weights(raw_weights)

        new_state_dict = apply_weighted_delta(
            global_state=global_state,
            client_updates=client_updates,
            weights=weights,
            param_names=param_names,
            base_state=base_state,
            strict=strict,
        )

        check_finite_state_dict(
            state_dict=new_state_dict,
            param_names=param_names,
        )

        diagnostics = self.build_diagnostics(
            client_updates=client_updates,
            weights=weights,
            param_names=param_names,
        )

        return AggregationResult(
            new_state_dict=new_state_dict,
            weights=weights,
            diagnostics=diagnostics,
        )

    def build_diagnostics(
        self,
        client_updates: Sequence[ClientUpdate],
        weights: Mapping[int, float],
        param_names: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """
        构建聚合诊断信息。

        这些信息后面会写入日志，方便确认：
            1. 当前用的是什么聚合方法
            2. 聚合的是 expert 还是 non_expert
            3. 有多少客户端参与
            4. 每个客户端权重是多少
        """
        param_count = None

        if param_names is not None:
            param_count = len(list(param_names))

        return {
            "method": self.method_name,
            "param_group": self.param_group_name,
            "num_clients": len(client_updates),
            "param_count": param_count,
            "weights": {
                int(client_id): float(weight)
                for client_id, weight in weights.items()
            },
        }

    def _validate_client_updates(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> None:
        """
        检查客户端更新是否合法。
        """
        if len(client_updates) == 0:
            raise ValueError("client_updates 不能为空。")

        seen_client_ids = set()

        for update in client_updates:
            if update.client_id in seen_client_ids:
                raise ValueError(f"重复的 client_id：{update.client_id}")

            seen_client_ids.add(update.client_id)

            if update.num_samples <= 0:
                raise ValueError(
                    f"客户端 {update.client_id} 的 num_samples 必须大于 0，"
                    f"当前值：{update.num_samples}"
                )

            if len(update.model_delta) == 0:
                raise ValueError(
                    f"客户端 {update.client_id} 的 model_delta 为空。"
                )


def build_uniform_weights(
    client_updates: Sequence[ClientUpdate],
) -> Dict[int, float]:
    """
    构建均匀权重。

    每个客户端权重相同。
    """
    if len(client_updates) == 0:
        raise ValueError("client_updates 不能为空。")

    weight = 1.0 / len(client_updates)

    return {
        int(update.client_id): weight
        for update in client_updates
    }


def build_sample_weights(
    client_updates: Sequence[ClientUpdate],
) -> Dict[int, float]:
    """
    构建按样本数加权的权重。

    注意：
        这里返回的是未必严格归一化前的权重。
        Aggregator.aggregate() 里会统一调用 normalize_weights。
    """
    if len(client_updates) == 0:
        raise ValueError("client_updates 不能为空。")

    weights: Dict[int, float] = {}

    for update in client_updates:
        if update.num_samples <= 0:
            raise ValueError(
                f"客户端 {update.client_id} 的 num_samples 必须大于 0，"
                f"当前值：{update.num_samples}"
            )

        weights[int(update.client_id)] = float(update.num_samples)

    return weights


# Fixed non-expert aggregation and expert extension interface.
@dataclass
class AggregatorBundle:
    """Fixed non-expert aggregator plus an injected expert aggregator."""

    non_expert: Aggregator
    expert: Aggregator


class FixedNonExpertUniformAggregator(Aggregator):
    """
    固定用于 non_expert 参数的直接平均聚合器。

    权重规则：
        每个参与客户端权重相同。

    公式：
        w_i = 1 / K

    其中：
        K 是本轮参与聚合的客户端数量。

    聚合公式：
        theta_new = theta_global + sum_i w_i * delta_i
    """

    @property
    def method_name(self) -> str:
        """返回当前聚合方法名称。"""
        return "uniform"

    def compute_weights(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, float]:
        """计算所有有效客户端的直接平均权重。"""
        return build_uniform_weights(client_updates)


ExpertAggregatorBuilder = Callable[[Any], Aggregator]


def build_aggregators(
    cfg: Any,
    expert_aggregator_builder: ExpertAggregatorBuilder,
    method_context: Optional[MethodContext] = None,
) -> AggregatorBundle:
    """Build fixed uniform non-expert aggregation and injected expert logic."""
    non_expert = FixedNonExpertUniformAggregator(
        cfg=cfg,
        param_group_name="non_expert",
    )
    expert = expert_aggregator_builder(cfg)

    if not isinstance(expert, Aggregator):
        raise TypeError(
            "expert_aggregator_builder must return an Aggregator, "
            f"got {type(expert).__name__}."
        )
    if expert.param_group_name != "expert":
        raise ValueError(
            "The injected aggregator must use param_group_name='expert'."
        )

    expert.set_method_context(method_context)

    return AggregatorBundle(non_expert=non_expert, expert=expert)


# ============================================================================
# Bundled from fl/server.py
# ============================================================================


@dataclass
class ServerTrainResult:
    """
    服务端完整训练结果。

    round_results:
        每一轮训练后的结果摘要。

    train_state:
        训练结束后的服务端状态。

    best_acc:
        历史最佳测试准确率。

    best_round:
        历史最佳准确率对应的轮数。
    """

    round_results: List[RoundResult]
    train_state: TrainState
    best_acc: float
    best_round: int

    def to_dict(self) -> Dict[str, Any]:
        """
        转成普通 dict，方便后续保存日志。
        """
        return {
            "best_acc": float(self.best_acc),
            "best_round": int(self.best_round),
            "round_results": [
                item.to_dict()
                for item in self.round_results
            ],
            "train_state": self.train_state.to_dict(),
        }


class FLServer:
    """
    联邦学习服务端。

    职责：
        1. 持有全局模型
        2. 选择每轮参与训练的客户端
        3. 收集客户端更新
        4. 分别聚合 non_expert 参数和 expert 参数
        5. 在服务器测试集上评估全局模型

    不负责：
        1. 数据集加载
        2. 数据划分
        3. 具体客户端本地训练细节
        4. 具体聚合权重算法细节
    """

    def __init__(
        self,
        cfg: Any,
        client_loaders: Sequence[DataLoader],
        test_loader: DataLoader,
        device: torch.device | str,
        expert_aggregator_builder: ExpertAggregatorBuilder,
        client_evidence_loaders: Optional[Sequence[DataLoader]] = None,
        expert_evidence_collector: Optional[ExpertEvidenceCollector] = None,
        method_client_diagnostics_builder: Optional[MethodClientDiagnosticsBuilder] = None,
        method_context: Optional[MethodContext] = None,
        global_model: Optional[nn.Module] = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)

        # 全局模型默认常驻 CPU。
        # 客户端训练时会 deepcopy 后移动到 GPU。
        # 服务端评估时临时移动到 GPU，评估后再移回 CPU。
        if global_model is None:
            self.global_model = build_model(cfg)
        else:
            self.global_model = global_model

        self.global_model.to("cpu")

        self.clients = build_clients(
            cfg=cfg,
            client_loaders=client_loaders,
            # 方法级 evidence 专用 loader。
            # 本地训练仍然使用 client_loaders；
            # 统计 方法插件 时由客户端使用 client_evidence_loaders。
            client_evidence_loaders=client_evidence_loaders,
            device=self.device,
            expert_evidence_collector=expert_evidence_collector,
        )

        self.test_loader = test_loader
        self.method_client_diagnostics_builder = method_client_diagnostics_builder
        self.method_context = method_context

        self.aggregators = build_aggregators(
            cfg=cfg,
            expert_aggregator_builder=expert_aggregator_builder,
            method_context=method_context,
        )

        self.param_groups = build_param_groups(
            model=self.global_model,
            expected_num_experts=int(_cfg_get(cfg, "num_experts", 0)),
            strict=True,
        )

        self.train_state = TrainState(
            round_id=0,
            best_acc=0.0,
            best_round=0,
            extra={},
        )
        self.round_results: List[RoundResult] = []

        self._validate_server_state()

    def train(self) -> ServerTrainResult:
        """
        执行完整 FL 训练流程。

        每一轮流程：
            1. 选择客户端
            2. 客户端本地训练
            3. 聚合 non_expert 参数
            4. 聚合 expert 参数
            5. 更新全局模型
            6. 在服务器测试集评估
            7. 记录 RoundResult
        """
        rounds = int(_cfg_get(self.cfg, "rounds", 50))
        frac = float(_cfg_get(self.cfg, "frac", 1.0))
        seed = int(_cfg_get(self.cfg, "seed", 0))

        logging_cfg = _cfg_get(self.cfg, "logging", {})
        log_every = int(_cfg_get(logging_cfg, "log_every", 1))

        # 控制台进度条开关。
        # progress_bar: 是否启用 tqdm。
        # progress_in_non_tty: 是否允许在 nohup / 重定向等非交互终端里显示进度条。
        progress_bar_enabled = _cfg_get_bool(
            logging_cfg,
            "progress_bar",
            True,
        )
        progress_in_non_tty = _cfg_get_bool(
            logging_cfg,
            "progress_in_non_tty",
            False,
        )

        # 控制台只打印短摘要，方便实时观察。
        console_round_summary = _cfg_get_bool(
            logging_cfg,
            "console_round_summary",
            True,
        )

        # train.log 写更详细的结构化日志，但不写进度条。
        file_round_detail = _cfg_get_bool(
            logging_cfg,
            "file_round_detail",
            True,
        )

        if rounds <= 0:
            raise ValueError(f"rounds 必须大于 0，当前值：{rounds}")

        self.print_startup_summary()

        # 预估整个实验的客户端训练步数。
        # 进度条单位：完成一个客户端本地训练。
        total_client_steps = sum(
            len(
                select_clients(
                    clients=self.clients,
                    frac=frac,
                    round_id=progress_round_id,
                    seed=seed,
                )
            )
            for progress_round_id in range(1, rounds + 1)
        )

        # 进度条只写到 Python 原始 stderr。
        # 这样即使 train.py 用 tee_output_to_file() 捕获 stdout / stderr，
        # tqdm 进度条也尽量不会进入 train.log。
        #
        # 注意：
        # utils/logging.py 里也建议对 stderr 做 tqdm 过滤，
        # 这里和 TeeStream 过滤是双保险。
        progress_file = getattr(sys, "__stderr__", sys.stderr)
        progress_is_tty = bool(
            getattr(progress_file, "isatty", lambda: False)()
        )
        progress_enabled = bool(progress_bar_enabled) and (
            progress_is_tty or bool(progress_in_non_tty)
        )

        progress_bar = tqdm(
            total=total_client_steps,
            desc="Training",
            dynamic_ncols=True,
            leave=True,
            file=progress_file,
            disable=not progress_enabled,
            mininterval=0.5,
        )

        try:
            for round_id in range(1, rounds + 1):
                selected_clients = select_clients(
                    clients=self.clients,
                    frac=frac,
                    round_id=round_id,
                    seed=seed,
                )

                client_updates: List[ClientUpdate] = []

                # 逐客户端训练，保证每完成一个客户端本地训练就更新一次总进度条。
                for client in selected_clients:
                    single_client_updates = train_selected_clients(
                        clients=[client],
                        global_model=self.global_model,
                        round_id=round_id,
                    )
                    client_updates.extend(single_client_updates)

                    if not progress_bar.disable:
                        progress_bar.set_postfix(
                            round=f"{round_id}/{rounds}",
                            client=int(client.client_id),
                            best=f"{self.train_state.best_acc:.2f}%",
                            refresh=False,
                        )
                        progress_bar.update(len(single_client_updates))

                # 聚合前先清掉控制台进度条，避免 print 的每轮摘要和 tqdm 混在一起。
                if not progress_bar.disable:
                    progress_bar.clear()

                aggregation_info = self.aggregate_client_updates(
                    client_updates=client_updates,
                )

                if not progress_bar.disable:
                    progress_bar.refresh()

                eval_result = self.evaluate_global_model()

                if eval_result.acc > self.train_state.best_acc:
                    self.train_state.best_acc = float(eval_result.acc)
                    self.train_state.best_round = int(round_id)

                self.train_state.round_id = int(round_id)

                round_result = self.build_round_result(
                    round_id=round_id,
                    selected_clients=selected_clients,
                    client_updates=client_updates,
                    eval_result=eval_result,
                    aggregation_info=aggregation_info,
                )
                self.round_results.append(round_result)

                avg_train_loss = round_result.aggregation_info.get(
                    "avg_train_loss",
                    None,
                )
                if avg_train_loss is None:
                    avg_train_loss_text = "nan"
                else:
                    avg_train_loss_text = f"{avg_train_loss:.4f}"

                if not progress_bar.disable:
                    progress_bar.set_postfix(
                        round=f"{round_id}/{rounds}",
                        client="done",
                        acc=f"{eval_result.acc:.2f}%",
                        best=f"{self.train_state.best_acc:.2f}%",
                        loss=avg_train_loss_text,
                        refresh=False,
                    )
                    progress_bar.refresh()

                if log_every > 0 and round_id % log_every == 0:
                    if not progress_bar.disable:
                        progress_bar.clear()

                    # 控制台短摘要：给人实时看。
                    # 由于 train.py 的 tee 会双写 stdout，
                    # 这一行也会进入 train.log，作为每轮简洁摘要。
                    if console_round_summary:
                        self.print_round_summary(round_result)

                    # 文件详细日志：只写 train.log，不污染控制台。
                    if file_round_detail:
                        self.print_file_round_detail(round_result)

                    if not progress_bar.disable:
                        progress_bar.refresh()

                self._cleanup_after_round()

        finally:
            progress_bar.close()

        return ServerTrainResult(
            round_results=list(self.round_results),
            train_state=self.train_state,
            best_acc=float(self.train_state.best_acc),
            best_round=int(self.train_state.best_round),
        )

    def aggregate_client_updates(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[str, Any]:
        """
        聚合客户端更新。

        聚合顺序：
            1. non_expert 参数
            2. expert 参数

        non_expert 固定使用 uniform；expert 使用启动文件注入的聚合器。
        """
        if len(client_updates) == 0:
            raise ValueError("client_updates 不能为空。")

        global_state_cpu = state_dict_to(
            self.global_model.state_dict(),
            device="cpu",
        )

        non_expert_result = self.aggregators.non_expert.aggregate(
            global_state=global_state_cpu,
            client_updates=client_updates,
            param_names=self.param_groups.non_expert,
            base_state=None,
            strict=True,
        )

        expert_result = self.aggregators.expert.aggregate(
            global_state=global_state_cpu,
            client_updates=client_updates,
            param_names=self.param_groups.expert,
            base_state=non_expert_result.new_state_dict,
            strict=True,
        )

        new_state_dict = expert_result.new_state_dict

        check_finite_state_dict(new_state_dict)

        self.global_model.load_state_dict(
            new_state_dict,
            strict=True,
        )
        self.global_model.to("cpu")

        return {
            "non_expert": non_expert_result.summary(),
            "expert": expert_result.summary(),
        }

    def evaluate_global_model(self) -> EvalResult:
        """
        在服务器测试集上评估全局模型。

        注意：
            测试集只在服务器使用。
            不参与客户端训练。
            不参与参数聚合。
        """
        self.global_model.to(self.device)
        result = evaluate(
            model=self.global_model,
            data_loader=self.test_loader,
            device=self.device,
        )
        self.global_model.to("cpu")
        return result

    def build_round_result(
        self,
        round_id: int,
        selected_clients: Sequence[FLClient],
        client_updates: Sequence[ClientUpdate],
        eval_result: EvalResult,
        aggregation_info: Dict[str, Any],
    ) -> RoundResult:
        """
        构建单轮训练结果摘要。
        """
        selected_client_ids = [
            int(client.client_id)
            for client in selected_clients
        ]

        avg_train_loss = average_client_metric(
            client_updates=list(client_updates),
            metric_name="train_loss",
            weighted=True,
            default=None,
        )
        avg_train_acc = average_client_metric(
            client_updates=list(client_updates),
            metric_name="train_acc",
            weighted=True,
            default=None,
        )

        full_aggregation_info = dict(aggregation_info)
        full_aggregation_info["avg_train_loss"] = avg_train_loss
        full_aggregation_info["avg_train_acc"] = avg_train_acc

        # 保存每个客户端的轻量诊断信息。
        # 注意：这里不保存 model_delta，也不保存 method_evidence 原始矩阵，
        # 避免 summary.json / train.log 过大。
        full_aggregation_info["client_diagnostics"] = (
            self._build_client_diagnostics(client_updates)
        )

        return RoundResult(
            round_id=int(round_id),
            selected_clients=selected_client_ids,
            test_loss=float(eval_result.loss),
            test_acc=float(eval_result.acc),
            best_acc=float(self.train_state.best_acc),
            client_metrics=collect_client_metrics(list(client_updates)),
            aggregation_info=full_aggregation_info,
        )

    def print_startup_summary(self) -> None:
        """
        打印训练开始前的摘要信息。

        这部分同时进入控制台和 train.log。
        """
        model_summary = summarize_model(self.global_model)
        param_summary = summarize_param_groups(
            state_dict=self.global_model.state_dict(),
            groups=self.param_groups,
        )

        print()
        print("=" * 80)
        print("[Server] FL training start")
        print(f"[Server] device: {self.device}")
        print(f"[Server] num_clients: {len(self.clients)}")
        print(f"[Server] rounds: {int(_cfg_get(self.cfg, 'rounds', 50))}")
        print(f"[Server] frac: {float(_cfg_get(self.cfg, 'frac', 1.0))}")
        print(f"[Server] model: {_cfg_get(self.cfg, 'model', 'unknown')}")
        print(
            "[Server] params: "
            f"total={model_summary['total_params']:,}, "
            f"trainable={model_summary['trainable_params']:,}"
        )
        print(
            "[Server] aggregators: "
            f"non_expert={self.aggregators.non_expert.method_name}, "
            f"expert={self.aggregators.expert.method_name}"
        )
        print(f"[Server] param_groups: {self.param_groups.summary()}")
        print(f"[Server] param_numel: {param_summary['floating_numel']}")
        print("=" * 80)
        print()

    def print_round_summary(
        self,
        round_result: RoundResult,
    ) -> None:
        """
        打印每轮训练短摘要。

        这部分适合控制台实时观察，所以尽量短。
        因为 train.py 使用 tee 输出，所以这一行也会进入 train.log。
        """
        avg_train_loss = round_result.aggregation_info.get(
            "avg_train_loss",
            None,
        )
        avg_train_acc = round_result.aggregation_info.get(
            "avg_train_acc",
            None,
        )

        if avg_train_loss is None:
            avg_train_loss_text = "nan"
        else:
            avg_train_loss_text = f"{avg_train_loss:.4f}"

        if avg_train_acc is None:
            avg_train_acc_text = "nan"
        else:
            avg_train_acc_text = f"{avg_train_acc:.2f}%"

        print(
            f"[Round {round_result.round_id:03d}] "
            f"train_loss={avg_train_loss_text} | "
            f"train_acc={avg_train_acc_text} | "
            f"test_loss={round_result.test_loss:.4f} | "
            f"test_acc={round_result.test_acc:.2f}% | "
            f"best_acc={round_result.best_acc:.2f}%"
        )

    def print_file_round_detail(
        self,
        round_result: RoundResult,
    ) -> None:
        """
        写入每轮详细日志。

        这部分只写 train.log，不打印到控制台。

        当前记录：
            1. 本轮整体 train/test 指标
            2. 本轮选择的客户端
            3. non_expert / expert 分别用的聚合方法
            4. non_expert / expert 每个客户端的聚合权重
            5. 每个客户端样本数、本地 train_loss/train_acc、expert_usage
        """
        logging_cfg = _cfg_get(self.cfg, "logging", {})

        log_round_clients = _cfg_get_bool(
            logging_cfg,
            "log_round_clients",
            True,
        )
        log_client_table = _cfg_get_bool(
            logging_cfg,
            "log_client_table",
            True,
        )
        log_agg_weights = _cfg_get_bool(
            logging_cfg,
            "log_agg_weights",
            True,
        )

        avg_train_loss = round_result.aggregation_info.get(
            "avg_train_loss",
            None,
        )
        avg_train_acc = round_result.aggregation_info.get(
            "avg_train_acc",
            None,
        )

        avg_train_loss_text = self._format_metric(
            avg_train_loss,
            fmt=".4f",
        )
        avg_train_acc_text = self._format_metric(
            avg_train_acc,
            fmt=".2f",
            suffix="%",
        )

        self._write_log_only(
            f"[RoundMetrics] "
            f"round={round_result.round_id} "
            f"train_loss={avg_train_loss_text} "
            f"train_acc={avg_train_acc_text} "
            f"test_loss={round_result.test_loss:.4f} "
            f"test_acc={round_result.test_acc:.2f}% "
            f"best_acc={round_result.best_acc:.2f}%"
        )

        if log_round_clients:
            self._write_log_only(
                f"[Clients] "
                f"round={round_result.round_id} "
                f"ids={self._format_client_ids(round_result.selected_clients)}"
            )

        # 聚合器摘要：方法、客户端数、参数数量、权重。
        self._write_aggregation_info_to_log(
            round_result=round_result,
            log_agg_weights=log_agg_weights,
        )

        # 每个客户端一行诊断信息：样本数、训练指标、聚合权重、expert usage。
        if log_client_table:
            self._write_client_table_to_log(round_result)

    def _build_client_diagnostics(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, Dict[str, Any]]:
        """从 ClientUpdate 中提取公共及方法插件提供的轻量诊断信息。"""
        diagnostics: Dict[int, Dict[str, Any]] = {}
        for update in client_updates:
            extra = dict(update.extra or {})
            client_diag: Dict[str, Any] = {
                "num_samples": int(update.num_samples),
                "metrics": dict(update.metrics or {}),
                "expert_usage": extra.get("expert_usage", None),
            }
            if self.method_client_diagnostics_builder is not None:
                method_diag = self.method_client_diagnostics_builder(update)
                if method_diag is None:
                    method_diag = {}
                if not isinstance(method_diag, Mapping):
                    raise TypeError("method_client_diagnostics_builder 必须返回 Mapping 或 None。")
                client_diag.update(dict(method_diag))
            diagnostics[int(update.client_id)] = client_diag
        return diagnostics

    def _write_aggregation_info_to_log(
        self,
        round_result: RoundResult,
        log_agg_weights: bool,
    ) -> None:
        """
        写入 non_expert / expert 聚合摘要。

        输出示例：
            [Agg][non_expert] round=1 method=uniform clients=10 params=121 weights=uniform(each=0.1000)
            [Agg][expert] round=1 method=<method> clients=10 params=16 weights={0:0.0812,1:0.1033,...}
        """
        logging_cfg = _cfg_get(self.cfg, "logging", {})
        compact_uniform_weights = _cfg_get_bool(
            logging_cfg,
            "compact_uniform_weights",
            True,
        )

        for group_name in ("non_expert", "expert"):
            agg_info = self._extract_aggregation_info(
                round_result=round_result,
                group_name=group_name,
            )
            if agg_info is None:
                continue

            if log_agg_weights:
                weights_text = self._format_weights(
                    agg_info.get("weights", None),
                    compact_uniform=compact_uniform_weights,
                )
            else:
                weights_text = "hidden"

            self._write_log_only(
                f"[Agg][{group_name}] "
                f"round={round_result.round_id} "
                f"method={agg_info.get('method', 'unknown')} "
                f"clients={agg_info.get('num_clients', 'unknown')} "
                f"params={agg_info.get('param_count', 'unknown')} "
                f"weights={weights_text}"
            )

    def _write_client_table_to_log(
        self,
        round_result: RoundResult,
    ) -> None:
        """
        写入每个客户端的一行诊断信息。

        每行包含：
            1. 客户端样本数
            2. 客户端本地 train_loss / train_acc
            3. non_expert 聚合权重
            4. expert 聚合权重
            5. expert_usage
        """
        client_diagnostics = round_result.aggregation_info.get(
            "client_diagnostics",
            {},
        )
        if not isinstance(client_diagnostics, Mapping):
            return

        non_expert_info = self._extract_aggregation_info(
            round_result=round_result,
            group_name="non_expert",
        )
        expert_info = self._extract_aggregation_info(
            round_result=round_result,
            group_name="expert",
        )

        non_expert_weights = {}
        expert_weights = {}

        if non_expert_info is not None:
            non_expert_weights = non_expert_info.get("weights", {}) or {}

        if expert_info is not None:
            expert_weights = expert_info.get("weights", {}) or {}

        for client_id in round_result.selected_clients:
            client_id = int(client_id)

            item = self._get_client_diagnostic(
                client_diagnostics,
                client_id,
            )

            if item is None:
                self._write_log_only(
                    f"[Client][{client_id}] "
                    f"round={round_result.round_id} "
                    f"missing_diagnostics=true"
                )
                continue

            metrics = item.get("metrics", {}) or {}

            num_samples = item.get("num_samples", "unknown")
            train_loss = self._format_metric(
                metrics.get("train_loss", None),
                fmt=".4f",
            )
            train_acc = self._format_metric(
                metrics.get("train_acc", None),
                fmt=".2f",
                suffix="%",
            )

            non_expert_weight = self._format_weight_value(
                self._get_weight_for_client(non_expert_weights, client_id)
            )
            expert_weight = self._format_weight_value(
                self._get_weight_for_client(expert_weights, client_id)
            )

            expert_usage_text = self._format_expert_usage(
                item.get("expert_usage", None)
            )

            self._write_log_only(
                f"[Client][{client_id}] "
                f"round={round_result.round_id} "
                f"samples={num_samples} "
                f"train_loss={train_loss} "
                f"train_acc={train_acc} "
                f"non_expert_w={non_expert_weight} "
                f"expert_w={expert_weight} "
                f"{expert_usage_text}"
            )

    def _extract_aggregation_info(
        self,
        round_result: RoundResult,
        group_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        提取某个参数组的聚合信息。

        当前 AggregationResult.summary() 常见结构：
            {
                "weights": {...},
                "diagnostics": {
                    "method": "uniform",
                    "param_group": "expert",
                    "num_clients": 10,
                    "param_count": 16,
                    ...
                }
            }

        这个函数会把外层 weights 和内层 diagnostics 合并成一个扁平 dict，
        方便日志打印。
        """
        summary = round_result.aggregation_info.get(group_name, None)
        if summary is None:
            return None

        if not isinstance(summary, Mapping):
            return {
                "method": "unknown",
                "param_group": group_name,
                "num_clients": "unknown",
                "param_count": "unknown",
                "weights": None,
                "raw_summary": summary,
            }

        diagnostics = summary.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            diagnostics = {}

        method = diagnostics.get(
            "method",
            summary.get(
                "method",
                summary.get("method_name", summary.get("aggregator", "unknown")),
            ),
        )
        param_group = diagnostics.get(
            "param_group",
            summary.get("param_group", group_name),
        )
        num_clients = diagnostics.get(
            "num_clients",
            summary.get(
                "num_clients",
                summary.get(
                    "effective_clients",
                    summary.get("num_effective_clients", "unknown"),
                ),
            ),
        )
        param_count = diagnostics.get(
            "param_count",
            summary.get("param_count", "unknown"),
        )

        weights = None

        for weight_key in (
            "weights",
            "client_weights",
            "sample_weights",
            "effective_weights",
        ):
            if weight_key in summary:
                weights = summary[weight_key]
                break

        if weights is None:
            for weight_key in (
                "weights",
                "client_weights",
                "sample_weights",
                "effective_weights",
            ):
                if weight_key in diagnostics:
                    weights = diagnostics[weight_key]
                    break

        return {
            "method": method,
            "param_group": param_group,
            "num_clients": num_clients,
            "param_count": param_count,
            "weights": weights,
            "diagnostics": dict(diagnostics),
        }

    def _write_log_only(self, message: str) -> None:
        """
        只写入 train.log，不打印到控制台。

        原理：
            utils/logging.py 的 TeeStream 会在 sys.stdout 上保存 log_file。
            如果当前确实处于 tee_output_to_file() 环境中，就直接写 log_file。
            如果没有使用 tee，则退化为普通 print，避免信息丢失。
        """
        stdout = sys.stdout

        log_file = getattr(stdout, "log_file", None)
        lock = getattr(stdout, "lock", None)

        if log_file is None:
            print(message)
            return

        if lock is None:
            log_file.write(message + "\n")
            log_file.flush()
            return

        with lock:
            log_file.write(message + "\n")
            log_file.flush()

    @staticmethod
    def _format_metric(
        value: Any,
        *,
        fmt: str,
        suffix: str = "",
    ) -> str:
        """
        格式化日志里的指标。

        value 为 None 时写 nan。
        """
        if value is None:
            return "nan"

        try:
            return f"{float(value):{fmt}}{suffix}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _format_weight_value(value: Any) -> str:
        """
        格式化单个客户端权重。

        例如：
            0.10000000000000002 -> 0.1000
        """
        if value is None:
            return "nan"

        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    def _format_weights(
        self,
        weights: Any,
        *,
        compact_uniform: bool,
    ) -> str:
        """
        格式化聚合权重。

        uniform 权重默认压缩成：
            uniform(each=0.1000)

        非 uniform 权重打印成：
            {0:0.1234,1:0.0987,...}
        """
        if weights is None:
            return "none"

        if not isinstance(weights, Mapping):
            return self._compact_log_value(weights)

        if len(weights) == 0:
            return "{}"

        numeric_items = []
        for key, value in weights.items():
            try:
                client_id = int(key)
                weight_value = float(value)
            except (TypeError, ValueError):
                return self._compact_log_value(weights)

            numeric_items.append((client_id, weight_value))

        numeric_items = sorted(
            numeric_items,
            key=lambda item: item[0],
        )

        if compact_uniform and self._is_uniform_weight_items(numeric_items):
            return f"uniform(each={numeric_items[0][1]:.4f})"

        body = ",".join(
            f"{client_id}:{weight_value:.4f}"
            for client_id, weight_value in numeric_items
        )
        return "{" + body + "}"

    @staticmethod
    def _is_uniform_weight_items(
        items: Sequence[tuple[int, float]],
        *,
        atol: float = 1.0e-10,
    ) -> bool:
        """
        判断权重是否近似均匀。

        用于把一长串 0.10000000000000002 压缩成 uniform(each=0.1000)。
        """
        if len(items) == 0:
            return False

        first_value = float(items[0][1])

        for _, value in items:
            if abs(float(value) - first_value) > atol:
                return False

        return True

    @staticmethod
    def _format_client_ids(client_ids: Sequence[int]) -> str:
        """
        格式化客户端 id 列表。

        输出：
            [0,4,9,6]
        """
        body = ",".join(
            str(int(client_id))
            for client_id in client_ids
        )
        return "[" + body + "]"

    @staticmethod
    def _get_client_diagnostic(
        client_diagnostics: Mapping[Any, Any],
        client_id: int,
    ) -> Optional[Mapping[str, Any]]:
        """
        兼容 int key / str key 两种客户端诊断字典。
        """
        if client_id in client_diagnostics:
            item = client_diagnostics[client_id]
            if isinstance(item, Mapping):
                return item

        str_client_id = str(client_id)
        if str_client_id in client_diagnostics:
            item = client_diagnostics[str_client_id]
            if isinstance(item, Mapping):
                return item

        return None

    @staticmethod
    def _get_weight_for_client(
        weights: Any,
        client_id: int,
    ) -> Optional[Any]:
        """
        从权重字典中读取某个客户端的权重。

        兼容：
            weights[0]
            weights["0"]
        """
        if not isinstance(weights, Mapping):
            return None

        if client_id in weights:
            return weights[client_id]

        str_client_id = str(client_id)
        if str_client_id in weights:
            return weights[str_client_id]

        return None

    def _format_expert_usage(
        self,
        expert_usage: Any,
    ) -> str:
        """
        格式化客户端 expert usage。

        输出示例：
            expert_active=4/4 expert_total=9600 expert_counts={0:2400,1:2381,2:2410,3:2409} expert_frac={0:0.250,1:0.248,2:0.251,3:0.251}

        如果没有采集：
            expert_usage=none

        如果模型不支持：
            expert_usage=unsupported(reason=...)
        """
        if expert_usage is None:
            return "expert_usage=none"

        if not isinstance(expert_usage, Mapping):
            return f"expert_usage={self._compact_log_value(expert_usage)}"

        supported = bool(expert_usage.get("supported", True))
        if not supported:
            reason = expert_usage.get("reason", "unknown")
            return (
                "expert_usage=unsupported"
                f"(reason={self._compact_log_value(reason, max_chars=160)})"
            )

        num_experts = expert_usage.get(
            "num_experts",
            _cfg_get(self.cfg, "num_experts", "unknown"),
        )
        active_experts = expert_usage.get("active_experts", "unknown")
        total_activations = expert_usage.get("total_activations", "unknown")
        expert_counts = expert_usage.get("expert_counts", None)
        expert_fraction = expert_usage.get("expert_fraction", None)
        dead_experts = expert_usage.get("dead_experts", [])

        counts_text = self._format_int_mapping(expert_counts)
        fraction_text = self._format_float_mapping(
            expert_fraction,
            precision=3,
        )

        return (
            f"expert_active={active_experts}/{num_experts} "
            f"expert_total={total_activations} "
            f"expert_counts={counts_text} "
            f"expert_frac={fraction_text} "
            f"dead={self._format_client_ids(dead_experts)}"
        )

    @staticmethod
    def _format_int_mapping(value: Any) -> str:
        """
        格式化 int 映射。

        输出：
            {0:120,1:130}
        """
        if not isinstance(value, Mapping):
            return "none"

        items = []
        for key, item_value in value.items():
            try:
                items.append((int(key), int(item_value)))
            except (TypeError, ValueError):
                return repr(value)

        items = sorted(
            items,
            key=lambda item: item[0],
        )

        body = ",".join(
            f"{key}:{item_value}"
            for key, item_value in items
        )
        return "{" + body + "}"

    @staticmethod
    def _format_float_mapping(
        value: Any,
        *,
        precision: int,
    ) -> str:
        """
        格式化 float 映射。

        输出：
            {0:0.250,1:0.248}
        """
        if not isinstance(value, Mapping):
            return "none"

        items = []
        for key, item_value in value.items():
            try:
                items.append((int(key), float(item_value)))
            except (TypeError, ValueError):
                return repr(value)

        items = sorted(
            items,
            key=lambda item: item[0],
        )

        body = ",".join(
            f"{key}:{item_value:.{precision}f}"
            for key, item_value in items
        )
        return "{" + body + "}"

    @staticmethod
    def _compact_log_value(
        value: Any,
        *,
        max_chars: int = 1200,
    ) -> str:
        """
        把日志字段压成一行，避免 train.log 被超长对象刷屏。
        """
        text = repr(value)

        if len(text) <= max_chars:
            return text

        return text[:max_chars] + "..."

    def _validate_server_state(self) -> None:
        """
        检查服务端初始化状态是否合法。
        """
        if len(self.clients) == 0:
            raise ValueError("服务端没有任何客户端。")

        if self.test_loader is None:
            raise ValueError("test_loader 不能为空。")

        if len(self.param_groups.expert) == 0:
            raise ValueError(
                "没有找到 expert 参数。"
                "请检查模型参数名是否包含 experts.。"
            )

        if len(self.param_groups.non_expert) == 0:
            raise ValueError("没有找到 non_expert 参数。")

    @staticmethod
    def _cleanup_after_round() -> None:
        """
        每轮结束后清理显存和 Python 垃圾对象。
        """
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_server(
    cfg: Any,
    client_loaders: Sequence[DataLoader],
    test_loader: DataLoader,
    device: torch.device | str,
    expert_aggregator_builder: ExpertAggregatorBuilder,
    client_evidence_loaders: Optional[Sequence[DataLoader]] = None,
    expert_evidence_collector: Optional[ExpertEvidenceCollector] = None,
    method_client_diagnostics_builder: Optional[MethodClientDiagnosticsBuilder] = None,
    method_context: Optional[MethodContext] = None,
) -> FLServer:
    """
    构建 FLServer。

    train.py 后面可以直接调用这个函数。

    client_evidence_loaders:
        方法级 evidence 专用 loader。
        如果为 None，后续 client.py 里会回退到普通 train_loader。
    """
    return FLServer(
        cfg=cfg,
        client_loaders=client_loaders,
        client_evidence_loaders=client_evidence_loaders,
        test_loader=test_loader,
        device=device,
        expert_aggregator_builder=expert_aggregator_builder,
        expert_evidence_collector=expert_evidence_collector,
        method_client_diagnostics_builder=method_client_diagnostics_builder,
        method_context=method_context,
    )


def resolve_device(cfg: Any) -> torch.device:
    """
    根据 cfg.device 解析训练设备。

    支持：
        auto
        cpu
        cuda
        mps
    """
    device_name = str(_cfg_get(cfg, "device", "auto")).lower().strip()

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置 device=cuda，但当前环境 CUDA 不可用。")
        return torch.device("cuda")

    if device_name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("配置 device=mps，但当前环境 MPS 不可用。")
        return torch.device("mps")

    if device_name == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"不支持的 device：{device_name}。"
        "当前支持：auto, cpu, cuda, mps"
    )


def _cfg_get_bool(
    cfg: Any,
    key: str,
    default: bool = False,
) -> bool:
    """
    从配置里读取 bool 值。

    支持 true / false、1 / 0、yes / no、on / off 等常见写法。
    """
    value = _cfg_get(cfg, key, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"1", "true", "yes", "y", "on"}:
            return True

        if normalized in {"0", "false", "no", "n", "off"}:
            return False

    return bool(value)


# ============================================================================
# Bundled from train.py (runtime entry and output)
# ============================================================================


MethodCliArgumentRegistrar = Callable[[argparse.ArgumentParser], None]
MethodCliOverridesBuilder = Callable[[argparse.Namespace], Mapping[str, Any]]


def parse_args(method_cli_argument_registrar: Optional[MethodCliArgumentRegistrar] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FL + MoE training entrypoint")
    parser.add_argument("--config", type=str, default=None, help="可选外部配置文件；显式 CLI 参数优先。")
    parser.add_argument("--dataset", type=normalize_dataset_name, choices=sorted(SUPPORTED_DATASETS), default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--frac", type=float, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--backbone", choices=list_supported_backbones(), default=None)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--optimizer", dest="optimizer_type", choices=("sgd","adam","adamw"), default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--server-evidence-size", type=int, default=None)
    parser.add_argument("--server-evidence-batch-size", type=int, default=None)
    if method_cli_argument_registrar is not None:
        method_cli_argument_registrar(parser)
    return parser.parse_args()


def build_common_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    top_level={
        "data_root":"data_root","num_clients":"num_clients","alpha":"alpha","frac":"frac",
        "rounds":"rounds","local_epochs":"local_epochs","batch_size":"batch_size",
        "test_batch_size":"test_batch_size","num_workers":"num_workers","num_experts":"num_experts",
        "topk":"topk","seed":"seed","device":"device","output_dir":"output_dir","run_name":"run_name",
    }
    for arg_name,config_key in top_level.items():
        value=getattr(args,arg_name,None)
        if value is not None: overrides[config_key]=value
    dataset=getattr(args,"dataset",None)
    if dataset is not None:
        dataset = normalize_dataset_name(dataset)
        overrides["dataset"]=dataset
        overrides["num_classes"]=_infer_num_classes(dataset)
        info=DATASET_INFO.get(dataset)
        if info is not None:
            overrides["input_shape"]=tuple(info["input_shape"])
    backbone=getattr(args,"backbone",None)
    if backbone is not None: overrides.setdefault("model_cfg",{})["backbone"]=backbone
    opt={}
    for an,ck in (("optimizer_type","type"),("lr","lr"),("momentum","momentum"),("weight_decay","weight_decay")):
        v=getattr(args,an,None)
        if v is not None: opt[ck]=v
    if opt: overrides["optimizer"]=opt

    server_evidence = {}
    server_evidence_size = getattr(args, "server_evidence_size", None)
    if server_evidence_size is not None:
        server_evidence["size"] = server_evidence_size
    server_evidence_batch_size = getattr(args, "server_evidence_batch_size", None)
    if server_evidence_batch_size is not None:
        server_evidence["batch_size"] = server_evidence_batch_size
    if server_evidence:
        overrides["server_evidence"] = server_evidence

    return overrides


def main(
    expert_aggregator_builder: ExpertAggregatorBuilder,
    embedded_method_config: Mapping[str, Any],
    expert_method_name: str,
    method_config_defaults: Mapping[str, Any] | None = None,
    method_config_validator: Optional[Callable[[Mapping[str, Any]], None]] = None,
    expert_evidence_collector: Optional[ExpertEvidenceCollector] = None,
    method_client_diagnostics_builder: Optional[MethodClientDiagnosticsBuilder] = None,
    method_cli_argument_registrar: Optional[MethodCliArgumentRegistrar] = None,
    method_cli_overrides_builder: Optional[MethodCliOverridesBuilder] = None,
) -> int:
    """
    训练入口。

    这里负责：
        1. 读取配置
        2. 设置随机种子
        3. 创建输出目录
        4. 打开 train.log 双写
        5. 调用 run_training() 执行实际训练

    日志说明：
        - 普通 print / 报错信息会写入控制台和 train.log。
        - tqdm 这类动态进度条只显示在控制台，不写入 train.log。
    """
    args = parse_args(method_cli_argument_registrar=method_cli_argument_registrar)
    config_overrides = build_common_cli_overrides(args)
    if method_cli_overrides_builder is not None:
        config_overrides = _deep_merge(base=config_overrides, override=dict(method_cli_overrides_builder(args) or {}))
    cfg = (
        load_config(args.config, method_defaults=method_config_defaults, method_validator=method_config_validator, config_overrides=config_overrides)
        if args.config is not None
        else load_embedded_config(embedded_method_config, method_defaults=method_config_defaults, method_validator=method_config_validator, config_overrides=config_overrides)
    )

    configured_non_expert_method = str(
        cfg.get("agg.non_expert.method", "uniform")
    ).lower().strip()
    if configured_non_expert_method != "uniform":
        raise ValueError(
            "base.py 已固定 non_expert 使用 uniform 聚合，"
            f"当前配置却是 {configured_non_expert_method!r}。"
        )

    configured_expert_method = str(
        cfg.get("agg.expert.method", "")
    ).lower().strip()
    if configured_expert_method != expert_method_name.lower().strip():
        raise ValueError(
            f"当前脚本实现的专家聚合是 {expert_method_name!r}，"
            f"但配置中 agg.expert.method={configured_expert_method!r}。"
        )

    set_seed(
        seed=int(cfg.seed),
        deterministic=bool(cfg.get("deterministic", True)),
    )

    run_dir = Path(ensure_run_dir(cfg))
    log_path = run_dir / "train.log"

    # 开启 stderr 进度条过滤。
    # 这样 tqdm 的动态刷新不会污染 train.log，
    # 但普通 stderr 报错和 traceback 仍然会写入 train.log。
    with tee_output_to_file(
        log_path,
        filter_stderr_progress=True,
    ):
        try:
            return run_training(
                args=args,
                cfg=cfg,
                run_dir=run_dir,
                expert_aggregator_builder=expert_aggregator_builder,
                expert_method_name=expert_method_name,
                expert_evidence_collector=expert_evidence_collector,
                method_client_diagnostics_builder=method_client_diagnostics_builder,
            )
        except Exception:
            print()
            print("=" * 80)
            print("[Train] Failed")
            print("=" * 80)
            traceback.print_exc()
            return 1


def run_training(
    args: argparse.Namespace,
    cfg: Any,
    run_dir: Path,
    expert_aggregator_builder: ExpertAggregatorBuilder,
    expert_method_name: str,
    expert_evidence_collector: Optional[ExpertEvidenceCollector] = None,
    method_client_diagnostics_builder: Optional[MethodClientDiagnosticsBuilder] = None,
) -> int:
    """
    实际训练流程。

    这个函数会被 main() 包在 tee_output_to_file() 里面，
    所以这里的普通 print 和报错会同时写入控制台和 train.log。

    注意：
        tqdm 进度条由 utils/logging.py 过滤，不写入 train.log。

    总流程：
        1. 保存配置
        2. 解析设备
        3. 加载数据集
        4. 划分客户端数据
        5. 创建 DataLoader
        6. 创建 FLServer
        7. 执行联邦训练
        8. 保存结果
    """
    if bool(cfg.get("logging.save_config", True)):
        save_config(
            cfg=cfg,
            output_path=run_dir / "config_used.yaml",
        )

    device = resolve_device(cfg)

    print()
    print("=" * 80)
    print("[Train] Start")
    config_source = (
        args.config
        if args.config is not None
        else f"<embedded:{expert_method_name}>"
    )
    print(f"[Train] config: {config_source}")
    print(f"[Train] run_name: {cfg.run_name}")
    print(f"[Train] run_dir: {cfg.run_dir}")
    print(f"[Train] log_file: {run_dir / 'train.log'}")
    print(f"[Train] device: {device}")
    print("=" * 80)
    print()

    dataset_bundle = build_datasets(cfg)

    print(
        "[Data] "
        f"dataset={dataset_bundle.name} | "
        f"num_classes={dataset_bundle.num_classes} | "
        f"input_shape={dataset_bundle.input_shape}"
    )
    print(
        "[Data] "
        f"train_size={len(dataset_bundle.train_dataset)} | "
        f"test_size={len(dataset_bundle.test_dataset)}"
    )
    if dataset_bundle.server_evidence_dataset is not None:
        print(
            "[Data] "
            f"server_evidence_size={len(dataset_bundle.server_evidence_dataset)}"
        )

    partition = partition_dataset(
        cfg=cfg,
        dataset=dataset_bundle.train_dataset,
    )

    save_partition_summary(
        partition=partition,
        output_path=run_dir / "partition_summary.json",
    )

    loader_bundle = build_dataloaders(
        cfg=cfg,
        train_dataset=dataset_bundle.train_dataset,
        # 方法级 evidence 专用数据集。
        # 它和 train_dataset 使用同一份原始样本，
        # 但 transform 在 data/datasets.py 中强制关闭随机数据增强。
        train_evidence_dataset=dataset_bundle.train_evidence_dataset,
        test_dataset=dataset_bundle.test_dataset,
        client_indices=partition.client_indices,
        server_evidence_dataset=dataset_bundle.server_evidence_dataset,
    )

    method_context = MethodContext(
        cfg=cfg,
        device=device,
        dataset_name=dataset_bundle.name,
        server_evidence_loader=loader_bundle.server_evidence_loader,
        model_builder=build_model,
    )

    server = build_server(
        cfg=cfg,
        client_loaders=loader_bundle.client_loaders,
        # 方法级 evidence 专用 loader。
        # 训练仍然走 client_loaders，统计 方法插件 时走 client_evidence_loaders。
        client_evidence_loaders=loader_bundle.client_evidence_loaders,
        test_loader=loader_bundle.test_loader,
        device=device,
        expert_aggregator_builder=expert_aggregator_builder,
        expert_evidence_collector=expert_evidence_collector,
        method_client_diagnostics_builder=method_client_diagnostics_builder,
        method_context=method_context,
    )

    train_result = server.train()

    save_train_outputs(
        train_result=train_result,
        output_dir=run_dir,
        save_csv=bool(cfg.get("logging.save_results_csv", True)),
    )

    print()
    print("=" * 80)
    print("[Train] Done")
    print(f"[Train] best_acc: {train_result.best_acc:.2f}%")
    print(f"[Train] best_round: {train_result.best_round}")
    print(f"[Train] outputs saved to: {cfg.run_dir}")
    print("=" * 80)

    return 0


def save_partition_summary(
    partition: Any,
    output_path: Path,
) -> None:
    """
    保存数据划分摘要。

    注意：
        不保存完整 client_indices。
        这里只保存每个客户端样本数、类别分布等轻量信息。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = partition_summary_to_dict(partition)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            make_json_safe(summary),
            f,
            ensure_ascii=False,
            indent=2,
        )


def save_train_outputs(
    train_result: Any,
    output_dir: Path,
    save_csv: bool = True,
) -> None:
    """
    保存训练输出。

    输出文件：
        summary.json:
            完整训练摘要。

        results.csv:
            每轮核心指标，方便直接画图或导入 Excel。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            make_json_safe(train_result.to_dict()),
            f,
            ensure_ascii=False,
            indent=2,
        )

    if save_csv:
        csv_path = output_dir / "results.csv"
        save_round_results_csv(
            round_results=train_result.round_results,
            output_path=csv_path,
        )


def save_round_results_csv(
    round_results: List[Any],
    output_path: Path,
) -> None:
    """
    保存每轮训练结果到 CSV。

    CSV 只保存最常用的核心指标：
        round_id
        selected_clients
        avg_train_loss
        avg_train_acc
        test_loss
        test_acc
        best_acc
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "round_id",
        "selected_clients",
        "avg_train_loss",
        "avg_train_acc",
        "test_loss",
        "test_acc",
        "best_acc",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for item in round_results:
            aggregation_info = item.aggregation_info

            row = {
                "round_id": int(item.round_id),
                "selected_clients": " ".join(
                    str(client_id) for client_id in item.selected_clients
                ),
                "avg_train_loss": aggregation_info.get(
                    "avg_train_loss",
                    "",
                ),
                "avg_train_acc": aggregation_info.get(
                    "avg_train_acc",
                    "",
                ),
                "test_loss": float(item.test_loss),
                "test_acc": float(item.test_acc),
                "best_acc": float(item.best_acc),
            }

            writer.writerow(row)


def make_json_safe(obj: Any) -> Any:
    """
    把对象转换成 JSON 可保存格式。

    主要处理：
        torch.Tensor
        torch.device
        Path
        dict
        list / tuple
    """
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()

    if isinstance(obj, torch.device):
        return str(obj)

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {
            str(key): make_json_safe(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            make_json_safe(value)
            for value in obj
        ]

    if hasattr(obj, "to_dict"):
        return make_json_safe(obj.to_dict())

    return obj

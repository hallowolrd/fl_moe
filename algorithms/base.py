"""联邦学习与 MoE 实验的公共基础模块。

负责统一配置管理、数据处理、模型构建、客户端训练、参数聚合、
服务端协调、模型评估、日志记录及实验结果输出。
专家聚合算法通过方法脚本注入，公共模块不直接实现具体专家聚合策略。
"""

from __future__ import annotations

# ============================================================================
# 0. 启动环境初始化（必须先于 torch 导入）
# ============================================================================
# 注意：
# 下面这一段必须放在 import torch 之前。
#
# 原因：
# 1. CUBLAS_WORKSPACE_CONFIG 必须在 CUDA / cuBLAS 初始化前设置。
# 2. PYTHONHASHSEED 必须在 Python 解释器启动前生效。
#
# 所以如果当前进程的 PYTHONHASHSEED 和配置文件里的 seed 不一致，
# 这里会自动重新执行一次当前 Python 命令。
# ============================================================================

import os
import sys
from pathlib import Path


# 公共实验配置的唯一默认值来源。
# 必须在 import torch 前定义，因为启动阶段需要读取默认 seed。
DEFAULT_CONFIG = {
    "dataset": "cifar10",
    "data_root": "./datasets",
    "num_clients": 5,
    "partition": {
        "method": "dirichlet",
        "dirichlet": {
            "alpha": 0.1,
            "sample_balance": "unbalanced",
        },
    },
    "min_samples_per_client": 1,
    "partition_max_retries": 100,
    "frac": 1.0,
    "rounds": 50,
    "local_epochs": 5,
    "local_steps": None,
    "batch_size": 64,
    "test_batch_size": 64,
    "num_workers": 2,
    "data_augmentation": True,
    "download_data": True,
    "drop_last": False,
    "pin_memory": None,
    "label_smooth": 0.0,
    "grad_clip": None,
    "model_cfg": {
        "transformer_layers": 6,
        "hidden_dim": 256,
        "attention_heads": 4,
        "patch_size": 4,
        "expert_hidden_dim": 1024,
        "moe_block_indices": [2, 4, 6],
        "routing": "top1",
        "aux_loss_coef": 0.0,
        "capacity_factor": 0.0,
        "token_dropping": False,
        "router_statistics": True,
        "dropout": 0.0,
    },
    "num_experts": 4,
    "optimizer": {
        "type": "sgd",
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "nesterov": False,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
    },
    "agg": {
        "non_expert": {"method": "uniform"},
        "expert": {"method": "uniform"},
    },
    # 此配置由所选算法或插件自行定义。
    # 公共框架只校验其为映射类型，不解析内部字段。
    "method_config": {},
    "seed": 0,
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
        "save_config": True,
        "save_results_csv": True,
        "progress_bar": True,
    },
}


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

    include_paths: list[Path] = []
    include_list_indent: int | None = None
    local_value: str | None = None

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            # 去掉行内注释。这里足够处理当前配置里的 seed / include。
            line = raw_line.split("#", 1)[0].rstrip()
            stripped_line = line.strip()

            if not stripped_line:
                continue

            indent = len(line) - len(line.lstrip())

            # 支持 include: 后面的 YAML 块列表，包括无额外缩进的列表。
            if include_list_indent is not None:
                if stripped_line.startswith("-") and indent >= include_list_indent:
                    include_value = _clean_simple_yaml_value(stripped_line[1:].strip())
                    if include_value:
                        include_paths.append((path.parent / include_value).resolve())
                    continue

                if indent <= include_list_indent:
                    include_list_indent = None
                else:
                    continue

            if ":" not in stripped_line:
                continue

            left, right = stripped_line.split(":", 1)
            name = left.strip()
            raw_value = right.strip()
            value = _clean_simple_yaml_value(raw_value)

            if name == "include" and indent == 0:
                # YAML 中重复的 include 键以最后一个为准。
                include_paths = []

                if not raw_value:
                    include_list_indent = indent
                elif value.startswith("[") and value.endswith("]"):
                    # 支持 include: [base.yaml, model/switch_vit.yaml]。
                    for item in value[1:-1].split(","):
                        item = _clean_simple_yaml_value(item)
                        if item:
                            include_paths.append((path.parent / item).resolve())
                else:
                    include_paths.append((path.parent / value).resolve())

                continue

            if name == key and indent == 0:
                local_value = value
                continue

    # 当前配置里的 seed 优先级高于 include；多个 include 按正式加载器的
    # 合并顺序处理，因此后面的 include 优先级更高。
    if local_value is not None:
        return local_value

    for include_path in reversed(include_paths):
        included_value = _read_top_level_scalar_from_yaml_like_file(
            path=include_path,
            key=key,
            visited=visited,
        )
        if included_value is not None:
            return included_value

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
            config_seed if config_seed is not None else DEFAULT_CONFIG["seed"]
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
# 依赖导入：必须放在启动环境初始化之后。
# 集中导入依赖；此代码块必须位于确定性环境初始化之后。
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
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
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
import torch.optim as optim
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm


# ============================================================================
# 1. 配置管理与注册（Configuration）
# 包含配置节点、默认配置、配置加载与合并、参数覆盖、校验及保存。
# ============================================================================


# ------------------------------------------------------------------------------
# 1.1 可扩展的合法取值注册区
# ------------------------------------------------------------------------------
# 后续新增数据集时，优先改这里。

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
    """将常见的数据集名称变体规范化为统一名称。"""
    key = str(dataset_name).strip().lower()
    return DATASET_ALIASES.get(key, key)


class ConfigError(Exception):
    """配置相关错误。"""

    pass


class ConfigNode:
    """
    轻量级配置对象。

    支持属性读取：
        cfg.dataset
        cfg.agg.non_expert.method
        cfg.agg.expert.method
        cfg.method_config

    也支持路径读取：
        cfg.get("agg.non_expert.method")
        cfg.get("agg.expert.method")
        cfg.get("method_config.some_option")
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
            cfg.get("agg.non_expert.method")
            cfg.get("agg.expert.method")
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


def _default_config_value(path: str) -> Any:
    """按点分隔路径读取默认配置，并返回独立副本。"""
    current: Any = DEFAULT_CONFIG
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"默认配置中不存在字段：{path}")
        current = current[part]
    return copy.deepcopy(current)


def load_embedded_config(
    method_overrides: Mapping[str, Any] | None = None,
    method_defaults: Mapping[str, Any] | None = None,
    method_validator: Optional[Callable[[Mapping[str, Any]], None]] = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> ConfigNode:
    """在不依赖 YAML 文件的情况下构建最终生效的配置。"""
    raw_cfg: Dict[str, Any] = {}

    if method_overrides is not None:
        raw_cfg = copy.deepcopy(dict(method_overrides))

    # 先应用算法专属默认值，再应用公共默认值。
    # 显式写入的内置配置优先级高于上述两层默认值。
    raw_cfg = _apply_method_defaults(raw_cfg, method_defaults)
    raw_cfg = _apply_defaults(raw_cfg)
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

    # 先应用算法专属默认值，再应用公共默认值。
    # YAML 中显式配置的优先级高于上述两层默认值。
    raw_cfg = _apply_method_defaults(raw_cfg, method_defaults)
    raw_cfg = _apply_defaults(raw_cfg)
    raw_cfg = _apply_config_overrides(raw_cfg, config_overrides)
    raw_cfg = _finalize_run_info(raw_cfg)
    _validate_config(raw_cfg)
    if method_validator is not None:
        method_validator(raw_cfg)
    return ConfigNode(raw_cfg)


def _build_method_config_record(
    raw: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    收集算法专属配置，不解析具体算法名称或其内部语义。

    新配置应将插件专属选项放入顶层通用配置 method_config。
    为兼容旧配置，未知的顶层字段也会保留在该映射中。

    公共框架字段以及派生字段、输入别名字段不会被收集。
    """
    framework_keys = set(DEFAULT_CONFIG)
    framework_keys.update(
        {
            "alpha",  # legacy alias for partition.dirichlet.alpha
            "data_augmentation",
            "grad_clip",
            "include",
            "input_shape",  # derived from the dataset unless explicitly set
            "lr",  # rejected legacy optimizer aliases
            "momentum",
            "num_classes",  # derived from the dataset unless explicitly set
            "partition_method",  # legacy alias for partition.method
            "run_dir",  # derived from output_dir and run_name
            "model",  # removed legacy selector; rejected by _validate_config
            "weight_decay",
        }
    )

    # 兼容旧版算法专属顶层字段，不在公共框架中绑定具体算法名称。
    method_config = {
        key: copy.deepcopy(value)
        for key, value in raw.items()
        if key not in framework_keys
    }

    explicit_method_config = raw.get("method_config", {})
    if explicit_method_config is None:
        explicit_method_config = {}
    if not isinstance(explicit_method_config, Mapping):
        raise ConfigError("method_config 必须是 dict。")

    # 显式配置的通用命名空间优先于旧版顶层字段。
    method_config.update(copy.deepcopy(dict(explicit_method_config)))
    return method_config


def _build_config_record(
    cfg: ConfigNode | Mapping[str, Any],
    model_builder: Optional[Callable[..., Any]] = None,
    resolved_device: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    构建便于阅读且能够复现实验的最终配置记录。

    记录最终生效的公共配置，包括默认值、显式覆盖值和由框架推导的
    DataLoader / 运行设备配置。算法插件专属配置仍统一放入
    aggregation.method_config。
    """
    if isinstance(cfg, ConfigNode):
        raw = cfg.to_dict()
    elif isinstance(cfg, Mapping):
        raw = copy.deepcopy(dict(cfg))
    else:
        raise TypeError("cfg 必须是 ConfigNode 或 Mapping。")

    def as_mapping(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    record: Dict[str, Any] = {}

    # 1. 数据配置
    data_cfg: Dict[str, Any] = {}
    for source_key, record_key in (
        ("dataset", "dataset"),
        ("data_root", "root"),
        ("num_clients", "clients"),
        ("frac", "participation"),
        ("num_workers", "workers"),
    ):
        value = raw.get(source_key, _default_config_value(source_key))
        if value is not None:
            data_cfg[record_key] = copy.deepcopy(value)

    data_cfg["augmentation"] = _cfg_get_bool(
        raw,
        "data_augmentation",
        _default_config_value("data_augmentation"),
    )
    data_cfg["download"] = _cfg_get_bool(
        raw,
        "download_data",
        _default_config_value("download_data"),
    )

    partition_cfg = _deep_merge(
        base=_default_config_value("partition"),
        override=as_mapping(raw.get("partition")),
    )
    partition_method = str(
        partition_cfg.get("method", _default_config_value("partition.method"))
    ).strip().lower()
    if partition_method:
        partition_record: Dict[str, Any] = {"method": partition_method}
        if partition_method == "dirichlet":
            dirichlet_cfg = as_mapping(partition_cfg.get("dirichlet"))
            alpha = dirichlet_cfg.get(
                "alpha",
                _default_config_value("partition.dirichlet.alpha"),
            )
            sample_balance = str(
                dirichlet_cfg.get(
                    "sample_balance",
                    _default_config_value("partition.dirichlet.sample_balance"),
                )
            ).strip().lower()
            partition_record["alpha"] = copy.deepcopy(alpha)
            partition_record["sample_balance"] = sample_balance
            partition_record["min_samples_per_client"] = int(
                raw.get(
                    "min_samples_per_client",
                    _default_config_value("min_samples_per_client"),
                )
            )
            if sample_balance == "unbalanced":
                partition_record["max_retries"] = int(
                    raw.get(
                        "partition_max_retries",
                        _default_config_value("partition_max_retries"),
                    )
                )
        data_cfg["partition"] = partition_record

    # 记录 DataLoader 的最终行为，而不是只记录用户显式传入的字段。
    # drop_last 仅控制客户端训练 loader；测试和 Evidence loader 固定为 False。
    num_workers = int(
        raw.get("num_workers", _default_config_value("num_workers")) or 0
    )
    data_cfg["loader"] = {
        "client_train_drop_last": _cfg_get_bool(
            raw,
            "drop_last",
            _default_config_value("drop_last"),
        ),
        "pin_memory": _infer_pin_memory(raw),
        "persistent_workers": num_workers > 0,
    }
    record["data"] = data_cfg

    # 2. 模型配置
    model_record: Dict[str, Any] = {}
    if model_builder is not None:
        builder_module = getattr(model_builder, "__module__", None)
        builder_name = getattr(
            model_builder,
            "__qualname__",
            getattr(model_builder, "__name__", type(model_builder).__qualname__),
        )
        model_record["builder"] = (
            f"{builder_module}.{builder_name}"
            if builder_module
            else str(builder_name)
        )
        model_record["architecture"] = (
            "SwitchViT"
            if model_builder is build_switch_vit_model
            else "custom"
        )

    model_record["num_experts"] = copy.deepcopy(
        raw.get("num_experts", _default_config_value("num_experts"))
    )
    dataset_name = normalize_dataset_name(
        raw.get("dataset", _default_config_value("dataset"))
    )
    dataset_info = DATASET_INFO.get(dataset_name)
    num_classes = raw.get("num_classes")
    if num_classes is None and dataset_info is not None:
        num_classes = int(dataset_info["num_classes"])
    if num_classes is not None:
        model_record["num_classes"] = copy.deepcopy(num_classes)
    input_shape = raw.get("input_shape")
    if input_shape is None and dataset_info is not None:
        input_shape = tuple(dataset_info["input_shape"])
    if input_shape is not None:
        model_record["input_shape"] = (
            list(input_shape)
            if isinstance(input_shape, (tuple, list))
            else copy.deepcopy(input_shape)
        )

    model_cfg = _deep_merge(
        base=_default_config_value("model_cfg"),
        override=as_mapping(raw.get("model_cfg")),
    )
    if model_cfg:
        model_record["config"] = copy.deepcopy(dict(model_cfg))
    if model_record:
        record["model"] = model_record

    # 3. 训练配置
    training_record: Dict[str, Any] = {}
    training_record["communication_rounds"] = copy.deepcopy(
        raw.get("rounds", _default_config_value("rounds"))
    )

    local_record: Dict[str, Any] = {}
    local_steps = raw.get("local_steps", _default_config_value("local_steps"))
    if local_steps is not None:
        local_record["steps"] = copy.deepcopy(local_steps)
    else:
        local_record["epochs"] = copy.deepcopy(
            raw.get("local_epochs", _default_config_value("local_epochs"))
        )

    local_record["batch_size"] = copy.deepcopy(
        raw.get("batch_size", _default_config_value("batch_size"))
    )
    training_record["local"] = local_record

    training_record["evaluation"] = {
        "batch_size": copy.deepcopy(
            raw.get("test_batch_size", _default_config_value("test_batch_size"))
        )
    }

    training_record["loss"] = {
        "name": "cross_entropy",
        "label_smoothing": copy.deepcopy(
            raw.get("label_smooth", _default_config_value("label_smooth"))
        ),
    }
    training_record["gradient_clip"] = copy.deepcopy(_get_grad_clip(raw))

    optimizer_cfg = _deep_merge(
        base=_default_config_value("optimizer"),
        override=as_mapping(raw.get("optimizer")),
    )
    if optimizer_cfg:
        optimizer_name = str(
            optimizer_cfg.get("type", _default_config_value("optimizer.type"))
        ).strip().lower()
        optimizer_record: Dict[str, Any] = {"name": optimizer_name}

        if optimizer_cfg.get("lr", _default_config_value("optimizer.lr")) is not None:
            optimizer_record["learning_rate"] = copy.deepcopy(optimizer_cfg["lr"])
        if "weight_decay" in optimizer_cfg:
            optimizer_record["weight_decay"] = copy.deepcopy(
                optimizer_cfg["weight_decay"]
            )

        if optimizer_name == "sgd":
            optimizer_record["momentum"] = copy.deepcopy(
                optimizer_cfg.get(
                    "momentum",
                    _default_config_value("optimizer.momentum"),
                )
            )
            optimizer_record["nesterov"] = _cfg_get_bool(
                optimizer_cfg,
                "nesterov",
                _default_config_value("optimizer.nesterov"),
            )
        elif optimizer_name in {"adam", "adamw"}:
            optimizer_record["betas"] = copy.deepcopy(
                list(
                    optimizer_cfg.get("betas", _default_config_value("optimizer.betas"))
                )
            )
            optimizer_record["eps"] = copy.deepcopy(
                optimizer_cfg.get("eps", _default_config_value("optimizer.eps"))
            )

        training_record["optimizer"] = optimizer_record

    record["training"] = training_record

    # 4. 聚合配置
    agg_cfg = _deep_merge(
        base=_default_config_value("agg"),
        override=as_mapping(raw.get("agg")),
    )
    aggregation_record: Dict[str, Any] = {}

    non_expert_cfg = as_mapping(
        agg_cfg.get("non_expert", _default_config_value("agg.non_expert"))
    )
    expert_cfg = as_mapping(
        agg_cfg.get("expert", _default_config_value("agg.expert"))
    )
    if non_expert_cfg.get("method") is not None:
        aggregation_record["non_expert"] = copy.deepcopy(
            non_expert_cfg["method"]
        )
    if expert_cfg.get("method") is not None:
        aggregation_record["expert"] = copy.deepcopy(expert_cfg["method"])

    method_config = _build_method_config_record(raw)
    if method_config:
        aggregation_record["method_config"] = method_config

    if aggregation_record:
        record["aggregation"] = aggregation_record

    # 5. 可复现性配置：False 也是有效实验设置，必须显式记录。
    record["reproducibility"] = {
        "seed": copy.deepcopy(raw.get("seed", _default_config_value("seed"))),
        "deterministic": _cfg_get_bool(
            raw,
            "deterministic",
            _default_config_value("deterministic"),
        ),
    }

    # 6. 运行环境：同时记录请求设备和本次实际解析出的设备。
    runtime_record: Dict[str, Any] = {
        "device": copy.deepcopy(
            raw.get("device", _default_config_value("device"))
        )
    }
    if resolved_device is not None:
        runtime_record["resolved_device"] = str(resolved_device)
    record["runtime"] = runtime_record

    # 7. Server Evidence：size=0 时也保留完整配置，明确表示该功能未启用。
    evidence_cfg = _deep_merge(
        base=_default_config_value("server_evidence"),
        override=as_mapping(raw.get("server_evidence")),
    )
    record["server_evidence"] = {
        "size": int(
            evidence_cfg.get("size", _default_config_value("server_evidence.size")) or 0
        ),
        "batch_size": copy.deepcopy(
            evidence_cfg.get(
                "batch_size",
                _default_config_value("server_evidence.batch_size"),
            )
        ),
        "class_balanced": _cfg_get_bool(
            evidence_cfg,
            "class_balanced",
            _default_config_value("server_evidence.class_balanced"),
        ),
    }

    # 8. 日志配置：True 和 False 均完整记录。
    logging_cfg = _deep_merge(
        base=_default_config_value("logging"),
        override=as_mapping(raw.get("logging")),
    )
    record["logging"] = {
        key: _cfg_get_bool(logging_cfg, key, _default_config_value(f"logging.{key}"))
        for key in ("save_config", "save_results_csv", "progress_bar")
    }

    # 9. 输出配置：包含实验命名策略，便于理解目录如何分配。
    run_cfg = _deep_merge(
        base=_default_config_value("run"),
        override=as_mapping(raw.get("run")),
    )
    auto_generated = _cfg_get_bool(run_cfg, "_auto_generated", False)
    unique_name = _cfg_get_bool(
        run_cfg,
        "unique_name",
        _default_config_value("run.unique_name"),
    )
    overwrite = _cfg_get_bool(
        run_cfg,
        "overwrite",
        _default_config_value("run.overwrite"),
    )
    output_record: Dict[str, Any] = {
        "root": str(raw.get("output_dir", _default_config_value("output_dir"))),
        "run_name": str(raw.get("run_name", _default_config_value("run_name"))),
    }
    output_record["auto_generated"] = auto_generated
    output_record["unique_name"] = unique_name
    output_record["effective_unique_name"] = (
        auto_generated or (unique_name and not overwrite)
    )
    output_record["overwrite"] = overwrite
    record["output"] = output_record

    return record


def save_config(
    cfg: ConfigNode | Mapping[str, Any],
    output_path: str | Path,
    model_builder: Optional[Callable[..., Any]] = None,
    resolved_device: Optional[Any] = None,
) -> None:
    """
    保存按功能分组的实验配置记录。

    生成的 config_used.yaml 用于查看和复现实验记录，
    不作为再次传入训练配置加载器的输入文件。模型构建器和解析后的设备
    由训练入口传入，以记录本次实际使用的模型构建接口与运行设备。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    record = _build_config_record(
        cfg,
        model_builder=model_builder,
        resolved_device=resolved_device,
    )
    yaml_text = yaml.safe_dump(
        record,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )

    # 各顶层配置分区之间保留空行，以提高可读性。
    yaml_text = re.sub(
        r"\n(?=[A-Za-z_][A-Za-z0-9_]*:\n)",
        "\n\n",
        yaml_text,
    )

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# 实验记录文件：仅用于查看，不作为训练输入配置。\n\n")
        f.write(yaml_text)


# 以下文件由实验运行器在每个实验目录中创建和管理。
# 覆盖实验目录时，用户自行创建的文件不会被删除。
RUN_OUTPUT_FILENAMES = (
    "train.log",
    "config_used.yaml",
    "partition_summary.json",
    "summary.json",
    "results.csv",
)


def _set_reserved_run_info(
    cfg: ConfigNode | MutableMapping[str, Any],
    run_name: str,
    run_dir: Path,
) -> None:
    """回写原子保留后的实验名称和目录，并标记目录已保留。"""
    if isinstance(cfg, ConfigNode):
        cfg.run_name = run_name
        cfg.run_dir = str(run_dir)
        run_cfg = cfg.get("run")
        if run_cfg is None:
            run_cfg = ConfigNode({})
            cfg.run = run_cfg
        if not isinstance(run_cfg, ConfigNode):
            raise ConfigError("run 必须是配置节点。")
        run_cfg._directory_reserved = True
        return

    if not isinstance(cfg, MutableMapping):
        raise ConfigError("原子分配实验目录时，配置对象必须可修改。")

    cfg["run_name"] = run_name
    cfg["run_dir"] = str(run_dir)
    run_cfg = cfg.get("run", _default_config_value("run"))
    if isinstance(run_cfg, Mapping) and not isinstance(run_cfg, MutableMapping):
        run_cfg = dict(run_cfg)
        cfg["run"] = run_cfg
    if not isinstance(run_cfg, MutableMapping):
        raise ConfigError("run 必须是可修改的 dict。")
    if "run" not in cfg:
        cfg["run"] = run_cfg
    run_cfg["_directory_reserved"] = True


def ensure_run_dir(cfg: ConfigNode | Mapping[str, Any]) -> Path:
    """
    原子创建实验目录，并在 run.overwrite=True 时重置框架管理的输出文件。

    自动名称以及启用 unique_name 的实验通过 mkdir(exist_ok=False) 原子保留目录。
    用户自行创建且不属于实验运行器管理范围的文件会被保留。
    """
    if isinstance(cfg, ConfigNode):
        run_dir = Path(cfg.run_dir)
        has_run_identity = hasattr(cfg, "output_dir") and hasattr(cfg, "run_name")
        output_dir = Path(getattr(cfg, "output_dir", run_dir.parent))
        run_name = str(getattr(cfg, "run_name", run_dir.name))
        overwrite = cfg.get("run.overwrite", _default_config_value("run.overwrite"))
        unique_name = cfg.get("run.unique_name", _default_config_value("run.unique_name"))
        auto_generated = cfg.get("run._auto_generated", False)
        directory_reserved = cfg.get("run._directory_reserved", False)
    else:
        run_dir = Path(cfg["run_dir"])
        has_run_identity = "output_dir" in cfg and "run_name" in cfg
        output_dir = Path(cfg.get("output_dir", run_dir.parent))
        run_name = str(cfg.get("run_name", run_dir.name))
        run_cfg = cfg.get("run", _default_config_value("run"))
        if not isinstance(run_cfg, Mapping):
            raise ConfigError("run 必须是 dict。")
        overwrite = run_cfg.get("overwrite", _default_config_value("run.overwrite"))
        unique_name = run_cfg.get("unique_name", _default_config_value("run.unique_name"))
        auto_generated = run_cfg.get("_auto_generated", False)
        directory_reserved = run_cfg.get("_directory_reserved", False)

    if has_run_identity and not isinstance(cfg, (ConfigNode, MutableMapping)):
        raise ConfigError("原子分配实验目录时，配置对象必须可修改。")
    if (
        isinstance(cfg, ConfigNode)
        and has_run_identity
        and cfg.get("run") is not None
        and not isinstance(cfg.get("run"), ConfigNode)
    ):
        raise ConfigError("run 必须是 dict。")

    _require_bool_value(overwrite, "run.overwrite")
    _require_bool_value(unique_name, "run.unique_name")
    _require_bool_value(auto_generated, "run._auto_generated")
    _require_bool_value(directory_reserved, "run._directory_reserved")

    if has_run_identity:
        # 同一个配置对象重复调用时，复用本进程已经保留的目录。
        if directory_reserved and run_dir.is_dir():
            pass
        elif auto_generated or (unique_name and not overwrite):
            # 自动名称始终唯一；手动名称仅在 unique_name=True 且
            # overwrite=False 时自动追加版本号。
            run_name, run_dir = _reserve_unique_run_dir(output_dir, run_name)
            _set_reserved_run_info(cfg, run_name, run_dir)
        elif not overwrite:
            # unique_name=False 且 overwrite=False：不追加版本号，
            # 但必须用独占创建避免并发进程同时复用同一目录。
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise ConfigError(
                    f"实验目录已存在：{run_dir}。"
                    "请启用 run.overwrite，或启用 run.unique_name "
                    "让框架自动创建新的实验目录。"
                ) from exc
            _set_reserved_run_info(cfg, run_name, run_dir)
        else:
            # 手动指定名称且 overwrite=True：按用户明确要求复用原目录。
            run_dir.mkdir(parents=True, exist_ok=True)
    else:
        # 兼容只提供 run_dir 的旧调用方式。
        run_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        output_paths = [
            run_dir / filename
            for filename in RUN_OUTPUT_FILENAMES
        ]

        # 删除文件前先执行预检查。保留路径若意外指向目录，
        # 不得对其执行递归删除。
        invalid_paths = [
            path
            for path in output_paths
            if path.exists()
            and not path.is_file()
            and not path.is_symlink()
        ]
        if invalid_paths:
            raise ConfigError(
                "以下实验输出路径存在但不是普通文件，拒绝覆盖："
                + ", ".join(str(path) for path in invalid_paths)
            )

        for path in output_paths:
            if path.is_file() or path.is_symlink():
                path.unlink()

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
          - model/switch_vit.yaml

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
    """应用算法专属默认值，同时保留用户显式指定的配置。"""
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
    """应用优先级最高的运行时覆盖项，例如命令行参数。"""
    if config_overrides is None:
        return copy.deepcopy(cfg)
    return _deep_merge(base=cfg, override=config_overrides)


def _apply_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """应用公共默认值，并将数据划分配置规范化为嵌套结构。"""
    explicit_cfg = copy.deepcopy(dict(cfg))
    cfg = _deep_merge(base=DEFAULT_CONFIG, override=explicit_cfg)

    # 标准配置结构：
    #   partition.method
    #   partition.dirichlet.alpha
    #   partition.dirichlet.sample_balance
    # 仍接受旧版顶层 partition_method / alpha 字段作为输入别名。
    explicit_partition = explicit_cfg.get("partition", {})
    if explicit_partition is None:
        explicit_partition = {}
    if not isinstance(explicit_partition, Mapping):
        raise ConfigError("partition 必须是 dict。")

    explicit_dirichlet = explicit_partition.get("dirichlet", {})
    if explicit_dirichlet is None:
        explicit_dirichlet = {}
    if not isinstance(explicit_dirichlet, Mapping):
        raise ConfigError("partition.dirichlet 必须是 dict。")

    default_partition = copy.deepcopy(DEFAULT_CONFIG["partition"])
    default_dirichlet = default_partition["dirichlet"]

    partition_method = explicit_partition.get(
        "method",
        explicit_cfg.get(
            "partition_method",
            default_partition["method"],
        ),
    )
    dirichlet_alpha = explicit_dirichlet.get(
        "alpha",
        explicit_cfg.get("alpha", default_dirichlet["alpha"]),
    )
    sample_balance = explicit_dirichlet.get(
        "sample_balance",
        default_dirichlet["sample_balance"],
    )

    normalized_partition = _deep_merge(
        base=default_partition,
        override=explicit_partition,
    )
    normalized_partition["method"] = str(partition_method).strip().lower()
    normalized_partition["dirichlet"] = _deep_merge(
        base=default_dirichlet,
        override=explicit_dirichlet,
    )
    normalized_partition["dirichlet"]["alpha"] = dirichlet_alpha
    normalized_partition["dirichlet"]["sample_balance"] = str(
        sample_balance
    ).strip().lower()
    cfg["partition"] = normalized_partition

    # 旧版字段仅用于兼容输入；保存和对外展示时只使用标准嵌套结构。
    cfg.pop("partition_method", None)
    cfg.pop("alpha", None)

    dataset_name = normalize_dataset_name(cfg["dataset"])
    cfg["dataset"] = dataset_name

    dataset_info = DATASET_INFO.get(dataset_name)
    if dataset_info is None:
        raise ConfigError(
            f"无法根据数据集 {dataset_name!r} 推断模型元信息。"
        )

    # 类别数和输入尺寸均由数据集注册信息决定；显式配置只允许与其一致。
    if cfg.get("num_classes") is None:
        cfg["num_classes"] = _infer_num_classes(dataset_name)
    if cfg.get("input_shape") is None:
        cfg["input_shape"] = tuple(dataset_info["input_shape"])

    return cfg


def _finalize_run_info(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成基础 run_name 和暂定 run_dir。

    规则：
    1. run_name 缺失 / 为空 / auto / null 时，使用专家聚合方法名。
    2. 自动名称始终唯一；目录名称在 ensure_run_dir 中通过原子创建确定。
    3. 手动名称的重名行为由 run.unique_name / run.overwrite 控制。
    """
    cfg = copy.deepcopy(cfg)

    raw_run_name = cfg.get("run_name", _default_config_value("run_name"))
    should_auto_name = _is_auto_run_name(raw_run_name)

    if should_auto_name:
        run_name = _build_auto_run_name(cfg)
    else:
        run_name = _safe_name(raw_run_name)

    output_dir = Path(cfg.get("output_dir", _default_config_value("output_dir")))
    run_cfg = cfg.get("run", _default_config_value("run"))
    if not isinstance(run_cfg, Mapping):
        raise ConfigError("run 必须是 dict。")

    unique_name = run_cfg.get("unique_name", _default_config_value("run.unique_name"))
    overwrite = run_cfg.get("overwrite", _default_config_value("run.overwrite"))

    _require_bool_value(unique_name, "run.unique_name")
    _require_bool_value(overwrite, "run.overwrite")

    # 只在配置中记录命名策略，不提前检查目录是否存在。
    # 真正创建目录时再执行原子保留，避免 exists() 与 mkdir() 之间的竞争窗口。
    run_state = dict(run_cfg)
    run_state["_auto_generated"] = should_auto_name
    run_state.pop("_directory_reserved", None)
    cfg["run"] = run_state

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
    自动实验名称只使用专家聚合方法名。

    完整实验参数统一保存在 config_used.yaml 中。
    """
    agg_cfg = cfg.get("agg", _default_config_value("agg"))
    expert_cfg = agg_cfg.get("expert", _default_config_value("agg.expert"))
    expert_method = expert_cfg.get("method", _default_config_value("agg.expert.method"))

    if not isinstance(expert_method, str) or not expert_method.strip():
        raise ConfigError(
            "自动生成 run_name 时，agg.expert.method 必须是非空字符串。"
        )

    return _safe_name(expert_method)


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


def _reserve_unique_run_dir(
    output_dir: Path,
    run_name: str,
) -> tuple[str, Path]:
    """
    通过原子目录创建保留唯一实验目录。

    依次尝试 run_name、run_name_v2、run_name_v3 等名称。
    mkdir(exist_ok=False) 将目录占用检查与创建合并为一个原子操作，
    避免并发实验进程选择到同一个目录。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    index = 1
    while True:
        candidate = (
            run_name
            if index == 1
            else f"{run_name}_v{index}"
        )
        run_dir = output_dir / candidate

        try:
            run_dir.mkdir(exist_ok=False)
        except FileExistsError:
            index += 1
            continue

        return candidate, run_dir


def _infer_num_classes(dataset: str) -> int:
    """根据数据集注册信息推断类别数量。"""
    dataset_name = normalize_dataset_name(dataset)
    info = DATASET_INFO.get(dataset_name)
    if info is None:
        raise ConfigError(
            f"无法根据数据集 {dataset_name!r} 推断类别数量。"
        )
    return int(info["num_classes"])


def _require_bool_value(
    value: Any,
    key: str,
    *,
    allow_none: bool = False,
) -> None:
    """严格校验配置值为 bool；仅显式声明可选的字段允许 None。"""
    if value is None and allow_none:
        return
    if type(value) is not bool:
        expected = "bool 或 None" if allow_none else "bool"
        raise ConfigError(
            f"{key} 必须是 {expected}，当前值：{value!r}"
            f"（类型：{type(value).__name__}）。"
        )


def _validate_logging_config(cfg: Mapping[str, Any]) -> None:
    """
    校验范围受控的公共日志配置项。

    目前只有三个日志选项允许通过配置修改。
    对旧版配置中已固定或不再支持的字段显式报错，
    避免过期 YAML 文件静默改变框架的固定日志行为。
    """
    logging_cfg = cfg.get("logging", _default_config_value("logging"))
    if not isinstance(logging_cfg, Mapping):
        raise ConfigError("logging 必须是 dict。")

    deprecated_keys = {
        "log_every",
        "progress_in_non_tty",
        "console_round_summary",
        "file_round_detail",
        "log_round_clients",
        "log_client_table",
        "log_client_metrics",
        "log_agg_weights",
        "compact_uniform_weights",
    }
    configured_deprecated = sorted(
        set(logging_cfg.keys()).intersection(deprecated_keys)
    )
    if configured_deprecated:
        fields = ", ".join(configured_deprecated)
        raise ConfigError(
            "logging 中包含已废弃的配置项："
            f"{fields}。这些行为现在由框架固定，不再支持通过 YAML 或命令行修改；"
            "请从配置文件中删除这些字段。"
        )

    supported_keys = {
        "save_config",
        "save_results_csv",
        "progress_bar",
    }
    for key in supported_keys:
        _require_bool_value(
            logging_cfg.get(key, _default_config_value(f"logging.{key}")),
            f"logging.{key}",
        )

    unsupported_keys = sorted(set(logging_cfg.keys()) - supported_keys)
    if unsupported_keys:
        fields = ", ".join(str(key) for key in unsupported_keys)
        supported = ", ".join(sorted(supported_keys))
        raise ConfigError(
            "logging 只允许配置以下字段："
            f"{supported}。检测到不支持的字段：{fields}。"
        )


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

    if "model" in cfg:
        raise ConfigError(
            "顶层 model 选择字段已移除；当前模型由 model_builder 决定。"
        )

    model_cfg = cfg.get("model_cfg", _default_config_value("model_cfg"))
    if not isinstance(model_cfg, Mapping):
        raise ConfigError("model_cfg 必须是 dict。")
    effective_model_cfg = _deep_merge(
        base=_default_config_value("model_cfg"),
        override=dict(model_cfg),
    )

    for key in (
        "data_augmentation",
        "download_data",
        "drop_last",
        "deterministic",
    ):
        if key in cfg:
            _require_bool_value(cfg[key], key)
    _require_bool_value(
        cfg.get("pin_memory", _default_config_value("pin_memory")),
        "pin_memory",
        allow_none=True,
    )
    for key in ("token_dropping", "router_statistics"):
        if key in model_cfg:
            _require_bool_value(model_cfg[key], f"model_cfg.{key}")

    removed_model_fields = {"backbone", "moe_hidden_dim"}.intersection(model_cfg)
    if removed_model_fields:
        raise ConfigError(
            "旧模型配置字段已移除："
            f"{sorted(removed_model_fields)}。当前使用 SwitchViT 的 model_cfg。"
        )

    method_config = cfg.get(
        "method_config",
        _default_config_value("method_config"),
    )
    if not isinstance(method_config, Mapping):
        raise ConfigError("method_config 必须是 dict。")

    agg_cfg = cfg.get("agg", _default_config_value("agg"))
    if isinstance(agg_cfg, Mapping):
        agg_cfg = _deep_merge(
            base=_default_config_value("agg"),
            override=agg_cfg,
        )
    non_expert_cfg = agg_cfg.get(
        "non_expert", _default_config_value("agg.non_expert")
    )
    expert_cfg = agg_cfg.get(
        "expert", _default_config_value("agg.expert")
    )
    non_expert_method = non_expert_cfg.get("method")
    expert_method = expert_cfg.get("method")

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
    dataset_info = DATASET_INFO.get(dataset)
    if dataset_info is None:
        raise ConfigError(
            f"数据集 {dataset!r} 缺少 DATASET_INFO 注册信息。"
        )

    expected_num_classes = int(dataset_info["num_classes"])
    if cfg["num_classes"] != expected_num_classes:
        raise ConfigError(
            f"num_classes 与数据集 {dataset!r} 不一致："
            f"配置值为 {cfg['num_classes']}，"
            f"数据集注册值为 {expected_num_classes}。"
            "请删除 num_classes 覆盖项，或改为数据集对应的类别数。"
        )

    expected_input_shape = tuple(dataset_info["input_shape"])
    input_shape = cfg.get("input_shape")
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 3:
        raise ConfigError(
            "input_shape 必须是包含 channels、height、width 的三元素 list/tuple，"
            f"当前值：{input_shape!r}。"
        )
    if any(not _is_positive_int_value(value) for value in input_shape):
        raise ConfigError(
            "input_shape 的 channels、height、width 必须全部是严格正整数，"
            f"当前值：{input_shape!r}。"
        )
    if tuple(input_shape) != expected_input_shape:
        raise ConfigError(
            f"input_shape 与数据集 {dataset!r} 不一致："
            f"配置值为 {tuple(input_shape)}，"
            f"数据集注册值为 {expected_input_shape}。"
            "请删除 input_shape 覆盖项，使用数据集默认输入尺寸。"
        )

    _validate_switch_model_config(
        model_cfg=effective_model_cfg,
        input_shape=expected_input_shape,
    )

    _require_positive_int(cfg, "num_clients")
    _require_positive_int(cfg, "rounds")
    _require_non_negative_int(cfg, "local_epochs")
    local_steps = cfg.get("local_steps", _default_config_value("local_steps"))
    if local_steps is None:
        if int(cfg.get("local_epochs", _default_config_value("local_epochs"))) <= 0:
            raise ConfigError(
                "local_steps 未设置时，local_epochs 必须是正整数。"
            )
    else:
        _require_positive_int(cfg, "local_steps")
    _require_positive_int(cfg, "batch_size")
    _require_positive_int(cfg, "test_batch_size")
    _require_non_negative_int(cfg, "num_workers")
    _require_positive_int(cfg, "num_experts")

    server_evidence_cfg = cfg.get(
        "server_evidence", _default_config_value("server_evidence")
    )
    if not isinstance(server_evidence_cfg, Mapping):
        raise ConfigError("server_evidence 必须是 dict。")

    _require_bool_value(
        server_evidence_cfg.get(
            "class_balanced",
            _default_config_value("server_evidence.class_balanced"),
        ),
        "server_evidence.class_balanced",
    )

    server_evidence_size = server_evidence_cfg.get(
        "size", _default_config_value("server_evidence.size")
    )
    if (
        isinstance(server_evidence_size, bool)
        or not isinstance(server_evidence_size, int)
        or server_evidence_size < 0
    ):
        raise ConfigError(
            "server_evidence.size 必须是非负整数，"
            f"当前值：{server_evidence_size}"
        )

    server_evidence_batch_size = server_evidence_cfg.get(
        "batch_size", _default_config_value("server_evidence.batch_size")
    )
    if (
        isinstance(server_evidence_batch_size, bool)
        or not isinstance(server_evidence_batch_size, int)
        or server_evidence_batch_size <= 0
    ):
        raise ConfigError(
            "server_evidence.batch_size 必须是正整数，"
            f"当前值：{server_evidence_batch_size}"
        )

    frac = float(cfg.get("frac", _default_config_value("frac")))
    if not (0.0 < frac <= 1.0):
        raise ConfigError(f"frac 必须在 (0, 1] 范围内，当前值：{frac}")

    partition_cfg = cfg.get(
        "partition", _default_config_value("partition")
    )
    if not isinstance(partition_cfg, Mapping):
        raise ConfigError("partition 必须是 dict。")

    partition_method = str(
        partition_cfg.get("method", _default_config_value("partition.method"))
    ).strip().lower()
    if partition_method not in {"iid", "dirichlet"}:
        raise ConfigError(
            f"不支持的数据划分方法：{partition_method}。当前支持：iid, dirichlet"
        )

    dirichlet_cfg = partition_cfg.get(
        "dirichlet", _default_config_value("partition.dirichlet")
    )
    if not isinstance(dirichlet_cfg, Mapping):
        raise ConfigError("partition.dirichlet 必须是 dict。")

    if partition_method == "dirichlet":
        alpha = float(
            dirichlet_cfg.get(
                "alpha", _default_config_value("partition.dirichlet.alpha")
            )
        )
        if not math.isfinite(alpha) or alpha <= 0:
            raise ConfigError(
                "partition.dirichlet.alpha 必须是有限的正数，"
                f"当前值：{alpha}"
            )

        sample_balance = str(
            dirichlet_cfg.get(
                "sample_balance",
                _default_config_value("partition.dirichlet.sample_balance"),
            )
        ).strip().lower()
        if sample_balance not in {"balanced", "unbalanced"}:
            raise ConfigError(
                "partition.dirichlet.sample_balance 只能是 balanced 或 unbalanced，"
                f"当前值：{sample_balance!r}"
            )

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

    optimizer_cfg = cfg.get("optimizer", _default_config_value("optimizer"))
    if not isinstance(optimizer_cfg, Mapping):
        raise ConfigError("optimizer 必须是 dict。")
    _require_bool_value(
        optimizer_cfg.get("nesterov", _default_config_value("optimizer.nesterov")),
        "optimizer.nesterov",
    )
    optimizer_type = optimizer_cfg.get("type", _default_config_value("optimizer.type"))

    if optimizer_type not in {"sgd", "adam", "adamw"}:
        raise ConfigError(
            f"不支持的优化器：{optimizer_type}。"
            f"当前支持：sgd, adam, adamw"
        )

    lr = float(optimizer_cfg.get("lr", _default_config_value("optimizer.lr")))
    if not math.isfinite(lr) or lr <= 0:
        raise ConfigError(
            f"optimizer.lr 必须是有限的正数，当前值：{lr}"
        )

    weight_decay = float(
        optimizer_cfg.get(
            "weight_decay",
            _default_config_value("optimizer.weight_decay"),
        )
    )
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ConfigError(
            f"optimizer.weight_decay 必须是有限的非负数，当前值：{weight_decay}"
        )

    run_cfg = cfg.get("run", _default_config_value("run"))
    if not isinstance(run_cfg, Mapping):
        raise ConfigError("run 必须是 dict。")
    _require_bool_value(
        run_cfg.get("unique_name", _default_config_value("run.unique_name")),
        "run.unique_name",
    )
    _require_bool_value(
        run_cfg.get("overwrite", _default_config_value("run.overwrite")),
        "run.overwrite",
    )

    _validate_logging_config(cfg)


def _is_positive_int_value(value: Any) -> bool:
    """判断值是否为严格意义上的正整数（不接受 bool）。"""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def _require_positive_int(cfg: Mapping[str, Any], key: str) -> None:
    """检查某个字段是否为正整数。"""
    value = cfg.get(key)

    if not _is_positive_int_value(value):
        raise ConfigError(f"{key} 必须是正整数，当前值：{value}")


def _require_non_negative_int(cfg: Mapping[str, Any], key: str) -> None:
    """检查某个字段是否为非负整数。"""
    value = cfg.get(key)

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{key} 必须是非负整数，当前值：{value}")


def _validate_switch_model_config(
    model_cfg: Mapping[str, Any],
    input_shape: Sequence[int],
) -> None:
    """严格校验 SwitchViT 的整数结构参数及其组合约束。"""
    if not isinstance(model_cfg, Mapping):
        raise ConfigError("model_cfg 必须是 dict。")

    integer_fields = (
        "transformer_layers",
        "hidden_dim",
        "attention_heads",
        "patch_size",
        "expert_hidden_dim",
    )
    for key in integer_fields:
        value = model_cfg.get(key)
        if not _is_positive_int_value(value):
            raise ConfigError(
                f"model_cfg.{key} 必须是严格正整数，"
                f"当前值：{value!r}（类型：{type(value).__name__}）。"
            )

    transformer_layers = model_cfg["transformer_layers"]
    hidden_dim = model_cfg["hidden_dim"]
    attention_heads = model_cfg["attention_heads"]
    patch_size = model_cfg["patch_size"]

    if hidden_dim % attention_heads != 0:
        raise ConfigError(
            "model_cfg.hidden_dim 必须能被 model_cfg.attention_heads 整除，"
            f"当前值：hidden_dim={hidden_dim}, "
            f"attention_heads={attention_heads}。"
        )

    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 3:
        raise ConfigError(
            "模型 input_shape 必须是包含 channels、height、width 的三元素序列。"
        )
    if any(not _is_positive_int_value(value) for value in input_shape):
        raise ConfigError(
            f"模型 input_shape 必须全部由严格正整数组成，当前值：{input_shape!r}。"
        )

    image_height, image_width = input_shape[1], input_shape[2]
    if image_height % patch_size != 0 or image_width % patch_size != 0:
        raise ConfigError(
            "数据集图像尺寸必须能被 model_cfg.patch_size 整除，"
            f"当前值：image_size={(image_height, image_width)}, "
            f"patch_size={patch_size}。"
        )

    moe_block_indices = model_cfg.get("moe_block_indices")
    if not isinstance(moe_block_indices, (list, tuple)) or not moe_block_indices:
        raise ConfigError(
            "model_cfg.moe_block_indices 必须是非空 list/tuple。"
        )
    if any(not _is_positive_int_value(index) for index in moe_block_indices):
        raise ConfigError(
            "model_cfg.moe_block_indices 中的每个层号都必须是严格正整数，"
            f"当前值：{moe_block_indices!r}。"
        )
    if len(set(moe_block_indices)) != len(moe_block_indices):
        raise ConfigError(
            "model_cfg.moe_block_indices 不允许出现重复层号，"
            f"当前值：{moe_block_indices!r}。"
        )
    invalid_indices = [
        index
        for index in moe_block_indices
        if index > transformer_layers
    ]
    if invalid_indices:
        raise ConfigError(
            "model_cfg.moe_block_indices 必须位于 "
            f"1 到 transformer_layers={transformer_layers} 的范围内，"
            f"非法层号：{invalid_indices}。"
        )


# 各公共功能模块统一通过此接口读取配置。
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """从 ConfigNode、dict 或普通对象中读取普通字段或点分隔路径字段。"""
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
# 2. 运行环境与随机种子（Runtime & Seed）
# 包含确定性设置、随机种子管理及 DataLoader 随机数生成器。
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
    deterministic: bool = DEFAULT_CONFIG["deterministic"],
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
# 3. 日志与终端输出（Logging）
# 包含控制台输出、日志文件写入及进度条过滤。
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

    # 每次训练启动时都创建新的日志文件，
    # 避免复用或覆盖实验目录后残留旧日志。
    with log_path.open(
        "w",
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
# 4. 联邦学习公共工具（FL Common Utilities）
# 4.1 State Dict 操作、参数差分与权重处理
# ============================================================================


StateDict = Mapping[str, torch.Tensor]
MutableStateDict = Dict[str, torch.Tensor]


def validate_state_dict_shapes(
    reference_state: StateDict,
    candidate_state: StateDict,
    *,
    context: str = "state_dict",
) -> None:
    """
    验证两个 state_dict 的参数名称和 Tensor Shape 完全一致。

    该检查用于聚合边界，避免不同客户端或自定义聚合器返回的错误维度
    借助 PyTorch broadcasting 进入后续计算。
    """
    if not isinstance(reference_state, Mapping):
        raise TypeError(
            f"{context} 的参考 state_dict 必须是 Mapping，"
            f"当前类型：{type(reference_state).__name__}"
        )
    if not isinstance(candidate_state, Mapping):
        raise TypeError(
            f"{context} 必须是 Mapping，"
            f"当前类型：{type(candidate_state).__name__}"
        )

    expected_names = set(reference_state)
    actual_names = set(candidate_state)
    if actual_names != expected_names:
        missing_names = sorted(expected_names - actual_names)
        extra_names = sorted(actual_names - expected_names)
        raise ValueError(
            f"{context} 的参数名称与参考 state_dict 不一致。"
            f" missing={missing_names[:10]}, extra={extra_names[:10]}"
        )

    for name, reference_tensor in reference_state.items():
        candidate_tensor = candidate_state[name]
        if not torch.is_tensor(reference_tensor):
            raise TypeError(
                f"参考 state_dict 中的 {name} 必须是 Tensor，"
                f"当前类型：{type(reference_tensor).__name__}"
            )
        if not torch.is_tensor(candidate_tensor):
            raise TypeError(
                f"{context} 中的 {name} 必须是 Tensor，"
                f"当前类型：{type(candidate_tensor).__name__}"
            )
        if tuple(candidate_tensor.shape) != tuple(reference_tensor.shape):
            raise ValueError(
                f"{context} 中参数 {name} 的 Tensor Shape 不匹配："
                f"expected={tuple(reference_tensor.shape)}, "
                f"actual={tuple(candidate_tensor.shape)}"
            )


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

        if not torch.is_tensor(local_tensor):
            raise TypeError(
                f"local_state 中的 {name} 必须是 Tensor，"
                f"当前类型：{type(local_tensor).__name__}"
            )
        if not torch.is_tensor(global_tensor):
            raise TypeError(
                f"global_state 中的 {name} 必须是 Tensor，"
                f"当前类型：{type(global_tensor).__name__}"
            )
        if tuple(local_tensor.shape) != tuple(global_tensor.shape):
            raise ValueError(
                f"本地模型与全局模型的 Tensor Shape 不一致：参数 {name}，"
                f"expected={tuple(global_tensor.shape)}, "
                f"actual={tuple(local_tensor.shape)}"
            )

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
        validate_state_dict_shapes(
            reference_state=global_state,
            candidate_state=base_state,
            context="加权聚合 base_state",
        )
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
            delta_value = model_delta[name]
            if not torch.is_tensor(delta_value):
                raise TypeError(
                    f"客户端 {client_id} 的 model_delta[{name}] 必须是 Tensor，"
                    f"当前类型：{type(delta_value).__name__}"
                )
            if tuple(delta_value.shape) != tuple(global_tensor.shape):
                raise ValueError(
                    f"客户端 {client_id} 的 model_delta 参数 {name} "
                    f"Tensor Shape 不匹配："
                    f"expected={tuple(global_tensor.shape)}, "
                    f"actual={tuple(delta_value.shape)}"
                )
            delta_tensor = delta_value.to(global_tensor.device)

            total_delta = total_delta + weight * delta_tensor

        new_state[name] = global_tensor + total_delta

    validate_state_dict_shapes(
        reference_state=global_state,
        candidate_state=new_state,
        context="加权聚合结果",
    )
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

    # 多个有限非负权重的和仍可能溢出为 inf。
    # 仅在溢出时缩放，保持正常权重的原有计算顺序及结果。
    if not math.isfinite(total):
        scale = max(float(weight) for weight in weights.values())
        scaled_weights = {
            int(client_id): float(weight) / scale
            for client_id, weight in weights.items()
        }
        scaled_total = sum(scaled_weights.values())
        return {
            client_id: value / scaled_total
            for client_id, value in scaled_weights.items()
        }

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
# 4.2 模型评估、Batch 解析与 logits 提取
# ============================================================================


@dataclass
class ModelOutput:
    """
    模型前向传播的统一输出接口。

    logits:
        分类任务输出，形状通常为 [batch_size, num_classes]。

    aux_loss:
        已经完成层内/层间加权的辅助损失标量。
        模型负责计算辅助损失及其权重，联邦训练框架只负责将其加入 task loss。

    aux_loss_by_layer:
        可选的逐层辅助损失，主要用于后续诊断；不参与二次加权。

    moe_stats:
        可选的 MoE 路由统计。应当是轻量统计信息，不应保存计算图。
        统计值应使用可跨 batch/client 累加的计数或求和量（例如 tokens_per_expert、router_prob_sum）；
        不要直接传入已经求平均的比例或均值。框架会在客户端和服务端逐级累加。
    """

    logits: torch.Tensor
    aux_loss: Optional[torch.Tensor] = None
    aux_loss_by_layer: Dict[str, torch.Tensor] = field(default_factory=dict)
    moe_stats: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.logits):
            raise TypeError("ModelOutput.logits 必须是 torch.Tensor。")

        if self.aux_loss is not None and not torch.is_tensor(self.aux_loss):
            raise TypeError("ModelOutput.aux_loss 必须是 torch.Tensor 或 None。")

        if not isinstance(self.aux_loss_by_layer, Mapping):
            raise TypeError("ModelOutput.aux_loss_by_layer 必须是 Mapping。")

        if not isinstance(self.moe_stats, Mapping):
            raise TypeError("ModelOutput.moe_stats 必须是 Mapping。")

        self.aux_loss_by_layer = dict(self.aux_loss_by_layer)
        self.moe_stats = dict(self.moe_stats)


@dataclass(frozen=True)
class ModelLoss:
    """一次前向传播对应的分类损失、辅助损失和总损失。"""

    total_loss: torch.Tensor
    task_loss: torch.Tensor
    aux_loss: torch.Tensor


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

    保留旧接口，兼容：
        1. 直接返回 logits tensor
        2. 返回对象，且对象有 .logits
        3. 返回 dict，且 dict["logits"]
        4. 返回 tuple/list，默认第一个元素是 logits
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


def extract_model_output(outputs: Any) -> ModelOutput:
    """
    将新旧模型的前向输出统一转换为 ModelOutput。

    旧模型只返回 Tensor、Tuple 或仅包含 logits 的 Dict 时，
    aux_loss 自动视为 None，不改变原有训练行为。
    """
    if isinstance(outputs, ModelOutput):
        return outputs

    logits = extract_logits(outputs)
    aux_loss = None
    aux_loss_by_layer: Mapping[str, torch.Tensor] = {}
    moe_stats: Mapping[str, Any] = {}

    if isinstance(outputs, Mapping):
        aux_loss = outputs.get("aux_loss")
        aux_loss_by_layer = outputs.get("aux_loss_by_layer", {})
        moe_stats = outputs.get("moe_stats", {})
    elif hasattr(outputs, "logits"):
        aux_loss = getattr(outputs, "aux_loss", None)
        aux_loss_by_layer = getattr(outputs, "aux_loss_by_layer", {})
        moe_stats = getattr(outputs, "moe_stats", {})

    if aux_loss_by_layer is None:
        aux_loss_by_layer = {}
    if moe_stats is None:
        moe_stats = {}

    return ModelOutput(
        logits=logits,
        aux_loss=aux_loss,
        aux_loss_by_layer=aux_loss_by_layer,
        moe_stats=moe_stats,
    )


def compute_model_loss(
    outputs: Any,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> ModelLoss:
    """
    统一计算模型训练损失。

    total_loss = task_loss + aux_loss

    aux_loss 必须是模型返回的最终加权标量。框架不再对它乘权重，
    避免模型层和联邦训练层重复加权。没有辅助损失时使用与 task_loss
    同设备、同 dtype 的零标量，保证普通分类模型行为不变。
    """
    model_output = extract_model_output(outputs)
    task_loss = criterion(model_output.logits, targets)

    if not torch.is_tensor(task_loss):
        raise TypeError("criterion 必须返回 torch.Tensor。")
    if task_loss.numel() != 1:
        raise ValueError(
            "criterion 必须返回标量损失；"
            f"当前 task_loss shape={tuple(task_loss.shape)}"
        )
    task_loss = task_loss.reshape(())

    if model_output.aux_loss is None:
        aux_loss = task_loss.new_zeros(())
    else:
        aux_loss = model_output.aux_loss
        if aux_loss.numel() != 1:
            raise ValueError(
                "ModelOutput.aux_loss 必须是标量；"
                f"当前 aux_loss shape={tuple(aux_loss.shape)}"
            )
        if not aux_loss.is_floating_point():
            raise TypeError("ModelOutput.aux_loss 必须是浮点 Tensor。")
        if aux_loss.device != task_loss.device:
            raise ValueError(
                "task_loss 和 aux_loss 必须位于同一设备；"
                f"task_loss={task_loss.device}, aux_loss={aux_loss.device}"
            )
        aux_loss = aux_loss.reshape(())

    return ModelLoss(
        total_loss=task_loss + aux_loss,
        task_loss=task_loss,
        aux_loss=aux_loss,
    )


def _normalize_moe_stat_value(value: Any, path: str = "moe_stats") -> Any:
    """分离运行时 MoE 统计数据，并转换为可安全写入 JSON 的 Python 类型。"""
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.is_complex():
            raise TypeError(f"{path} 不支持复数 Tensor。")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{path} 包含 NaN 或 Inf。")
        if tensor.numel() == 1:
            return tensor.item()
        return tensor.tolist()
    if isinstance(value, np.ndarray):
        return _normalize_moe_stat_value(value.tolist(), path=path)
    if isinstance(value, np.generic):
        return _normalize_moe_stat_value(value.item(), path=path)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for key, item in value.items():
            safe_key = str(key)
            if safe_key in normalized:
                raise ValueError(f"{path} 存在转换后重复的键：{safe_key!r}")
            normalized[safe_key] = _normalize_moe_stat_value(
                item, path=f"{path}.{safe_key}"
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_moe_stat_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} 包含 NaN 或 Inf。")
        return float(value)
    raise TypeError(f"{path} 包含不支持的统计类型：{type(value).__name__}")


def _normalize_moe_stats(stats: Mapping[str, Any] | None) -> Dict[str, Any]:
    """将 ModelOutput.moe_stats 映射规范化为已分离的 Python 数据。"""
    if stats is None:
        return {}
    if not isinstance(stats, Mapping):
        raise TypeError("ModelOutput.moe_stats 必须是 Mapping。")
    return dict(_normalize_moe_stat_value(stats))


def _merge_additive_moe_stats(total: Any, current: Any, path: str = "moe_stats") -> Any:
    """
    累加本地 Batch 和客户端产生的 MoE 统计数据。

    数值标量和序列按元素求和；重复出现的字符串或布尔元数据必须保持一致。
    模型应在 moe_stats 中提供计数值或累加值（例如 tokens_per_expert、
    router_prob_sum），而不是预先计算的平均值。
    """
    if total is None:
        return copy.deepcopy(current)
    if current is None:
        return copy.deepcopy(total)
    if isinstance(total, Mapping) and isinstance(current, Mapping):
        merged = copy.deepcopy(dict(total))
        for key, value in current.items():
            if key in merged:
                merged[key] = _merge_additive_moe_stats(
                    merged[key], value, path=f"{path}.{key}"
                )
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    if isinstance(total, list) and isinstance(current, list):
        if len(total) != len(current):
            raise ValueError(
                f"{path} 的序列长度在不同 batch/client 间不一致："
                f"{len(total)} != {len(current)}"
            )
        return [
            _merge_additive_moe_stats(left, right, path=f"{path}[{index}]")
            for index, (left, right) in enumerate(zip(total, current))
        ]
    if (
        isinstance(total, (int, float)) and not isinstance(total, bool)
        and isinstance(current, (int, float)) and not isinstance(current, bool)
    ):
        value = total + current
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path} 累加后出现 NaN 或 Inf。")
        return value
    if isinstance(total, bool) and isinstance(current, bool) and total == current:
        return total
    if isinstance(total, str) and isinstance(current, str) and total == current:
        return total
    raise TypeError(
        f"{path} 在不同 batch/client 间的数据类型或元数据不一致："
        f"{type(total).__name__} vs {type(current).__name__}"
    )


def _extract_aux_loss_by_layer(values: Mapping[str, Any] | None) -> Dict[str, float]:
    """校验并分离各层的标量辅助损失。"""
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("ModelOutput.aux_loss_by_layer 必须是 Mapping。")
    result: Dict[str, float] = {}
    for key, value in values.items():
        layer_name = str(key)
        if torch.is_tensor(value):
            if value.numel() != 1 or not value.is_floating_point():
                raise ValueError(
                    f"aux_loss_by_layer[{layer_name!r}] 必须是浮点标量 Tensor。"
                )
            scalar = float(value.detach().cpu().item())
        elif isinstance(value, np.generic):
            scalar = float(value.item())
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            scalar = float(value)
        else:
            raise TypeError(
                f"aux_loss_by_layer[{layer_name!r}] 必须是数值或浮点标量 Tensor。"
            )
        if not math.isfinite(scalar):
            raise ValueError(f"aux_loss_by_layer[{layer_name!r}] 包含 NaN 或 Inf。")
        result[layer_name] = scalar
    return result


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


# ============================================================================
# 5. 数据集（Dataset）
# 包含数据集注册、构建、数据变换及服务端 Evidence 数据。
# ============================================================================


# ------------------------------------------------------------------------------
# 5.1 数据集元信息注册区
# ------------------------------------------------------------------------------
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
        # LEAF FEMNIST 的像素值范围为 [0, 1]，此处保持原始数值范围。
        "mean": (0.0,),
        "std": (1.0,),
    },
}



class _ArrayImageDataset(Dataset):
    """使用 uint8 NumPy 数组和整数标签存储图像的数据集。"""

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
    """直接读取 Tiny-ImageNet 原始验证集目录结构，不要求重新整理文件。"""

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
    """根据常见目录结构定位用户手动下载的数据集目录。"""
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
    """定位 LEAF FEMNIST 预处理后的 train/test JSON 目录。"""
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
    """将 LEAF 某个数据划分中的所有用户样本合并为集中式 uint8 图像和标签数组。"""
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
                if (float(array.max()) if array.size else 0.0) <= 1.0:
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
    """为当前基准数据集构建集中式训练、Evidence 和测试数据集。"""
    dataset_name = normalize_dataset_name(str(cfg.dataset))
    data_root = Path(cfg.data_root).expanduser()

    if dataset_name not in DATASET_INFO:
        raise ValueError(
            f"不支持的数据集：{dataset_name}。当前支持：{sorted(DATASET_INFO.keys())}"
        )

    info = DATASET_INFO[dataset_name]
    train_transform = build_train_transform(
        dataset_name=dataset_name,
        use_augmentation=_cfg_get_bool(cfg, "data_augmentation", True),
    )
    train_evidence_transform = build_train_transform(
        dataset_name=dataset_name,
        use_augmentation=False,
    )
    test_transform = build_test_transform(dataset_name=dataset_name)
    download = _cfg_get_bool(cfg, "download_data", True)

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
            # 同时兼容用户已按类别目录重新整理的验证集。
            test_dataset = datasets.ImageFolder(val_root, transform=test_transform)

    elif dataset_name == "femnist":
        # 方案 A：加载后不再区分书写者身份，
        # 直接复用本模块后续的集中式 IID/Dirichlet 划分逻辑。
        root = _resolve_femnist_root(data_root)
        train_data, train_targets = _load_femnist_split_storage(root / "train")
        test_data, test_targets = _load_femnist_split_storage(root / "test")

        train_dataset = _ArrayImageDataset(
            data=train_data,
            targets=train_targets,
            transform=train_transform,
            image_mode="L",
        )
        # 训练集与 Evidence 数据集共享 uint8 底层数组，仅数据变换不同。
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
    可选地从训练数据中划分确定性的服务端 Evidence 子集。

    默认 server_evidence.size=0，此时不执行任何划分，直接返回原始 train 和
    train_evidence 数据集对象。启用后，相同索引会从客户端训练数据视图中移除，
    保留的子集则从确定性的 train_evidence_dataset 视图中提取。
    """
    evidence_cfg = _cfg_get(
        cfg,
        "server_evidence",
        _default_config_value("server_evidence"),
    )
    evidence_size = int(
        _cfg_get(evidence_cfg, "size", _default_config_value("server_evidence.size"))
    )

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

    num_clients = int(
        _cfg_get(cfg, "num_clients", _default_config_value("num_clients"))
    )
    if train_size - evidence_size < num_clients:
        raise ValueError(
            "server evidence holdout 后剩余训练样本少于客户端数量："
            f"remaining={train_size - evidence_size}, num_clients={num_clients}"
        )

    targets = get_dataset_targets(bundle.train_evidence_dataset)
    class_balanced = _cfg_get_bool(
        evidence_cfg,
        "class_balanced",
        _default_config_value("server_evidence.class_balanced"),
    )
    seed = int(_cfg_get(cfg, "seed", _default_config_value("seed")))

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
    """在不影响全局随机数状态的前提下，选择确定性的留出样本索引。"""
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
    """根据数据集类型构建训练变换，同时保持 CIFAR 原有处理流程。"""
    dataset_name = normalize_dataset_name(dataset_name)
    mean, std = get_normalization_stats(dataset_name)
    input_shape = tuple(DATASET_INFO[dataset_name]["input_shape"])
    image_size = int(input_shape[1])

    transform_list: List[Callable] = []

    if use_augmentation:
        if dataset_name in {"cifar10", "cifar100", "cinic10"}:
            # 保持 CIFAR 数据集原有的数据增强流程不变。
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
            # 衣物图像和字符字形不执行水平翻转，仅使用平移裁剪。
            transform_list.append(transforms.RandomCrop(image_size, padding=4))

    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    return transforms.Compose(transform_list)


def build_test_transform(dataset_name: str) -> Callable:
    """构建用于评估和 Evidence 的确定性预处理流程。"""
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
# 6. 客户端数据划分（Data Partition）
# 包含 IID、普通/均衡 Dirichlet 划分及划分结果校验。
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

    sample_balance:
        Dirichlet 客户端样本数策略：balanced / unbalanced；IID 时为 none。

    dirichlet_alpha:
        Dirichlet 集中参数；IID 时为 None。
    """

    client_indices: List[List[int]]
    client_sample_counts: Dict[int, int]
    client_class_counts: Dict[int, Dict[int, int]]
    num_classes: int
    partition_method: str
    sample_balance: str = "none"
    dirichlet_alpha: Optional[float] = None


def partition_dataset(cfg: Any, dataset: Any) -> PartitionResult:
    """根据嵌套 partition 配置划分训练集。"""
    num_clients = int(cfg.num_clients)
    seed = int(cfg.seed)

    partition_cfg = _cfg_get(cfg, "partition", _default_config_value("partition"))
    partition_method = str(
        _cfg_get(partition_cfg, "method", _default_config_value("partition.method"))
    ).strip().lower()
    dirichlet_cfg = _cfg_get(
        partition_cfg,
        "dirichlet",
        _default_config_value("partition.dirichlet"),
    )
    sample_balance = str(
        _cfg_get(dirichlet_cfg, "sample_balance", _default_config_value("partition.dirichlet.sample_balance"))
    ).strip().lower()

    targets = get_dataset_targets(dataset)
    num_classes = infer_num_classes(targets)
    dirichlet_alpha: Optional[float] = None

    if partition_method == "iid":
        client_indices = build_iid_partition(
            targets=targets,
            num_clients=num_clients,
            seed=seed,
        )
        sample_balance = "none"

    elif partition_method == "dirichlet":
        dirichlet_alpha = float(
            _cfg_get(
                dirichlet_cfg,
                "alpha",
                _default_config_value("partition.dirichlet.alpha"),
            )
        )
        min_samples_per_client = int(
            _cfg_get(
                cfg,
                "min_samples_per_client",
                _default_config_value("min_samples_per_client"),
            )
        )

        if sample_balance == "balanced":
            client_indices = build_balanced_dirichlet_partition(
                targets=targets,
                num_clients=num_clients,
                alpha=dirichlet_alpha,
                seed=seed,
                min_samples_per_client=min_samples_per_client,
            )
        elif sample_balance == "unbalanced":
            max_retries = int(
                _cfg_get(
                    cfg,
                    "partition_max_retries",
                    _default_config_value("partition_max_retries"),
                )
            )
            client_indices = build_dirichlet_partition(
                targets=targets,
                num_clients=num_clients,
                alpha=dirichlet_alpha,
                seed=seed,
                min_samples_per_client=min_samples_per_client,
                max_retries=max_retries,
            )
        else:
            raise ValueError(
                "partition.dirichlet.sample_balance 只能是 balanced 或 unbalanced，"
                f"当前值：{sample_balance!r}"
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
        sample_balance=sample_balance,
        dirichlet_alpha=dirichlet_alpha,
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


def _allocate_class_samples_with_capacity(
    num_samples: int,
    class_proportions: np.ndarray,
    remaining_capacity: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """在客户端容量约束下分配一个类别的样本。"""
    allocation = np.zeros_like(remaining_capacity, dtype=np.int64)
    pending = int(num_samples)

    while pending > 0:
        active_clients = np.flatnonzero(remaining_capacity > 0)
        if active_clients.size == 0:
            raise RuntimeError(
                "balanced Dirichlet 分配失败：样本仍未分配，但所有客户端容量已用尽。"
            )

        # Dirichlet 概率决定类别偏好；剩余容量用于避免客户端超额。
        weights = (
            class_proportions[active_clients]
            * remaining_capacity[active_clients].astype(np.float64)
        )
        weight_sum = float(weights.sum())
        if not math.isfinite(weight_sum) or weight_sum <= 0.0:
            weights = remaining_capacity[active_clients].astype(np.float64)
            weight_sum = float(weights.sum())

        probabilities = weights / weight_sum
        proposed = rng.multinomial(pending, probabilities)
        accepted = np.minimum(
            proposed,
            remaining_capacity[active_clients],
        )

        allocation[active_clients] += accepted
        remaining_capacity[active_clients] -= accepted
        assigned = int(accepted.sum())
        if assigned <= 0:
            raise RuntimeError(
                "balanced Dirichlet 分配停滞：本轮没有样本能够写入剩余客户端容量。"
            )
        pending -= assigned

    return allocation


def build_balanced_dirichlet_partition(
    targets: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples_per_client: int = 1,
) -> List[List[int]]:
    """
    容量约束的 balanced Dirichlet Non-IID 划分。

    对每个类别独立采样 Dirichlet(alpha) 客户端偏好，再在客户端剩余容量
    约束下分配该类别样本。所有样本都会被分配；若数据集大小不能被客户端数
    整除，各客户端样本数只会相差 1。

    这是一种带容量约束的 Dirichlet 分配：类别比例受均衡容量约束，
    不等同于不受约束的逐类独立 multinomial 划分。
    """
    _validate_num_clients(num_clients)

    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(
            f"alpha 必须是有限的正数，当前值：{alpha}"
        )
    if min_samples_per_client < 0:
        raise ValueError(
            "min_samples_per_client 不能小于 0，"
            f"当前值：{min_samples_per_client}"
        )

    targets_array = np.asarray(targets, dtype=np.int64)
    dataset_size = int(targets_array.size)
    if dataset_size < num_clients:
        raise ValueError(
            "balanced Dirichlet 要求训练样本数不少于客户端数量，"
            f"当前样本数={dataset_size}，客户端数={num_clients}。"
        )

    rng = np.random.default_rng(seed)

    # 将总样本数精确分成 floor(N/M) 或 ceil(N/M)。
    base_capacity, remainder = divmod(dataset_size, num_clients)
    client_capacities = np.full(
        num_clients,
        base_capacity,
        dtype=np.int64,
    )
    if remainder > 0:
        extra_clients = rng.choice(
            num_clients,
            size=remainder,
            replace=False,
        )
        client_capacities[extra_clients] += 1

    min_capacity = int(client_capacities.min())
    if min_capacity < min_samples_per_client:
        raise ValueError(
            "balanced Dirichlet 的目标客户端容量低于 min_samples_per_client："
            f"最小容量={min_capacity}，要求至少={min_samples_per_client}。"
        )

    remaining_capacity = client_capacities.copy()
    client_indices: List[List[int]] = [
        [] for _ in range(num_clients)
    ]

    # 随机化类别处理顺序，避免固定类别总是优先占用容量。
    class_ids = np.unique(targets_array)
    for class_id in rng.permutation(class_ids):
        class_indices = np.flatnonzero(targets_array == class_id)
        rng.shuffle(class_indices)

        class_proportions = rng.dirichlet(
            alpha=np.full(num_clients, alpha, dtype=np.float64)
        )
        class_allocation = _allocate_class_samples_with_capacity(
            num_samples=len(class_indices),
            class_proportions=class_proportions,
            remaining_capacity=remaining_capacity,
            rng=rng,
        )

        cursor = 0
        for client_id, count in enumerate(class_allocation):
            count = int(count)
            if count > 0:
                client_indices[client_id].extend(
                    class_indices[cursor:cursor + count].astype(int).tolist()
                )
            cursor += count

        if cursor != len(class_indices):
            raise RuntimeError(
                f"类别 {int(class_id)} 的样本分配数量不正确："
                f"已分配={cursor}，实际={len(class_indices)}。"
            )

    if np.any(remaining_capacity != 0):
        raise RuntimeError(
            "balanced Dirichlet 分配结束后仍有未填满的客户端容量："
            f"{remaining_capacity.tolist()}"
        )

    for indices in client_indices:
        rng.shuffle(indices)

    sample_counts = [len(indices) for indices in client_indices]
    if max(sample_counts) - min(sample_counts) > 1:
        raise RuntimeError(
            "balanced Dirichlet 未能满足客户端样本数最多相差 1 的约束："
            f"{sample_counts}"
        )

    return client_indices


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

    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(
            f"alpha 必须是有限的正数，当前值：{alpha}"
        )

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
    sample_counts = list(partition.client_sample_counts.values())
    return {
        "partition_method": partition.partition_method,
        "sample_balance": partition.sample_balance,
        "dirichlet_alpha": partition.dirichlet_alpha,
        "num_classes": partition.num_classes,
        "num_clients": len(partition.client_indices),
        "client_sample_counts": partition.client_sample_counts,
        "client_sample_count_min": min(sample_counts) if sample_counts else 0,
        "client_sample_count_max": max(sample_counts) if sample_counts else 0,
        "client_sample_count_spread": (
            max(sample_counts) - min(sample_counts) if sample_counts else 0
        ),
        "client_class_counts": partition.client_class_counts,
    }


def _validate_num_clients(num_clients: int) -> None:
    """检查客户端数量是否合法。"""
    if not isinstance(num_clients, int) or num_clients <= 0:
        raise ValueError(f"num_clients 必须是正整数，当前值：{num_clients}")


# ============================================================================
# 7. 数据加载器（DataLoader）
# 包含客户端训练与 Evidence、服务端 Evidence 及测试 DataLoader。
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
    drop_last = _cfg_get_bool(cfg, "drop_last", False)
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
    """构建可选的确定性服务端算法 Evidence DataLoader。"""
    if server_evidence_dataset is None:
        return None

    if len(server_evidence_dataset) <= 0:
        raise ValueError("server_evidence_dataset 不能为空。")

    evidence_cfg = _cfg_get(
        cfg,
        "server_evidence",
        _default_config_value("server_evidence"),
    )
    batch_size = int(
        _cfg_get(
            evidence_cfg,
            "batch_size",
            _default_config_value("server_evidence.batch_size"),
        )
    )
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
    explicit_pin_memory = _cfg_get(
        cfg,
        "pin_memory",
        _default_config_value("pin_memory"),
    )
    if explicit_pin_memory is not None:
        _require_bool_value(explicit_pin_memory, "pin_memory")
        return explicit_pin_memory

    device = str(
        _cfg_get(cfg, "device", _default_config_value("device"))
    ).lower()
    if device == "cpu":
        return False

    return True


# ============================================================================
# 8. 模型构建与管理（Model）
# 包含内置 SwitchViT 模型实现及可注入的 ModelBuilder 接口。
# ============================================================================

ModelBuilder = Callable[[Any], nn.Module]


# ============================================================================
# 内置图像分类 Switch Transformer。
# 模型实现在 base.py 中，联邦学习框架无需依赖外部
# 模型包即可使用。配置中的 MoE Block 编号从 1 开始。
# ============================================================================

def _switch_model_config(cfg: Any) -> Dict[str, Any]:
    raw = _cfg_get(cfg, "model_cfg", _default_config_value("model_cfg"))
    if hasattr(raw, "to_dict"):
        raw = raw.to_dict()
    if not isinstance(raw, Mapping):
        raise TypeError("model_cfg 必须是 Mapping。")
    return _deep_merge(
        base=DEFAULT_CONFIG["model_cfg"],
        override=dict(raw),
    )


class SwitchFeedForward(nn.Module):
    """供非 MoE Block 使用的标准 Dense Transformer FFN。"""

    def __init__(
        self,
        hidden_dim: int,
        expert_hidden_dim: int,
        dropout: float = DEFAULT_CONFIG["model_cfg"]["dropout"],
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, expert_hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(expert_hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.fc1(inputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        outputs = self.fc2(outputs)
        return self.dropout(outputs)


class SwitchExpert(nn.Module):
    """由两层前馈网络构成的独立 Expert。"""

    def __init__(
        self,
        hidden_dim: int,
        expert_hidden_dim: int,
        dropout: float = DEFAULT_CONFIG["model_cfg"]["dropout"],
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, expert_hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(expert_hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.fc1(inputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        outputs = self.fc2(outputs)
        return self.dropout(outputs)


class SwitchRouter(nn.Module):
    """执行 Top-1 Token 路由，并返回路由概率和选中的 Expert ID。"""

    def __init__(self, hidden_dim: int, num_experts: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, num_experts)
        self.num_experts = int(num_experts)

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.projection(tokens)
        # 使用 FP32 计算路由概率，以提高数值稳定性。
        router_probabilities = torch.softmax(router_logits.float(), dim=-1)
        top1_probabilities, top1_experts = torch.max(
            router_probabilities,
            dim=-1,
        )
        return router_probabilities, top1_probabilities, top1_experts


class SwitchMoE(nn.Module):
    """
    支持容量限制和溢出处理的 Top-1 Switch FFN。

    启用 Expert Capacity 后，Token 首先尝试分配给路由概率最高的 Expert。
    发生容量溢出时，Token 按剩余专家的 Router 概率从高到低依次尝试。
    如果所有 Expert 均已满载，仅在 token_dropping=True 时丢弃 Token；
    否则配置必须保证所有 Token 都有足够的总容量可供分配。
    """

    def __init__(
        self,
        hidden_dim: int,
        expert_hidden_dim: int,
        num_experts: int,
        dropout: float = DEFAULT_CONFIG["model_cfg"]["dropout"],
        collect_router_statistics: bool = (
            DEFAULT_CONFIG["model_cfg"]["router_statistics"]
        ),
        aux_loss_coef: float = DEFAULT_CONFIG["model_cfg"]["aux_loss_coef"],
        capacity_factor: float = DEFAULT_CONFIG["model_cfg"]["capacity_factor"],
        token_dropping: bool = DEFAULT_CONFIG["model_cfg"]["token_dropping"],
    ) -> None:
        super().__init__()
        if not _is_positive_int_value(num_experts):
            raise ValueError("num_experts 必须是正整数。")
        if type(collect_router_statistics) is not bool:
            raise TypeError(
                "collect_router_statistics 必须是 bool，"
                f"当前值：{collect_router_statistics!r}"
            )
        if type(token_dropping) is not bool:
            raise TypeError(
                f"token_dropping 必须是 bool，当前值：{token_dropping!r}"
            )

        aux_loss_coef = float(aux_loss_coef)
        capacity_factor = float(capacity_factor)
        if not math.isfinite(aux_loss_coef) or aux_loss_coef < 0.0:
            raise ValueError("aux_loss_coef 必须是非负有限数。")
        if not math.isfinite(capacity_factor) or capacity_factor < 0.0:
            raise ValueError("capacity_factor 必须是非负有限数。")
        expert_capacity = capacity_factor > 0.0
        auxiliary_loss = aux_loss_coef > 0.0
        if token_dropping and not expert_capacity:
            raise ValueError(
                "token_dropping=True 时 capacity_factor 必须大于 0。"
            )
        if expert_capacity and not token_dropping and capacity_factor < 1.0:
            raise ValueError(
                "启用容量限制但关闭 Token Dropping 时，capacity_factor "
                "必须 >= 1.0，以保证总容量足以容纳全部 Token。"
            )

        self.num_experts = num_experts
        self.collect_router_statistics = collect_router_statistics
        self.auxiliary_loss = auxiliary_loss
        self.aux_loss_coef = aux_loss_coef
        self.expert_capacity = expert_capacity
        self.capacity_factor = capacity_factor
        self.token_dropping = token_dropping

        self.router = SwitchRouter(
            hidden_dim=hidden_dim,
            num_experts=self.num_experts,
        )
        self.experts = nn.ModuleDict({
            f"expert_{expert_id}": SwitchExpert(
                hidden_dim=hidden_dim,
                expert_hidden_dim=expert_hidden_dim,
                dropout=dropout,
            )
            for expert_id in range(self.num_experts)
        })

    def _assign_experts_with_capacity(
        self,
        router_probabilities: torch.Tensor,
        top1_experts: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        在遵守各 Expert 容量限制的前提下，按候选优先级分轮分配 Token。

        每个 Expert 的队列保留 Token 展平后的原始顺序。
        只有当前候选 Expert 已无剩余容量时，Token 才会尝试下一个候选 Expert。
        """
        num_tokens = int(top1_experts.numel())
        capacity = int(math.ceil(
            (num_tokens / self.num_experts) * self.capacity_factor
        ))

        # 将排名第一的专家固定为 torch.max 返回的 Top-1 结果。
        # 同时避免 argmax 与 argsort 在概率相同时采用不同的排序规则。
        remaining_probabilities = router_probabilities.clone()
        remaining_probabilities.scatter_(
            1,
            top1_experts.unsqueeze(1),
            float("-inf"),
        )
        remaining_order = torch.argsort(
            remaining_probabilities,
            dim=-1,
            descending=True,
        )[:, : self.num_experts - 1]
        routing_order = torch.cat(
            (top1_experts.unsqueeze(1), remaining_order),
            dim=1,
        )

        assigned_experts = torch.full_like(top1_experts, -1)
        expert_loads = [0 for _ in range(self.num_experts)]
        active_indices = torch.arange(
            num_tokens,
            device=top1_experts.device,
            dtype=torch.long,
        )

        for preference_rank in range(self.num_experts):
            if active_indices.numel() == 0:
                break

            proposed_experts = routing_order.index_select(
                0,
                active_indices,
            )[:, preference_rank]
            accepted_mask = torch.zeros(
                active_indices.numel(),
                dtype=torch.bool,
                device=active_indices.device,
            )

            for expert_id in range(self.num_experts):
                candidate_positions = torch.nonzero(
                    proposed_experts == expert_id,
                    as_tuple=False,
                ).flatten()
                if candidate_positions.numel() == 0:
                    continue

                remaining_capacity = capacity - expert_loads[expert_id]
                if remaining_capacity <= 0:
                    continue

                accepted_positions = candidate_positions[:remaining_capacity]
                accepted_indices = active_indices.index_select(
                    0,
                    accepted_positions,
                )
                assigned_experts[accepted_indices] = expert_id
                expert_loads[expert_id] += int(accepted_indices.numel())
                accepted_mask[accepted_positions] = True

            # 保持被拒绝 Token 的原始相对顺序，
            # 以便进入下一轮候选专家分配。
            active_indices = active_indices[~accepted_mask]

        if active_indices.numel() > 0 and not self.token_dropping:
            raise RuntimeError(
                "No-Token-Left-Behind 路由未能分配全部 Token；"
                "请检查 capacity_factor 和专家容量配置。"
            )

        return assigned_experts, capacity

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Optional[torch.Tensor]]:
        if inputs.ndim != 3:
            raise ValueError(
                "SwitchMoE 输入必须是 [batch, tokens, hidden_dim]，"
                f"当前 shape={tuple(inputs.shape)}"
            )

        batch_size, token_count, hidden_dim = inputs.shape
        flat_tokens = inputs.reshape(batch_size * token_count, hidden_dim)
        num_tokens = int(flat_tokens.shape[0])
        if num_tokens == 0:
            raise ValueError("SwitchMoE 不支持空 Token 输入。")

        router_probabilities, top1_probabilities, top1_experts = self.router(
            flat_tokens
        )

        if self.expert_capacity:
            assigned_experts, capacity = self._assign_experts_with_capacity(
                router_probabilities,
                top1_experts,
            )
        else:
            assigned_experts = top1_experts
            capacity = None

        # Auxiliary Loss 使用容量溢出二次路由之前的原始 Top-1 选择结果，
        # 以及当前层的平均 Router 概率。
        aux_loss: Optional[torch.Tensor] = None
        if self.auxiliary_loss:
            first_choice_counts = torch.bincount(
                top1_experts,
                minlength=self.num_experts,
            ).to(dtype=router_probabilities.dtype)
            expert_fraction = first_choice_counts / float(num_tokens)
            mean_router_probability = router_probabilities.mean(dim=0)
            aux_loss = (
                self.aux_loss_coef
                * self.num_experts
                * torch.sum(expert_fraction * mean_router_probability)
            )

        # 每个成功分配的 Token 只执行一次派发。
        # 经过所有候选轮次仍未分配的 Token，仅在启用 Token Dropping 时丢弃。
        flat_outputs = torch.zeros_like(flat_tokens)
        for expert_id, expert in enumerate(self.experts.values()):
            token_indices = torch.nonzero(
                assigned_experts == expert_id,
                as_tuple=False,
            ).flatten()
            if token_indices.numel() == 0:
                continue

            expert_inputs = flat_tokens.index_select(0, token_indices)
            expert_outputs = expert(expert_inputs)
            gate = router_probabilities[
                token_indices,
                expert_id,
            ].to(dtype=expert_outputs.dtype).unsqueeze(-1)
            flat_outputs = flat_outputs.index_copy(
                0,
                token_indices,
                expert_outputs * gate,
            )

        outputs = flat_outputs.reshape(batch_size, token_count, hidden_dim)

        stats: Dict[str, Any] = {}
        if self.collect_router_statistics:
            tokens_per_expert = torch.bincount(
                top1_experts,
                minlength=self.num_experts,
            )
            processed_tokens_per_expert = torch.bincount(
                assigned_experts[assigned_experts >= 0],
                minlength=self.num_experts,
            )
            rerouted_mask = (
                (assigned_experts >= 0)
                & (assigned_experts != top1_experts)
            )
            dropped_mask = assigned_experts < 0
            stats = {
                # 为兼容旧接口，此字段表示各 Expert 原始 Top-1 路由需求量。
                "tokens_per_expert": tokens_per_expert,
                "router_prob_sum": router_probabilities.sum(dim=0),
                "top1_prob_sum": top1_probabilities.sum(),
                "num_tokens": num_tokens,
                # 表示容量限制和溢出处理后的实际派发数量。
                "processed_tokens_per_expert": processed_tokens_per_expert,
                "rerouted_tokens": int(rerouted_mask.sum().item()),
                "rerouted_tokens_per_expert": torch.bincount(
                    assigned_experts[rerouted_mask],
                    minlength=self.num_experts,
                ),
                "dropped_tokens": int(dropped_mask.sum().item()),
            }
            if capacity is not None:
                stats["overflow_tokens_per_expert"] = (
                    tokens_per_expert - capacity
                ).clamp_min(0)
            else:
                stats["overflow_tokens_per_expert"] = torch.zeros_like(
                    tokens_per_expert
                )

        return outputs, stats, aux_loss


class SwitchTransformerBlock(nn.Module):
    """采用 Pre-LN 结构、并包含 Dense FFN 或 Switch MoE FFN 的 Transformer Block。"""

    def __init__(
        self,
        hidden_dim: int,
        attention_heads: int,
        expert_hidden_dim: int,
        num_experts: int,
        use_moe: bool,
        dropout: float = DEFAULT_CONFIG["model_cfg"]["dropout"],
        collect_router_statistics: bool = (
            DEFAULT_CONFIG["model_cfg"]["router_statistics"]
        ),
        aux_loss_coef: float = DEFAULT_CONFIG["model_cfg"]["aux_loss_coef"],
        capacity_factor: float = DEFAULT_CONFIG["model_cfg"]["capacity_factor"],
        token_dropping: bool = DEFAULT_CONFIG["model_cfg"]["token_dropping"],
    ) -> None:
        super().__init__()
        self.use_moe = bool(use_moe)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)

        if self.use_moe:
            self.mlp = SwitchMoE(
                hidden_dim=hidden_dim,
                expert_hidden_dim=expert_hidden_dim,
                num_experts=num_experts,
                dropout=dropout,
                collect_router_statistics=collect_router_statistics,
                aux_loss_coef=aux_loss_coef,
                capacity_factor=capacity_factor,
                token_dropping=token_dropping,
            )
        else:
            self.mlp = SwitchFeedForward(
                hidden_dim=hidden_dim,
                expert_hidden_dim=expert_hidden_dim,
                dropout=dropout,
            )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Optional[torch.Tensor]]:
        normalized = self.norm1(inputs)
        attention_outputs, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        outputs = inputs + self.attention_dropout(attention_outputs)

        if self.use_moe:
            mlp_outputs, router_stats, aux_loss = self.mlp(self.norm2(outputs))
        else:
            mlp_outputs = self.mlp(self.norm2(outputs))
            router_stats = {}
            aux_loss = None

        return outputs + mlp_outputs, router_stats, aux_loss


class SwitchViT(nn.Module):
    """
    用于图像分类的 Switch ViT。

    架构默认参数统一由 DEFAULT_CONFIG["model_cfg"] 提供；
    数据集类别数和输入尺寸由数据集注册信息推导。
    """

    def __init__(
        self,
        num_classes: int,
        input_shape: Sequence[int],
        transformer_layers: int = DEFAULT_CONFIG["model_cfg"]["transformer_layers"],
        hidden_dim: int = DEFAULT_CONFIG["model_cfg"]["hidden_dim"],
        attention_heads: int = DEFAULT_CONFIG["model_cfg"]["attention_heads"],
        patch_size: int = DEFAULT_CONFIG["model_cfg"]["patch_size"],
        expert_hidden_dim: int = DEFAULT_CONFIG["model_cfg"]["expert_hidden_dim"],
        num_experts: int = DEFAULT_CONFIG["num_experts"],
        moe_block_indices: Sequence[int] = tuple(
            DEFAULT_CONFIG["model_cfg"]["moe_block_indices"]
        ),
        dropout: float = DEFAULT_CONFIG["model_cfg"]["dropout"],
        router_statistics: bool = DEFAULT_CONFIG["model_cfg"]["router_statistics"],
        aux_loss_coef: float = DEFAULT_CONFIG["model_cfg"]["aux_loss_coef"],
        capacity_factor: float = DEFAULT_CONFIG["model_cfg"]["capacity_factor"],
        token_dropping: bool = DEFAULT_CONFIG["model_cfg"]["token_dropping"],
    ) -> None:
        super().__init__()

        if len(input_shape) != 3:
            raise ValueError(
                "input_shape 必须为 (channels, height, width)，"
                f"当前值：{input_shape}"
            )
        in_channels, image_height, image_width = (
            int(input_shape[0]),
            int(input_shape[1]),
            int(input_shape[2]),
        )
        if min(in_channels, image_height, image_width) <= 0:
            raise ValueError(f"非法 input_shape：{input_shape}")
        if min(
            transformer_layers,
            hidden_dim,
            attention_heads,
            patch_size,
            expert_hidden_dim,
            num_classes,
        ) <= 0:
            raise ValueError(
                "transformer_layers、hidden_dim、attention_heads、patch_size、"
                "expert_hidden_dim 和 num_classes 必须为正整数。"
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout 必须位于 [0, 1)，当前值：{dropout}")
        if image_height % patch_size != 0 or image_width % patch_size != 0:
            raise ValueError(
                f"图像尺寸 {(image_height, image_width)} 必须能被 "
                f"patch_size={patch_size} 整除。"
            )
        if hidden_dim % attention_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} 必须能被 "
                f"attention_heads={attention_heads} 整除。"
            )

        normalized_moe_blocks = tuple(int(index) for index in moe_block_indices)
        if (
            not normalized_moe_blocks
            or len(set(normalized_moe_blocks)) != len(normalized_moe_blocks)
            or any(index < 1 or index > transformer_layers for index in normalized_moe_blocks)
        ):
            raise ValueError(
                "moe_block_indices 必须是无重复且位于模型层数范围内的 1-based 层号。"
            )

        self.num_classes = int(num_classes)
        self.input_shape = (in_channels, image_height, image_width)
        self.transformer_layers = int(transformer_layers)
        self.hidden_dim = int(hidden_dim)
        self.num_experts = num_experts
        self.moe_block_indices = normalized_moe_blocks
        self.patch_size = int(patch_size)

        grid_height = image_height // patch_size
        grid_width = image_width // patch_size
        num_patches = grid_height * grid_width

        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, hidden_dim)
        )
        self.pos_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SwitchTransformerBlock(
                hidden_dim=hidden_dim,
                attention_heads=attention_heads,
                expert_hidden_dim=expert_hidden_dim,
                num_experts=num_experts,
                use_moe=(block_index in normalized_moe_blocks),
                dropout=dropout,
                collect_router_statistics=router_statistics,
                aux_loss_coef=aux_loss_coef,
                capacity_factor=capacity_factor,
                token_dropping=token_dropping,
            )
            for block_index in range(1, transformer_layers + 1)
        ])

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.head = nn.Linear(hidden_dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> ModelOutput:
        if images.ndim != 4:
            raise ValueError(
                "SwitchViT 输入必须是 [batch, channels, height, width]，"
                f"当前 shape={tuple(images.shape)}"
            )
        if tuple(images.shape[1:]) != self.input_shape:
            raise ValueError(
                f"输入图像 shape={tuple(images.shape[1:])} 与模型配置 "
                f"input_shape={self.input_shape} 不一致。"
            )

        outputs = self.patch_embed(images)
        outputs = outputs.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(outputs.shape[0], -1, -1)
        outputs = torch.cat((cls_tokens, outputs), dim=1)
        outputs = self.pos_dropout(outputs + self.pos_embed)

        moe_stats: Dict[str, Any] = {}
        aux_loss_by_layer: Dict[str, torch.Tensor] = {}
        for block_index, block in enumerate(self.blocks, start=1):
            outputs, layer_stats, layer_aux_loss = block(outputs)
            layer_name = f"blocks.{block_index - 1}.mlp"
            if layer_stats:
                # 使用与 Expert 参数相同的模块路径，
                # 使逐层诊断信息能够对应联邦参数分组。
                moe_stats[layer_name] = layer_stats
            if layer_aux_loss is not None:
                aux_loss_by_layer[layer_name] = layer_aux_loss

        cls_features = self.norm(outputs[:, 0])
        logits = self.head(cls_features)
        aux_loss = (
            torch.stack(list(aux_loss_by_layer.values())).sum()
            if aux_loss_by_layer
            else None
        )

        return ModelOutput(
            logits=logits,
            aux_loss=aux_loss,
            aux_loss_by_layer=aux_loss_by_layer,
            moe_stats=moe_stats,
        )


def build_switch_vit_model(cfg: Any) -> nn.Module:
    """根据公共实验配置构建默认 SwitchViT 模型。"""
    model_cfg = _switch_model_config(cfg)

    if str(model_cfg["routing"]).strip().lower() != "top1":
        raise ValueError("当前 SwitchViT 仅实现 Top-1 Routing。")

    # MoE 自身的选项约束由 SwitchMoE.__init__ 统一负责。
    token_dropping = model_cfg["token_dropping"]
    router_statistics = model_cfg["router_statistics"]
    aux_loss_coef = model_cfg["aux_loss_coef"]
    capacity_factor = model_cfg["capacity_factor"]

    num_experts = _cfg_get(
        cfg,
        "num_experts",
        _default_config_value("num_experts"),
    )
    dataset_name = normalize_dataset_name(
        _cfg_get(cfg, "dataset", _default_config_value("dataset"))
    )
    dataset_info = DATASET_INFO.get(dataset_name)
    if dataset_info is None:
        raise ValueError(
            f"无法根据数据集 {dataset_name!r} 推断模型输入尺寸和类别数量。"
        )

    # 数据集注册信息是类别数和输入尺寸的唯一来源。
    input_shape = tuple(dataset_info["input_shape"])
    num_classes = int(dataset_info["num_classes"])
    _validate_switch_model_config(
        model_cfg=model_cfg,
        input_shape=input_shape,
    )

    return SwitchViT(
        num_classes=int(num_classes),
        input_shape=input_shape,
        transformer_layers=int(model_cfg["transformer_layers"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        attention_heads=int(model_cfg["attention_heads"]),
        patch_size=int(model_cfg["patch_size"]),
        expert_hidden_dim=int(model_cfg["expert_hidden_dim"]),
        num_experts=num_experts,
        moe_block_indices=model_cfg["moe_block_indices"],
        dropout=float(model_cfg["dropout"]),
        router_statistics=router_statistics,
        aux_loss_coef=aux_loss_coef,
        capacity_factor=capacity_factor,
        token_dropping=token_dropping,
    )


def build_model(cfg: Any, model_builder: ModelBuilder) -> nn.Module:
    """通过外部注入的 ModelBuilder 创建模型。"""
    if not callable(model_builder):
        raise TypeError("model_builder 必须是可调用的模型构建器。")
    model = model_builder(cfg)
    if not isinstance(model, nn.Module):
        raise TypeError(
            "model_builder 必须返回 torch.nn.Module，"
            f"当前返回：{type(model).__name__}"
        )
    return model


def count_parameters(
    model: nn.Module,
    trainable_only: bool = False,
) -> int:
    """统计模型参数量。"""
    total = 0
    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        total += int(param.numel())
    return total


def summarize_model(model: nn.Module) -> Dict[str, int]:
    """返回模型总参数量和可训练参数量。"""
    return {
        "total_params": count_parameters(model=model, trainable_only=False),
        "trainable_params": count_parameters(model=model, trainable_only=True),
    }


# ============================================================================
# 9. 模型参数分组（Parameter Groups）
# 负责 Expert 参数识别、分组构建、结构校验及统计。
# ============================================================================


# 同时支持 ModuleList 风格的 experts.0 和 ModuleDict 风格的
# experts.expert_0（常见 Switch Transformer 实现采用此结构）。
# 统一使用此解析逻辑提取 Expert ID 和 MoE 层编号，
# 避免不同位置的解析结果不一致。
_EXPERT_PATH_PATTERN = re.compile(
    r"^(?P<layer>.*?)(?:^|\.)experts\."
    r"(?P<expert_key>\d+|expert_\d+)(?=\.|$)"
)


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
        跨 MoE 层按 expert id 汇总的参数名，保留原有接口。

    expert_by_moe_layer:
        按 MoE 模块路径分组的专家参数名。

    expert_by_layer_and_id:
        按 MoE 模块路径和 expert id 二级分组的专家参数名。
    """

    all: List[str]
    non_expert: List[str]
    expert: List[str]
    expert_by_id: Dict[int, List[str]] = field(default_factory=dict)
    expert_by_moe_layer: Dict[str, List[str]] = field(default_factory=dict)
    expert_by_layer_and_id: Dict[str, Dict[int, List[str]]] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """转成普通 dict，方便写日志或调试。"""
        return {
            "all": list(self.all),
            "non_expert": list(self.non_expert),
            "expert": list(self.expert),
            "expert_by_id": {
                int(expert_id): list(names)
                for expert_id, names in self.expert_by_id.items()
            },
            "expert_by_moe_layer": {
                str(layer_name): list(names)
                for layer_name, names in self.expert_by_moe_layer.items()
            },
            "expert_by_layer_and_id": {
                str(layer_name): {
                    int(expert_id): list(names)
                    for expert_id, names in expert_groups.items()
                }
                for layer_name, expert_groups in self.expert_by_layer_and_id.items()
            },
        }

    def summary(self) -> Dict[str, Any]:
        """返回轻量摘要，不包含完整参数名列表。"""
        return {
            "num_all": len(self.all),
            "num_non_expert": len(self.non_expert),
            "num_expert": len(self.expert),
            "num_experts_found": len(self.expert_by_id),
            "num_expert_instances_found": sum(
                len(expert_groups)
                for expert_groups in self.expert_by_layer_and_id.values()
            ),
            "num_moe_layers_found": len(self.expert_by_moe_layer),
            "num_experts_by_moe_layer": {
                layer_name: len(expert_groups)
                for layer_name, expert_groups in self.expert_by_layer_and_id.items()
            },
            "num_params_by_expert": {
                int(expert_id): len(names)
                for expert_id, names in self.expert_by_id.items()
            },
            "num_params_by_moe_layer": {
                str(layer_name): len(names)
                for layer_name, names in self.expert_by_moe_layer.items()
            },
            "num_params_by_layer_and_id": {
                str(layer_name): {
                    int(expert_id): len(names)
                    for expert_id, names in expert_groups.items()
                }
                for layer_name, expert_groups in self.expert_by_layer_and_id.items()
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
        支持 ModuleList 的 experts.<id>，以及 ModuleDict 的
        experts.expert_<id>。两种命名统一解析为整数 expert id。

    例如：
        moe.experts.0.fc1.weight                  -> expert 0
        blocks.0.moe.experts.expert_3.wi.weight   -> expert 3

    其他参数都归为 non_expert。
    """
    all_names = list(state_dict.keys())

    non_expert_names: List[str] = []
    expert_names: List[str] = []
    expert_by_id: Dict[int, List[str]] = {}
    expert_by_moe_layer: Dict[str, List[str]] = {}
    expert_by_layer_and_id: Dict[str, Dict[int, List[str]]] = {}

    for name in all_names:
        expert_id = get_expert_id_from_name(name)

        if expert_id is None:
            non_expert_names.append(name)
            continue

        moe_layer = get_moe_layer_from_name(name)
        if moe_layer is None:
            raise ValueError(
                "无法从专家参数名解析 MoE 层路径："
                f"{name!r}"
            )

        expert_names.append(name)
        expert_by_id.setdefault(expert_id, []).append(name)
        expert_by_moe_layer.setdefault(moe_layer, []).append(name)
        expert_by_layer_and_id.setdefault(moe_layer, {}).setdefault(
            expert_id,
            [],
        ).append(name)

    groups = ParamGroups(
        all=all_names,
        non_expert=non_expert_names,
        expert=expert_names,
        expert_by_id={
            expert_id: names
            for expert_id, names in sorted(expert_by_id.items())
        },
        expert_by_moe_layer={
            layer_name: names
            for layer_name, names in sorted(expert_by_moe_layer.items())
        },
        expert_by_layer_and_id={
            layer_name: {
                expert_id: names
                for expert_id, names in sorted(expert_groups.items())
            }
            for layer_name, expert_groups in sorted(
                expert_by_layer_and_id.items()
            )
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

    支持两种常见容器命名：
        ModuleList: experts.<id>
        ModuleDict: experts.expert_<id>

    示例：
        "moe.experts.0.fc.weight"                 -> 0
        "moe.experts.expert_3.wi.weight"          -> 3
        "backbone.conv.weight"                    -> None
    """
    match = _EXPERT_PATH_PATTERN.match(name)

    if match is None:
        return None

    expert_key = match.group("expert_key")
    if expert_key.startswith("expert_"):
        expert_key = expert_key[len("expert_"):]

    return int(expert_key)


def get_moe_layer_from_name(name: str) -> str | None:
    """
    从专家参数名中解析所属 MoE 模块路径。

    示例：
        "blocks.2.moe.experts.0.fc.weight"
            -> "blocks.2.moe"
        "encoder.block.0.mlp.experts.expert_3.wi.weight"
            -> "encoder.block.0.mlp"
        "experts.expert_0.fc.weight" -> "<root>"

    对于直接挂在模型根节点的 experts，使用 "<root>" 作为层标识。
    """
    match = _EXPERT_PATH_PATTERN.match(name)

    if match is None:
        return None

    return match.group("layer") or "<root>"


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
        4. expert_by_id、expert_by_moe_layer 和 expert_by_layer_and_id 是否一致
        5. 如果 strict=True，要求至少找到一个 expert 参数
        6. 如果传入 expected_num_experts，则逐层检查专家编号是否完整
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

    # 验证按 MoE 层分组：每个专家参数必须归属到正确的 MoE 模块。
    expert_by_layer_names = set()
    for layer_name, names in groups.expert_by_moe_layer.items():
        if not isinstance(layer_name, str) or not layer_name:
            raise ValueError(f"MoE 层名称必须是非空字符串：{layer_name!r}")

        for name in names:
            parsed_layer = get_moe_layer_from_name(name)
            if parsed_layer != layer_name:
                raise ValueError(
                    f"expert_by_moe_layer 分组错误：参数 {name} 被放到 "
                    f"{layer_name!r}，但解析结果是 {parsed_layer!r}。"
                )
            expert_by_layer_names.add(name)

    if expert_by_layer_names != expert_names:
        missing = sorted(expert_names - expert_by_layer_names)
        extra = sorted(expert_by_layer_names - expert_names)
        raise ValueError(
            "expert_by_moe_layer 和 expert 参数组不一致。"
            f" missing={missing[:10]}, extra={extra[:10]}"
        )

    if set(groups.expert_by_moe_layer) != set(groups.expert_by_layer_and_id):
        raise ValueError(
            "expert_by_moe_layer 和 expert_by_layer_and_id 的 MoE 层集合不一致。"
        )

    # 验证 MoE 层 + expert id 二级分组。
    expert_by_layer_and_id_names = set()
    for layer_name, expert_groups in groups.expert_by_layer_and_id.items():
        if layer_name not in groups.expert_by_moe_layer:
            raise ValueError(
                "expert_by_layer_and_id 包含未知 MoE 层："
                f"{layer_name!r}"
            )

        for expert_id, names in expert_groups.items():
            if expert_id < 0:
                raise ValueError(
                    f"expert_id 不能小于 0，当前值：{expert_id}"
                )

            for name in names:
                parsed_layer = get_moe_layer_from_name(name)
                parsed_expert_id = get_expert_id_from_name(name)
                if parsed_layer != layer_name or parsed_expert_id != expert_id:
                    raise ValueError(
                        "expert_by_layer_and_id 分组错误："
                        f"参数 {name} 被放到 ({layer_name!r}, {expert_id})，"
                        f"实际解析结果为 ({parsed_layer!r}, {parsed_expert_id!r})。"
                    )
                expert_by_layer_and_id_names.add(name)

    if expert_by_layer_and_id_names != expert_names:
        missing = sorted(expert_names - expert_by_layer_and_id_names)
        extra = sorted(expert_by_layer_and_id_names - expert_names)
        raise ValueError(
            "expert_by_layer_and_id 和 expert 参数组不一致。"
            f" missing={missing[:10]}, extra={extra[:10]}"
        )

    if strict and len(expert_names) == 0:
        raise ValueError(
            "没有找到任何 expert 参数。"
            "请确认模型中的 expert 参数名是否符合 experts.<id> 或 experts.expert_<id>。"
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

        # 多层 MoE 必须逐层检查，不能让某一层缺失的专家被其他层掩盖。
        if strict:
            for layer_name, expert_groups in groups.expert_by_layer_and_id.items():
                layer_expert_ids = sorted(expert_groups.keys())
                if layer_expert_ids != expected_expert_ids:
                    raise ValueError(
                        f"MoE 层 {layer_name!r} 的 expert id 与期望不一致。"
                        f" found={layer_expert_ids}, expected={expected_expert_ids}"
                    )


@dataclass(frozen=True)
class FederatedOptimizationGroups:
    """
    面向联邦优化职责的参数分组。

    non_expert_parameters:
        Backbone / Router 等非专家可训练参数。
    expert_parameters:
        全部可训练专家参数。具体采用欧氏更新还是流形更新，由专家聚合算法决定。
    *_state_names:
        对应 state_dict 分组，包含需要处理的参数和 buffer。
    """

    non_expert_state_names: List[str]
    expert_state_names: List[str]
    non_expert_parameters: List[str]
    expert_parameters: List[str]
    buffers: List[str] = field(default_factory=list)
    frozen_parameters: List[str] = field(default_factory=list)
    expert_state_names_by_moe_layer: Dict[str, List[str]] = field(
        default_factory=dict
    )
    expert_state_names_by_layer_and_id: Dict[str, Dict[int, List[str]]] = field(
        default_factory=dict
    )
    expert_parameters_by_moe_layer: Dict[str, List[str]] = field(
        default_factory=dict
    )
    expert_parameters_by_layer_and_id: Dict[str, Dict[int, List[str]]] = field(
        default_factory=dict
    )

    @property
    def expert_parameter_names(self) -> List[str]:
        return list(self.expert_parameters)

    def summary(self) -> Dict[str, Any]:
        return {
            "num_non_expert_parameters": len(self.non_expert_parameters),
            "num_expert_parameters": len(self.expert_parameters),
            "num_buffers": len(self.buffers),
            "num_frozen_parameters": len(self.frozen_parameters),
            "num_moe_layers": len(self.expert_parameters_by_moe_layer),
            "num_expert_parameters_by_moe_layer": {
                layer_name: len(names)
                for layer_name, names in self.expert_parameters_by_moe_layer.items()
            },
            "num_expert_parameters_by_layer_and_id": {
                layer_name: {
                    int(expert_id): len(names)
                    for expert_id, names in expert_groups.items()
                }
                for layer_name, expert_groups in (
                    self.expert_parameters_by_layer_and_id.items()
                )
            },
        }


def build_federated_optimization_groups(
    model: nn.Module,
    param_groups: Optional[ParamGroups] = None,
    expected_num_experts: Optional[int] = None,
    strict: bool = True,
) -> FederatedOptimizationGroups:
    """
    按模型参数归属拆分 non_expert 与 expert。

    本函数只识别参数属于哪个专家，不决定其几何类型或约束方式。
    专家参数的更新与约束行为由注入的训练策略和聚合器决定。
    """
    if param_groups is None:
        param_groups = build_param_groups(
            model=model,
            expected_num_experts=expected_num_experts,
            strict=strict,
        )

    state_dict = model.state_dict()
    named_parameters = dict(model.named_parameters())
    parameter_names = set(named_parameters)
    trainable_names = {
        name
        for name, parameter in named_parameters.items()
        if parameter.requires_grad
    }
    frozen_names = [
        name
        for name, parameter in named_parameters.items()
        if not parameter.requires_grad
    ]

    expert_state_names = set(param_groups.expert)
    non_expert_state_names = set(param_groups.non_expert)

    non_expert_parameters = [
        name
        for name, parameter in named_parameters.items()
        if name in non_expert_state_names and parameter.requires_grad
    ]
    expert_parameters = [
        name
        for name, parameter in named_parameters.items()
        if name in expert_state_names and parameter.requires_grad
    ]
    buffers = [
        name
        for name in state_dict
        if name not in parameter_names
    ]

    if set(non_expert_parameters) | set(expert_parameters) != trainable_names:
        missing = sorted(
            trainable_names - set(non_expert_parameters) - set(expert_parameters)
        )
        raise ValueError(
            "联邦优化分组未覆盖全部可训练参数；"
            f"missing={missing[:10]}"
        )
    if set(non_expert_parameters) & set(expert_parameters):
        raise ValueError("non_expert 与 expert 参数分组发生重叠。")

    expert_state_names_by_moe_layer = {
        layer_name: list(names)
        for layer_name, names in param_groups.expert_by_moe_layer.items()
    }
    expert_state_names_by_layer_and_id = {
        layer_name: {
            expert_id: list(names)
            for expert_id, names in expert_groups.items()
        }
        for layer_name, expert_groups in param_groups.expert_by_layer_and_id.items()
    }

    trainable_name_set = set(trainable_names)
    expert_parameters_by_moe_layer = {
        layer_name: [
            name for name in names if name in trainable_name_set
        ]
        for layer_name, names in param_groups.expert_by_moe_layer.items()
    }
    expert_parameters_by_layer_and_id = {
        layer_name: {
            expert_id: [
                name for name in names if name in trainable_name_set
            ]
            for expert_id, names in expert_groups.items()
        }
        for layer_name, expert_groups in param_groups.expert_by_layer_and_id.items()
    }

    return FederatedOptimizationGroups(
        non_expert_state_names=list(param_groups.non_expert),
        expert_state_names=list(param_groups.expert),
        non_expert_parameters=non_expert_parameters,
        expert_parameters=expert_parameters,
        buffers=buffers,
        frozen_parameters=frozen_names,
        expert_state_names_by_moe_layer=expert_state_names_by_moe_layer,
        expert_state_names_by_layer_and_id=expert_state_names_by_layer_and_id,
        expert_parameters_by_moe_layer=expert_parameters_by_moe_layer,
        expert_parameters_by_layer_and_id=expert_parameters_by_layer_and_id,
    )


def count_tensors_by_group(groups: ParamGroups) -> Dict[str, Any]:
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
        "expert_by_moe_layer": {
            layer_name: len(names)
            for layer_name, names in groups.expert_by_moe_layer.items()
        },
        "expert_by_layer_and_id": {
            layer_name: {
                int(expert_id): len(names)
                for expert_id, names in expert_groups.items()
            }
            for layer_name, expert_groups in groups.expert_by_layer_and_id.items()
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
        "expert_by_moe_layer": {
            layer_name: _count_numel(
                state_dict=state_dict,
                names=names,
                only_floating=only_floating,
            )
            for layer_name, names in groups.expert_by_moe_layer.items()
        },
        "expert_by_layer_and_id": {
            layer_name: {
                int(expert_id): _count_numel(
                    state_dict=state_dict,
                    names=names,
                    only_floating=only_floating,
                )
                for expert_id, names in expert_groups.items()
            }
            for layer_name, expert_groups in groups.expert_by_layer_and_id.items()
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
# 10. 联邦学习核心数据结构（FL Core）
# 定义 ClientUpdate、AggregationResult、RoundResult 及 TrainState。
# ============================================================================


TensorDict = Dict[str, torch.Tensor]


@dataclass
class ClientUpdate:
    """
    客户端每一轮训练后上传给服务端的结果。

    这个结构是 client 和 server 之间的统一接口。

    后续新增的聚合方法、训练策略和诊断插件都应复用这个结构，
    而不是让 client.py 和 server.py 互相强耦合。
    """

    # 客户端编号
    client_id: int

    # 当前联邦训练轮数
    round_id: int

    # 客户端训练集的数据规模，用于联邦聚合权重；不是本轮实际访问次数。
    num_samples: int

    # 本地模型相对全局模型的可聚合浮点 state_dict 差分。
    # 普通训练路径包含全部可聚合浮点状态；启用专家训练策略时由策略契约决定：
    # includes_expert_parameters_in_model_delta=True 时包含专家与非专家状态，
    # False 时只包含 non_expert 状态，专家更新通过 algorithm_payload 上传。
    # 非浮点 state（例如 BatchNorm.num_batches_tracked）不进入 model_delta。
    model_delta: TensorDict

    # 客户端本地训练指标
    # 例如：
    # train_loss
    # train_acc
    metrics: Dict[str, float] = field(default_factory=dict)

    # 客户端本地训练期间逐层辅助损失的样本加权平均。
    avg_aux_loss_by_layer: Dict[str, float] = field(default_factory=dict)

    # 客户端本地训练期间累加的 MoE 运行时统计；字段值必须遵循 ModelOutput 的 additive 约定。
    moe_stats: Dict[str, Any] = field(default_factory=dict)

    # 预留扩展字段
    # 后面可以放：
    # 算法专属 Evidence / 统计信息
    # method_evidence
    # method_evidence_summary
    # sgld_mean
    # sgld_var
    # router_stats
    extra: Dict[str, Any] = field(default_factory=dict)

    # 专家算法专属载荷，例如 Gradient Stream；不与轻量诊断 extra 混用。
    # model_delta 仍保留，用于 non_expert 的常规参数聚合。
    algorithm_payload: Dict[str, Any] = field(default_factory=dict)

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
            "avg_aux_loss_by_layer_keys": sorted(self.avg_aux_loss_by_layer.keys()),
            "moe_stats_keys": sorted(self.moe_stats.keys()),
            "extra_keys": sorted(self.extra.keys()),
            "algorithm_payload_keys": sorted(self.algorithm_payload.keys()),
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
    # num_state_entries：选中的 state_dict 条目数量，包括 buffer。
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

    各类数据按职责分别保存：
        train_metrics: 客户端本地训练指标的样本数加权平均，
            包含总训练损失、分类损失、辅助损失和准确率。
        test_metrics: 服务器全局模型的测试集指标。
        aggregation_info: non_expert / expert 聚合结果与算法诊断。
        client_diagnostics: 每个客户端的完整轻量诊断信息。
        avg_train_aux_loss_by_layer: 按实际处理样本数加权的逐层辅助损失。
        moe_stats: 客户端上报的可加统计量在本轮内累加。
    """

    # 当前轮数
    round_id: int

    # 本轮参与训练的客户端编号
    selected_clients: List[int]

    # 客户端本地训练指标的加权平均
    train_metrics: Dict[str, Optional[float]]

    # 服务器全局模型测试指标
    test_metrics: Dict[str, float]

    # 当前历史最佳准确率
    best_acc: float

    # 本轮聚合信息摘要，不包含客户端诊断和训练指标
    aggregation_info: Dict[str, Any] = field(default_factory=dict)

    # 每个客户端的完整轻量诊断信息
    client_diagnostics: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    avg_train_aux_loss_by_layer: Dict[str, float] = field(default_factory=dict)
    moe_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为用于 summary.json 的普通字典。"""
        return {
            "round_id": int(self.round_id),
            "selected_clients": list(self.selected_clients),
            "train_metrics": dict(self.train_metrics),
            "test_metrics": dict(self.test_metrics),
            "best_acc": float(self.best_acc),
            "aggregation_info": dict(self.aggregation_info),
            "client_diagnostics": dict(self.client_diagnostics),
            "avg_train_aux_loss_by_layer": dict(self.avg_train_aux_loss_by_layer),
            "moe_stats": dict(self.moe_stats),
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
    # History Filter 状态
    # Bayes 先验状态
    # 算法专属运行状态
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(
        self,
        include_best_metrics: bool = True,
    ) -> Dict[str, Any]:
        """
        转成普通 dict。

        include_best_metrics=False 时，仅输出运行状态和额外状态，
        避免 ServerTrainResult 顶层与嵌套 train_state 重复保存最佳指标。
        """
        result = {
            "round_id": int(self.round_id),
        }
        if include_best_metrics:
            result["best_acc"] = float(self.best_acc)
            result["best_round"] = int(self.best_round)
        result["extra"] = self.extra
        return result


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
        weighted=True 时优先按 metrics["num_processed_samples"]（本地实际处理的
        样本次数）加权；旧版 ClientUpdate 没有该字段时回退到 num_samples。
        这与聚合器使用的 num_samples（客户端数据集规模）是两个不同概念。
    """
    values = []

    for update in client_updates:
        if metric_name not in update.metrics:
            continue

        value = float(update.metrics[metric_name])
        if weighted:
            weight_value = update.metrics.get(
                "num_processed_samples",
                update.num_samples,
            )
            try:
                weight_float = float(weight_value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"客户端 {update.client_id} 的 num_processed_samples "
                    f"不是有效数值：{weight_value!r}"
                ) from exc
            if (
                not math.isfinite(weight_float)
                or weight_float <= 0
                or not weight_float.is_integer()
            ):
                raise ValueError(
                    f"客户端 {update.client_id} 的 num_processed_samples "
                    f"必须是正整数，当前值：{weight_value!r}"
                )
            weight = int(weight_float)
        else:
            weight = 1

        values.append((value, weight))

    if len(values) == 0:
        return default

    total_weight = sum(weight for _, weight in values)

    if total_weight <= 0:
        return default

    return sum(value * weight for value, weight in values) / total_weight


def average_client_aux_loss_by_layer(
    client_updates: Sequence[ClientUpdate],
) -> Dict[str, float]:
    """根据实际处理样本数加权平均各层 Auxiliary Loss。"""
    weighted_sums: Dict[str, float] = {}
    total_weights: Dict[str, int] = {}
    for update in client_updates:
        if not update.avg_aux_loss_by_layer:
            continue
        raw_weight = update.metrics.get("num_processed_samples", update.num_samples)
        try:
            weight_float = float(raw_weight)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"客户端 {update.client_id} 的 num_processed_samples 无效：{raw_weight!r}"
            ) from exc
        if not math.isfinite(weight_float) or weight_float <= 0 or not weight_float.is_integer():
            raise ValueError(
                f"客户端 {update.client_id} 的 num_processed_samples 必须是正整数：{raw_weight!r}"
            )
        weight = int(weight_float)
        for layer_name, raw_value in update.avg_aux_loss_by_layer.items():
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"客户端 {update.client_id} 的 {layer_name} 辅助损失非有限。")
            weighted_sums[layer_name] = weighted_sums.get(layer_name, 0.0) + value * weight
            total_weights[layer_name] = total_weights.get(layer_name, 0) + weight
    return {
        name: weighted_sums[name] / total_weights[name]
        for name in weighted_sums if total_weights[name] > 0
    }


def aggregate_client_moe_stats(
    client_updates: Sequence[ClientUpdate],
) -> Dict[str, Any]:
    """累加参与客户端产生的 MoE 运行时统计数据。"""
    aggregated: Any = None
    for update in client_updates:
        current = _normalize_moe_stats(update.moe_stats)
        if current:
            aggregated = _merge_additive_moe_stats(aggregated, current)
    return {} if aggregated is None else dict(aggregated)


# ============================================================================
# 11. 联邦客户端（FL Client）
# 包含客户端构建与选择、本地训练及客户端更新信息生成。
# ============================================================================


@dataclass(frozen=True)
class ClientTrainStats:
    """客户端本地训练损失、逐层辅助损失和 MoE 运行时统计。"""

    avg_loss: float
    train_acc: float
    num_samples: int
    num_batches: int
    avg_task_loss: float = 0.0
    avg_aux_loss: float = 0.0
    avg_aux_loss_by_layer: Dict[str, float] = field(default_factory=dict)
    moe_stats: Dict[str, Any] = field(default_factory=dict)

    def to_metrics(self) -> Dict[str, float]:
        """转成 ClientUpdate.metrics 使用的标量训练指标。"""
        return {
            "train_loss": float(self.avg_loss),
            "train_task_loss": float(self.avg_task_loss),
            "train_aux_loss": float(self.avg_aux_loss),
            "train_acc": float(self.train_acc),
            "num_batches": float(self.num_batches),
            "num_processed_samples": float(self.num_samples),
        }


ExpertEvidenceCollector = Callable[..., Mapping[str, Any]]
MethodClientDiagnosticsBuilder = Callable[[ClientUpdate], Mapping[str, Any]]


@dataclass
class ExpertClientTrainContext:
    """专家客户端训练策略在一轮本地训练中可访问的公共上下文。"""

    client_id: int
    round_id: int
    cfg: Any
    device: torch.device
    model: nn.Module
    global_state: Mapping[str, torch.Tensor]
    train_loader: DataLoader
    criterion: nn.Module
    non_expert_optimizer: optim.Optimizer
    optimization_groups: FederatedOptimizationGroups
    local_epochs: int
    local_steps: Optional[int]
    non_expert_grad_clip: Optional[float]

    @property
    def expert_parameters(self) -> Dict[str, nn.Parameter]:
        parameter_map = dict(self.model.named_parameters())
        return {
            name: parameter_map[name]
            for name in self.optimization_groups.expert_parameter_names
        }


@dataclass
class ExpertClientBatchContext:
    """一次本地 batch 反向传播后的专家策略上下文。"""

    train_context: ExpertClientTrainContext
    epoch: int
    batch_index: int
    local_step: int
    images: torch.Tensor
    targets: torch.Tensor
    logits: torch.Tensor
    loss: torch.Tensor


@dataclass(frozen=True)
class ExpertClientTrainResult:
    stats: ClientTrainStats
    algorithm_payload: Dict[str, Any]


class ExpertClientTrainingStrategy(ABC):
    """
    专家本地训练策略接口。

    框架先执行 forward/backward；step() 在反向传播后、non_expert
    optimizer.step() 前调用。策略负责专家参数的本地更新，并可在结束时
    上传 algorithm_payload。non_expert 参数仍由框架的普通优化器更新。

    默认由框架对全部模型梯度统一执行 grad_clip；若策略声明自己负责
    expert 梯度裁剪，则框架只裁剪 non_expert 梯度。额外的参数约束或回缩
    由具体策略决定，模型本身不触发。
    """

    @property
    def strategy_name(self) -> str:
        return self.__class__.__name__

    @property
    def includes_expert_parameters_in_model_delta(self) -> bool:
        """
        策略是否已在本地更新专家参数，并要求框架把专家参数差一并上传。

        若策略要求框架通过 model_delta 上传专家参数差分，则返回 True；
        若策略通过 algorithm_payload 上传专家更新，则保持 False。
        """
        return False

    @property
    def handles_expert_grad_clip(self) -> bool:
        """策略是否自行裁剪 expert 梯度；默认由框架统一裁剪全部模型梯度。"""
        return False

    def on_train_start(self, context: ExpertClientTrainContext) -> None:
        """每个客户端每轮训练开始时调用，可用于初始化本轮状态。"""

    @abstractmethod
    def step(self, context: ExpertClientBatchContext) -> None:
        """读取 expert 梯度并执行算法定义的专家本地更新。"""
        raise NotImplementedError

    def on_train_end(
        self,
        context: ExpertClientTrainContext,
    ) -> Mapping[str, Any]:
        """返回本轮需要上传的专家算法载荷。"""
        return {}


ExpertClientTrainingStrategyBuilder = Callable[
    [Any, int],
    ExpertClientTrainingStrategy,
]


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
        expert_client_training_strategy: Optional[ExpertClientTrainingStrategy] = None,
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
        self.expert_client_training_strategy = expert_client_training_strategy

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
        global_state_cpu = clone_state_dict(
            state_dict_to(
                global_model.state_dict(),
                device="cpu",
            )
        )

        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)
        local_model.train()

        criterion = build_criterion(self.cfg)
        local_epochs = int(
            _cfg_get(
                self.cfg,
                "local_epochs",
                _default_config_value("local_epochs"),
            )
        )
        configured_local_steps = _cfg_get(self.cfg, "local_steps", _default_config_value("local_steps"))
        local_steps = (
            None if configured_local_steps is None
            else int(configured_local_steps)
        )
        grad_clip = _get_grad_clip(self.cfg)
        algorithm_payload: Dict[str, Any] = {}
        optimization_groups: Optional[FederatedOptimizationGroups] = None

        if self.expert_client_training_strategy is None:
            # 旧算法路径保持不变：普通优化器更新整个模型。
            optimizer = build_optimizer(
                model=local_model,
                cfg=self.cfg,
            )
            stats = train_local_model(
                model=local_model,
                train_loader=self.train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                local_epochs=local_epochs,
                local_steps=local_steps,
                grad_clip=grad_clip,
            )
        else:
            # 专家算法路径：non_expert 使用原优化器，expert 交给注入策略。
            optimization_groups = build_federated_optimization_groups(
                model=local_model,
                expected_num_experts=int(_cfg_get(self.cfg, "num_experts", _default_config_value("num_experts"))),
                strict=True,
            )
            optimizer = build_optimizer(
                model=local_model,
                cfg=self.cfg,
                param_names=optimization_groups.non_expert_parameters,
            )
            train_result = train_local_model_with_expert_strategy(
                model=local_model,
                train_loader=self.train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                local_epochs=local_epochs,
                local_steps=local_steps,
                grad_clip=grad_clip,
                client_id=self.client_id,
                round_id=int(round_id),
                cfg=self.cfg,
                global_state=global_state_cpu,
                optimization_groups=optimization_groups,
                strategy=self.expert_client_training_strategy,
            )
            stats = train_result.stats
            algorithm_payload = train_result.algorithm_payload

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

        include_expert_delta = (
            self.expert_client_training_strategy is not None
            and self.expert_client_training_strategy.includes_expert_parameters_in_model_delta
        )
        if self.expert_client_training_strategy is not None and not isinstance(
            include_expert_delta,
            bool,
        ):
            raise TypeError(
                "ExpertClientTrainingStrategy.includes_expert_parameters_in_model_delta "
                "必须返回 bool。"
            )

        model_delta = subtract_state_dict(
            local_state=local_state_cpu,
            global_state=global_state_cpu,
            param_names=(
                optimization_groups.non_expert_state_names
                if optimization_groups is not None and not include_expert_delta
                else None
            ),
            strict=True,
        )

        check_finite_state_dict(model_delta)

        update = ClientUpdate(
            client_id=self.client_id,
            round_id=int(round_id),
            num_samples=self.num_samples,
            model_delta=model_delta,
            metrics=stats.to_metrics(),
            avg_aux_loss_by_layer=dict(stats.avg_aux_loss_by_layer),
            moe_stats=dict(stats.moe_stats),
            extra={
                "optimizer": get_optimizer_type(self.cfg),
                "local_epochs": int(local_epochs),
                "configured_local_steps": (
                    int(local_steps) if local_steps is not None else None
                ),
                "local_training_mode": (
                    "steps" if local_steps is not None else "epochs"
                ),
                "actual_local_steps": int(stats.num_batches),
                "grad_clip": float(grad_clip) if grad_clip is not None else None,
                "expert_training_strategy": (
                    self.expert_client_training_strategy.strategy_name
                    if self.expert_client_training_strategy is not None
                    else None
                ),
                **method_extra,
            },
            algorithm_payload=algorithm_payload,
        )

        del local_model
        del optimizer
        del criterion

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return update


def _iter_local_training_batches(
    train_loader: DataLoader,
    local_epochs: int,
    local_steps: Optional[int],
):
    """
    统一生成本地训练 batch。

    local_steps 非 None 时，严格生成 K 个 batch；DataLoader 耗尽后重新迭代。
    local_steps 为 None 时，完整遍历 local_epochs 个 epoch。
    """
    if (
        isinstance(local_epochs, bool)
        or not isinstance(local_epochs, int)
        or local_epochs < 0
    ):
        raise ValueError(
            "local_epochs 必须是非负整数，"
            f"当前值：{local_epochs!r}"
        )

    if local_steps is not None:
        if (
            isinstance(local_steps, bool)
            or not isinstance(local_steps, int)
            or local_steps <= 0
        ):
            raise ValueError(
                "local_steps 必须是正整数或 None，"
                f"当前值：{local_steps!r}"
            )

        batches_per_epoch = len(train_loader)
        if batches_per_epoch <= 0:
            raise ValueError(
                "客户端 DataLoader 没有可训练 batch，无法执行固定 local_steps。"
            )

        iterator = iter(train_loader)
        for step_index in range(local_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise ValueError(
                        "客户端 DataLoader 为空，无法执行固定 local_steps。"
                    ) from exc

            yield (
                step_index // batches_per_epoch,
                step_index % batches_per_epoch,
                batch,
            )
        return

    if local_epochs <= 0:
        raise ValueError(
            "local_steps 为 None 时，local_epochs 必须大于 0；"
            f"当前值：{local_epochs}"
        )

    for epoch in range(local_epochs):
        for batch_index, batch in enumerate(train_loader):
            yield epoch, batch_index, batch


def train_local_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    local_epochs: int,
    grad_clip: Optional[float] = None,
    local_steps: Optional[int] = None,
) -> ClientTrainStats:
    """
    训练一个客户端本地模型。

    local_steps 优先于 local_epochs；为 None 时按完整 epoch 训练。
    总训练损失由分类损失和模型提供的辅助损失组成。
    普通模型没有 aux_loss 时，训练行为等价于原来的 CrossEntropyLoss 路径。
    """
    model.train()

    total_loss = 0.0
    total_task_loss = 0.0
    total_aux_loss = 0.0
    aux_loss_by_layer_sums: Dict[str, float] = {}
    aux_loss_by_layer_weights: Dict[str, int] = {}
    moe_stats: Any = None
    total_correct = 0
    total_samples = 0
    total_batches = 0

    for _, _, batch in _iter_local_training_batches(
        train_loader=train_loader,
        local_epochs=local_epochs,
        local_steps=local_steps,
    ):
        images, targets = unpack_batch(batch)

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        model_output = extract_model_output(outputs)
        loss_output = compute_model_loss(
            outputs=model_output,
            targets=targets,
            criterion=criterion,
        )
        logits = model_output.logits
        batch_size = int(targets.size(0))
        for layer_name, layer_loss in _extract_aux_loss_by_layer(
            model_output.aux_loss_by_layer
        ).items():
            aux_loss_by_layer_sums[layer_name] = (
                aux_loss_by_layer_sums.get(layer_name, 0.0) + layer_loss * batch_size
            )
            aux_loss_by_layer_weights[layer_name] = (
                aux_loss_by_layer_weights.get(layer_name, 0) + batch_size
            )
        batch_moe_stats = _normalize_moe_stats(model_output.moe_stats)
        if batch_moe_stats:
            moe_stats = _merge_additive_moe_stats(moe_stats, batch_moe_stats)

        loss_output.total_loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(grad_clip),
            )

        optimizer.step()

        total_loss += float(loss_output.total_loss.item()) * batch_size
        total_task_loss += float(loss_output.task_loss.item()) * batch_size
        total_aux_loss += float(loss_output.aux_loss.item()) * batch_size
        total_correct += int(logits.argmax(dim=1).eq(targets).sum().item())
        total_samples += batch_size
        total_batches += 1

    if total_samples <= 0:
        raise ValueError("客户端本地训练没有处理任何样本。")

    return ClientTrainStats(
        avg_loss=total_loss / total_samples,
        train_acc=100.0 * total_correct / total_samples,
        num_samples=total_samples,
        num_batches=total_batches,
        avg_task_loss=total_task_loss / total_samples,
        avg_aux_loss=total_aux_loss / total_samples,
        avg_aux_loss_by_layer={
            name: aux_loss_by_layer_sums[name] / aux_loss_by_layer_weights[name]
            for name in aux_loss_by_layer_sums if aux_loss_by_layer_weights[name] > 0
        },
        moe_stats={} if moe_stats is None else dict(moe_stats),
    )


def _detach_algorithm_payload_to_cpu(value: Any) -> Any:
    """递归 detach 算法载荷中的 Tensor，并移至 CPU。"""
    if torch.is_tensor(value):
        return value.detach().to("cpu").clone()
    if isinstance(value, Mapping):
        return {
            key: _detach_algorithm_payload_to_cpu(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_detach_algorithm_payload_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_algorithm_payload_to_cpu(item) for item in value)
    return value


def _validate_algorithm_payload_finite(
    value: Any,
    *,
    client_id: int,
    path: str = "algorithm_payload",
) -> None:
    """递归检查算法载荷中的浮点 Tensor / 数组 / 标量是否包含 NaN 或 Inf。"""
    if torch.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            tensor_values = (
                value.coalesce().values()
                if value.layout == torch.sparse_coo
                else value
            )
            if not bool(torch.isfinite(tensor_values).all().item()):
                raise ValueError(
                    f"客户端 {client_id} 的算法载荷 {path} 包含 NaN 或 Inf。"
                )
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_algorithm_payload_finite(
                item,
                client_id=client_id,
                path=f"{path}[{key!r}]",
            )
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_algorithm_payload_finite(
                item,
                client_id=client_id,
                path=f"{path}[{index}]",
            )
        return

    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(
                f"客户端 {client_id} 的算法载荷 {path} 包含 NaN 或 Inf。"
            )
        return

    if isinstance(value, (complex, np.complexfloating)):
        numeric_value = complex(value)
        if not (
            math.isfinite(numeric_value.real)
            and math.isfinite(numeric_value.imag)
        ):
            raise ValueError(
                f"客户端 {client_id} 的算法载荷 {path} 包含 NaN 或 Inf。"
            )
        return

    if isinstance(value, np.ndarray):
        if (
            np.issubdtype(value.dtype, np.floating)
            or np.issubdtype(value.dtype, np.complexfloating)
        ) and not bool(np.isfinite(value).all()):
            raise ValueError(
                f"客户端 {client_id} 的算法载荷 {path} 包含 NaN 或 Inf。"
            )


def train_local_model_with_expert_strategy(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    local_epochs: int,
    grad_clip: Optional[float],
    client_id: int,
    round_id: int,
    cfg: Any,
    global_state: Mapping[str, torch.Tensor],
    optimization_groups: FederatedOptimizationGroups,
    strategy: ExpertClientTrainingStrategy,
    local_steps: Optional[int] = None,
) -> ExpertClientTrainResult:
    """
    使用专家训练策略更新 expert，同时用普通 optimizer 更新 non_expert。

    策略 step() 在 loss.backward() 之后调用；传入的 optimizer 只包含
    non_expert 可训练参数，因此策略不会与普通 optimizer 重复更新 expert。
    """
    if (
        isinstance(local_epochs, bool)
        or not isinstance(local_epochs, int)
        or local_epochs < 0
    ):
        raise ValueError(
            "local_epochs 必须是非负整数，"
            f"当前值：{local_epochs!r}"
        )
    if not isinstance(strategy, ExpertClientTrainingStrategy):
        raise TypeError(
            "strategy 必须是 ExpertClientTrainingStrategy，"
            f"当前类型：{type(strategy).__name__}"
        )

    parameter_map = dict(model.named_parameters())
    expert_parameter_names = optimization_groups.expert_parameter_names
    if not expert_parameter_names:
        raise ValueError("专家训练策略已启用，但模型没有可训练 expert 参数。")

    expert_parameters = [
        parameter_map[name]
        for name in expert_parameter_names
    ]
    non_expert_parameters = [
        parameter_map[name]
        for name in optimization_groups.non_expert_parameters
    ]
    if not non_expert_parameters:
        raise ValueError("模型没有可供普通本地优化器更新的 non_expert 参数。")

    context = ExpertClientTrainContext(
        client_id=int(client_id),
        round_id=int(round_id),
        cfg=cfg,
        device=torch.device(device),
        model=model,
        global_state=global_state,
        train_loader=train_loader,
        criterion=criterion,
        non_expert_optimizer=optimizer,
        optimization_groups=optimization_groups,
        local_epochs=int(local_epochs),
        local_steps=local_steps,
        non_expert_grad_clip=grad_clip,
    )

    model.train()
    strategy.on_train_start(context)

    total_loss = 0.0
    total_task_loss = 0.0
    total_aux_loss = 0.0
    aux_loss_by_layer_sums: Dict[str, float] = {}
    aux_loss_by_layer_weights: Dict[str, int] = {}
    moe_stats: Any = None
    total_correct = 0
    total_samples = 0
    total_batches = 0

    def run_batch(
        batch: Any,
        epoch: int,
        batch_index: int,
    ) -> None:
        nonlocal total_loss, total_task_loss, total_aux_loss, moe_stats
        nonlocal total_correct, total_samples, total_batches

        images, targets = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 清空全部参数梯度；optimizer 本身只管理 non_expert 参数。
        model.zero_grad(set_to_none=True)

        outputs = model(images)
        model_output = extract_model_output(outputs)
        loss_output = compute_model_loss(
            outputs=model_output,
            targets=targets,
            criterion=criterion,
        )
        logits = model_output.logits
        batch_size = int(targets.size(0))
        for layer_name, layer_loss in _extract_aux_loss_by_layer(
            model_output.aux_loss_by_layer
        ).items():
            aux_loss_by_layer_sums[layer_name] = (
                aux_loss_by_layer_sums.get(layer_name, 0.0) + layer_loss * batch_size
            )
            aux_loss_by_layer_weights[layer_name] = (
                aux_loss_by_layer_weights.get(layer_name, 0) + batch_size
            )
        batch_moe_stats = _normalize_moe_stats(model_output.moe_stats)
        if batch_moe_stats:
            moe_stats = _merge_additive_moe_stats(moe_stats, batch_moe_stats)

        loss_output.total_loss.backward()

        if grad_clip is not None and grad_clip > 0:
            handles_expert_grad_clip = strategy.handles_expert_grad_clip
            if not isinstance(handles_expert_grad_clip, bool):
                raise TypeError(
                    "ExpertClientTrainingStrategy.handles_expert_grad_clip "
                    "必须返回 bool。"
                )
            clip_parameters = (
                non_expert_parameters
                if handles_expert_grad_clip
                else model.parameters()
            )
            torch.nn.utils.clip_grad_norm_(
                clip_parameters,
                max_norm=float(grad_clip),
            )

        strategy.step(
            ExpertClientBatchContext(
                train_context=context,
                epoch=int(epoch),
                batch_index=int(batch_index),
                local_step=int(total_batches + 1),
                images=images,
                targets=targets,
                logits=logits,
                loss=loss_output.total_loss,
            )
        )

        # 防止专家梯度跨 batch 意外累积；策略需要保留时应自行 detach/clone。
        for parameter in expert_parameters:
            parameter.grad = None

        optimizer.step()

        total_loss += float(loss_output.total_loss.item()) * batch_size
        total_task_loss += float(loss_output.task_loss.item()) * batch_size
        total_aux_loss += float(loss_output.aux_loss.item()) * batch_size
        total_correct += int(logits.argmax(dim=1).eq(targets).sum().item())
        total_samples += batch_size
        total_batches += 1

    for epoch, batch_index, batch in _iter_local_training_batches(
        train_loader=train_loader,
        local_epochs=local_epochs,
        local_steps=local_steps,
    ):
        run_batch(
            batch=batch,
            epoch=epoch,
            batch_index=batch_index,
        )

    if total_samples <= 0:
        raise ValueError("专家策略本地训练没有处理任何样本。")

    payload = strategy.on_train_end(context)
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise TypeError(
            "strategy.on_train_end() 必须返回 Mapping 或 None，"
            f"当前类型：{type(payload).__name__}"
        )

    stats = ClientTrainStats(
        avg_loss=total_loss / total_samples,
        train_acc=100.0 * total_correct / total_samples,
        num_samples=total_samples,
        num_batches=total_batches,
        avg_task_loss=total_task_loss / total_samples,
        avg_aux_loss=total_aux_loss / total_samples,
        avg_aux_loss_by_layer={
            name: aux_loss_by_layer_sums[name] / aux_loss_by_layer_weights[name]
            for name in aux_loss_by_layer_sums if aux_loss_by_layer_weights[name] > 0
        },
        moe_stats={} if moe_stats is None else dict(moe_stats),
    )
    return ExpertClientTrainResult(
        stats=stats,
        algorithm_payload=_detach_algorithm_payload_to_cpu(dict(payload)),
    )


def build_criterion(cfg: Any) -> nn.Module:
    """
    构建本地训练 loss 函数。

    第一版只使用 CrossEntropyLoss。
    """
    label_smoothing = float(
        _cfg_get(cfg, "label_smooth", _default_config_value("label_smooth"))
    )

    return nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
    )


def build_optimizer(
    model: nn.Module,
    cfg: Any,
    param_names: Optional[Iterable[str]] = None,
) -> optim.Optimizer:
    """
    根据 cfg.optimizer 构建优化器。

    当前支持：
        sgd
        adam
        adamw
    """
    optimizer_type = get_optimizer_type(cfg)
    optimizer_cfg = _cfg_get(
        cfg,
        "optimizer",
        _default_config_value("optimizer"),
    )

    lr = float(
        _cfg_get(optimizer_cfg, "lr", _default_config_value("optimizer.lr"))
    )
    weight_decay = float(
        _cfg_get(
            optimizer_cfg,
            "weight_decay",
            _default_config_value("optimizer.weight_decay"),
        )
    )

    if param_names is None:
        params = [
            param
            for param in model.parameters()
            if param.requires_grad
        ]
    else:
        named_parameters = dict(model.named_parameters())
        selected_names = list(param_names)
        if len(selected_names) != len(set(selected_names)):
            raise ValueError("build_optimizer 的 param_names 不能包含重复名称。")

        missing_names = [
            name for name in selected_names
            if name not in named_parameters
        ]
        if missing_names:
            raise KeyError(
                f"build_optimizer 找不到指定 Parameter：{missing_names}"
            )

        params = [
            named_parameters[name]
            for name in selected_names
            if named_parameters[name].requires_grad
        ]

    if len(params) == 0:
        raise ValueError("模型没有可训练参数。")

    if optimizer_type == "sgd":
        momentum = float(
            _cfg_get(
                optimizer_cfg,
                "momentum",
                _default_config_value("optimizer.momentum"),
            )
        )
        nesterov = _cfg_get_bool(
            optimizer_cfg,
            "nesterov",
            _default_config_value("optimizer.nesterov"),
        )

        return optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    if optimizer_type == "adam":
        betas = _cfg_get(
            optimizer_cfg,
            "betas",
            _default_config_value("optimizer.betas"),
        )
        eps = float(
            _cfg_get(
                optimizer_cfg,
                "eps",
                _default_config_value("optimizer.eps"),
            )
        )

        return optim.Adam(
            params,
            lr=lr,
            betas=tuple(betas),
            eps=eps,
            weight_decay=weight_decay,
        )

    if optimizer_type == "adamw":
        betas = _cfg_get(
            optimizer_cfg,
            "betas",
            _default_config_value("optimizer.betas"),
        )
        eps = float(
            _cfg_get(
                optimizer_cfg,
                "eps",
                _default_config_value("optimizer.eps"),
            )
        )

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
    optimizer_cfg = _cfg_get(
        cfg,
        "optimizer",
        _default_config_value("optimizer"),
    )
    optimizer_type = _cfg_get(optimizer_cfg, "type", _default_config_value("optimizer.type"))

    return str(optimizer_type).lower().strip()


def build_clients(
    cfg: Any,
    client_loaders: Sequence[DataLoader],
    device: torch.device | str,
    client_evidence_loaders: Optional[Sequence[DataLoader]] = None,
    expert_evidence_collector: Optional[ExpertEvidenceCollector] = None,
    expert_client_training_strategy_builder: Optional[
        ExpertClientTrainingStrategyBuilder
    ] = None,
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

        expert_client_training_strategy = None
        if expert_client_training_strategy_builder is not None:
            expert_client_training_strategy = expert_client_training_strategy_builder(
                cfg,
                int(client_id),
            )
            if not isinstance(
                expert_client_training_strategy,
                ExpertClientTrainingStrategy,
            ):
                raise TypeError(
                    "expert_client_training_strategy_builder 必须返回 "
                    "ExpertClientTrainingStrategy，"
                    f"当前类型：{type(expert_client_training_strategy).__name__}"
                )

        clients.append(
            FLClient(
                client_id=client_id,
                train_loader=train_loader,
                evidence_loader=evidence_loader,
                cfg=cfg,
                device=device,
                expert_evidence_collector=expert_evidence_collector,
                expert_client_training_strategy=expert_client_training_strategy,
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
    optimizer_cfg = _cfg_get(
        cfg,
        "optimizer",
        _default_config_value("optimizer"),
    )

    value = _cfg_get(
        optimizer_cfg,
        "grad_clip",
        _default_config_value("grad_clip"),
    )

    if value is None:
        value = _cfg_get(
            cfg,
            "grad_clip",
            _default_config_value("grad_clip"),
        )

    if value is None:
        return None

    value = float(value)

    if value <= 0:
        return None

    return value


# ============================================================================
# 12. 参数聚合（Aggregation）
# 包含聚合接口、权重处理、Non-Expert 参数聚合及算法插件接口。
# ============================================================================


@dataclass(frozen=True)
class MethodContext:
    """
    提供给专家聚合插件使用的通用服务端运行上下文。

    base.py 只管理公共资源及其生命周期。
    算法插件可以读取所需资源，而无需在 base.py 中引入具体算法名称分支。
    """

    cfg: Any
    device: torch.device
    dataset_name: str
    server_evidence_loader: Optional[DataLoader] = None
    model_builder: Optional[ModelBuilder] = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    global_model: Optional[nn.Module] = None
    param_groups: Optional[ParamGroups] = None
    optimization_groups: Optional[FederatedOptimizationGroups] = None


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
        """在插件构建完成后注入公共服务端资源。"""
        if method_context is not None and not isinstance(method_context, MethodContext):
            raise TypeError(
                "method_context 必须是 MethodContext 或 None，"
                f"当前类型：{type(method_context).__name__}"
            )
        self.method_context = method_context

    def initialize_model(self, model: nn.Module) -> None:
        """在需要时，对全局模型执行算法专属初始化。"""

    def after_aggregate(self, model: nn.Module) -> None:
        """在需要时，执行算法专属的聚合后处理操作。"""

    def build_post_aggregate_diagnostics(
        self,
        model: nn.Module,
    ) -> Dict[str, Any]:
        """在 after_aggregate 执行完成后，根据服务端模型构建诊断信息。"""
        return {}

    @property
    @abstractmethod
    def method_name(self) -> str:
        """
        当前聚合方法名称。

        子类返回自己启动文件对应的方法名称。
        """
        raise NotImplementedError

    @property
    def consumes_algorithm_payload(self) -> bool:
        """
        是否消费 ClientUpdate.algorithm_payload。

        默认 False。使用专家自定义训练策略时，客户端 model_delta 只包含
        non_expert 状态，因此对应专家聚合器必须覆盖 aggregate() 并声明 True。
        """
        return False

    @property
    def handles_manifold_retraction(self) -> bool:
        """
        算法是否在自己的服务器聚合生命周期中负责流形更新与回缩。

        base.py 不再根据模型类型推断或自动执行 QR / Retract；
        算法可在 aggregate() 或 after_aggregate() 中实现自己的几何操作。
        """
        return False

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

        这是所有普通加权 delta 聚合方法的公共流程。

        专家聚合插件可以覆盖本方法，直接读取 ClientUpdate.algorithm_payload
        并返回新的 AggregationResult；non_expert 聚合仍由固定 uniform 聚合器负责。

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

        # 聚合、校验和诊断复用相同的参数名称，
        # 因此一次性迭代对象只应物化一次。
        if param_names is not None:
            param_names = list(param_names)

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
        """构建通用且与具体算法无关的聚合诊断信息。

        weights 参数保留在接口中，以兼容现有聚合器实现；但规范的聚合权重
        只存储在 AggregationResult.weights 中。num_state_entries 统计选中的
        state_dict 条目数量（包括 buffer），并不代表可训练参数数量。
        """
        _ = weights
        num_state_entries = (
            len(list(param_names)) if param_names is not None else None
        )

        return {
            "method": self.method_name,
            "param_group": self.param_group_name,
            "num_clients": len(client_updates),
            "num_state_entries": num_state_entries,
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

            if not isinstance(update.algorithm_payload, Mapping):
                raise TypeError(
                    f"客户端 {update.client_id} 的 algorithm_payload 必须是 Mapping，"
                    f"当前类型：{type(update.algorithm_payload).__name__}"
                )

            _validate_algorithm_payload_finite(
                update.algorithm_payload,
                client_id=int(update.client_id),
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



# 固定的 Non-Expert 聚合逻辑及 Expert 扩展接口。
@dataclass
class AggregatorBundle:
    """由固定的 Non-Expert 聚合器和注入式 Expert 聚合器组成的聚合器集合。"""

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
    """构建固定的 Non-Expert 等权聚合器及注入式 Expert 聚合逻辑。"""
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
# 13. 联邦服务端（FL Server）
# 包含全局模型管理、训练轮次协调、参数聚合调用及全局评估。
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

    @property
    def best_acc(self) -> float:
        """最终最佳准确率的唯一数据源为 TrainState。"""
        return float(self.train_state.best_acc)

    @property
    def best_round(self) -> int:
        """最终最佳轮次的唯一数据源为 TrainState。"""
        return int(self.train_state.best_round)

    def to_dict(self) -> Dict[str, Any]:
        """转换为最终实验摘要，最佳指标只在顶层保存一次。"""
        return {
            "best_acc": self.best_acc,
            "best_round": self.best_round,
            "round_results": [
                item.to_dict()
                for item in self.round_results
            ],
            "train_state": self.train_state.to_dict(
                include_best_metrics=False,
            ),
        }


def collect_client_update_steps(
    client_updates: Sequence[ClientUpdate],
) -> Dict[int, int]:
    """
    校验并收集本轮每个客户端的实际本地梯度更新次数。

    Epoch 和 Steps 两种模式都必须满足更新次数为合法正整数值；
    是否要求客户端之间一致，由训练模式单独决定。
    """
    if not client_updates:
        raise ValueError("client_updates 不能为空。")

    steps_by_client: Dict[int, int] = {}
    for update in client_updates:
        raw_client_id = update.client_id
        if (
            isinstance(raw_client_id, bool)
            or not isinstance(raw_client_id, int)
            or raw_client_id < 0
        ):
            raise ValueError(
                f"客户端 ID 必须是非负整数，当前值：{raw_client_id!r}。"
            )
        client_id = raw_client_id
        if client_id in steps_by_client:
            raise ValueError(f"本轮出现重复的客户端更新：client_id={client_id}。")

        metrics = update.metrics
        if not isinstance(metrics, Mapping):
            raise ValueError(
                f"客户端 {client_id} 的 metrics 必须是 Mapping，"
                f"当前类型：{type(metrics).__name__}。"
            )
        raw_steps = metrics.get("num_batches")
        valid_steps = (
            not isinstance(raw_steps, bool)
            and (
                isinstance(raw_steps, int)
                or (
                    isinstance(raw_steps, float)
                    and math.isfinite(raw_steps)
                    and raw_steps.is_integer()
                )
            )
            and int(raw_steps) > 0
        )
        if not valid_steps:
            raise ValueError(
                f"客户端 {client_id} 的实际本地更新次数无效："
                f"metrics['num_batches']={raw_steps!r}；"
                "必须是有限的正整数值。"
            )
        steps_by_client[client_id] = int(raw_steps)

    return steps_by_client


def _validate_equal_client_update_steps_by_client(
    steps_by_client: Mapping[int, int],
    expected_steps: Optional[int] = None,
) -> int:
    """检查 Steps 模式的客户端实际更新次数一致性及配置值约束。"""
    if not steps_by_client:
        raise ValueError("steps_by_client 不能为空。")
    if expected_steps is not None and not _is_positive_int_value(expected_steps):
        raise ValueError(
            f"expected_steps 必须是严格正整数，当前值：{expected_steps!r}。"
        )

    if expected_steps is not None:
        mismatched_steps = {
            client_id: steps
            for client_id, steps in steps_by_client.items()
            if steps != expected_steps
        }
        if mismatched_steps:
            raise ValueError(
                "Steps 模式要求每个客户端的实际更新次数严格等于 "
                f"local_steps={expected_steps}；"
                f"当前不符合要求的客户端更新次数：{mismatched_steps}。"
            )

    unique_steps = set(steps_by_client.values())
    if len(unique_steps) != 1:
        raise ValueError(
            "Steps 模式要求本轮所有参与客户端执行相同数量的本地梯度更新；"
            f"当前实际更新次数：{dict(steps_by_client)}。"
        )

    return unique_steps.pop()


def validate_equal_client_update_steps(
    client_updates: Sequence[ClientUpdate],
    expected_steps: Optional[int] = None,
) -> int:
    """
    检查 Steps 模式下所有客户端的实际本地梯度更新次数。

    可选 expected_steps 用于进一步确保实际更新次数严格等于配置的 local_steps。
    """
    steps_by_client = collect_client_update_steps(client_updates)
    return _validate_equal_client_update_steps_by_client(
        steps_by_client,
        expected_steps=expected_steps,
    )


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
        model_builder: Optional[ModelBuilder] = None,
        expert_client_training_strategy_builder: Optional[
            ExpertClientTrainingStrategyBuilder
        ] = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)

        # ModelBuilder 优先从显式参数注入，也允许从 MethodContext 传入。
        effective_model_builder = model_builder
        if effective_model_builder is None and method_context is not None:
            effective_model_builder = method_context.model_builder

        # 全局模型默认常驻 CPU；客户端训练时 deepcopy 后移动到 GPU。
        if global_model is None:
            if effective_model_builder is None:
                raise ValueError(
                    "FLServer 未收到 global_model 或 model_builder；请通过外部 ModelBuilder 注入模型。"
                )
            self.global_model = build_model(cfg, effective_model_builder)
        else:
            self.global_model = global_model

        self.global_model.to("cpu")

        self.param_groups = build_param_groups(
            model=self.global_model,
            expected_num_experts=int(_cfg_get(cfg, "num_experts", _default_config_value("num_experts"))),
            strict=True,
        )
        self.optimization_groups = build_federated_optimization_groups(
            model=self.global_model,
            param_groups=self.param_groups,
            strict=True,
        )

        self.clients = build_clients(
            cfg=cfg,
            client_loaders=client_loaders,
            # 方法级 evidence 专用 loader。
            # 本地训练仍然使用 client_loaders；
            # 统计 方法插件 时由客户端使用 client_evidence_loaders。
            client_evidence_loaders=client_evidence_loaders,
            device=self.device,
            expert_evidence_collector=expert_evidence_collector,
            expert_client_training_strategy_builder=expert_client_training_strategy_builder,
        )

        self.test_loader = test_loader
        self.method_client_diagnostics_builder = method_client_diagnostics_builder
        if method_context is None:
            self.method_context = None
        else:
            self.method_context = MethodContext(
                cfg=method_context.cfg,
                device=method_context.device,
                dataset_name=method_context.dataset_name,
                server_evidence_loader=method_context.server_evidence_loader,
                model_builder=effective_model_builder,
                global_model=self.global_model,
                param_groups=self.param_groups,
                optimization_groups=self.optimization_groups,
                extra=method_context.extra,
            )

        self.aggregators = build_aggregators(
            cfg=cfg,
            expert_aggregator_builder=expert_aggregator_builder,
            method_context=self.method_context,
        )
        expert_aggregator = self.aggregators.expert
        expert_aggregator.initialize_model(self.global_model)
        strategy_enabled = expert_client_training_strategy_builder is not None
        consumes_algorithm_payload = expert_aggregator.consumes_algorithm_payload
        handles_manifold_retraction = expert_aggregator.handles_manifold_retraction

        if not isinstance(consumes_algorithm_payload, bool):
            raise TypeError(
                "Aggregator.consumes_algorithm_payload 必须返回 bool。"
            )
        if not isinstance(handles_manifold_retraction, bool):
            raise TypeError(
                "Aggregator.handles_manifold_retraction 必须返回 bool。"
            )

        if consumes_algorithm_payload and not strategy_enabled:
            raise ValueError(
                "专家聚合器声明 consumes_algorithm_payload=True，"
                "但没有启用 expert_client_training_strategy_builder；"
                "客户端不会生成对应算法载荷。"
            )
        if strategy_enabled and not consumes_algorithm_payload:
            invalid_clients = []
            for client in self.clients:
                strategy = client.expert_client_training_strategy
                includes_delta = (
                    strategy.includes_expert_parameters_in_model_delta
                    if strategy is not None
                    else False
                )
                if not isinstance(includes_delta, bool):
                    raise TypeError(
                        "ExpertClientTrainingStrategy.includes_expert_parameters_in_model_delta "
                        "必须返回 bool。"
                    )
                if not includes_delta:
                    invalid_clients.append(client.client_id)
            if invalid_clients:
                raise ValueError(
                    "当前专家聚合器不消费 algorithm_payload，因此启用的专家训练策略"
                    "必须声明 includes_expert_parameters_in_model_delta=True；"
                    f"不满足的客户端：{invalid_clients}"
                )
        if (
            consumes_algorithm_payload
            and type(expert_aggregator).aggregate is Aggregator.aggregate
        ):
            raise TypeError(
                "声明 consumes_algorithm_payload=True 的专家聚合器必须"
                "真正覆盖 aggregate()，不能使用 Aggregator 的默认 Delta 聚合实现。"
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
        rounds = int(
            _cfg_get(self.cfg, "rounds", _default_config_value("rounds"))
        )
        frac = float(
            _cfg_get(self.cfg, "frac", _default_config_value("frac"))
        )
        seed = int(
            _cfg_get(self.cfg, "seed", _default_config_value("seed"))
        )

        logging_cfg = _cfg_get(
            self.cfg,
            "logging",
            _default_config_value("logging"),
        )

        # 日志行为固定：每轮记录，不提供配置入口。
        log_every = 1
        console_round_summary = True
        file_round_detail = True
        progress_in_non_tty = False

        # 仅进度条显示本身保留为可配置项。
        progress_bar_enabled = _cfg_get_bool(
            logging_cfg,
            "progress_bar",
            _default_config_value("logging.progress_bar"),
        )

        if rounds <= 0:
            raise ValueError(f"rounds 必须大于 0，当前值：{rounds}")

        self.print_startup_summary()

        # 预估整个实验的客户端训练步数。
        # 进度条单位：完成一个客户端本地训练。
        total_client_steps = len(
            select_clients(
                clients=self.clients,
                frac=frac,
                round_id=1,
                seed=seed,
            )
        ) * rounds

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

                # Epoch 模式下，每个客户端的实际更新次数由 DataLoader 长度决定。
                # 在开始本轮本地训练前先检查，避免完成昂贵训练后才发现步数不一致。
                if _cfg_get(self.cfg, "local_steps", _default_config_value("local_steps")) is None:
                    local_epochs = int(
            _cfg_get(
                self.cfg,
                "local_epochs",
                _default_config_value("local_epochs"),
            )
        )
                    planned_steps_by_client = {
                        int(client.client_id): (
                            local_epochs * len(client.train_loader)
                        )
                        for client in selected_clients
                    }
                    if any(step_count <= 0 for step_count in planned_steps_by_client.values()):
                        raise ValueError(
                            "Epoch 本地训练模式下，参与客户端必须至少有一个训练 batch；"
                            f"当前预计更新次数：{planned_steps_by_client}。"
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

                avg_train_loss = round_result.train_metrics.get(
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

        # 两种模式统一严格校验更新次数合法性；只有 Steps 模式要求客户端一致。
        actual_local_steps_by_client = collect_client_update_steps(client_updates)
        configured_local_steps = _cfg_get(
            self.cfg,
            "local_steps",
            _default_config_value("local_steps"),
        )
        if configured_local_steps is not None:
            actual_local_steps = _validate_equal_client_update_steps_by_client(
                actual_local_steps_by_client,
                expected_steps=configured_local_steps,
            )
        else:
            # Epoch 模式允许客户端因本地样本量不同而执行不同更新次数。
            # 只有步数恰好一致时才保留标量记录，否则使用 None。
            unique_steps = set(actual_local_steps_by_client.values())
            actual_local_steps = (
                unique_steps.pop() if len(unique_steps) == 1 else None
            )

        global_state_cpu = clone_state_dict(
            state_dict_to(
                self.global_model.state_dict(),
                device="cpu",
            )
        )

        non_expert_result = self.aggregators.non_expert.aggregate(
            global_state=global_state_cpu,
            client_updates=client_updates,
            param_names=self.param_groups.non_expert,
            base_state=None,
            strict=True,
        )
        if not isinstance(non_expert_result, AggregationResult):
            raise TypeError(
                "非专家聚合器 aggregate() 必须返回 AggregationResult，"
                f"当前类型：{type(non_expert_result).__name__}"
            )
        validate_state_dict_shapes(
            reference_state=global_state_cpu,
            candidate_state=non_expert_result.new_state_dict,
            context="non_expert 聚合结果",
        )

        expert_result = self.aggregators.expert.aggregate(
            global_state=global_state_cpu,
            client_updates=client_updates,
            param_names=self.param_groups.expert,
            base_state=non_expert_result.new_state_dict,
            strict=True,
        )
        if not isinstance(expert_result, AggregationResult):
            raise TypeError(
                "专家聚合器 aggregate() 必须返回 AggregationResult，"
                f"当前类型：{type(expert_result).__name__}"
            )
        if not isinstance(expert_result.new_state_dict, Mapping):
            raise TypeError(
                "AggregationResult.new_state_dict 必须是 Mapping，"
                f"当前类型：{type(expert_result.new_state_dict).__name__}"
            )

        validate_state_dict_shapes(
            reference_state=global_state_cpu,
            candidate_state=expert_result.new_state_dict,
            context="expert 聚合结果",
        )
        new_state_dict = expert_result.new_state_dict

        check_finite_state_dict(new_state_dict)

        self.global_model.load_state_dict(
            new_state_dict,
            strict=True,
        )

        # 聚合后的 QR / Retract 等操作由当前专家聚合算法决定。
        self.aggregators.expert.after_aggregate(self.global_model)
        check_finite_state_dict(self.global_model.state_dict())

        # 只有在最终服务端模型状态加载完成、
        # 且算法专属聚合后处理全部执行完毕后，才收集诊断信息。
        post_diagnostics = (
            self.aggregators.expert.build_post_aggregate_diagnostics(
                self.global_model,
            )
        )
        if not isinstance(post_diagnostics, Mapping):
            raise TypeError(
                "build_post_aggregate_diagnostics() 必须返回 Mapping，"
                f"当前类型：{type(post_diagnostics).__name__}"
            )
        if post_diagnostics:
            current_diagnostics = expert_result.diagnostics
            if not isinstance(current_diagnostics, Mapping):
                raise TypeError(
                    "AggregationResult.diagnostics 必须是 Mapping，"
                    f"当前类型：{type(current_diagnostics).__name__}"
                )
            duplicate_keys = (
                set(current_diagnostics).intersection(post_diagnostics)
            )
            if duplicate_keys:
                raise ValueError(
                    "聚合前诊断与聚合后诊断存在重复字段："
                    f"{sorted(str(key) for key in duplicate_keys)}"
                )
            merged_diagnostics = dict(current_diagnostics)
            merged_diagnostics.update(dict(post_diagnostics))
            expert_result.diagnostics = merged_diagnostics

        self.global_model.to("cpu")

        return {
            "local_update_steps": (
                int(actual_local_steps) if actual_local_steps is not None else None
            ),
            "local_update_steps_by_client": actual_local_steps_by_client,
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
        avg_train_task_loss = average_client_metric(
            client_updates=list(client_updates),
            metric_name="train_task_loss",
            weighted=True,
            default=None,
        )
        avg_train_aux_loss = average_client_metric(
            client_updates=list(client_updates),
            metric_name="train_aux_loss",
            weighted=True,
            default=None,
        )
        avg_train_acc = average_client_metric(
            client_updates=list(client_updates),
            metric_name="train_acc",
            weighted=True,
            default=None,
        )

        # 不同职责的数据分别存储，aggregation_info 只包含聚合结果与诊断。
        # 所有本地训练损失均按实际处理样本数加权。
        train_metrics = {
            "avg_train_loss": avg_train_loss,
            "avg_train_task_loss": avg_train_task_loss,
            "avg_train_aux_loss": avg_train_aux_loss,
            "avg_train_acc": avg_train_acc,
        }
        test_metrics = {
            "test_loss": float(eval_result.loss),
            "test_acc": float(eval_result.acc),
        }
        client_diagnostics = self._build_client_diagnostics(client_updates)
        avg_train_aux_loss_by_layer = average_client_aux_loss_by_layer(client_updates)
        moe_stats = aggregate_client_moe_stats(client_updates)

        return RoundResult(
            round_id=int(round_id),
            selected_clients=selected_client_ids,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            best_acc=float(self.train_state.best_acc),
            aggregation_info=dict(aggregation_info),
            client_diagnostics=client_diagnostics,
            avg_train_aux_loss_by_layer=avg_train_aux_loss_by_layer,
            moe_stats=moe_stats,
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
        rounds = int(
            _cfg_get(self.cfg, "rounds", _default_config_value("rounds"))
        )
        local_steps = _cfg_get(
            self.cfg,
            "local_steps",
            _default_config_value("local_steps"),
        )
        local_epochs = int(
            _cfg_get(
                self.cfg,
                "local_epochs",
                _default_config_value("local_epochs"),
            )
        )
        frac = float(
            _cfg_get(self.cfg, "frac", _default_config_value("frac"))
        )
        print(f"[Server] rounds: {rounds}")
        if local_steps is None:
            print(f"[Server] local_training: epochs={local_epochs}")
        else:
            print(
                f"[Server] local_training: steps={int(local_steps)} "
                "(local_epochs ignored)"
            )
        print(f"[Server] frac: {frac}")
        model_type = type(self.global_model)
        print(
            "[Server] model_type: "
            f"{model_type.__module__}.{model_type.__qualname__}"
        )
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
        avg_train_loss = round_result.train_metrics.get(
            "avg_train_loss",
            None,
        )
        avg_train_task_loss = round_result.train_metrics.get(
            "avg_train_task_loss",
            None,
        )
        avg_train_aux_loss = round_result.train_metrics.get(
            "avg_train_aux_loss",
            None,
        )
        avg_train_acc = round_result.train_metrics.get(
            "avg_train_acc",
            None,
        )

        avg_train_loss_text = self._format_metric(avg_train_loss, fmt=".4f")
        avg_train_task_loss_text = self._format_metric(
            avg_train_task_loss,
            fmt=".4f",
        )
        avg_train_aux_loss_text = self._format_metric(
            avg_train_aux_loss,
            fmt=".4f",
        )
        avg_train_acc_text = self._format_metric(
            avg_train_acc,
            fmt=".2f",
            suffix="%",
        )

        print(
            f"[Round {round_result.round_id:03d}] "
            f"train_loss={avg_train_loss_text} | "
            f"task_loss={avg_train_task_loss_text} | "
            f"aux_loss={avg_train_aux_loss_text} | "
            f"train_acc={avg_train_acc_text} | "
            f"test_loss={round_result.test_metrics['test_loss']:.4f} | "
            f"test_acc={round_result.test_metrics['test_acc']:.2f}% | "
            f"best_acc={round_result.best_acc:.2f}%"
        )

    def print_file_round_detail(
        self,
        round_result: RoundResult,
    ) -> None:
        """
        写入每轮详细日志。

        详细日志行为固定：记录整体指标、参与客户端、聚合权重和客户端诊断表。
        """
        avg_train_loss = round_result.train_metrics.get(
            "avg_train_loss",
            None,
        )
        avg_train_task_loss = round_result.train_metrics.get(
            "avg_train_task_loss",
            None,
        )
        avg_train_aux_loss = round_result.train_metrics.get(
            "avg_train_aux_loss",
            None,
        )
        avg_train_acc = round_result.train_metrics.get(
            "avg_train_acc",
            None,
        )

        avg_train_loss_text = self._format_metric(
            avg_train_loss,
            fmt=".4f",
        )
        avg_train_task_loss_text = self._format_metric(
            avg_train_task_loss,
            fmt=".4f",
        )
        avg_train_aux_loss_text = self._format_metric(
            avg_train_aux_loss,
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
            f"task_loss={avg_train_task_loss_text} "
            f"aux_loss={avg_train_aux_loss_text} "
            f"train_acc={avg_train_acc_text} "
            f"test_loss={round_result.test_metrics['test_loss']:.4f} "
            f"test_acc={round_result.test_metrics['test_acc']:.2f}% "
            f"best_acc={round_result.best_acc:.2f}%"
        )

        self._write_log_only(
            f"[Clients] "
            f"round={round_result.round_id} "
            f"ids={self._format_client_ids(round_result.selected_clients)}"
        )

        # 固定记录 non_expert / expert 聚合方法与客户端权重。
        self._write_aggregation_info_to_log(
            round_result=round_result,
        )

        if round_result.avg_train_aux_loss_by_layer:
            self._write_log_only(
                f"[AuxLossByLayer] round={round_result.round_id} "
                f"values={self._compact_log_value(round_result.avg_train_aux_loss_by_layer)}"
            )
        if round_result.moe_stats:
            self._write_log_only(
                f"[MoEStats] round={round_result.round_id} "
                f"values={self._compact_log_value(round_result.moe_stats)}"
            )

        # 固定记录每个客户端的诊断信息。
        self._write_client_table_to_log(round_result)

    def _build_client_diagnostics(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> Dict[int, Dict[str, Any]]:
        """从 ClientUpdate 中提取公共及方法插件提供的轻量诊断信息。"""
        diagnostics: Dict[int, Dict[str, Any]] = {}
        for update in client_updates:
            client_diag: Dict[str, Any] = {
                "num_samples": int(update.num_samples),
                "metrics": dict(update.metrics or {}),
                "avg_aux_loss_by_layer": dict(update.avg_aux_loss_by_layer or {}),
                "moe_stats": copy.deepcopy(update.moe_stats or {}),
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
    ) -> None:
        """
        写入 non_expert / expert 聚合摘要及客户端权重。

        权重日志固定开启；均匀权重固定使用紧凑格式。
        """
        for group_name in ("non_expert", "expert"):
            agg_info = self._extract_aggregation_info(
                round_result=round_result,
                group_name=group_name,
            )
            if agg_info is None:
                continue

            weights_text = self._format_weights(
                agg_info.get("weights", None),
            )

            self._write_log_only(
                f"[Agg][{group_name}] "
                f"round={round_result.round_id} "
                f"method={agg_info.get('method', 'unknown')} "
                f"clients={agg_info.get('num_clients', 'unknown')} "
                f"state_entries={agg_info.get('num_state_entries', 'unknown')} "
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
            2. 客户端本地总损失、分类损失、辅助损失和准确率
            3. non_expert 聚合权重
            4. expert 聚合权重
        """
        client_diagnostics = round_result.client_diagnostics
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
            train_task_loss = self._format_metric(
                metrics.get("train_task_loss", None),
                fmt=".4f",
            )
            train_aux_loss = self._format_metric(
                metrics.get("train_aux_loss", None),
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

            self._write_log_only(
                f"[Client][{client_id}] "
                f"round={round_result.round_id} "
                f"samples={num_samples} "
                f"train_loss={train_loss} "
                f"task_loss={train_task_loss} "
                f"aux_loss={train_aux_loss} "
                f"train_acc={train_acc} "
                f"aux_loss_by_layer={self._compact_log_value(item.get('avg_aux_loss_by_layer', {}))} "
                f"moe_stats={self._compact_log_value(item.get('moe_stats', {}))} "
                f"non_expert_w={non_expert_weight} "
                f"expert_w={expert_weight}"
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
                    "num_state_entries": 16,
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
                "num_state_entries": "unknown",
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
        num_state_entries = diagnostics.get(
            "num_state_entries",
            diagnostics.get(
                "param_count",
                summary.get(
                    "num_state_entries",
                    summary.get("param_count", "unknown"),
                ),
            ),
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
            "num_state_entries": num_state_entries,
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
    ) -> str:
        """
        格式化聚合权重。

        uniform 权重固定压缩成：
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

        if self._is_uniform_weight_items(numeric_items):
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
    expert_client_training_strategy_builder: Optional[
        ExpertClientTrainingStrategyBuilder
    ] = None,
    model_builder: Optional[ModelBuilder] = None,
    global_model: Optional[nn.Module] = None,
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
        expert_client_training_strategy_builder=expert_client_training_strategy_builder,
        model_builder=model_builder,
        global_model=global_model,
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
    device_name = str(
        _cfg_get(cfg, "device", _default_config_value("device"))
    ).lower().strip()

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
    """严格读取布尔配置；字符串和整数形式的真假值不做隐式转换。"""
    value = _cfg_get(cfg, key, default)
    _require_bool_value(value, key)
    return value


# ============================================================================
# 14. 训练入口与结果输出（Training Entry & Output）
# 包含 CLI 参数解析、配置覆盖、训练启动及实验结果保存。
# ============================================================================


MethodCliArgumentRegistrar = Callable[[argparse.ArgumentParser], None]
MethodCliOverridesBuilder = Callable[[argparse.Namespace], Mapping[str, Any]]


def _parse_bool_cli_value(value: str) -> bool:
    """解析 True/False、true/false 等显式命令行布尔值。"""
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"布尔参数只接受 True/False，当前值：{value!r}"
    )


def parse_args(method_cli_argument_registrar: Optional[MethodCliArgumentRegistrar] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FL + MoE training entrypoint")
    parser.add_argument("--config", type=str, default=None, help="可选外部配置文件；显式 CLI 参数优先。")
    parser.add_argument("--dataset", type=normalize_dataset_name, choices=sorted(SUPPORTED_DATASETS), default=None)
    parser.add_argument(
        "--data-augmentation",
        type=_parse_bool_cli_value,
        default=None,
        metavar="True|False",
        help="是否开启训练集随机数据增强，例如：--data-augmentation False。",
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument(
        "--partition-method",
        choices=("iid", "dirichlet"),
        default=None,
        help="数据划分方法；覆盖 partition.method。",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Dirichlet alpha；覆盖 partition.dirichlet.alpha。",
    )
    parser.add_argument(
        "--dirichlet-sample-balance",
        choices=("balanced", "unbalanced"),
        default=None,
        help="Dirichlet 客户端样本数均衡策略；覆盖 partition.dirichlet.sample_balance。",
    )
    parser.add_argument("--frac", type=float, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument(
        "--local-steps",
        type=int,
        default=None,
        help="每个客户端每轮固定本地梯度更新次数；设置后优先于 local_epochs。",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-experts", type=int, default=None)

    # SwitchViT 结构参数统一归属于 model_cfg 配置节点。
    parser.add_argument("--transformer-layers", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--attention-heads", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--expert-hidden-dim", type=int, default=None)
    parser.add_argument(
        "--moe-block-indices",
        type=int,
        nargs="+",
        default=None,
        metavar="BLOCK",
        help="使用 MoE FFN 的 Transformer Block 编号，例如：--moe-block-indices 2 4 6。",
    )
    parser.add_argument("--routing", choices=("top1",), default=None)
    parser.add_argument("--dropout", type=float, default=None)

    # 数值为 0 时关闭对应功能；正数表示启用。
    parser.add_argument(
        "--aux-loss-coef",
        type=float,
        default=None,
        help="Auxiliary Loss 系数；0 表示关闭，正数表示启用。",
    )
    parser.add_argument(
        "--capacity-factor",
        type=float,
        default=None,
        help="Expert Capacity 系数；0 表示关闭，正数表示启用。",
    )
    parser.add_argument(
        "--token-dropping",
        type=_parse_bool_cli_value,
        default=None,
        metavar="True|False",
        help="容量耗尽且所有专家均无法接收 Token 时，是否丢弃 Token。",
    )
    parser.add_argument(
        "--router-statistics",
        type=_parse_bool_cli_value,
        default=None,
        metavar="True|False",
        help="是否收集 Router 统计。",
    )

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
        "model":"model","data_root":"data_root","num_clients":"num_clients","frac":"frac",
        "rounds":"rounds","local_epochs":"local_epochs","local_steps":"local_steps","batch_size":"batch_size",
        "test_batch_size":"test_batch_size","num_workers":"num_workers","num_experts":"num_experts",
        "seed":"seed","device":"device","output_dir":"output_dir","run_name":"run_name",
    }
    for arg_name,config_key in top_level.items():
        value=getattr(args,arg_name,None)
        if value is not None: overrides[config_key]=value

    data_augmentation = getattr(args, "data_augmentation", None)
    if data_augmentation is not None:
        overrides["data_augmentation"] = data_augmentation

    partition_method = getattr(args, "partition_method", None)
    if partition_method is not None:
        overrides.setdefault("partition", {})["method"] = partition_method

    alpha = getattr(args, "alpha", None)
    if alpha is not None:
        overrides.setdefault("partition", {}).setdefault("dirichlet", {})[
            "alpha"
        ] = alpha

    sample_balance = getattr(args, "dirichlet_sample_balance", None)
    if sample_balance is not None:
        overrides.setdefault("partition", {}).setdefault("dirichlet", {})[
            "sample_balance"
        ] = sample_balance

    dataset=getattr(args,"dataset",None)
    if dataset is not None:
        dataset = normalize_dataset_name(dataset)
        overrides["dataset"]=dataset
        overrides["num_classes"]=_infer_num_classes(dataset)
        info=DATASET_INFO.get(dataset)
        if info is not None:
            overrides["input_shape"]=tuple(info["input_shape"])
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

    model_cfg_overrides: Dict[str, Any] = {}
    model_cfg_cli_fields = (
        "transformer_layers",
        "hidden_dim",
        "attention_heads",
        "patch_size",
        "expert_hidden_dim",
        "moe_block_indices",
        "routing",
        "dropout",
        "aux_loss_coef",
        "capacity_factor",
        "token_dropping",
        "router_statistics",
    )
    for field_name in model_cfg_cli_fields:
        value = getattr(args, field_name, None)
        if value is not None:
            model_cfg_overrides[field_name] = value
    if model_cfg_overrides:
        overrides["model_cfg"] = model_cfg_overrides

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
    expert_client_training_strategy_builder: Optional[
        ExpertClientTrainingStrategyBuilder
    ] = None,
    model_builder: Optional[ModelBuilder] = None,
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

    # 默认使用 base.py 内置的图像分类 SwitchViT；外部仍可显式注入
    # 自定义 ModelBuilder，不改变已有的模型解耦与测试接口。
    if model_builder is None:
        model_builder = build_switch_vit_model
    if not callable(model_builder):
        raise TypeError("model_builder 必须是可调用对象。")
    config_overrides = build_common_cli_overrides(args)
    if method_cli_overrides_builder is not None:
        config_overrides = _deep_merge(base=config_overrides, override=dict(method_cli_overrides_builder(args) or {}))
    cfg = (
        load_config(args.config, method_defaults=method_config_defaults, method_validator=method_config_validator, config_overrides=config_overrides)
        if args.config is not None
        else load_embedded_config(embedded_method_config, method_defaults=method_config_defaults, method_validator=method_config_validator, config_overrides=config_overrides)
    )

    configured_non_expert_method = str(
        cfg.get(
            "agg.non_expert.method",
            _default_config_value("agg.non_expert.method"),
        )
    ).lower().strip()
    if configured_non_expert_method != "uniform":
        raise ValueError(
            "base.py 已固定 non_expert 使用 uniform 聚合，"
            f"当前配置却是 {configured_non_expert_method!r}。"
        )

    configured_expert_method = str(
        cfg.get(
            "agg.expert.method",
            _default_config_value("agg.expert.method"),
        )
    ).lower().strip()
    if configured_expert_method != expert_method_name.lower().strip():
        raise ValueError(
            f"当前脚本实现的专家聚合是 {expert_method_name!r}，"
            f"但配置中 agg.expert.method={configured_expert_method!r}。"
        )

    set_seed(
        seed=int(cfg.seed),
        deterministic=_cfg_get_bool(
            cfg,
            "deterministic",
            _default_config_value("deterministic"),
        ),
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
                model_builder=model_builder,
                expert_evidence_collector=expert_evidence_collector,
                method_client_diagnostics_builder=method_client_diagnostics_builder,
                expert_client_training_strategy_builder=expert_client_training_strategy_builder,
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
    model_builder: ModelBuilder,
    expert_evidence_collector: Optional[ExpertEvidenceCollector] = None,
    method_client_diagnostics_builder: Optional[MethodClientDiagnosticsBuilder] = None,
    expert_client_training_strategy_builder: Optional[
        ExpertClientTrainingStrategyBuilder
    ] = None,
) -> int:
    """
    实际训练流程。

    这个函数会被 main() 包在 tee_output_to_file() 里面，
    所以这里的普通 print 和报错会同时写入控制台和 train.log。

    注意：
        tqdm 进度条由 utils/logging.py 过滤，不写入 train.log。

    总流程：
        1. 解析设备并保存完整配置
        2. 加载数据集
        3. 划分客户端数据
        4. 创建 DataLoader
        5. 创建 FLServer
        6. 执行联邦训练
        7. 保存结果
    """
    device = resolve_device(cfg)

    if _cfg_get_bool(cfg, "logging.save_config", True):
        save_config(
            cfg=cfg,
            output_path=run_dir / "config_used.yaml",
            model_builder=model_builder,
            resolved_device=device,
        )

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
        model_builder=model_builder,
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
        expert_client_training_strategy_builder=expert_client_training_strategy_builder,
        model_builder=model_builder,
    )

    train_result = server.train()

    save_train_outputs(
        train_result=train_result,
        output_dir=run_dir,
        save_csv=_cfg_get_bool(cfg, "logging.save_results_csv", True),
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
            allow_nan=False,
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
            allow_nan=False,
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
        avg_train_task_loss
        avg_train_aux_loss
        avg_train_aux_loss_by_layer_json
        moe_stats_json
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
        "avg_train_task_loss",
        "avg_train_aux_loss",
        "avg_train_aux_loss_by_layer_json",
        "moe_stats_json",
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
            train_metrics = item.train_metrics
            test_metrics = item.test_metrics

            row = {
                "round_id": int(item.round_id),
                "selected_clients": json.dumps(
                    list(item.selected_clients),
                    ensure_ascii=False,
                ),
                "avg_train_loss": train_metrics.get(
                    "avg_train_loss",
                    "",
                ),
                "avg_train_task_loss": train_metrics.get(
                    "avg_train_task_loss",
                    "",
                ),
                "avg_train_aux_loss": train_metrics.get(
                    "avg_train_aux_loss",
                    "",
                ),
                "avg_train_aux_loss_by_layer_json": json.dumps(
                    make_json_safe(item.avg_train_aux_loss_by_layer),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "moe_stats_json": json.dumps(
                    make_json_safe(item.moe_stats),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "avg_train_acc": train_metrics.get(
                    "avg_train_acc",
                    "",
                ),
                "test_loss": float(test_metrics["test_loss"]),
                "test_acc": float(test_metrics["test_acc"]),
                "best_acc": float(item.best_acc),
            }

            writer.writerow(row)


def make_json_safe(obj: Any, path: str = "$") -> Any:
    """
    递归转换为严格 JSON 可序列化的数据。

    支持：
        Tensor / torch.device / torch.dtype
        NumPy ndarray / scalar
        Mapping / list / tuple / set / frozenset
        Enum / dataclass / to_dict 对象
        Path 和 JSON 基本类型

    NaN / Inf 不会被静默替换为 null，而是抛出带字段路径的 ValueError，
    避免实验数值异常在保存阶段被隐藏。
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(
                f"JSON 数据在 {path} 处包含非有限浮点数：{obj!r}"
            )
        return obj

    if isinstance(obj, Enum):
        return make_json_safe(obj.value, path=path)

    if isinstance(obj, torch.Tensor):
        tensor = obj.detach().cpu()
        if tensor.is_complex():
            raise TypeError(
                f"JSON 数据在 {path} 处包含复数 Tensor，无法直接序列化。"
            )
        if tensor.numel() == 1:
            return make_json_safe(tensor.item(), path=path)
        return make_json_safe(tensor.tolist(), path=path)

    if isinstance(obj, (torch.device, torch.dtype)):
        return str(obj)

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist(), path=path)

    if isinstance(obj, np.generic):
        return make_json_safe(obj.item(), path=path)

    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return make_json_safe(obj.to_dict(), path=path)

    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            item.name: make_json_safe(
                getattr(obj, item.name),
                path=f"{path}.{item.name}",
            )
            for item in fields(obj)
        }

    if isinstance(obj, Mapping):
        result = {}
        for key, value in obj.items():
            safe_key = str(key)
            if safe_key in result:
                raise ValueError(
                    f"JSON 数据在 {path} 处出现转换后重复的字典键：{safe_key!r}"
                )
            result[safe_key] = make_json_safe(
                value,
                path=f"{path}.{safe_key}",
            )
        return result

    if isinstance(obj, (list, tuple)):
        return [
            make_json_safe(value, path=f"{path}[{index}]")
            for index, value in enumerate(obj)
        ]

    if isinstance(obj, (set, frozenset)):
        safe_values = [
            make_json_safe(value, path=f"{path}{{item}}")
            for value in obj
        ]
        return sorted(
            safe_values,
            key=lambda value: json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            ),
        )

    raise TypeError(
        f"JSON 数据在 {path} 处包含不支持的对象类型："
        f"{type(obj).__module__}.{type(obj).__qualname__}"
    )

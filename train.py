from __future__ import annotations

import csv
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from aggregation import build_expert_aggregator
from federated import (
    ClientUpdate,
    aggregate_global_model,
    evaluate_model,
    sample_clients,
    train_client,
)
from models.backbones.resnet18_gn import build_resnet18_gn
from models.sparse_moe import build_sparse_moe


@dataclass
class ExperimentConfig:
    """
    实验配置。

    本项目不使用 argparse。修改实验参数时，直接编辑此 dataclass
    中的默认值，确保一次运行对应一套明确配置。
    """

    # --------------------------------------------------------
    # 实验与复现
    # --------------------------------------------------------

    seed: int = 0
    deterministic: bool = True
    device: str = "auto"

    data_dir: str = "./data"
    output_root: str = "./outputs"

    dataset_name: str = "cifar10"
    backbone_name: str = "resnet18_gn"
    # uniform、activation_weighted
    expert_aggregation: str = "uniform"

    # --------------------------------------------------------
    # 联邦学习
    # --------------------------------------------------------

    num_clients: int = 10
    participation_rate: float = 1.0
    num_rounds: int = 200
    local_epochs: int = 1

    dirichlet_alpha: float = 0.1

    client_batch_size: int = 64
    test_batch_size: int = 256
    drop_last: bool = False

    # --------------------------------------------------------
    # 模型
    # --------------------------------------------------------

    num_classes: int = 10
    num_experts: int = 4
    top_k: int = 1

    moe_dim: int = 512
    expert_hidden_dim: int = 1024

    small_image_stem: bool = True
    max_gn_groups: int = 32
    zero_init_residual: bool = False

    balance_loss_weight: float = 0.01

    # --------------------------------------------------------
    # 本地优化
    # --------------------------------------------------------

    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4

    use_amp: bool = False
    max_grad_norm: float | None = None


def main() -> None:
    config = ExperimentConfig()
    validate_config(config)

    set_reproducibility(
        seed=config.seed,
        deterministic=config.deterministic,
    )

    device = resolve_device(config.device)
    output_dir = create_output_directory(config)

    logger = create_logger(output_dir / "train.log")

    save_json(
        output_dir / "config.json",
        {
            **asdict(config),
            "resolved_device": str(device),
        },
    )

    logger.info("Output directory: %s", output_dir)
    logger.info("Device: %s", device)

    train_dataset, test_dataset = build_cifar10_datasets(
        data_dir=config.data_dir,
    )

    client_indices = dirichlet_label_partition(
        labels=np.asarray(train_dataset.targets),
        num_clients=config.num_clients,
        alpha=config.dirichlet_alpha,
        seed=config.seed,
    )

    save_partition(
        path=output_dir / "partition.json",
        client_indices=client_indices,
        labels=np.asarray(train_dataset.targets),
        config=config,
    )

    client_loaders = build_client_loaders(
        dataset=train_dataset,
        client_indices=client_indices,
        batch_size=config.client_batch_size,
        drop_last=config.drop_last,
        seed=config.seed,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        drop_last=False,
    )

    backbone = build_resnet18_gn(
        in_channels=3,
        small_image_stem=config.small_image_stem,
        max_gn_groups=config.max_gn_groups,
        zero_init_residual=config.zero_init_residual,
    )

    global_model = build_sparse_moe(
        backbone=backbone,
        num_classes=config.num_classes,
        num_experts=config.num_experts,
        top_k=config.top_k,
        moe_dim=config.moe_dim,
        expert_hidden_dim=config.expert_hidden_dim,
    )

    global_model.validate_parameter_partition()
    global_model.cpu()

    logger.info(
        "Model parameters | total=%s | shared=%s | per_expert=%s",
        f"{global_model.count_total_parameters():,}",
        f"{global_model.count_shared_parameters():,}",
        f"{global_model.count_expert_parameters(0):,}",
    )

    expert_aggregator = build_expert_aggregator(
        config.expert_aggregation,
    )

    optimizer_factory = build_optimizer_factory(config)

    client_sampling_generator = torch.Generator()
    client_sampling_generator.manual_seed(config.seed)

    metrics_path = output_dir / "metrics.csv"
    start_time = time.perf_counter()

    # 不从多轮测试结果中选择最高值作为最终结果。
    # 保存每轮测试指标，训练结束后报告最后一轮结果，
    # 并统计最后若干轮的均值和标准差。
    test_accuracy_history: list[float] = []
    test_total_loss_history: list[float] = []
    summary_window = 10

    final_metrics: dict | None = None

    with metrics_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as metrics_file:
        fieldnames = [
            "round",
            "mean_client_loss",
            "mean_client_accuracy",
            "test_total_loss",
            "test_classification_loss",
            "test_balance_loss",
            "test_accuracy",
            "route_distribution",
            "expert_participant_counts",
        ]

        writer = csv.DictWriter(
            metrics_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        metrics_file.flush()

        for round_idx in range(config.num_rounds):
            selected_client_ids = sample_clients(
                num_clients=config.num_clients,
                participation_rate=config.participation_rate,
                generator=client_sampling_generator,
            )

            client_updates: list[ClientUpdate] = []

            for client_id in selected_client_ids:
                update = train_client(
                    client_id=client_id,
                    global_model=global_model,
                    train_loader=client_loaders[client_id],
                    optimizer_factory=optimizer_factory,
                    device=device,
                    local_epochs=config.local_epochs,
                    balance_loss_weight=(
                        config.balance_loss_weight
                    ),
                    use_amp=config.use_amp,
                    max_grad_norm=config.max_grad_norm,
                )

                # Dirichlet 划分允许空客户端。
                if update is not None:
                    client_updates.append(update)

            if not client_updates:
                raise RuntimeError(
                    f"Round {round_idx + 1} produced no valid "
                    "client updates."
                )

            aggregation_summary = aggregate_global_model(
                global_model=global_model,
                client_updates=client_updates,
                expert_aggregator=expert_aggregator,
                round_idx=round_idx,
            )

            test_metrics = evaluate_model(
                model=global_model,
                data_loader=test_loader,
                device=device,
                balance_loss_weight=(
                    config.balance_loss_weight
                ),
                use_amp=config.use_amp,
            )

            test_accuracy_history.append(
                float(test_metrics.accuracy)
            )
            test_total_loss_history.append(
                float(test_metrics.loss)
            )

            # 保持服务器全局模型位于 CPU，减少串行客户端训练时
            # GPU 上同时存在全局模型与本地模型的显存开销。
            global_model.cpu()

            mean_client_loss = mean(
                update.train_loss
                for update in client_updates
            )
            mean_client_accuracy = mean(
                update.accuracy
                for update in client_updates
            )

            route_distribution = [
                round(float(value), 6)
                for value in test_metrics.route_distribution
            ]
            expert_participant_counts = (
                aggregation_summary
                .expert_participant_counts
                .tolist()
            )

            current_round = round_idx + 1

            row = {
                "round": current_round,
                "mean_client_loss": (
                    f"{mean_client_loss:.8f}"
                ),
                "mean_client_accuracy": (
                    f"{mean_client_accuracy:.8f}"
                ),
                "test_total_loss": f"{test_metrics.loss:.8f}",
                "test_classification_loss": (
                    f"{test_metrics.classification_loss:.8f}"
                ),
                "test_balance_loss": (
                    f"{test_metrics.balance_loss:.8f}"
                ),
                "test_accuracy": (
                    f"{test_metrics.accuracy:.8f}"
                ),
                "route_distribution": json.dumps(
                    route_distribution,
                    ensure_ascii=False,
                ),
                "expert_participant_counts": json.dumps(
                    expert_participant_counts,
                    ensure_ascii=False,
                ),
            }

            writer.writerow(row)
            metrics_file.flush()

            logger.info(
                "Round %03d/%03d | "
                "mean_client_loss=%.6f | "
                "mean_client_accuracy=%.4f | "
                "test_total_loss=%.6f | "
                "test_accuracy=%.4f | "
                "route_distribution=%s | "
                "expert_participant_counts=%s",
                current_round,
                config.num_rounds,
                mean_client_loss,
                mean_client_accuracy,
                test_metrics.loss,
                test_metrics.accuracy,
                route_distribution,
                expert_participant_counts,
            )

            final_metrics = {
                "round": current_round,
                "mean_client_loss": mean_client_loss,
                "mean_client_accuracy": (
                    mean_client_accuracy
                ),
                "test": test_metrics.to_dict(),
                "aggregation": (
                    aggregation_summary.to_dict()
                ),
            }

    elapsed_seconds = time.perf_counter() - start_time

    if final_metrics is None:
        raise RuntimeError(
            "Training ended without producing metrics."
        )

    num_summary_rounds = min(
        summary_window,
        len(test_accuracy_history),
    )

    last_test_accuracies = test_accuracy_history[
        -num_summary_rounds:
    ]
    last_test_total_losses = test_total_loss_history[
        -num_summary_rounds:
    ]

    final_test_accuracy = test_accuracy_history[-1]
    final_test_total_loss = test_total_loss_history[-1]

    # 这里的标准差表示同一次训练最后若干轮之间的波动，
    # 不等同于多个随机种子实验之间的标准差。
    last_rounds_mean_accuracy = float(
        np.mean(last_test_accuracies)
    )
    last_rounds_std_accuracy = float(
        np.std(last_test_accuracies)
    )
    last_rounds_mean_total_loss = float(
        np.mean(last_test_total_losses)
    )
    last_rounds_std_total_loss = float(
        np.std(last_test_total_losses)
    )

    summary = {
        "output_directory": str(output_dir),
        "seed": config.seed,
        "num_rounds": config.num_rounds,
        "final_test_accuracy": final_test_accuracy,
        "final_test_total_loss": final_test_total_loss,
        "last_rounds_summary": {
            "num_rounds": num_summary_rounds,
            "mean_test_accuracy": (
                last_rounds_mean_accuracy
            ),
            "std_test_accuracy": (
                last_rounds_std_accuracy
            ),
            "mean_test_total_loss": last_rounds_mean_total_loss,
            "std_test_total_loss": last_rounds_std_total_loss,
        },
        "final_metrics": final_metrics,
        "elapsed_seconds": elapsed_seconds,
    }

    save_json(
        output_dir / "summary.json",
        summary,
    )

    logger.info(
        "Finished | "
        "final_test_accuracy=%.4f | "
        "last_%d_rounds_mean_accuracy=%.4f | "
        "last_%d_rounds_std_accuracy=%.4f | "
        "elapsed_seconds=%.2f",
        final_test_accuracy,
        num_summary_rounds,
        last_rounds_mean_accuracy,
        num_summary_rounds,
        last_rounds_std_accuracy,
        elapsed_seconds,
    )


def validate_config(config: ExperimentConfig) -> None:
    if config.dataset_name.lower() != "cifar10":
        raise ValueError(
            "The current train.py only supports CIFAR-10."
        )

    if config.backbone_name.lower() != "resnet18_gn":
        raise ValueError(
            "The current train.py only constructs ResNet18-GN."
        )

    if config.num_clients <= 0:
        raise ValueError("num_clients must be greater than 0.")

    if not 0.0 < config.participation_rate <= 1.0:
        raise ValueError(
            "participation_rate must be in (0, 1]."
        )

    if config.num_rounds <= 0:
        raise ValueError("num_rounds must be greater than 0.")

    if config.local_epochs <= 0:
        raise ValueError("local_epochs must be greater than 0.")

    if config.dirichlet_alpha <= 0.0:
        raise ValueError(
            "dirichlet_alpha must be greater than 0."
        )

    if config.client_batch_size <= 0:
        raise ValueError(
            "client_batch_size must be greater than 0."
        )

    if config.test_batch_size <= 0:
        raise ValueError(
            "test_batch_size must be greater than 0."
        )

    if config.num_classes != 10:
        raise ValueError(
            "CIFAR-10 requires num_classes=10."
        )

    if config.num_experts <= 1:
        raise ValueError(
            "num_experts must be greater than 1."
        )

    if not 1 <= config.top_k <= config.num_experts:
        raise ValueError(
            "top_k must satisfy "
            "1 <= top_k <= num_experts."
        )

    if config.learning_rate <= 0.0:
        raise ValueError(
            "learning_rate must be greater than 0."
        )

    if config.momentum < 0.0:
        raise ValueError("momentum must be non-negative.")

    if config.weight_decay < 0.0:
        raise ValueError(
            "weight_decay must be non-negative."
        )

    if config.balance_loss_weight < 0.0:
        raise ValueError(
            "balance_loss_weight must be non-negative."
        )

    if (
        config.max_grad_norm is not None
        and config.max_grad_norm <= 0.0
    ):
        raise ValueError(
            "max_grad_norm must be positive or None."
        )


def set_reproducibility(
    *,
    seed: int,
    deterministic: bool,
) -> None:
    """
    设置 Python、NumPy 和 PyTorch 随机种子。

    deterministic=True 时:
        - cudnn.deterministic = True
        - cudnn.benchmark = False
        - deterministic algorithms 使用 warn_only 模式
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def resolve_device(device_name: str) -> torch.device:
    normalized = device_name.lower().strip()

    if normalized == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(normalized)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    return device


def create_output_directory(
    config: ExperimentConfig,
) -> Path:
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_dir = (
        Path(config.output_root)
        / config.dataset_name.lower()
        / config.backbone_name.lower()
        / config.expert_aggregation.lower()
        / f"seed_{config.seed}"
        / timestamp
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    return output_dir


def create_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("federated_sparse_moe")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 防止交互环境重复运行时重复添加 handler。
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(message)s")
    )

    file_handler = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def build_cifar10_datasets(
    *,
    data_dir: str,
):
    """
    CIFAR-10 训练集使用标准数据增强，测试集不增强。
    """
    mean_values = (0.4914, 0.4822, 0.4465)
    std_values = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(
                32,
                padding=4,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean_values,
                std_values,
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean_values,
                std_values,
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=test_transform,
        download=True,
    )

    return train_dataset, test_dataset


def dirichlet_label_partition(
    *,
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
) -> list[list[int]]:
    """
    按类别分别采样 Dirichlet 比例，将训练集划分给客户端。

    特性:
        - 不保证每个客户端至少有一定数量样本；
        - 允许空客户端；
        - 允许客户端缺少任意类别；
        - 每个训练样本只分配给一个客户端；
        - 所有训练样本都会被分配。

    对每个类别 c:
        proportions_c ~ Dirichlet(alpha, ..., alpha)
        counts_c ~ Multinomial(n_c, proportions_c)
    """
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array.")

    if num_clients <= 0:
        raise ValueError(
            "num_clients must be greater than 0."
        )

    if alpha <= 0.0:
        raise ValueError("alpha must be greater than 0.")

    rng = np.random.default_rng(seed)
    client_indices: list[list[int]] = [
        [] for _ in range(num_clients)
    ]

    unique_classes = np.unique(labels)

    for class_id in unique_classes:
        class_indices = np.flatnonzero(
            labels == class_id
        )
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(
            np.full(num_clients, alpha)
        )

        counts = rng.multinomial(
            n=len(class_indices),
            pvals=proportions,
        )

        boundaries = np.concatenate(
            [
                np.array([0], dtype=np.int64),
                np.cumsum(counts),
            ]
        )

        for client_id in range(num_clients):
            start = int(boundaries[client_id])
            end = int(boundaries[client_id + 1])

            client_indices[client_id].extend(
                class_indices[start:end].tolist()
            )

    for indices in client_indices:
        rng.shuffle(indices)

    assigned_indices = [
        index
        for indices in client_indices
        for index in indices
    ]

    if len(assigned_indices) != len(labels):
        raise RuntimeError(
            "Dirichlet partition did not assign every sample."
        )

    if len(set(assigned_indices)) != len(labels):
        raise RuntimeError(
            "Dirichlet partition assigned duplicate samples."
        )

    return client_indices


def build_client_loaders(
    *,
    dataset,
    client_indices: Sequence[Sequence[int]],
    batch_size: int,
    drop_last: bool,
    seed: int,
) -> list[DataLoader]:
    loaders: list[DataLoader] = []

    for client_id, indices in enumerate(client_indices):
        subset = Subset(
            dataset,
            list(indices),
        )

        generator = torch.Generator()
        generator.manual_seed(
            seed + 10_000 + client_id
        )

        # RandomSampler 不接受空数据集，因此空客户端不 shuffle。
        shuffle = len(subset) > 0

        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            generator=generator,
        )

        loaders.append(loader)

    return loaders


def build_optimizer_factory(
    config: ExperimentConfig,
):
    """
    共享参数和专家参数使用完全相同的 SGD 配置。
    """

    def optimizer_factory(
        model: torch.nn.Module,
    ) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    return optimizer_factory


def save_partition(
    *,
    path: Path,
    client_indices: Sequence[Sequence[int]],
    labels: np.ndarray,
    config: ExperimentConfig,
) -> None:
    clients: Dict[str, dict] = {}

    for client_id, indices in enumerate(client_indices):
        class_counts = {
            str(class_id): 0
            for class_id in range(config.num_classes)
        }

        for sample_index in indices:
            class_id = int(labels[sample_index])
            class_counts[str(class_id)] += 1

        clients[str(client_id)] = {
            "num_samples": len(indices),
            "class_counts": class_counts,
            "indices": [int(index) for index in indices],
        }

    payload = {
        "dataset": config.dataset_name,
        "partition_method": "dirichlet_label",
        "alpha": config.dirichlet_alpha,
        "seed": config.seed,
        "num_clients": config.num_clients,
        "num_total_samples": int(len(labels)),
        "clients": clients,
    }

    save_json(path, payload)


def save_json(
    path: Path,
    payload: dict,
) -> None:
    with path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    main()

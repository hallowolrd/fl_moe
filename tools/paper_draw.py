#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import yaml


# =========================
# 论文绘图样式
# =========================

# MATLAB 规范：宽 14.2 cm，高 8.0 cm
FIG_WIDTH = 14.2 / 2.54
FIG_HEIGHT = 8.0 / 2.54

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        # 更接近 GAPSL 成图比例
        "font.size": 15,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
        "axes.linewidth": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.frameon": True,
    }
)


# =========================
# 曲线样式
# =========================
PLOT_STYLES = [
    {
        "color": "#1689D8",
        "linestyle": (0, (5.0, 1.3, 1.6, 1.3)),
        "linewidth": 1.8,
    },
    {
        "color": "#A500A5",
        "linestyle": "-",
        "linewidth": 1.8,
    },
    {
        "color": "#19C4C7",
        "linestyle": (0, (1.0, 1.15)),
        "linewidth": 1.8,
    },
    {
        "color": "#F4BE00",
        "linestyle": (0, (5.0, 1.3, 1.4, 1.3)),
        "linewidth": 1.8,
    },
    {
        "color": "#8A8A8A",
        "linestyle": (0, (3.2, 1.8)),
        "linewidth": 1.8,
    },
]

# 颜色与原始线型绑定：
# 调换 METHOD_COLORS 中的颜色时，对应线型会一起跟随该颜色。
STYLE_BY_COLOR = {
    style["color"]: style
    for style in PLOT_STYLES
}

# =========================
# 每个算法对应的颜色
#
# 以后如果只想调换颜色，只修改这里即可。
# 使用脚本内部固定名称。
# =========================
METHOD_COLORS = {
    "Uniform": "#8A8A8A",
    "Fisher/K-FAC": "#A500A5",
    "Fed-MoE": "#19C4C7",
    "FedMoE-DA": "#F4BE00",
    "SOMFed": "#1689D8",
}


# =========================
# 图例显示名称
#
# 以后如果只想修改图例里的算法名称，
# 只需要修改这里右侧的字符串即可。
# 左侧名称是脚本内部用于识别算法的固定名称，不要改。
# =========================
LEGEND_NAMES = {
    "Uniform": "FedAvg-MoE",
    "Fisher/K-FAC": "GEA-FedMoE",
    "Fed-MoE": "Fed-MoE",
    "FedMoE-DA": "FedMoE-DA",
    "SOMFed": "SOMFed",
}

# =========================
# 图例显示顺序
#
# 这里使用的是脚本内部固定名称。
# 如果以后只改“图例名字”，不需要改这里。
# =========================
LEGEND_ORDER = [
    "Fisher/K-FAC",
    "Fed-MoE",
    "Uniform",
    "SOMFed",
    "FedMoE-DA",
]

# =========================
# 不绘制的算法
#
# 使用脚本内部固定名称。
# 空集合表示全部绘制。
# =========================
HIDDEN_METHODS = {
    "Uniform",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare test_acc curves from federated-learning results.csv files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="总实验目录，例如 outputs/cifar10_resnet_cifar",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="test_acc 滑动平均窗口，默认 5；设为 1 表示不平滑。",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="可选。只比较指定 dataset；默认从 config_used.yaml 自动识别。",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="可选。只比较指定 backbone；默认从 config_used.yaml 自动识别。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="可选。图片保存目录；默认保存到项目根目录 pictures。",
    )
    parser.add_argument(
        "--max-round",
        type=int,
        default=None,
        help="可选。只绘制到指定通信轮数，例如 50 或 60；默认绘制全部轮数。",
    )
    return parser.parse_args()


def load_run_metadata(csv_path: Path) -> Tuple[str | None, str | None, str]:
    config_path = csv_path.parent / "config_used.yaml"

    if not config_path.is_file():
        return None, None, csv_path.parent.name

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    dataset = cfg.get("dataset")

    model_cfg = cfg.get("model_cfg", {}) or {}
    backbone = model_cfg.get("backbone") if isinstance(model_cfg, dict) else None

    agg_cfg = cfg.get("agg", {}) or {}
    expert_cfg = agg_cfg.get("expert", {}) if isinstance(agg_cfg, dict) else {}
    method = expert_cfg.get("method") if isinstance(expert_cfg, dict) else None

    if not method:
        method = csv_path.parent.name

    return (
        str(dataset) if dataset is not None else None,
        str(backbone) if backbone is not None else None,
        str(method),
    )


def read_test_acc(csv_path: Path) -> Tuple[List[int], List[float]]:
    rounds: List[int] = []
    test_acc: List[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        required = {"round_id", "test_acc"}
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"{csv_path} 缺少必要列：{sorted(missing)}；"
                f"当前列：{reader.fieldnames}"
            )

        for row in reader:
            round_text = str(row.get("round_id", "")).strip()
            acc_text = str(row.get("test_acc", "")).strip()

            if not round_text or not acc_text:
                continue

            rounds.append(int(float(round_text)))
            test_acc.append(float(acc_text))

    if not rounds:
        raise ValueError(f"{csv_path} 中没有有效的 round_id/test_acc 数据。")

    pairs = sorted(zip(rounds, test_acc), key=lambda item: item[0])
    return [item[0] for item in pairs], [item[1] for item in pairs]


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 0:
        raise ValueError(f"window 必须大于 0，当前值：{window}")

    result: List[float] = []
    running_sum = 0.0

    for index, value in enumerate(values):
        running_sum += value

        if index >= window:
            running_sum -= values[index - window]

        current_count = min(index + 1, window)
        result.append(running_sum / current_count)

    return result


def friendly_method_name(method: str) -> str:
    name = method.lower().strip()

    if name == "uniform" or "uniform" in name:
        return "Uniform"
    if "fisher" in name or "kfac" in name:
        return "Fisher/K-FAC"
    if "fedmoe_da" in name or "fed_moe_da" in name or "fedmoeda" in name:
        return "FedMoE-DA"
    if "somfed" in name:
        return "SOMFed"
    if "fed_moe" in name or "fedmoe" in name:
        return "Fed-MoE"

    return method


def method_order(method: str) -> Tuple[int, str]:
    label = friendly_method_name(method)
    preferred = {
        "Uniform": 0,
        "Fisher/K-FAC": 1,
        "Fed-MoE": 2,
        "FedMoE-DA": 3,
        "SOMFed": 4,
    }
    return preferred.get(label, 100), label.lower()


def choose_dataset_backbone(
    runs: List[Dict[str, object]],
    dataset_arg: str | None,
    backbone_arg: str | None,
) -> Tuple[str, str]:
    known_pairs = {
        (str(run["dataset"]), str(run["backbone"]))
        for run in runs
        if run["dataset"] is not None and run["backbone"] is not None
    }

    if dataset_arg is not None and backbone_arg is not None:
        return dataset_arg, backbone_arg

    if dataset_arg is not None:
        backbones = sorted({b for d, b in known_pairs if d == dataset_arg})
        if len(backbones) == 1:
            return dataset_arg, backbones[0]
        raise ValueError(
            f"dataset={dataset_arg!r} 对应的 backbone 无法唯一确定：{backbones}。"
            "请同时传 --backbone。"
        )

    if backbone_arg is not None:
        datasets = sorted({d for d, b in known_pairs if b == backbone_arg})
        if len(datasets) == 1:
            return datasets[0], backbone_arg
        raise ValueError(
            f"backbone={backbone_arg!r} 对应的 dataset 无法唯一确定：{datasets}。"
            "请同时传 --dataset。"
        )

    if len(known_pairs) == 1:
        return next(iter(known_pairs))

    if len(known_pairs) == 0:
        raise ValueError(
            "没有从 config_used.yaml 中识别出 dataset/backbone。"
            "请显式传 --dataset 和 --backbone。"
        )

    choices = ", ".join(f"{d}+{b}" for d, b in sorted(known_pairs))
    raise ValueError(
        "输入目录中存在多个 dataset/backbone 组合："
        f"{choices}。请显式传 --dataset 和 --backbone。"
    )


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    if args.window <= 0:
        raise ValueError(f"--window 必须大于 0，当前值：{args.window}")

    if args.max_round is not None and args.max_round <= 0:
        raise ValueError(
            f"--max-round 必须大于 0，当前值：{args.max_round}"
        )

    csv_files = sorted(input_dir.rglob("results.csv"))
    if not csv_files:
        raise FileNotFoundError(f"在 {input_dir} 下没有找到任何 results.csv。")

    runs: List[Dict[str, object]] = []
    for csv_path in csv_files:
        dataset, backbone, method = load_run_metadata(csv_path)
        runs.append(
            {
                "csv": csv_path,
                "dataset": dataset,
                "backbone": backbone,
                "method": method,
            }
        )

    dataset, backbone = choose_dataset_backbone(
        runs=runs,
        dataset_arg=args.dataset,
        backbone_arg=args.backbone,
    )

    matching_runs = [
        run
        for run in runs
        if run["dataset"] == dataset and run["backbone"] == backbone
    ]

    if not matching_runs:
        raise ValueError(
            f"没有找到 dataset={dataset!r}, backbone={backbone!r} 的 results.csv。"
        )

    latest_by_method: Dict[str, Path] = {}
    for run in matching_runs:
        method = str(run["method"])
        csv_path = Path(run["csv"])
        previous = latest_by_method.get(method)
        if previous is None or csv_path.stat().st_mtime > previous.stat().st_mtime:
            latest_by_method[method] = csv_path

    selected = sorted(
        latest_by_method.items(),
        key=lambda item: method_order(item[0]),
    )

    if len(selected) < 5:
        print(
            f"[Warning] 当前只识别到 {len(selected)} 个算法结果；"
            "将对已找到的结果作图。"
        )
    elif len(selected) > 5:
        print(
            f"[Warning] 当前识别到 {len(selected)} 个算法结果；"
            "图中会全部绘制。"
        )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    print()
    print(f"Dataset : {dataset}")
    print(f"Backbone: {backbone}")
    print(f"Window  : {args.window}")
    print("Results :")

    all_smoothed: List[float] = []
    max_round = 0

    for index, (method, csv_path) in enumerate(selected):
        label = friendly_method_name(method)

        if label in HIDDEN_METHODS:
            continue

        rounds, test_acc = read_test_acc(csv_path)

        if args.max_round is not None:
            filtered = [
                (round_id, acc)
                for round_id, acc in zip(rounds, test_acc)
                if round_id <= args.max_round
            ]

            if not filtered:
                raise ValueError(
                    f"{csv_path} 在 round <= {args.max_round} 范围内没有数据。"
                )

            rounds = [item[0] for item in filtered]
            test_acc = [item[1] for item in filtered]

        smoothed = moving_average(test_acc, args.window)

        fallback_style = PLOT_STYLES[index % len(PLOT_STYLES)]
        color = METHOD_COLORS.get(label, fallback_style["color"])
        style = STYLE_BY_COLOR.get(color, fallback_style)

        ax.plot(
            rounds,
            smoothed,
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            color=color,
            label=label,
            solid_capstyle="butt",
            dash_capstyle="butt",
            zorder=3,
        )

        all_smoothed.extend(smoothed)
        if rounds:
            max_round = max(max_round, max(rounds))

        print(
            f"  {label:<14} "
            f"rounds={len(rounds):>3} "
            f"final={test_acc[-1]:>6.2f}% "
            f"best={max(test_acc):>6.2f}% "
            f"<- {csv_path}"
        )

    ax.set_xlabel("Training Round", fontsize=21, labelpad=6)
    ax.set_ylabel("Test Accuracy (%)", fontsize=21, labelpad=6)

    x_ticks = [
        max_round * i / 5
        for i in range(6)
    ]
    x_tick_labels = [
        f"{int(round(tick))}" if abs(tick - round(tick)) < 1e-9 else f"{tick:g}"
        for tick in x_ticks
    ]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlim(0, max_round)

    if not all_smoothed:
        raise ValueError("没有可用于绘图的 test_acc 数据。")

    y_min = min(all_smoothed)
    y_max = max(all_smoothed)

    # GAPSL 风格：纵轴下边界从较整齐的主刻度开始；
    # 内部主刻度尽量使用固定漂亮步长，最上面的刻度直接贴到上边框。
    y_lower = math.floor(y_min / 5) * 5

    # 顶部只保留少量余量，并把上边框写成实际的纵坐标值。
    y_upper = math.ceil((y_max + 1.0) / 1.0)
    if y_upper - y_max < 1.0:
        y_upper += 1

    # 内部刻度尽量控制在约 3 个区间，最后一个刻度单独用 y_upper。
    raw_step = (y_upper - y_lower) / 3
    nice_steps = [5, 10, 15, 20, 25, 30]
    y_step = min(
        nice_steps,
        key=lambda step: abs(step - raw_step),
    )

    y_ticks = list(
        range(
            int(y_lower),
            int(y_upper),
            int(y_step),
        )
    )

    if not y_ticks:
        y_ticks = [int(y_lower)]

    # 如果最后一个常规刻度离上边框太近，则删除该刻度，
    # 避免出现类似 70 和 73 同时挤在顶部的情况。
    if y_ticks and (y_upper - y_ticks[-1]) < 0.5 * y_step:
        y_ticks.pop()

    # 最后一个纵轴刻度始终等于上边框的实际纵坐标值。
    if not y_ticks or y_ticks[-1] != int(y_upper):
        y_ticks.append(int(y_upper))

    ax.set_ylim(y_lower, y_upper)
    ax.set_yticks(y_ticks)

    ax.grid(
        which="major",
        axis="both",
        visible=True,
        linestyle="-",
        linewidth=1.0,
        color="#D9D9D9",
        alpha=0.9,
        zorder=0,
    )
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color("black")

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        width=1.8,
        length=3.5,
        top=True,
        right=True,
        pad=4,
    )

    handles, labels = ax.get_legend_handles_labels()

    # 图例顺序和图例显示名称彻底解耦。
    # 以后只改 LEGEND_NAMES，就能修改图例文字；
    # 不需要再改 friendly_method_name()、method_order() 或这里的排序逻辑。
    legend_items = {
        label: handle
        for handle, label in zip(handles, labels)
    }

    ordered_internal_labels = [
        label
        for label in LEGEND_ORDER
        if label in legend_items
    ]

    # 如果出现不在 LEGEND_ORDER 中的新算法，也保留在图例中。
    for label in labels:
        if label not in ordered_internal_labels:
            ordered_internal_labels.append(label)

    ordered_handles = [
        legend_items[label]
        for label in ordered_internal_labels
    ]

    display_labels = [
        LEGEND_NAMES.get(label, label)
        for label in ordered_internal_labels
    ]

    legend = ax.legend(
        ordered_handles,
        display_labels,
        loc="lower right",
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
        borderpad=0.22,
        labelspacing=0.18,
        columnspacing=0.8,
        handlelength=1.65,
        handletextpad=0.25,
        fontsize=14,
    )
    legend.get_frame().set_linewidth(1.6)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else Path(__file__).resolve().parent.parent / "pictures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{dataset}_{backbone}.png"
    pdf_path = output_dir / f"{dataset}_{backbone}.pdf"

    fig.tight_layout()

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

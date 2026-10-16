#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


# ============================================================
# 只需要修改这里：横轴、算法名、数值、颜色
# ============================================================

ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]

METHODS = [
    {
        "name": "FedAvg-MoE",
        "values": [62.93, 78.15, 82.71, 85.35, 86.50],
    },
    {
        "name": "SOMFed",
        "values": [60.94, 77.29, 82.51, 84.98, 86.03],
    },
    {
        "name": "Fed-MoE",
        "values": [58.26, 75.58, 80.61, 84.99, 86.30],
    },
    {
        "name": "FedMoE-DA",
        "values": [55.85, 75.60, 82.32, 84.42, 85.86],
    },
    {
        "name": "CurvFedMoE",
        "values": [69.49, 79.33, 85.09, 86.60, 87.48],
    },
]

# ============================================================
# 颜色配置
#
# 保持下面 5 种颜色不变时，
# 以后只需要修改 METHOD_COLORS 中的对应关系即可调换颜色。
# ============================================================

COLORS = {
    "purple": "#A000A0",
    "gray": "#8A8A8A",
    "blue": "#1689D8",
    "cyan": "#38C7CB",
    "yellow": "#FFD200",
}

METHOD_COLORS = {
    "FedAvg-MoE": COLORS["gray"],
    "SOMFed": COLORS["purple"],
    "Fed-MoE": COLORS["blue"],
    "FedMoE-DA": COLORS["cyan"],
    "CurvFedMoE": COLORS["yellow"],
}

# ============================================================
# 不绘制的算法
# 空集合表示全部绘制
# ============================================================

HIDDEN_METHODS = {
    "FedAvg-MoE",
}

# ============================================================
# 图例显示名称
#
# 左侧是脚本内部固定名称，不要改。
# 以后如果只想修改图例里的显示名称，
# 只需要修改右侧字符串即可。
# ============================================================

LEGEND_NAMES = {
    "FedAvg-MoE": "FedAvg-MoE",
    "SOMFed": "SOMFed",
    "Fed-MoE": "Fed-MoE",
    "FedMoE-DA": "FedMoE-DA",
    "CurvFedMoE": "GEA-FedMoE",
}

# ============================================================
# 图例显示顺序
#
# 两列图例下，下面的顺序会显示为：
# CurvFedMoE    SOMFed
# Fed-MoE       FedMoE-DA
# ============================================================

LEGEND_ORDER = [
    "CurvFedMoE",
    "Fed-MoE",
    "SOMFed",
    "FedMoE-DA",
]


# ============================================================
# 输出设置
# ============================================================

OUTPUT_DIR = Path("./paper_pic/bar")
OUTPUT_NAME = "cifar10_resnet18_dirichlet_best_accuracy_r40"


# ============================================================
# 图像尺寸与字体
# ============================================================

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
        "mathtext.fontset": "stix",
        "font.size": 15,
        "axes.labelsize": 21,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 13,
        "axes.linewidth": 1.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "legend.frameon": True,
    }
)


def validate_data() -> None:
    if not ALPHAS:
        raise ValueError("ALPHAS 不能为空。")

    if not METHODS:
        raise ValueError("METHODS 不能为空。")

    names = set()

    for method in METHODS:
        name = method["name"]
        values = method["values"]

        if name in names:
            raise ValueError(f"存在重复算法名：{name}")

        names.add(name)

        if len(values) != len(ALPHAS):
            raise ValueError(
                f"{name} 的数据数量为 {len(values)}，"
                f"但 ALPHAS 数量为 {len(ALPHAS)}。"
            )


def main() -> int:
    validate_data()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    x = np.arange(
        len(ALPHAS),
        dtype=float,
    )

    active_methods = [
        method
        for method in METHODS
        if method["name"] not in HIDDEN_METHODS
    ]

    # 参考图中 5 根柱子较紧凑，
    # 但柱子之间仍保留细小空隙。
    bar_width = 0.115
    bar_step = 0.135

    offsets = (
        np.arange(
            len(active_methods),
            dtype=float,
        )
        - (len(active_methods) - 1) / 2.0
    ) * bar_step

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH, FIG_HEIGHT)
    )

    # ========================================================
    # 分组柱状图
    # ========================================================

    for index, method in enumerate(active_methods):
        ax.bar(
            x + offsets[index],
            method["values"],
            width=bar_width,
            color=METHOD_COLORS[method["name"]],
            edgecolor="black",
            linewidth=1.15,
            zorder=3,
        )

    # ========================================================
    # 坐标轴
    # ========================================================

    ax.set_xlabel(
        r"Dirichlet $\alpha$",
        labelpad=2,
    )

    ax.set_ylabel(
        "Best Accuracy (%)",
        labelpad=5,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            f"{alpha:.1f}"
            for alpha in ALPHAS
        ]
    )

    # 与参考图一致
    ax.set_ylim(50, 98)

    ax.set_yticks(
        [50, 60, 70, 80, 90]
    )

    ax.set_xlim(
        -0.48,
        len(ALPHAS) - 1 + 0.48,
    )

    # ========================================================
    # 背景网格
    # 只有横向灰色网格
    # ========================================================

    ax.grid(
        which="major",
        axis="y",
        visible=True,
        linestyle="-",
        linewidth=1.0,
        color="#C8C8C8",
        alpha=0.70,
        zorder=0,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    ax.set_axisbelow(True)

    # ========================================================
    # 黑色坐标轴边框
    # ========================================================

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("black")

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        width=1.6,
        length=4.0,
        top=True,
        right=True,
        pad=5,
    )

    # ========================================================
    # 图例
    # ========================================================

    legend_methods = [
        next(
            method
            for method in active_methods
            if method["name"] == name
        )
        for name in LEGEND_ORDER
        if any(
            method["name"] == name
            for method in active_methods
        )
    ]

    legend_handles = [
        Patch(
            facecolor=METHOD_COLORS[method["name"]],
            edgecolor="black",
            linewidth=1.0,
            label=method["name"],
        )
        for method in legend_methods
    ]

    legend_labels = [
        LEGEND_NAMES.get(
            method["name"],
            method["name"],
        )
        for method in legend_methods
    ]

    legend = ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper right",
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
        borderpad=0.12,
        labelspacing=0.08,
        columnspacing=0.45,
        handlelength=0.50,
        handleheight=0.95,
        handletextpad=0.12,
        fontsize=12.5,
    )

    legend.get_frame().set_linewidth(1.2)

    # ========================================================
    # 保存
    # ========================================================

    fig.tight_layout(
        pad=0.35
    )

    png_path = (
        OUTPUT_DIR
        / f"{OUTPUT_NAME}.png"
    )

    pdf_path = (
        OUTPUT_DIR
        / f"{OUTPUT_NAME}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

    print(
        f"Saved PNG: "
        f"{png_path.resolve()}"
    )

    print(
        f"Saved PDF: "
        f"{pdf_path.resolve()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

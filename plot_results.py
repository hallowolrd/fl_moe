from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import matplotlib

# 使用无界面后端，适合服务器和 SSH 环境。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare smoothed test-accuracy curves from two "
            "federated-learning metrics CSV files."
        )
    )

    parser.add_argument(
        "--csv-a",
        type=Path,
        required=True,
        help="Path to the first metrics.csv file.",
    )
    parser.add_argument(
        "--label-a",
        type=str,
        required=True,
        help="Legend label for the first experiment.",
    )
    parser.add_argument(
        "--csv-b",
        type=Path,
        required=True,
        help="Path to the second metrics.csv file.",
    )
    parser.add_argument(
        "--label-b",
        type=str,
        required=True,
        help="Legend label for the second experiment.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Trailing moving-average window size. Default: 5.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("accuracy_comparison.png"),
        help="Output PNG path. Default: accuracy_comparison.png.",
    )

    return parser.parse_args()


def load_accuracy_metrics(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    读取 metrics.csv 中的 round 和 test_accuracy。

    返回:
        rounds:
            严格递增的一维整数数组。
        accuracies:
            与 rounds 对应的一维浮点数组，取值位于 [0, 1]。
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"CSV file does not exist: {csv_path}"
        )

    rounds: list[int] = []
    accuracies: list[float] = []

    with csv_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV file has no header: {csv_path}"
            )

        required_fields = {"round", "test_accuracy"}
        missing_fields = (
            required_fields - set(reader.fieldnames)
        )

        if missing_fields:
            missing_text = ", ".join(
                sorted(missing_fields)
            )
            raise ValueError(
                f"CSV file {csv_path} is missing required "
                f"column(s): {missing_text}"
            )

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            round_text = (row.get("round") or "").strip()
            accuracy_text = (
                row.get("test_accuracy") or ""
            ).strip()

            if not round_text and not accuracy_text:
                continue

            try:
                round_value = int(round_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid round value at "
                    f"{csv_path}:{line_number}: "
                    f"{round_text!r}"
                ) from error

            try:
                accuracy_value = float(accuracy_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid test_accuracy value at "
                    f"{csv_path}:{line_number}: "
                    f"{accuracy_text!r}"
                ) from error

            if round_value <= 0:
                raise ValueError(
                    f"Round must be positive at "
                    f"{csv_path}:{line_number}."
                )

            if not math.isfinite(accuracy_value):
                raise ValueError(
                    f"test_accuracy must be finite at "
                    f"{csv_path}:{line_number}."
                )

            if not 0.0 <= accuracy_value <= 1.0:
                raise ValueError(
                    f"test_accuracy must be in [0, 1] at "
                    f"{csv_path}:{line_number}; "
                    f"received {accuracy_value}."
                )

            rounds.append(round_value)
            accuracies.append(accuracy_value)

    if not rounds:
        raise ValueError(
            f"CSV file contains no experiment rows: "
            f"{csv_path}"
        )

    rounds_array = np.asarray(
        rounds,
        dtype=np.int64,
    )
    accuracies_array = np.asarray(
        accuracies,
        dtype=np.float64,
    )

    order = np.argsort(
        rounds_array,
        kind="stable",
    )
    rounds_array = rounds_array[order]
    accuracies_array = accuracies_array[order]

    if np.any(np.diff(rounds_array) == 0):
        duplicate_rounds = sorted(
            {
                int(round_value)
                for round_value in rounds_array[
                    1:
                ][
                    np.diff(rounds_array) == 0
                ]
            }
        )
        raise ValueError(
            f"CSV file contains duplicate round values: "
            f"{duplicate_rounds}"
        )

    return rounds_array, accuracies_array


def trailing_moving_average(
    values: Sequence[float] | np.ndarray,
    window: int,
) -> np.ndarray:
    """
    计算向后滑动平均。

    前 window - 1 个位置使用当前已有的全部历史数据，
    因此不会丢弃训练前几轮。
    """
    if window <= 0:
        raise ValueError(
            "Moving-average window must be greater than 0."
        )

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(
            "Moving-average input must be one-dimensional."
        )

    if array.size == 0:
        raise ValueError(
            "Moving-average input must not be empty."
        )

    cumulative_sum = np.cumsum(
        array,
        dtype=np.float64,
    )
    result = np.empty_like(array)

    for index in range(array.size):
        start = max(
            0,
            index - window + 1,
        )
        total = cumulative_sum[index]

        if start > 0:
            total -= cumulative_sum[start - 1]

        count = index - start + 1
        result[index] = total / count

    return result


def normalize_output_path(
    output_path: Path,
) -> Path:
    """
    强制输出为 PNG。

    未提供扩展名时自动补充 .png；使用其他扩展名时直接报错。
    """
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".png")
    elif output_path.suffix.lower() != ".png":
        raise ValueError(
            "The output file must use the .png extension."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    return output_path


def plot_accuracy_comparison(
    *,
    rounds_a: np.ndarray,
    accuracies_a: np.ndarray,
    label_a: str,
    rounds_b: np.ndarray,
    accuracies_b: np.ndarray,
    label_b: str,
    window: int,
    output_path: Path,
) -> None:
    if not label_a.strip():
        raise ValueError(
            "label-a must not be empty."
        )

    if not label_b.strip():
        raise ValueError(
            "label-b must not be empty."
        )

    smoothed_a = trailing_moving_average(
        accuracies_a,
        window,
    )
    smoothed_b = trailing_moving_average(
        accuracies_b,
        window,
    )

    figure, axis = plt.subplots(
        figsize=(10, 6),
    )

    axis.plot(
        rounds_a,
        smoothed_a,
        linewidth=2.2,
        label=label_a,
    )
    axis.plot(
        rounds_b,
        smoothed_b,
        linewidth=2.2,
        label=label_b,
    )

    axis.set_title(
        f"Test Accuracy Comparison "
        f"(Moving Average Window = {window})"
    )
    axis.set_xlabel("Communication Round")
    axis.set_ylabel("Test Accuracy")
    axis.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=0,
        )
    )

    axis.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.5,
    )
    axis.legend()
    axis.margins(x=0.01)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()

    if args.window <= 0:
        raise ValueError(
            "--window must be greater than 0."
        )

    output_path = normalize_output_path(
        args.output
    )

    rounds_a, accuracies_a = load_accuracy_metrics(
        args.csv_a
    )
    rounds_b, accuracies_b = load_accuracy_metrics(
        args.csv_b
    )

    plot_accuracy_comparison(
        rounds_a=rounds_a,
        accuracies_a=accuracies_a,
        label_a=args.label_a,
        rounds_b=rounds_b,
        accuracies_b=accuracies_b,
        label_b=args.label_b,
        window=args.window,
        output_path=output_path,
    )

    print(f"Saved comparison plot to: {output_path}")


if __name__ == "__main__":
    main()

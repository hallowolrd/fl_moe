#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import random
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev

TARGET_SCRIPT = "experiments/expert_equal_local_kfac_whiten_layer_projection.py"
BASELINE_SCRIPT = "experiments/expert_equal_uniform.py"

STAGE_SPECS = {
    1: {"rounds": 40, "seeds": [0], "keep": 8},
    2: {"rounds": 80, "seeds": [0], "keep": 4},
    3: {"rounds": 150, "seeds": [0, 1], "keep": 2},
    4: {"rounds": 200, "seeds": [0, 1, 2], "keep": 1},
}

SEARCH_SPACE = {
    "learning_rate": [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03],
    "client_batch_size": [32, 64, 128],
    "local_epochs": [1, 2, 3, 4, 5],
    "momentum": [0.0, 0.5, 0.9, 0.95],
    "weight_decay": [0.0, 1e-4, 5e-4, 1e-3],
}

ANCHORS = [
    (0.01, 64, 1, 0.9, 5e-4),
    (0.01, 64, 2, 0.9, 1e-4),
    (0.01, 64, 3, 0.9, 1e-4),
    (0.01, 64, 4, 0.9, 1e-4),
    (0.005, 64, 3, 0.9, 1e-4),
    (0.005, 64, 4, 0.9, 1e-4),
    (0.01, 64, 3, 0.5, 1e-4),
    (0.01, 32, 3, 0.9, 1e-4),
    (0.01, 128, 3, 0.9, 1e-4),
    (0.02, 64, 2, 0.5, 1e-4),
    (0.015, 64, 2, 0.9, 1e-4),
    (0.008, 64, 4, 0.5, 1e-4),
]


@dataclass(frozen=True)
class Candidate:
    learning_rate: float
    client_batch_size: int
    local_epochs: int
    momentum: float
    weight_decay: float

    @property
    def config_id(self) -> str:
        text = (
            f"lr={self.learning_rate}|bs={self.client_batch_size}|"
            f"le={self.local_epochs}|mom={self.momentum}|wd={self.weight_decay}"
        )
        short_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        return (
            f"le{self.local_epochs}_lr{fmt(self.learning_rate)}_"
            f"bs{self.client_batch_size}_m{fmt(self.momentum)}_"
            f"wd{fmt(self.weight_decay)}_{short_hash}"
        )

    def cli_args(self) -> list[str]:
        return [
            "--learning-rate", str(self.learning_rate),
            "--client-batch-size", str(self.client_batch_size),
            "--local-epochs", str(self.local_epochs),
            "--momentum", str(self.momentum),
            "--weight-decay", str(self.weight_decay),
        ]


@dataclass
class CurveStats:
    final: float
    best: float
    all_mean: float
    tail_mean: float
    tail_std: float
    tail_slope: float


@dataclass
class PairResult:
    stage: int
    rounds: int
    seed: int
    physical_gpu: int
    config_id: str
    candidate: Candidate
    target: CurveStats
    baseline: CurveStats
    target_summary: str
    baseline_summary: str
    target_diagnostics: str | None

    @property
    def tail_gap(self) -> float:
        return self.target.tail_mean - self.baseline.tail_mean

    @property
    def auc_gap(self) -> float:
        return self.target.all_mean - self.baseline.all_mean

    @property
    def stability_penalty(self) -> float:
        return 0.5 * (self.target.tail_std + self.baseline.tail_std)

    @property
    def score(self) -> float:
        return 0.75 * self.tail_gap + 0.25 * self.auc_gap - 0.10 * self.stability_penalty


def fmt(value: float) -> str:
    return format(value, ".6g").replace("-", "m").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("1", "2", "3", "4", "all"), default="1")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--min-free-mib", type=int, default=3072)
    parser.add_argument("--memory-poll-seconds", type=float, default=15.0)
    parser.add_argument("--output-root", default="outputs_cifar10_kfac_hparam_search_dualgpu")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--stage1-candidates", type=int, default=22)
    parser.add_argument("--search-seed", type=int, default=20260810)
    parser.add_argument("--balance-loss-weight", type=float, default=0.0)
    parser.add_argument("--extra-common-args", default="")
    return parser.parse_args()


def parse_gpu_ids(text: str) -> list[int]:
    ids = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not ids:
        raise ValueError("--gpus must contain at least one GPU id.")
    if len(ids) != len(set(ids)):
        raise ValueError("--gpus contains duplicate ids.")
    if any(gpu < 0 for gpu in ids):
        raise ValueError("GPU ids must be non-negative.")
    return ids


def candidate_from_tuple(values) -> Candidate:
    lr, bs, le, mom, wd = values
    return Candidate(float(lr), int(bs), int(le), float(mom), float(wd))


def generate_stage1_candidates(total: int, seed: int) -> list[Candidate]:
    if total < len(ANCHORS):
        raise ValueError(f"--stage1-candidates must be >= {len(ANCHORS)}.")
    rng = random.Random(seed)
    candidates: dict[str, Candidate] = {}
    for values in ANCHORS:
        c = candidate_from_tuple(values)
        candidates[c.config_id] = c
    while len(candidates) < total:
        c = Candidate(
            rng.choice(SEARCH_SPACE["learning_rate"]),
            rng.choice(SEARCH_SPACE["client_batch_size"]),
            rng.choice(SEARCH_SPACE["local_epochs"]),
            rng.choice(SEARCH_SPACE["momentum"]),
            rng.choice(SEARCH_SPACE["weight_decay"]),
        )
        candidates[c.config_id] = c
    return sorted(candidates.values(), key=lambda c: c.config_id)


def query_gpu_free_mib(physical_gpu: int) -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={physical_gpu}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).strip()
        return int(out.splitlines()[0].strip())
    except Exception:
        return None


def wait_for_gpu_memory(gpu: int, min_free_mib: int, poll_seconds: float) -> None:
    while True:
        free = query_gpu_free_mib(gpu)
        if free is None:
            print(f"[GPU {gpu}] memory query unavailable; launch without guard.", flush=True)
            return
        if free >= min_free_mib:
            print(f"[GPU {gpu}] free={free} MiB >= {min_free_mib} MiB; launch.", flush=True)
            return
        print(f"[GPU {gpu}] free={free} MiB < {min_free_mib} MiB; wait {poll_seconds:g}s.", flush=True)
        time.sleep(poll_seconds)


def before_after_new_summary(root: Path, before: set[Path]) -> Path:
    after = {p.resolve() for p in root.rglob("summary.json") if p.is_file()}
    new = sorted(after - before, key=lambda p: p.stat().st_mtime_ns)
    if len(new) != 1:
        raise RuntimeError(f"Expected one new summary.json under {root}, got {len(new)}.")
    return new[0]


def run_experiment(
    *,
    script: Path,
    role: str,
    candidate: Candidate,
    rounds: int,
    seed: int,
    physical_gpu: int,
    min_free_mib: int,
    memory_poll_seconds: float,
    pair_root: Path,
    project_root: Path,
    balance_loss_weight: float,
    extra_common_args: list[str],
) -> tuple[Path, Path | None]:
    pair_root.mkdir(parents=True, exist_ok=True)
    before = {p.resolve() for p in pair_root.rglob("summary.json") if p.is_file()}

    wait_for_gpu_memory(physical_gpu, min_free_mib, memory_poll_seconds)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)

    command = [
        sys.executable,
        str(script),
        "--dataset-name", "cifar10",
        "--device", "cuda:0",
        "--output-root", str(pair_root),
        "--num-rounds", str(rounds),
        "--seed", str(seed),
        "--balance-loss-weight", str(balance_loss_weight),
        *candidate.cli_args(),
        *extra_common_args,
    ]

    print(
        f"[GPU {physical_gpu}] START {role} | rounds={rounds} | seed={seed} | {candidate.config_id}",
        flush=True,
    )
    subprocess.run(command, cwd=project_root, env=env, check=True)

    summary = before_after_new_summary(pair_root, before)
    diagnostics = None
    if role == "target":
        files = list(summary.parent.glob("kfac_projection_diagnostics.jsonl"))
        if files:
            diagnostics = files[0]

    print(f"[GPU {physical_gpu}] DONE {role} | {candidate.config_id}", flush=True)
    return summary, diagnostics


def read_accuracy_curve(summary_path: Path) -> list[float]:
    metrics = summary_path.parent / "metrics.csv"
    values = []
    with metrics.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            v = float(row["test_accuracy"])
            if not math.isfinite(v):
                raise RuntimeError(f"Non-finite accuracy in {metrics}")
            values.append(v)
    if not values:
        raise RuntimeError(f"No accuracy rows in {metrics}")
    return values


def linear_slope(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    n = len(values)
    mx = (n - 1) / 2.0
    my = mean(values)
    num = sum((i - mx) * (v - my) for i, v in enumerate(values))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0


def curve_stats(values: list[float], tail_window: int = 10) -> CurveStats:
    tail = values[-min(tail_window, len(values)):]
    return CurveStats(
        final=values[-1],
        best=max(values),
        all_mean=mean(values),
        tail_mean=mean(tail),
        tail_std=pstdev(tail) if len(tail) > 1 else 0.0,
        tail_slope=linear_slope(tail),
    )


def run_pair(
    *,
    stage: int,
    candidate: Candidate,
    rounds: int,
    seed: int,
    physical_gpu: int,
    min_free_mib: int,
    memory_poll_seconds: float,
    output_root: Path,
    project_root: Path,
    target_script: Path,
    baseline_script: Path,
    balance_loss_weight: float,
    extra_common_args: list[str],
) -> PairResult:
    pair_root = output_root / f"stage_{stage}" / candidate.config_id / f"seed_{seed}"

    baseline_summary, _ = run_experiment(
        script=baseline_script, role="baseline", candidate=candidate, rounds=rounds,
        seed=seed, physical_gpu=physical_gpu, min_free_mib=min_free_mib,
        memory_poll_seconds=memory_poll_seconds, pair_root=pair_root,
        project_root=project_root, balance_loss_weight=balance_loss_weight,
        extra_common_args=extra_common_args,
    )
    target_summary, diagnostics = run_experiment(
        script=target_script, role="target", candidate=candidate, rounds=rounds,
        seed=seed, physical_gpu=physical_gpu, min_free_mib=min_free_mib,
        memory_poll_seconds=memory_poll_seconds, pair_root=pair_root,
        project_root=project_root, balance_loss_weight=balance_loss_weight,
        extra_common_args=extra_common_args,
    )

    target_curve = read_accuracy_curve(target_summary)
    baseline_curve = read_accuracy_curve(baseline_summary)
    if len(target_curve) != rounds or len(baseline_curve) != rounds:
        raise RuntimeError("Unexpected metrics length.")

    return PairResult(
        stage, rounds, seed, physical_gpu, candidate.config_id, candidate,
        curve_stats(target_curve), curve_stats(baseline_curve),
        str(target_summary), str(baseline_summary),
        str(diagnostics) if diagnostics else None,
    )


def pair_result_row(r: PairResult) -> dict[str, object]:
    c = r.candidate
    return {
        "stage": r.stage,
        "rounds": r.rounds,
        "seed": r.seed,
        "physical_gpu": r.physical_gpu,
        "config_id": r.config_id,
        "learning_rate": c.learning_rate,
        "client_batch_size": c.client_batch_size,
        "local_epochs": c.local_epochs,
        "momentum": c.momentum,
        "weight_decay": c.weight_decay,
        "target_final": r.target.final,
        "baseline_final": r.baseline.final,
        "final_gap": r.target.final - r.baseline.final,
        "target_best": r.target.best,
        "baseline_best": r.baseline.best,
        "target_tail_mean": r.target.tail_mean,
        "baseline_tail_mean": r.baseline.tail_mean,
        "tail_gap": r.tail_gap,
        "target_tail_std": r.target.tail_std,
        "baseline_tail_std": r.baseline.tail_std,
        "target_tail_slope": r.target.tail_slope,
        "baseline_tail_slope": r.baseline.tail_slope,
        "target_all_mean": r.target.all_mean,
        "baseline_all_mean": r.baseline.all_mean,
        "auc_gap": r.auc_gap,
        "score": r.score,
        "target_summary": r.target_summary,
        "baseline_summary": r.baseline_summary,
        "target_diagnostics": r.target_diagnostics or "",
    }


def aggregate_candidate_results(results: list[PairResult]) -> list[dict[str, object]]:
    grouped: dict[str, list[PairResult]] = {}
    for r in results:
        grouped.setdefault(r.config_id, []).append(r)

    rows = []
    for config_id, items in grouped.items():
        c = items[0].candidate
        scores = [x.score for x in items]
        gaps = [x.tail_gap for x in items]
        aucs = [x.auc_gap for x in items]
        tmeans = [x.target.tail_mean for x in items]
        bmeans = [x.baseline.tail_mean for x in items]
        tslopes = [x.target.tail_slope for x in items]
        bslopes = [x.baseline.tail_slope for x in items]
        tstds = [x.target.tail_std for x in items]
        bstds = [x.baseline.tail_std for x in items]
        converged = (
            max(abs(v) for v in tslopes) <= 0.0015
            and max(abs(v) for v in bslopes) <= 0.0015
            and mean(tstds) <= 0.02
            and mean(bstds) <= 0.02
        )
        rows.append({
            "config_id": config_id,
            **asdict(c),
            "num_seeds": len(items),
            "mean_score": mean(scores),
            "score_std": pstdev(scores) if len(scores) > 1 else 0.0,
            "mean_tail_gap": mean(gaps),
            "tail_gap_std": pstdev(gaps) if len(gaps) > 1 else 0.0,
            "min_tail_gap": min(gaps),
            "max_tail_gap": max(gaps),
            "mean_auc_gap": mean(aucs),
            "mean_target_tail": mean(tmeans),
            "mean_baseline_tail": mean(bmeans),
            "mean_target_tail_std": mean(tstds),
            "mean_baseline_tail_std": mean(bstds),
            "max_abs_target_tail_slope": max(abs(v) for v in tslopes),
            "max_abs_baseline_tail_slope": max(abs(v) for v in bslopes),
            "wins": sum(g > 0 for g in gaps),
            "losses": sum(g < 0 for g in gaps),
            "converged_flag": converged,
        })

    rows.sort(
        key=lambda x: (float(x["mean_score"]), float(x["mean_tail_gap"]), float(x["mean_target_tail"])),
        reverse=True,
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def candidate_from_row(row: dict[str, str]) -> Candidate:
    return Candidate(
        float(row["learning_rate"]),
        int(row["client_batch_size"]),
        int(row["local_epochs"]),
        float(row["momentum"]),
        float(row["weight_decay"]),
    )


def load_top_candidates(path: Path, keep: int) -> list[Candidate]:
    result = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            result.append(candidate_from_row(row))
            if len(result) >= keep:
                break
    if not result:
        raise RuntimeError(f"No candidates in {path}")
    return result


def print_ranking(rows: list[dict[str, object]], top: int = 10) -> None:
    print("\nRANKING")
    for i, row in enumerate(rows[:top], 1):
        print(
            f"{i:>2}. {row['config_id']} | "
            f"gap={float(row['mean_tail_gap'])*100:+.3f} pp | "
            f"target={float(row['mean_target_tail'])*100:.3f}% | "
            f"uniform={float(row['mean_baseline_tail'])*100:.3f}% | "
            f"conv={row['converged_flag']}"
        )


def run_stage_parallel(
    *,
    stage: int,
    candidates: list[Candidate],
    args: argparse.Namespace,
    gpu_ids: list[int],
    project_root: Path,
    output_root: Path,
    target_script: Path,
    baseline_script: Path,
    extra_common_args: list[str],
) -> list[dict[str, object]]:
    spec = STAGE_SPECS[stage]
    rounds = int(spec["rounds"])
    seeds = list(spec["seeds"])

    jobs: queue.Queue = queue.Queue()
    for c in candidates:
        for seed in seeds:
            jobs.put((c, seed))

    results: list[PairResult] = []
    errors = []
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            item = jobs.get()
            try:
                if item is None:
                    return
                c, seed = item
                try:
                    r = run_pair(
                        stage=stage, candidate=c, rounds=rounds, seed=seed,
                        physical_gpu=gpu, min_free_mib=args.min_free_mib,
                        memory_poll_seconds=args.memory_poll_seconds,
                        output_root=output_root, project_root=project_root,
                        target_script=target_script, baseline_script=baseline_script,
                        balance_loss_weight=args.balance_loss_weight,
                        extra_common_args=extra_common_args,
                    )
                    with lock:
                        results.append(r)
                        print(
                            f"[PAIR DONE GPU {gpu}] {c.config_id} seed={seed} | "
                            f"gap={r.tail_gap*100:+.3f} pp",
                            flush=True,
                        )
                except Exception as exc:
                    with lock:
                        errors.append((gpu, c.config_id, seed, repr(exc)))
                        print(f"[PAIR FAILED GPU {gpu}] {c.config_id} seed={seed} | {exc!r}", flush=True)
            finally:
                jobs.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpu_ids]
    for t in threads:
        t.start()

    jobs.join()
    for _ in threads:
        jobs.put(None)
    for t in threads:
        t.join()

    if errors:
        raise RuntimeError(f"{len(errors)} pair jobs failed: {errors}")

    report = output_root / "_search_reports" / f"stage_{stage}"
    pair_rows = [pair_result_row(r) for r in results]
    pair_rows.sort(key=lambda x: (str(x["config_id"]), int(x["seed"])))
    write_csv(report / "pair_results.csv", pair_rows)

    ranking = aggregate_candidate_results(results)
    write_csv(report / "candidate_ranking.csv", ranking)

    with (report / "stage_config.json").open("w", encoding="utf-8") as f:
        json.dump({
            "stage": stage,
            "rounds": rounds,
            "seeds": seeds,
            "num_candidates": len(candidates),
            "physical_gpus": gpu_ids,
            "max_search_processes_per_gpu": 1,
            "min_free_mib_before_launch": args.min_free_mib,
            "keep_for_next_stage": spec["keep"],
            "balance_loss_weight": args.balance_loss_weight,
            "search_space": SEARCH_SPACE,
        }, f, ensure_ascii=False, indent=2)

    print_ranking(ranking)
    print(f"\nReports: {report}")
    return ranking


def main() -> None:
    args = parse_args()
    gpu_ids = parse_gpu_ids(args.gpus)

    project_root = Path(args.project_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()

    target_script = (project_root / TARGET_SCRIPT).resolve()
    baseline_script = (project_root / BASELINE_SCRIPT).resolve()

    if not target_script.is_file():
        raise FileNotFoundError(target_script)
    if not baseline_script.is_file():
        raise FileNotFoundError(baseline_script)

    extra_common_args = shlex.split(args.extra_common_args)
    output_root.mkdir(parents=True, exist_ok=True)

    stages = [1, 2, 3, 4] if args.stage == "all" else [int(args.stage)]

    for stage in stages:
        if stage == 1:
            candidates = generate_stage1_candidates(args.stage1_candidates, args.search_seed)
        else:
            prev = stage - 1
            prev_csv = output_root / "_search_reports" / f"stage_{prev}" / "candidate_ranking.csv"
            candidates = load_top_candidates(prev_csv, int(STAGE_SPECS[prev]["keep"]))

        run_stage_parallel(
            stage=stage, candidates=candidates, args=args, gpu_ids=gpu_ids,
            project_root=project_root, output_root=output_root,
            target_script=target_script, baseline_script=baseline_script,
            extra_common_args=extra_common_args,
        )


if __name__ == "__main__":
    main()

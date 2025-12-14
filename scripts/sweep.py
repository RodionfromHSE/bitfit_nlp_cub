#!/usr/bin/env python
"""Grid search runner for hyperparameter tuning."""

import argparse
import subprocess
from itertools import product


def run_sweep(
    tasks: list[str],
    methods: list[str],
    seeds: list[int],
    tracker: str = "none",
    max_steps: int = -1,
) -> None:
    """Run experiments for all combinations of tasks, methods, and seeds."""
    combinations = list(product(tasks, methods, seeds))
    total = len(combinations)

    for i, (task, method, seed) in enumerate(combinations, 1):
        print(f"\n{'='*60}")
        print(f"Running experiment {i}/{total}: task={task}, method={method}, seed={seed}")
        print(f"{'='*60}\n")

        cmd = [
            "uv",
            "run",
            "python",
            "scripts/train.py",
            f"task={task}",
            f"method={method}",
            f"training.seed={seed}",
            f"tracker={tracker}",
        ]

        if max_steps > 0:
            cmd.append(f"training.max_steps={max_steps}")

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"WARNING: Experiment failed with return code {result.returncode}")


def run_lr_sweep(task: str, method: str, lrs: list[float], tracker: str = "none") -> None:
    """Run learning rate sweep for a specific task and method."""
    for lr in lrs:
        print(f"\n{'='*60}")
        print(f"Running LR sweep: task={task}, method={method}, lr={lr}")
        print(f"{'='*60}\n")

        cmd = [
            "uv",
            "run",
            "python",
            "scripts/train.py",
            f"task={task}",
            f"method={method}",
            f"method.lr={lr}",
            f"tracker={tracker}",
        ]

        subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment sweeps")
    parser.add_argument("--mode", choices=["full", "lr", "quick"], default="quick")
    parser.add_argument("--tracker", default="none")
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    if args.mode == "quick":
        run_sweep(
            tasks=["sst2"],
            methods=["full_ft", "bitfit", "bitfit_subset", "lora", "prompt_tuning"],
            seeds=[42],
            tracker=args.tracker,
            max_steps=args.max_steps if args.max_steps > 0 else 100,
        )
    elif args.mode == "full":
        run_sweep(
            tasks=["sst2", "mrpc", "rte"],
            methods=["full_ft", "bitfit", "bitfit_subset", "lora", "prompt_tuning"],
            seeds=[0, 1, 2],
            tracker=args.tracker,
            max_steps=args.max_steps,
        )
    elif args.mode == "lr":
        for method in ["full_ft", "bitfit", "lora", "prompt_tuning"]:
            lrs = {
                "full_ft": [1e-5, 2e-5, 3e-5, 5e-5],
                "bitfit": [1e-4, 4e-4, 7e-4, 1e-3],
                "lora": [1e-4, 2e-4, 5e-4],
                "prompt_tuning": [1e-4, 2e-4, 5e-4, 1e-3],
            }
            run_lr_sweep("sst2", method, lrs[method], args.tracker)


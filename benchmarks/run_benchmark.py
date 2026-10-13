"""Measure the single-run success rate of master-tree training.

For every target function and master depth, ``N`` independent training runs (seeds
``seed, ..., seed + N - 1``, no multistart) are performed; a run succeeds if its snapped tree
reproduces the data with max error < tol. The success rate is reported with a 95 % Wilson score
interval and written to CSV. Nothing is pre-filled: every number comes from the runs.

Usage::

    python benchmarks/run_benchmark.py --quick
    python benchmarks/run_benchmark.py --runs 50 --out benchmarks/results/full.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from eml_sr.core.tree import Node, from_rpn
from eml_sr.model.trainer import TrainConfig, train_once

QUICK_RUNS = 5
DEFAULT_RUNS = 30
X_MIN, X_MAX, N_POINTS = 0.5, 2.0, 40


@dataclass(frozen=True)
class Target:
    """Benchmark target with a known shortest tree (used only to report representability)."""

    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    known_tree: Node

    @property
    def min_depth(self) -> int:
        """Depth of the known tree, i.e. the smallest master depth able to represent it."""
        return self.known_tree.depth()


TARGETS: tuple[Target, ...] = (
    Target("exp(x)", np.exp, from_rpn("x 1 E")),
    Target("ln(x)", np.log, from_rpn("1 1 x E 1 E E")),
    Target("exp(exp(x))", lambda z: np.exp(np.exp(z)), from_rpn("x 1 E 1 E")),
    Target("e - ln(x)", lambda z: np.e - np.log(z), from_rpn("1 x E")),
    Target("exp(x) - 1", lambda z: np.exp(z) - 1.0, from_rpn("x 1 1 E E")),
    Target("exp(x) - ln(x)", lambda z: np.exp(z) - np.log(z), from_rpn("x x E")),
)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion.

    Args:
        successes: Number of successes.
        n: Number of trials (``> 0``).
        z: Normal quantile (default: 95 % two-sided).

    Returns:
        ``(low, high)``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def run(
    runs: int, depths: list[int], seed: int, config: TrainConfig, targets: tuple[Target, ...]
) -> list[dict[str, object]]:
    """Run the benchmark and return one row per (target, depth)."""
    x = np.linspace(X_MIN, X_MAX, N_POINTS)
    rows: list[dict[str, object]] = []
    for target in targets:
        y = target.fn(x)
        for depth in depths:
            start = time.perf_counter()
            successes = 0
            for i in range(runs):
                successes += int(train_once(x, y, depth, seed + i, config).success)
            seconds = time.perf_counter() - start
            low, high = wilson_interval(successes, runs)
            row: dict[str, object] = {
                "target": target.name,
                "depth": depth,
                "representable": target.min_depth <= depth,
                "runs": runs,
                "successes": successes,
                "rate": successes / runs,
                "ci95_low": low,
                "ci95_high": high,
                "mean_seconds_per_run": seconds / runs,
                "epochs": config.epochs,
            }
            rows.append(row)
            print(
                f"{target.name:16s} d={depth} {successes:3d}/{runs:<3d} "
                f"rate={successes / runs:.2f} CI95=[{low:.2f}, {high:.2f}] "
                f"{seconds / runs:.2f}s/run",
                flush=True,
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the benchmark and write the CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help=f"{QUICK_RUNS} runs per cell")
    parser.add_argument("--runs", type=int, default=None, help=f"default {DEFAULT_RUNS}")
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--epochs", type=int, default=TrainConfig().epochs)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    runs = args.runs if args.runs is not None else (QUICK_RUNS if args.quick else DEFAULT_RUNS)
    out = args.out or Path("benchmarks/results") / ("quick.csv" if args.quick else "benchmark.csv")
    config = TrainConfig(epochs=args.epochs)
    print(
        f"python {platform.python_version()}, torch {torch.__version__}, runs={runs}, "
        f"depths={args.depths}, epochs={config.epochs}, x in [{X_MIN}, {X_MAX}] ({N_POINTS} pts)"
    )
    rows = run(runs, args.depths, args.seed, config, TARGETS)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

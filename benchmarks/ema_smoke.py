"""Reproducible paired EMA pilot; not a statistical confirmation."""

import csv
from pathlib import Path

import numpy as np
import torch

from eml_sr.model.trainer import TrainConfig, snap_errors, train_once


def main() -> None:
    """Run fixed paired seeds and save actual observations."""
    torch.set_num_threads(1)
    x = np.linspace(0.5, 2, 40)
    test_x = np.linspace(0.503, 1.997, 301)
    rows = []
    for name, target in (("exp", np.exp), ("log", np.log)):
        for seed in range(10000, 10005):
            for beta in (0.0, 0.99):
                result = train_once(x, target(x), 3, seed, TrainConfig(ema_beta=beta))
                row = result.as_dict()
                _, error = snap_errors(result.tree, test_x, target(test_x))
                row.update(
                    target=name, ema_beta=beta, test_max_error=error, test_success=error < 1e-8
                )
                rows.append(row)
    path = Path("benchmarks/results/ema_smoke.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()

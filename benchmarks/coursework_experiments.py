"""Run preregistered coursework experiments with resumable per-task CSV files."""
# ruff: noqa: ANN

import argparse
import csv
import json
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from eml_sr.core.tree import ONE, Eml, X
from eml_sr.model.initialization import embed_tree
from eml_sr.model.trainer import TrainConfig, train_once


def targets():
    """Return independently verified target expressions."""
    return {
        "exp_minus_one": (lambda x: np.exp(x) - 1, Eml(X, Eml(ONE, ONE)), 2),
        "log": (np.log, Eml(ONE, Eml(Eml(ONE, X), ONE)), 3),
        "exp_e_minus_x": (lambda x: np.exp(np.e - x), Eml(Eml(ONE, Eml(X, ONE)), ONE), 3),
        "e_minus_exp": (lambda x: np.e - np.exp(x), Eml(ONE, Eml(Eml(X, ONE), ONE)), 3),
    }


def error(tree, x, y):
    """Evaluate without guarded operations; nonfinite values fail."""
    err = np.abs(tree.evaluate(x, safe=False) - y)
    return float(np.max(err)) if np.isfinite(err).all() else float("inf")


def task(args):
    """Execute one independent episode, preserving all round observations."""
    name, method, beta, seed, epochs, out = args
    torch.set_num_threads(1)
    path = Path(out) / f"{name}_{method}_{beta}_{seed}.csv"
    if path.exists():
        return str(path)
    fn, reference, depth = targets()[name]
    x = np.linspace(0.5, 2, 40)
    inside = np.linspace(0.503, 1.997, 301)
    outside = np.linspace(2.01, 2.5, 301)
    cfg = TrainConfig(epochs=epochs, ema_beta=beta)
    initial = None
    rounds = 3 if method in ("IR", "RF") else 1
    if method.startswith("k"):
        initial = embed_tree(reference, depth, mistakes=int(method[1:]), seed=seed)
    rows = []
    previous_logits = None
    for r in range(rounds):
        if method == "RF" and r:
            assert previous_logits is not None
            gen = torch.Generator().manual_seed(seed + 300000 + r)
            initial = [
                p + torch.randn(p.shape, generator=gen, dtype=p.dtype) * (0.5 / r)
                for p in previous_logits
            ]
        result = train_once(x, fn(x), depth, seed + 100000 * r, cfg, initial)
        previous_logits = result.logits
        row = result.as_dict()
        row.update(
            target=name,
            method=method,
            beta=beta,
            episode_seed=seed,
            round=r,
            depth=depth,
            structural=result.tree == reference,
            interpolation_error=error(result.tree, inside, fn(inside)),
            extrapolation_error=error(result.tree, outside, fn(outside)),
        )
        row["verified_success"] = (
            max(row["interpolation_error"], row["extrapolation_error"]) < cfg.tol
        )
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def main():
    """Validate references, run workers, and consolidate the raw observations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", default="benchmarks/results/full")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for fn, tree, _ in targets().values():
        grid = np.linspace(0.5, 2.5, 301)
        np.testing.assert_allclose(tree.evaluate(grid, safe=False), fn(grid), atol=1e-9, rtol=0)
    meta = dict(
        args=vars(args),
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        platform=platform.platform(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        config=asdict(TrainConfig(epochs=args.epochs)),
        started=time.time(),
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    jobs = [
        (name, method, beta, seed, args.epochs, str(out))
        for name in targets()
        for method, beta in [
            ("IR", 0.0),
            ("IR", 0.99),
            ("RF", 0.0),
            ("RF", 0.99),
            ("k0", 0.0),
            ("k1", 0.0),
            ("k3", 0.0),
        ]
        for seed in range(20000, 20000 + args.runs)
    ]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, _ in enumerate(pool.map(task, jobs)):
            if (i + 1) % 30 == 0:
                print(f"{i + 1}/{len(jobs)} episodes", flush=True)
    rows = []
    for job in jobs:
        name, method, beta, seed, _, _ = job
        rows.extend(csv.DictReader((out / f"{name}_{method}_{beta}_{seed}.csv").open()))
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta["finished"] = time.time()
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Finished {len(rows)} runs", flush=True)


if __name__ == "__main__":
    main()

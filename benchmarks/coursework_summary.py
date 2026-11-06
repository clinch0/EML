"""Summarize raw coursework experiments without consulting test data for selection."""

# ruff: noqa: ANN001, ANN201
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from coursework_experiments import targets


def wilson(k, n):
    """Return the 95 percent Wilson score interval."""
    z = 1.959963984540054
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def paired(a, b):
    """Paired bootstrap difference, undefined interval with no discordances."""
    d = np.array(a, dtype=float) - np.array(b, dtype=float)
    if not np.any(d):
        return dict(delta=0.0, low=None, high=None, discordant=0)
    rng = np.random.default_rng(729)
    boot = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return dict(
        delta=float(d.mean()), low=float(lo), high=float(hi), discordant=int(np.count_nonzero(d))
    )


def main():
    """Create audit, aggregate data, baseline and publication figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmarks/results/full")
    out = Path(parser.parse_args().out)
    rows = list(csv.DictReader((out / "runs.csv").open()))
    episodes = defaultdict(list)
    for r in rows:
        episodes[(r["target"], r["method"], float(r["beta"]), int(r["episode_seed"]))].append(r)
    group = defaultdict(list)
    for (target, method, beta, seed), rounds in episodes.items():
        best = min(rounds, key=lambda r: float(r["max_error"]))
        group[(target, method, beta)].append(
            dict(
                seed=seed,
                success=best["verified_success"] == "True",
                structural=best["structural"] == "True",
                seconds=sum(float(r["seconds"]) for r in rounds),
                epochs=sum(int(r["epochs"]) for r in rounds),
                rounds=len(rounds),
                tree=best["tree"],
                train_error=float(best["max_error"]),
                interpolation=float(best["interpolation_error"]),
                extrapolation=float(best["extrapolation_error"]),
            )
        )
        if method == "IR" and beta == 0:
            first = min(rounds, key=lambda r: int(r["round"]))
            group[(target, "cold", 0.0)].append(
                dict(
                    seed=seed,
                    success=first["verified_success"] == "True",
                    structural=first["structural"] == "True",
                    seconds=float(first["seconds"]),
                    epochs=int(first["epochs"]),
                    rounds=1,
                )
            )
    summary = []
    for (target, method, beta), items in group.items():
        items.sort(key=lambda x: x["seed"])
        k = sum(x["success"] for x in items)
        lo, hi = wilson(k, len(items))
        summary.append(
            dict(
                target=target,
                method=method,
                beta=beta,
                k=k,
                n=len(items),
                rate=k / len(items),
                low=lo,
                high=hi,
                structural=sum(x["structural"] for x in items),
                seconds=float(np.median([x["seconds"] for x in items])),
                min_epochs=min(x["epochs"] for x in items),
                max_epochs=max(x["epochs"] for x in items),
            )
        )
    comparisons = []
    for name in targets():
        for label, a, b in [
            ("EMA", ("IR", 0.99), ("IR", 0.0)),
            ("RF", ("RF", 0.0), ("IR", 0.0)),
            ("RF_EMA", ("RF", 0.99), ("IR", 0.99)),
            ("basin", ("k0", 0.0), ("k3", 0.0)),
        ]:
            av = group[(name, *a)]
            bv = group[(name, *b)]
            assert [v["seed"] for v in av] == [v["seed"] for v in bv]
            comparisons.append(
                dict(
                    target=name,
                    comparison=label,
                    **paired([v["success"] for v in av], [v["success"] for v in bv]),
                )
            )
    baseline = []
    for name, (fn, _, _) in targets().items():
        x = np.linspace(0.5, 2, 40)
        polynomial = np.polynomial.Polynomial.fit(x, fn(x), 5)
        inside = np.linspace(0.503, 1.997, 301)
        outside = np.linspace(2.01, 2.5, 301)
        baseline.append(
            dict(
                target=name,
                interpolation=float(np.max(np.abs(polynomial(inside) - fn(inside)))),
                extrapolation=float(np.max(np.abs(polynomial(outside) - fn(outside)))),
            )
        )
    audit = dict(
        rows=len(rows),
        episodes=len(episodes),
        early_stops=sum(int(r["epochs"]) != 1000 for r in rows),
        total_training_seconds=sum(float(r["seconds"]) for r in rows),
        all_budgets_equal=all(
            v["min_epochs"] == v["max_epochs"] == (3000 if v["method"] in ("IR", "RF") else 1000)
            for v in summary
        ),
    )
    result = dict(summary=summary, comparisons=comparisons, baseline=baseline, audit=audit)
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for filename, data in [
        ("summary", summary),
        ("comparisons", comparisons),
        ("polynomial", baseline),
    ]:
        with (out / f"{filename}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, name in zip(axes.flat, targets(), strict=True):
        cells = [
            next(r for r in summary if (r["target"], r["method"], r["beta"]) == (name, m, b))
            for m, b in [("IR", 0.0), ("IR", 0.99), ("RF", 0.0), ("RF", 0.99)]
        ]
        rates = np.array([r["rate"] for r in cells])
        lo = np.array([r["low"] for r in cells])
        hi = np.array([r["high"] for r in cells])
        ax.bar(range(4), rates, color=["#59697d", "#6198b6", "#4e8973", "#9b8c59"])
        ax.errorbar(
            range(4),
            rates,
            yerr=[np.maximum(0, rates - lo), np.maximum(0, hi - rates)],
            fmt="none",
            color="black",
            capsize=4,
        )
        ax.set_xticks(range(4), ["IR", "IR+EMA", "RF", "RF+EMA"])
        ax.set_ylim(0, 1.08)
        ax.set_title(
            {
                "exp_minus_one": "exp(x)−1",
                "log": "ln(x)",
                "exp_e_minus_x": "exp(e−x)",
                "e_minus_exp": "e−exp(x)",
            }[name]
        )
        ax.set_ylabel("Доля успеха")
    fig.tight_layout()
    fig.savefig(out / "methods.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, name in zip(axes.flat, targets(), strict=True):
        cells = [
            next(r for r in summary if (r["target"], r["method"], r["beta"]) == (name, m, 0.0))
            for m in ["k0", "k1", "k3"]
        ]
        rates = np.array([r["rate"] for r in cells])
        lo = np.array([r["low"] for r in cells])
        hi = np.array([r["high"] for r in cells])
        lo[0] = rates[0]
        hi[0] = rates[0]
        ax.errorbar(
            [0, 1, 3],
            rates,
            yerr=[np.maximum(0, rates - lo), np.maximum(0, hi - rates)],
            fmt="o-",
            capsize=4,
        )
        cold = next(r for r in summary if r["target"] == name and r["method"] == "cold")
        ax.axhline(cold["rate"], color="gray", linestyle="--", label="Холодный старт")
        ax.set_ylim(-0.03, 1.08)
        ax.set_xticks([0, 1, 3])
        ax.set_title(
            {
                "exp_minus_one": "exp(x)−1",
                "log": "ln(x)",
                "exp_e_minus_x": "exp(e−x)",
                "e_minus_exp": "e−exp(x)",
            }[name]
        )
        ax.set_xlabel("Число повреждённых входов k")
        ax.set_ylabel("Доля успеха")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "basin.png", dpi=200)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

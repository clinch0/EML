"""Training of the master tree: soft phase, hardening, snapping, multistart.

Phases of one run (:func:`train_once`):

1. **Soft training** for ``soft_fraction * epochs`` epochs at temperature ``tau0``.
2. **Hardening** for the remaining epochs: ``tau`` decays geometrically from ``tau0`` to
   ``tau_min``; optionally an entropy penalty on the input weights is ramped linearly from 0 to
   ``entropy_weight``.
3. **Snapping**: every input is rounded to its ``argmax`` choice, giving an exact EML tree that is
   evaluated on the data. The run succeeds if ``max |tree(x) - y| < tol``.

Loss: ``mean |pred - y|**2`` with real ``y``; optimiser: Adam. If the loss becomes non-finite the
run stops early (it is reported, never silently patched) and the current logits are snapped.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch

from eml_sr.core.tree import Node
from eml_sr.model.master_tree import MasterTree


@dataclass(frozen=True)
class TrainConfig:
    """Hyper-parameters of one training run.

    Attributes:
        epochs: Total number of optimisation steps (soft + hardening).
        lr: Adam learning rate.
        tau0: Initial softmax temperature.
        tau_min: Final softmax temperature.
        soft_fraction: Fraction of ``epochs`` spent in the soft phase.
        entropy_weight: Final weight of the entropy penalty (0 disables it).
        sigma: Standard deviation of the initial logits.
        tol: Success threshold on the max absolute error of the snapped tree.
    """

    epochs: int = 2000
    lr: float = 0.05
    tau0: float = 1.0
    tau_min: float = 0.01
    soft_fraction: float = 0.5
    entropy_weight: float = 0.0
    sigma: float = 1.0
    tol: float = 1e-8

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not 0.0 < self.tau_min <= self.tau0:
            raise ValueError("need 0 < tau_min <= tau0")
        if not 0.0 <= self.soft_fraction <= 1.0:
            raise ValueError("soft_fraction must be in [0, 1]")

    def tau_at(self, epoch: int) -> float:
        """Return the temperature used at ``epoch``."""
        soft = int(round(self.soft_fraction * self.epochs))
        if epoch < soft:
            return self.tau0
        hard = self.epochs - soft
        t = (epoch - soft) / max(1, hard - 1)
        return self.tau0 * (self.tau_min / self.tau0) ** min(1.0, t)

    def entropy_at(self, epoch: int) -> float:
        """Return the entropy-penalty weight used at ``epoch``."""
        soft = int(round(self.soft_fraction * self.epochs))
        if self.entropy_weight == 0.0 or epoch < soft:
            return 0.0
        t = (epoch - soft) / max(1, self.epochs - soft - 1)
        return self.entropy_weight * min(1.0, t)


@dataclass
class RunResult:
    """Outcome of a single training run.

    Attributes:
        seed: Seed of the run.
        success: Whether the snapped tree reproduces the data within ``tol``.
        loss_before_snap: Soft-model loss after the last epoch.
        loss_after_snap: Loss ``mean |tree(x) - y|**2`` of the snapped tree.
        epochs: Number of epochs actually performed.
        seconds: Wall-clock duration.
        max_error: ``max |tree(x) - y|`` of the snapped tree.
        tree: The snapped tree.
    """

    seed: int
    success: bool
    loss_before_snap: float
    loss_after_snap: float
    epochs: int
    seconds: float
    max_error: float = math.inf
    tree: Node | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, object]:
        """Return the fields as a JSON-friendly dict (tree as a string)."""
        return {
            "seed": self.seed,
            "success": self.success,
            "loss_before_snap": self.loss_before_snap,
            "loss_after_snap": self.loss_after_snap,
            "epochs": self.epochs,
            "seconds": self.seconds,
            "max_error": self.max_error,
            "tree": None if self.tree is None else self.tree.to_str(),
        }


@dataclass
class MultiStartResult:
    """Outcome of :func:`fit_multistart`.

    Attributes:
        best: The first successful run, or the run with the smallest max error.
        runs: All runs in the order they were performed.
    """

    best: RunResult
    runs: list[RunResult]

    @property
    def success(self) -> bool:
        """Whether some run succeeded."""
        return self.best.success


def _to_tensor(a: npt.ArrayLike) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float64).reshape(-1))


def _entropy(weights: list[torch.Tensor]) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64)
    for w in weights:
        total = total - (w * torch.log(w.clamp_min(1e-300))).sum(dim=-1).mean()
    return total / len(weights)


def snap_errors(tree: Node, x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[float, float]:
    """Return ``(mse, max_abs_error)`` of ``tree`` on the data (non-finite -> ``inf``)."""
    pred = tree.evaluate(np.asarray(x, dtype=np.float64).reshape(-1))
    err = np.abs(pred - np.asarray(y, dtype=np.float64).reshape(-1))
    if not np.all(np.isfinite(err)):
        return math.inf, math.inf
    return float(np.mean(err**2)), float(np.max(err))


def train_once(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    depth: int,
    seed: int,
    config: TrainConfig | None = None,
) -> RunResult:
    """Train one master tree from the seed ``seed`` and snap it.

    Args:
        x: Inputs, shape ``(n,)``.
        y: Real targets, shape ``(n,)``.
        depth: Master-tree depth.
        seed: Seed for the logit initialisation.
        config: Hyper-parameters (defaults to :class:`TrainConfig`).

    Returns:
        The :class:`RunResult`.
    """
    cfg = config or TrainConfig()
    start = time.perf_counter()
    xt, yt = _to_tensor(x), _to_tensor(y)
    if xt.shape != yt.shape:
        raise ValueError("x and y must have the same number of samples")
    model = MasterTree(depth, sigma=cfg.sigma, seed=seed, tau=cfg.tau0)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_value = math.inf
    epochs_done = 0
    for epoch in range(cfg.epochs):
        model.tau = cfg.tau_at(epoch)
        optimizer.zero_grad()
        pred = model(xt)
        loss = torch.mean(torch.abs(pred - yt) ** 2)
        objective = loss
        lam = cfg.entropy_at(epoch)
        if lam > 0.0:
            objective = loss + lam * _entropy(model.weights())
        loss_value = float(loss.detach())
        if not math.isfinite(float(objective.detach())):
            break
        objective.backward()
        optimizer.step()
        epochs_done = epoch + 1
    with torch.no_grad():
        final = torch.mean(torch.abs(model(xt) - yt) ** 2)
    if math.isfinite(float(final)):
        loss_value = float(final)
    tree = model.to_discrete()
    mse, max_err = snap_errors(tree, x, y)
    return RunResult(
        seed=seed,
        success=max_err < cfg.tol,
        loss_before_snap=loss_value,
        loss_after_snap=mse,
        epochs=epochs_done,
        seconds=time.perf_counter() - start,
        max_error=max_err,
        tree=tree,
    )


def fit_multistart(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    depth: int,
    n_restarts: int = 50,
    seed: int = 0,
    config: TrainConfig | None = None,
) -> MultiStartResult:
    """Run :func:`train_once` with seeds ``seed, seed+1, ...`` until the first success.

    Args:
        x: Inputs, shape ``(n,)``.
        y: Real targets, shape ``(n,)``.
        depth: Master-tree depth.
        n_restarts: Maximum number of runs.
        seed: First seed.
        config: Hyper-parameters shared by all runs.

    Returns:
        :class:`MultiStartResult` whose ``best`` is the first successful run or, if none
        succeeded, the run with the smallest max error of the snapped tree.
    """
    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")
    runs: list[RunResult] = []
    for i in range(n_restarts):
        run = train_once(x, y, depth, seed + i, config)
        runs.append(run)
        if run.success:
            return MultiStartResult(best=run, runs=runs)
    best = min(runs, key=lambda r: (r.max_error, r.loss_after_snap))
    return MultiStartResult(best=best, runs=runs)

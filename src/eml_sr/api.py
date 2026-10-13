"""High-level estimator :class:`EMLRegressor`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import sympy as sp

from eml_sr.core.tree import Node
from eml_sr.model.trainer import RunResult, TrainConfig, fit_multistart
from eml_sr.search.enumerate import DEFAULT_MAX_LEAVES
from eml_sr.search.library import build_library
from eml_sr.symbolic.convert import convert


def _as_1d_x(X: npt.ArrayLike) -> np.ndarray:
    xa = np.asarray(X, dtype=np.float64)
    if xa.ndim == 2:
        if xa.shape[1] != 1:
            raise NotImplementedError(
                f"EMLRegressor supports exactly one input variable, but X has {xa.shape[1]} "
                "columns. Multivariate EML trees are not implemented; "
                "pass X of shape (n,) or (n, 1)."
            )
        xa = xa[:, 0]
    elif xa.ndim != 1:
        raise ValueError(f"X must have shape (n,) or (n, 1), got {xa.shape}")
    return xa


class EMLRegressor:
    """Symbolic regressor that fits an exact EML tree to one-dimensional data.

    After :meth:`fit` the following attributes are available:

    * ``tree_`` - snapped EML tree of the best run (:class:`~eml_sr.core.tree.Node`),
    * ``formula_`` - formula as a string (readable if possible, otherwise raw),
    * ``formula_expr_`` - the same formula as a sympy expression,
    * ``readable_`` - whether ``formula_`` is a verified readable form,
    * ``success_`` - whether some run reproduced the data within ``tol``,
    * ``max_error_`` - max absolute error of ``tree_`` on the training data,
    * ``runs_`` - list of :class:`~eml_sr.model.trainer.RunResult` of all runs.
    """

    def __init__(
        self,
        depth: int = 3,
        n_restarts: int = 50,
        epochs: int = 2000,
        tau0: float = 1.0,
        tau_min: float = 0.01,
        sigma: float = 1.0,
        seed: int = 0,
        lr: float = 0.05,
        entropy_weight: float = 0.0,
        tol: float = 1e-8,
        library_leaves: int = DEFAULT_MAX_LEAVES,
    ) -> None:
        """Store hyper-parameters.

        Args:
            depth: Master-tree depth.
            n_restarts: Maximum number of training runs (multistart).
            epochs: Epochs per run (half soft phase, half hardening).
            tau0: Initial softmax temperature.
            tau_min: Final softmax temperature.
            sigma: Standard deviation of the initial logits.
            seed: Seed of the first run; run ``i`` uses ``seed + i``.
            lr: Adam learning rate.
            entropy_weight: Final weight of the entropy penalty in the hardening phase.
            tol: Success threshold on the max absolute error of the snapped tree.
            library_leaves: Enumeration limit of the library used to recognise subtrees.
        """
        self.depth = depth
        self.n_restarts = n_restarts
        self.epochs = epochs
        self.tau0 = tau0
        self.tau_min = tau_min
        self.sigma = sigma
        self.seed = seed
        self.lr = lr
        self.entropy_weight = entropy_weight
        self.tol = tol
        self.library_leaves = library_leaves

    def _config(self) -> TrainConfig:
        return TrainConfig(
            epochs=self.epochs,
            lr=self.lr,
            tau0=self.tau0,
            tau_min=self.tau_min,
            entropy_weight=self.entropy_weight,
            sigma=self.sigma,
            tol=self.tol,
        )

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> EMLRegressor:
        """Fit an EML tree to the data.

        Args:
            X: Inputs of shape ``(n,)`` or ``(n, 1)``; the supported domain is ``x > 0``.
            y: Real targets of shape ``(n,)``.

        Returns:
            ``self``.

        Raises:
            NotImplementedError: If ``X`` has more than one column.
            ValueError: If the shapes do not match or the data is not finite.
        """
        x = _as_1d_x(X)
        ya = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.shape[0] != ya.shape[0]:
            raise ValueError(f"X has {x.shape[0]} samples but y has {ya.shape[0]}")
        if x.shape[0] == 0:
            raise ValueError("Empty data")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(ya))):
            raise ValueError("X and y must be finite")
        result = fit_multistart(
            x, ya, self.depth, n_restarts=self.n_restarts, seed=self.seed, config=self._config()
        )
        best = result.best
        assert best.tree is not None
        conversion = convert(best.tree, build_library(self.library_leaves), points=x)
        self.tree_: Node = best.tree
        self.formula_expr_: sp.Expr = conversion.expr
        self.formula_: str = conversion.text
        self.readable_: bool = conversion.readable
        self.success_: bool = result.success
        self.max_error_: float = best.max_error
        self.runs_: list[RunResult] = result.runs
        return self

    def _check_fitted(self) -> Node:
        tree = getattr(self, "tree_", None)
        if tree is None:
            raise RuntimeError("EMLRegressor is not fitted yet; call fit(X, y) first")
        return tree

    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        """Return the real part of the fitted tree evaluated at ``X``.

        Args:
            X: Inputs of shape ``(n,)`` or ``(n, 1)``.

        Returns:
            float64 array of shape ``(n,)``.
        """
        tree = self._check_fitted()
        return tree.evaluate(_as_1d_x(X)).real

    def score(self, X: npt.ArrayLike, y: npt.ArrayLike) -> float:
        """Return the coefficient of determination R^2 of :meth:`predict` on ``(X, y)``."""
        ya = np.asarray(y, dtype=np.float64).reshape(-1)
        pred = self.predict(X)
        ss_res = float(np.sum((ya - pred) ** 2))
        ss_tot = float(np.sum((ya - ya.mean()) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot

"""Recover exp(x) from samples with a depth-2 master tree and save the figures."""

from pathlib import Path

import numpy as np

from eml_sr import EMLRegressor
from eml_sr.viz.plot import plot_fit, plot_tree

OUT = Path(__file__).resolve().parent / "output"


def main() -> None:
    """Fit, print the result and write ``exp_tree.png`` and ``exp_fit.png``."""
    x = np.linspace(0.5, 2.5, 40)
    y = np.exp(x)
    model = EMLRegressor(depth=2, n_restarts=50, seed=0).fit(x, y)
    print(f"formula: {model.formula_} (readable={model.readable_})")
    print(f"tree: {model.tree_}  success={model.success_}  max_error={model.max_error_:.2e}")
    print(f"runs used: {len(model.runs_)}  R^2={model.score(x, y):.12f}")
    OUT.mkdir(exist_ok=True)
    plot_tree(model.tree_, path=OUT / "exp_tree.png")
    plot_fit(x, y, model.tree_, label=model.formula_, path=OUT / "exp_fit.png")
    print(f"saved figures to {OUT}")


if __name__ == "__main__":
    main()

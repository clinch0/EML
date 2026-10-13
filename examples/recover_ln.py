"""Recover ln(x) from samples with a depth-3 master tree and save the figures.

Depth 3 is the smallest master depth that can represent ln(x) = eml(1, eml(eml(1, x), 1)).
"""

from pathlib import Path

import numpy as np

from eml_sr import EMLRegressor
from eml_sr.viz.plot import plot_fit, plot_tree

OUT = Path(__file__).resolve().parent / "output"


def main() -> None:
    """Fit, print the result and write ``ln_tree.png`` and ``ln_fit.png``."""
    x = np.linspace(0.5, 3.0, 40)
    y = np.log(x)
    model = EMLRegressor(depth=3, n_restarts=50, seed=0).fit(x, y)
    print(f"formula: {model.formula_} (readable={model.readable_})")
    print(f"tree: {model.tree_}  success={model.success_}  max_error={model.max_error_:.2e}")
    print(f"runs used: {len(model.runs_)}  R^2={model.score(x, y):.12f}")
    OUT.mkdir(exist_ok=True)
    plot_tree(model.tree_, path=OUT / "ln_tree.png")
    plot_fit(x, y, model.tree_, label=model.formula_, path=OUT / "ln_fit.png")
    print(f"saved figures to {OUT}")


if __name__ == "__main__":
    main()

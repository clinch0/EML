"""The smallest complete EML symbolic-regression example."""

import numpy as np

from eml_sr import EMLRegressor


def main() -> None:
    """Fit the smallest useful master tree and print the recovered expression."""
    x = np.linspace(0.5, 2.0, 40)
    y = np.exp(x)
    model = EMLRegressor(depth=2, n_restarts=20, epochs=1000, seed=0).fit(x, y)
    print(f"tree: {model.tree_}")
    print(f"formula: {model.formula_}")
    print(f"success: {model.success_}, max error: {model.max_error_:.2e}")


if __name__ == "__main__":
    main()

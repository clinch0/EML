"""Command-line interface: ``eml-sr fit data.csv --depth 3 --restarts 50 --out result/``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from eml_sr.api import EMLRegressor
from eml_sr.viz.plot import plot_fit, plot_tree


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column CSV (``x, y``); a non-numeric first row is treated as a header.

    Raises:
        NotImplementedError: If the file has more than two columns (several input variables).
        ValueError: If the file has fewer than two columns or no data rows.
    """
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")
    try:
        [float(v) for v in lines[0].split(",")]
    except ValueError:
        lines = lines[1:]
    if not lines:
        raise ValueError(f"{path} has no data rows")
    data = np.array([[float(v) for v in ln.split(",")] for ln in lines], dtype=np.float64)
    if data.shape[1] > 2:
        raise NotImplementedError(
            f"{path} has {data.shape[1]} columns; only one input column x and one target column y "
            "are supported."
        )
    if data.shape[1] < 2:
        raise ValueError(f"{path} must have two columns: x, y")
    return data[:, 0], data[:, 1]


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    parser = argparse.ArgumentParser(prog="eml-sr", description="Symbolic regression on EML trees.")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="fit an EML tree to a CSV file with columns x, y")
    fit.add_argument("data", type=Path, help="CSV file with columns x, y (optional header)")
    fit.add_argument("--depth", type=int, default=3)
    fit.add_argument("--restarts", type=int, default=50)
    fit.add_argument("--epochs", type=int, default=2000)
    fit.add_argument("--lr", type=float, default=0.05)
    fit.add_argument("--tau0", type=float, default=1.0)
    fit.add_argument("--tau-min", type=float, default=0.01)
    fit.add_argument("--sigma", type=float, default=1.0)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--out", type=Path, default=Path("result"))
    return parser


def run_fit(args: argparse.Namespace) -> int:
    """Execute the ``fit`` command.

    Returns:
        0 if the snapped tree reproduces the data within tolerance, 1 otherwise, 2 on input errors.
    """
    try:
        x, y = load_csv(args.data)
    except (NotImplementedError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    model = EMLRegressor(
        depth=args.depth,
        n_restarts=args.restarts,
        epochs=args.epochs,
        tau0=args.tau0,
        tau_min=args.tau_min,
        sigma=args.sigma,
        seed=args.seed,
        lr=args.lr,
    )
    model.fit(x, y)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    plot_tree(model.tree_, path=out / "tree.png")
    plot_fit(x, y, model.tree_, label=model.formula_, path=out / "fit.png")
    summary = {
        "formula": model.formula_,
        "readable": model.readable_,
        "success": model.success_,
        "max_error": model.max_error_,
        "tree": model.tree_.to_str(),
        "rpn": " ".join(model.tree_.to_rpn()),
        "r2": model.score(x, y),
        "runs": [r.as_dict() for r in model.runs_],
    }
    (out / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"formula: {model.formula_}")
    print(f"readable: {model.readable_}")
    print(f"success: {model.success_} (max error {model.max_error_:.3e}, runs {len(model.runs_)})")
    print(f"tree: {model.tree_.to_str()}")
    print(f"saved: {out / 'tree.png'}, {out / 'fit.png'}, {out / 'result.json'}")
    return 0 if model.success_ else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``eml-sr`` console script."""
    args = build_parser().parse_args(argv)
    if args.command == "fit":
        return run_fit(args)
    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())

"""Matplotlib drawings of EML trees and of data versus the found formula.

Only the object-oriented matplotlib API (:class:`matplotlib.figure.Figure`) is used, so no global
pyplot state or GUI backend is touched and graphviz is not needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from eml_sr.core.tree import Eml, Leaf, Node

NODE_COLORS = {"eml": "#4C72B0", "one": "#DD8452", "x": "#55A868"}


def _layout(tree: Node) -> tuple[list[tuple[float, float, str, str]], list[tuple[int, int]]]:
    """Return node positions ``(x, y, label, kind)`` and edges ``(parent, child)`` by index."""
    nodes: list[tuple[float, float, str, str]] = []
    edges: list[tuple[int, int]] = []
    next_leaf = [0.0]

    def visit(node: Node, level: int) -> int:
        if isinstance(node, Leaf):
            idx = len(nodes)
            nodes.append((next_leaf[0], -float(level), node.to_str(), node.kind))
            next_leaf[0] += 1.0
            return idx
        assert isinstance(node, Eml)
        left = visit(node.left, level + 1)
        right = visit(node.right, level + 1)
        idx = len(nodes)
        nodes.append(((nodes[left][0] + nodes[right][0]) / 2.0, -float(level), "eml", "eml"))
        edges.extend([(idx, left), (idx, right)])
        return idx

    visit(tree, 0)
    return nodes, edges


def plot_tree(tree: Node, path: str | Path | None = None, title: str | None = None) -> Figure:
    """Draw ``tree`` (eml nodes and leaves ``1`` / ``x``).

    Args:
        tree: Tree to draw.
        path: If given, the figure is saved there (format from the suffix, e.g. ``.png``).
        title: Optional title (defaults to the nested notation of the tree).

    Returns:
        The matplotlib figure.
    """
    nodes, edges = _layout(tree)
    n_leaves = sum(1 for n in nodes if n[3] != "eml")
    width = max(4.0, 1.1 * n_leaves + 1.0)
    height = max(3.0, 1.1 * (tree.depth() + 1) + 1.0)
    fig = Figure(figsize=(width, height))
    ax = fig.add_subplot()
    for parent, child in edges:
        ax.plot(
            [nodes[parent][0], nodes[child][0]],
            [nodes[parent][1], nodes[child][1]],
            color="0.4",
            zorder=1,
        )
    for nx, ny, label, kind in nodes:
        ax.scatter([nx], [ny], s=900 if kind == "eml" else 600, color=NODE_COLORS[kind], zorder=2)
        ax.text(nx, ny, label, ha="center", va="center", color="white", fontsize=10, zorder=3)
    ax.set_title(title if title is not None else tree.to_str(), fontsize=9)
    ax.set_xlim(-0.8, max(n[0] for n in nodes) + 0.8)
    ax.set_ylim(-tree.depth() - 0.6, 0.6)
    ax.axis("off")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig


def plot_fit(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    tree: Node,
    label: str | None = None,
    path: str | Path | None = None,
    n_grid: int = 300,
) -> Figure:
    """Plot the data points and the real part of ``tree`` on a dense grid.

    Args:
        x: Data inputs.
        y: Data targets.
        tree: Found tree.
        label: Legend label of the curve (defaults to the nested notation of the tree).
        path: If given, the figure is saved there.
        n_grid: Number of grid points for the curve.

    Returns:
        The matplotlib figure.
    """
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    grid = np.linspace(xa.min(), xa.max(), n_grid)
    curve = tree.evaluate(grid).real
    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot()
    ax.scatter(xa, ya, s=18, color="black", label="data", zorder=2)
    ax.plot(grid, curve, color="#C44E52", label=label or tree.to_str(), zorder=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig

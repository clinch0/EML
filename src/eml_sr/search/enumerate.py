"""Bottom-up enumeration of EML trees with fingerprint de-duplication.

``trees(1) = {1, x}`` and ``trees(n) = {eml(a, b) : a in trees(i), b in trees(n - i)}``. Every tree
is evaluated (with plain, unguarded ``exp``/``log``) at the fingerprint points; trees producing
``nan``/``inf`` are dropped, and among trees with the same rounded fingerprint only the first one
(i.e. one with the fewest leaves) is kept. Children are drawn only from the de-duplicated sets.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from eml_sr.core.tree import ONE, Eml, Node, X

FINGERPRINT_POINTS: tuple[float, ...] = (0.5, 1.3, 2.1, 3.7, 4.4)
FINGERPRINT_DECIMALS: int = 9
DEFAULT_MAX_LEAVES: int = 8
MATCH_RTOL: float = 1e-9
MATCH_ATOL: float = 1e-9


def fingerprint_values(
    tree: Node, points: tuple[float, ...] = FINGERPRINT_POINTS, safe: bool = False
) -> np.ndarray:
    """Evaluate ``tree`` at the fingerprint points.

    Args:
        tree: Tree to evaluate.
        points: Evaluation points (``x > 0``).
        safe: Use guarded operations (default: plain ``exp``/``log``).

    Returns:
        complex128 array of shape ``(len(points),)``.
    """
    return tree.evaluate(np.asarray(points, dtype=np.float64), safe=safe)


def fingerprint_key(values: np.ndarray, decimals: int = FINGERPRINT_DECIMALS) -> bytes | None:
    """Return a hashable fingerprint of ``values`` or ``None`` if any value is not finite.

    Real and imaginary parts are rounded to ``decimals`` digits after the decimal point.
    """
    keys = _round_keys(np.atleast_2d(values), decimals)
    return None if keys is None else keys[0]


def _round_keys(values: np.ndarray, decimals: int) -> list[bytes] | None:
    if not np.all(np.isfinite(values)):
        return None
    parts = np.concatenate([values.real, values.imag], axis=1)
    rounded = np.round(parts, decimals) + 0.0
    return [row.tobytes() for row in rounded]


@dataclass
class EnumerationResult:
    """Unique trees found by :func:`enumerate_trees`.

    Attributes:
        trees: Unique trees ordered by number of leaves (shortest first).
        n_leaves: Number of leaves of each tree in ``trees``.
        values: Fingerprint values, shape ``(len(trees), len(points))``.
        points: Fingerprint points.
        max_leaves: Largest number of leaves that was enumerated.
    """

    trees: list[Node]
    n_leaves: list[int]
    values: np.ndarray
    points: tuple[float, ...]
    max_leaves: int
    _index: dict[bytes, int] = field(default_factory=dict, repr=False)

    def by_leaves(self, n: int) -> list[Node]:
        """Return the unique trees with exactly ``n`` leaves."""
        return [t for t, k in zip(self.trees, self.n_leaves, strict=True) if k == n]

    def find_values(self, target: np.ndarray) -> Node | None:
        """Return the shortest tree whose fingerprint matches ``target`` within tolerance.

        Args:
            target: Values at :attr:`points` (complex or real).

        Returns:
            The matching tree or ``None`` ("not found") if no enumerated tree matches.
        """
        tgt = np.asarray(target, dtype=np.complex128)
        if not np.all(np.isfinite(tgt)) or len(self.trees) == 0:
            return None
        close = np.isclose(self.values, tgt[None, :], rtol=MATCH_RTOL, atol=MATCH_ATOL)
        hits = np.flatnonzero(np.all(close, axis=1))
        return self.trees[int(hits[0])] if hits.size else None

    def find_function(self, fn: Callable[[np.ndarray], np.ndarray]) -> Node | None:
        """Return the shortest tree equal to ``fn`` on the fingerprint points, or ``None``."""
        pts = np.asarray(self.points, dtype=np.complex128)
        with np.errstate(all="ignore"):
            return self.find_values(np.asarray(fn(pts), dtype=np.complex128))


def enumerate_trees(
    max_leaves: int = DEFAULT_MAX_LEAVES,
    points: tuple[float, ...] = FINGERPRINT_POINTS,
    decimals: int = FINGERPRINT_DECIMALS,
) -> EnumerationResult:
    """Enumerate EML trees with up to ``max_leaves`` leaves, keeping one tree per fingerprint.

    Args:
        max_leaves: Maximum number of leaves (default 8).
        points: Fingerprint points (``x > 0``).
        decimals: Rounding of the fingerprint (9 -> ``1e-9``).

    Returns:
        :class:`EnumerationResult` with the unique trees, shortest first.
    """
    if max_leaves < 1:
        raise ValueError("max_leaves must be >= 1")
    pts = np.asarray(points, dtype=np.complex128)
    index: dict[bytes, int] = {}
    all_trees: list[Node] = []
    all_leaves: list[int] = []
    all_values: list[np.ndarray] = []
    level_trees: dict[int, list[Node]] = {}
    level_values: dict[int, np.ndarray] = {}

    def add_level(n: int, candidates: list[Node], values: np.ndarray) -> None:
        kept_trees: list[Node] = []
        kept_rows: list[int] = []
        finite = np.all(np.isfinite(values), axis=1)
        rows = np.flatnonzero(finite)
        keys = _round_keys(values[rows], decimals) if rows.size else []
        for row, key in zip(rows, keys or [], strict=True):
            if key in index:
                continue
            index[key] = len(all_trees)
            all_trees.append(candidates[row])
            all_leaves.append(n)
            kept_trees.append(candidates[row])
            kept_rows.append(int(row))
        level_trees[n] = kept_trees
        level_values[n] = values[kept_rows] if kept_rows else np.empty((0, len(pts)), complex)
        all_values.append(level_values[n])

    add_level(1, [ONE, X], np.stack([np.ones_like(pts), pts]))
    with np.errstate(all="ignore"):
        for n in range(2, max_leaves + 1):
            cand_trees: list[Node] = []
            cand_values: list[np.ndarray] = []
            for i in range(1, n):
                lefts, rights = level_trees[i], level_trees[n - i]
                if not lefts or not rights:
                    continue
                exp_a = np.exp(level_values[i])
                log_b = np.log(level_values[n - i])
                vals = (exp_a[:, None, :] - log_b[None, :, :]).reshape(-1, len(pts))
                cand_values.append(vals)
                cand_trees.extend(Eml(a, b) for a in lefts for b in rights)
            if cand_trees:
                add_level(n, cand_trees, np.concatenate(cand_values))
            else:
                add_level(n, [], np.empty((0, len(pts)), complex))

    values = np.concatenate(all_values) if all_values else np.empty((0, len(pts)), complex)
    return EnumerationResult(
        trees=all_trees,
        n_leaves=all_leaves,
        values=values,
        points=tuple(points),
        max_leaves=max_leaves,
        _index=index,
    )

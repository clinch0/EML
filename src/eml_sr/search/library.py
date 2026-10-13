"""Library of known functions and their shortest EML trees.

Each :class:`KnownFunction` has a readable name, a NumPy implementation (complex128 in, complex128
out) and a sympy expression. :func:`build_library` matches every known function against the
enumerated trees by fingerprint; functions without a match within ``max_leaves`` leaves are
reported as not found (``tree is None``) instead of being approximated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sympy as sp

from eml_sr.core.tree import Node
from eml_sr.search.enumerate import (
    DEFAULT_MAX_LEAVES,
    FINGERPRINT_POINTS,
    MATCH_ATOL,
    MATCH_RTOL,
    EnumerationResult,
    enumerate_trees,
)

X_SYMBOL = sp.Symbol("x", positive=True)


@dataclass(frozen=True)
class KnownFunction:
    """A named function with NumPy and sympy implementations."""

    name: str
    numpy_fn: Callable[[np.ndarray], np.ndarray]
    sympy_fn: Callable[[sp.Symbol], sp.Expr]

    def values(self, points: tuple[float, ...] = FINGERPRINT_POINTS) -> np.ndarray:
        """Evaluate the function at ``points`` in complex128."""
        pts = np.asarray(points, dtype=np.complex128)
        with np.errstate(all="ignore"):
            return np.asarray(self.numpy_fn(pts), dtype=np.complex128) * np.ones_like(pts)

    def expr(self, x: sp.Symbol = X_SYMBOL) -> sp.Expr:
        """Return the sympy expression in terms of ``x``."""
        return self.sympy_fn(x)


def _const(value: complex) -> Callable[[np.ndarray], np.ndarray]:
    return lambda z: np.full_like(z, value)


KNOWN_FUNCTIONS: tuple[KnownFunction, ...] = (
    KnownFunction("x", lambda z: z, lambda x: x),
    KnownFunction("e", _const(np.e), lambda x: sp.E),
    KnownFunction("0", _const(0.0), lambda x: sp.Integer(0)),
    KnownFunction("-1", _const(-1.0), lambda x: sp.Integer(-1)),
    KnownFunction("2", _const(2.0), lambda x: sp.Integer(2)),
    KnownFunction("exp(x)", np.exp, sp.exp),
    KnownFunction("ln(x)", np.log, sp.log),
    KnownFunction("x - 1", lambda z: z - 1, lambda x: x - 1),
    KnownFunction("x + 1", lambda z: z + 1, lambda x: x + 1),
    KnownFunction("2*x", lambda z: 2 * z, lambda x: 2 * x),
    KnownFunction("x^2", lambda z: z**2, lambda x: x**2),
    KnownFunction("-x", lambda z: -z, lambda x: -x),
    KnownFunction("1/x", lambda z: 1 / z, lambda x: 1 / x),
    KnownFunction("sqrt(x)", np.sqrt, sp.sqrt),
    KnownFunction("exp(x) - 1", lambda z: np.exp(z) - 1, lambda x: sp.exp(x) - 1),
    KnownFunction("exp(exp(x))", lambda z: np.exp(np.exp(z)), lambda x: sp.exp(sp.exp(x))),
    KnownFunction("e - ln(x)", lambda z: np.e - np.log(z), lambda x: sp.E - sp.log(x)),
    KnownFunction(
        "exp(x) - ln(x)", lambda z: np.exp(z) - np.log(z), lambda x: sp.exp(x) - sp.log(x)
    ),
)


@dataclass(frozen=True)
class LibraryEntry:
    """Result of matching one known function: its shortest tree or ``None`` if not found."""

    function: KnownFunction
    tree: Node | None

    @property
    def found(self) -> bool:
        """Whether a tree was found within the enumeration limit."""
        return self.tree is not None


class Library:
    """Known functions matched against an enumeration, with fingerprint lookup."""

    def __init__(
        self,
        enumeration: EnumerationResult,
        functions: tuple[KnownFunction, ...] = KNOWN_FUNCTIONS,
    ) -> None:
        """Match ``functions`` against ``enumeration``.

        Args:
            enumeration: Result of :func:`enumerate_trees`.
            functions: Known functions to match.
        """
        self.enumeration = enumeration
        self.functions = functions
        self.points = enumeration.points
        self._values = np.stack([f.values(self.points) for f in functions])
        self.entries: dict[str, LibraryEntry] = {
            f.name: LibraryEntry(f, enumeration.find_values(v))
            for f, v in zip(functions, self._values, strict=True)
        }

    def tree_for(self, name: str) -> Node | None:
        """Return the shortest tree of the known function ``name`` or ``None`` (not found).

        Raises:
            KeyError: If ``name`` is not a known function.
        """
        return self.entries[name].tree

    def match_values(self, values: np.ndarray) -> KnownFunction | None:
        """Return the first known function whose fingerprint matches ``values``, else ``None``.

        Args:
            values: Values at :attr:`points`.
        """
        vals = np.asarray(values, dtype=np.complex128)
        if vals.shape != (len(self.points),) or not np.all(np.isfinite(vals)):
            return None
        with np.errstate(all="ignore"):
            close = np.isclose(self._values, vals[None, :], rtol=MATCH_RTOL, atol=MATCH_ATOL)
        hits = np.flatnonzero(np.all(close, axis=1))
        return self.functions[int(hits[0])] if hits.size else None

    def match_tree(self, tree: Node) -> KnownFunction | None:
        """Return the known function equal to ``tree`` on the fingerprint points, else ``None``."""
        values = tree.evaluate(np.asarray(self.points, dtype=np.float64), safe=False)
        return self.match_values(values)


def build_library(
    max_leaves: int = DEFAULT_MAX_LEAVES,
    functions: tuple[KnownFunction, ...] = KNOWN_FUNCTIONS,
) -> Library:
    """Enumerate trees up to ``max_leaves`` leaves and match the known functions.

    Args:
        max_leaves: Enumeration limit (default 8).
        functions: Known functions to match.

    Returns:
        The :class:`Library`.
    """
    return Library(enumerate_trees(max_leaves), functions)

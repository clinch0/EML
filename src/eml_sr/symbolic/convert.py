"""Conversion of EML trees into readable sympy formulas.

Steps of :func:`convert`:

1. Raw translation: ``eml(a, b) -> exp(a) - log(b)``, leaves ``1`` and ``x`` (``x`` is a positive
   real sympy symbol).
2. Recognition (top-down): every subtree whose fingerprint matches a known function from the
   :class:`~eml_sr.search.library.Library` is replaced by that function's readable expression.
3. ``sympy.simplify``.
4. Verification: the resulting expression is evaluated numerically and compared with the tree on
   the verification points (fingerprint points by default). If it disagrees (or the tree is not
   finite there), the raw expression is returned with ``readable=False``.

``readable`` is true only if verification passed and either the whole tree was recognised or the
simplified expression has at most ``max_ops`` operations (``sympy.count_ops``). No formula is ever
guessed: the returned expression is always an exact rewrite of the tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import sympy as sp

from eml_sr.core.tree import Eml, Leaf, Node
from eml_sr.search.library import X_SYMBOL, Library, build_library

DEFAULT_MAX_OPS: int = 12
VERIFY_RTOL: float = 1e-8
VERIFY_ATOL: float = 1e-10


@dataclass
class ConversionResult:
    """Result of :func:`convert`.

    Attributes:
        expr: Readable expression if ``readable`` else the raw expression.
        raw: Raw translation of the tree.
        readable: Whether a readable, verified form was obtained.
        recognized: Names of the library functions substituted for subtrees.
    """

    expr: sp.Expr
    raw: sp.Expr
    readable: bool
    recognized: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Return ``str(expr)``."""
        return str(self.expr)


def tree_to_sympy(tree: Node, x: sp.Symbol = X_SYMBOL) -> sp.Expr:
    """Translate ``tree`` literally: ``eml(a, b) -> exp(a) - log(b)``."""
    if isinstance(tree, Leaf):
        return sp.Integer(1) if tree.kind == "one" else x
    if isinstance(tree, Eml):
        return sp.exp(tree_to_sympy(tree.left, x)) - sp.log(tree_to_sympy(tree.right, x))
    raise TypeError(f"Unsupported node type {type(tree).__name__}")


def _recognize(
    tree: Node, library: Library, x: sp.Symbol, names: list[str]
) -> tuple[sp.Expr, bool]:
    if isinstance(tree, Leaf):
        return tree_to_sympy(tree, x), True
    assert isinstance(tree, Eml)
    known = library.match_tree(tree)
    if known is not None:
        names.append(known.name)
        return known.expr(x), True
    left, _ = _recognize(tree.left, library, x, names)
    right, _ = _recognize(tree.right, library, x, names)
    return sp.exp(left) - sp.log(right), False


def _matches_tree(expr: sp.Expr, tree: Node, x: sp.Symbol, points: np.ndarray) -> bool:
    expected = tree.evaluate(points, safe=False)
    if not np.all(np.isfinite(expected)):
        return False
    try:
        fn = sp.lambdify(x, expr, modules="numpy")
        with np.errstate(all="ignore"):
            got = np.asarray(fn(points.astype(np.complex128)), dtype=np.complex128)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False
    got = got * np.ones_like(expected)
    if not np.all(np.isfinite(got)):
        return False
    return bool(np.allclose(got, expected, rtol=VERIFY_RTOL, atol=VERIFY_ATOL))


def convert(
    tree: Node,
    library: Library | None = None,
    max_ops: int = DEFAULT_MAX_OPS,
    x: sp.Symbol = X_SYMBOL,
    points: npt.ArrayLike | None = None,
) -> ConversionResult:
    """Convert ``tree`` into a (possibly readable) sympy expression.

    Args:
        tree: Tree to convert.
        library: Library used to recognise subtrees (built with default settings if ``None``).
        max_ops: Maximum ``sympy.count_ops`` of a simplified expression considered readable.
        x: Symbol used for the variable.
        points: Points (``x > 0``) where the expression is verified against the tree; the tree
            must be finite at all of them. Defaults to the library fingerprint points.

    Returns:
        :class:`ConversionResult`.
    """
    lib = library if library is not None else build_library()
    raw = tree_to_sympy(tree, x)
    names: list[str] = []
    recognized_expr, whole = _recognize(tree, lib, x, names)
    simplified = sp.simplify(recognized_expr)
    check = np.asarray(lib.points if points is None else points, dtype=np.float64).reshape(-1)
    if check.size == 0 or not _matches_tree(simplified, tree, x, check):
        return ConversionResult(expr=raw, raw=raw, readable=False, recognized=[])
    readable = whole or sp.count_ops(simplified) <= max_ops
    return ConversionResult(
        expr=simplified if readable else raw, raw=raw, readable=readable, recognized=names
    )

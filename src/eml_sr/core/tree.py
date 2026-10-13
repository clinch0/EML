"""Immutable EML expression trees.

Grammar: ``S -> 1 | x | eml(S, S)`` with ``eml(a, b) = exp(a) - ln(b)``. Values are computed in
complex128 with the principal branch of the logarithm.

RPN (reverse Polish notation) uses the tokens ``"1"``, ``"x"`` and ``"E"`` (an eml node), e.g.
``ln(x) = eml(1, eml(eml(1, x), 1))`` is ``1 1 x E 1 E E``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from eml_sr.core.safe_ops import EXP_LIMIT, LOG_EPS, safe_exp, safe_log

RPN_ONE = "1"
RPN_X = "x"
RPN_EML = "E"


class Node:
    """Base class of EML tree nodes. Use :class:`Leaf` and :class:`Eml`."""

    def evaluate(
        self,
        x: npt.ArrayLike,
        safe: bool = True,
        exp_limit: float = EXP_LIMIT,
        log_eps: float = LOG_EPS,
    ) -> np.ndarray:
        """Evaluate the tree at the points ``x``.

        Args:
            x: Input values (scalar or array).
            safe: If true, use :func:`safe_exp` / :func:`safe_log`; otherwise use plain
                ``numpy.exp`` / ``numpy.log`` (may produce ``inf``/``nan``; warnings suppressed).
            exp_limit: Clamp bound passed to :func:`safe_exp`.
            log_eps: Threshold passed to :func:`safe_log`.

        Returns:
            complex128 array with the shape of ``x``.
        """
        xa = np.asarray(x, dtype=np.complex128)
        with np.errstate(all="ignore"):
            return self._eval(xa, safe, exp_limit, log_eps)

    def _eval(self, x: np.ndarray, safe: bool, exp_limit: float, log_eps: float) -> np.ndarray:
        raise NotImplementedError

    def size(self) -> int:
        """Return the total number of nodes (leaves + eml nodes), i.e. the RPN length."""
        return len(self.to_rpn())

    def n_leaves(self) -> int:
        """Return the number of leaves."""
        return sum(1 for token in self.to_rpn() if token != RPN_EML)

    def depth(self) -> int:
        """Return the depth: 0 for a leaf, ``1 + max(child depths)`` for an eml node."""
        raise NotImplementedError

    def to_rpn(self) -> list[str]:
        """Return the reverse Polish notation as a list of tokens."""
        raise NotImplementedError

    def to_str(self) -> str:
        """Return the nested functional notation, e.g. ``eml(1, x)``."""
        raise NotImplementedError

    def __str__(self) -> str:
        return self.to_str()


@dataclass(frozen=True, eq=True)
class Leaf(Node):
    """Leaf node: the constant ``1`` (``kind="one"``) or the variable ``x`` (``kind="x"``)."""

    kind: Literal["one", "x"]

    def __post_init__(self) -> None:
        if self.kind not in ("one", "x"):
            raise ValueError(f"Leaf kind must be 'one' or 'x', got {self.kind!r}")

    def _eval(self, x: np.ndarray, safe: bool, exp_limit: float, log_eps: float) -> np.ndarray:
        if self.kind == "one":
            return np.ones_like(x)
        return x.copy()

    def depth(self) -> int:
        """Return 0."""
        return 0

    def to_rpn(self) -> list[str]:
        """Return ``["1"]`` or ``["x"]``."""
        return [RPN_ONE if self.kind == "one" else RPN_X]

    def to_str(self) -> str:
        """Return ``"1"`` or ``"x"``."""
        return "1" if self.kind == "one" else "x"


@dataclass(frozen=True, eq=True)
class Eml(Node):
    """Internal node ``eml(left, right) = exp(left) - ln(right)``."""

    left: Node
    right: Node

    def _eval(self, x: np.ndarray, safe: bool, exp_limit: float, log_eps: float) -> np.ndarray:
        a = self.left._eval(x, safe, exp_limit, log_eps)
        b = self.right._eval(x, safe, exp_limit, log_eps)
        if safe:
            return safe_exp(a, exp_limit) - safe_log(b, log_eps)
        return np.exp(a) - np.log(b)

    def depth(self) -> int:
        """Return ``1 + max(left.depth(), right.depth())``."""
        return 1 + max(self.left.depth(), self.right.depth())

    def to_rpn(self) -> list[str]:
        """Return ``left_rpn + right_rpn + ["E"]``."""
        return [*self.left.to_rpn(), *self.right.to_rpn(), RPN_EML]

    def to_str(self) -> str:
        """Return ``"eml(<left>, <right>)"``."""
        return f"eml({self.left.to_str()}, {self.right.to_str()})"


ONE = Leaf("one")
X = Leaf("x")


def from_rpn(tokens: list[str] | str) -> Node:
    """Build a tree from RPN tokens.

    Args:
        tokens: List of tokens or a whitespace-separated string of ``1``, ``x`` and ``E``.

    Returns:
        The parsed tree.

    Raises:
        ValueError: If the token sequence is not a valid RPN expression.
    """
    items = tokens.split() if isinstance(tokens, str) else list(tokens)
    stack: list[Node] = []
    for token in items:
        if token == RPN_ONE:
            stack.append(ONE)
        elif token == RPN_X:
            stack.append(X)
        elif token == RPN_EML:
            if len(stack) < 2:
                raise ValueError("Invalid RPN: 'E' needs two operands")
            right = stack.pop()
            left = stack.pop()
            stack.append(Eml(left, right))
        else:
            raise ValueError(f"Invalid RPN token {token!r}")
    if len(stack) != 1:
        raise ValueError(f"Invalid RPN: {len(stack)} values left on the stack")
    return stack[0]


def to_rpn(tree: Node) -> list[str]:
    """Return the RPN tokens of ``tree`` (functional alias of :meth:`Node.to_rpn`)."""
    return tree.to_rpn()

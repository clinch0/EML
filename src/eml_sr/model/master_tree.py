"""Differentiable "master tree" of EML nodes.

A master tree of depth ``d`` is a full binary tree of ``2**d - 1`` eml nodes. Level 0 is the root,
level ``d - 1`` is the bottom level. Every node has two inputs (left, right):

* bottom-level input: ``a*1 + b*x`` (two logits),
* other inputs: ``a*1 + b*x + c*f`` (three logits), where ``f`` is the output of the child eml node
  on the same side (node ``j`` of level ``k`` has children ``2j`` and ``2j + 1`` on level ``k+1``).

The weights ``(a, b[, c])`` are ``softmax(logits / tau)``. The number of logits is
``4 * 2**(d-1) + 6 * (2**(d-1) - 1) = 5 * 2**d - 6``. All computations are complex128 and use the
guarded :func:`safe_exp` / :func:`safe_log`.
"""

from __future__ import annotations

import torch
from torch import nn

from eml_sr.core.safe_ops import EXP_LIMIT, LOG_EPS, safe_exp, safe_log
from eml_sr.core.tree import ONE, Eml, Node, X

CHOICE_ONE = 0
CHOICE_X = 1
CHOICE_CHILD = 2


def expected_logit_count(depth: int) -> int:
    """Return ``5 * 2**depth - 6``, the number of logits of a depth-``depth`` master tree."""
    return 5 * 2**depth - 6


class MasterTree(nn.Module):
    """Soft EML master tree with softmax-weighted node inputs.

    Attributes:
        depth: Number of eml levels.
        tau: Softmax temperature used by :meth:`forward` (set by the trainer).
        levels: ``levels[k]`` holds the logits of level ``k``, shape ``(2**k, 2, c)`` with
            ``c = 2`` on the bottom level and ``c = 3`` elsewhere (axis 1 = left/right input,
            axis 2 = choice ``1``, ``x``, child).
    """

    def __init__(
        self,
        depth: int,
        sigma: float = 1.0,
        seed: int = 0,
        tau: float = 1.0,
        exp_limit: float = EXP_LIMIT,
        log_eps: float = LOG_EPS,
    ) -> None:
        """Create a master tree with logits drawn from ``N(0, sigma**2)``.

        Args:
            depth: Number of eml levels (``>= 1``).
            sigma: Standard deviation of the initial logits.
            seed: Seed of the private random generator used for initialisation.
            tau: Initial softmax temperature.
            exp_limit: Clamp bound of :func:`safe_exp`.
            log_eps: Threshold of :func:`safe_log`.
        """
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.tau = tau
        self.exp_limit = exp_limit
        self.log_eps = log_eps
        gen = torch.Generator().manual_seed(seed)
        params = []
        for k in range(depth):
            n_choices = 2 if k == depth - 1 else 3
            init = torch.randn(2**k, 2, n_choices, generator=gen, dtype=torch.float64) * sigma
            params.append(nn.Parameter(init))
        self.levels = nn.ParameterList(params)

    def n_logits(self) -> int:
        """Return the total number of logits."""
        return sum(p.numel() for p in self.levels)

    def weights(self, tau: float | None = None) -> list[torch.Tensor]:
        """Return the softmax input weights of every level (same shapes as :attr:`levels`)."""
        t = self.tau if tau is None else tau
        return [torch.softmax(p / t, dim=-1) for p in self.levels]

    def forward(self, x: torch.Tensor, tau: float | None = None) -> torch.Tensor:
        """Evaluate the soft tree.

        Args:
            x: Inputs of shape ``(n,)`` (real or complex).
            tau: Temperature override (defaults to :attr:`tau`).

        Returns:
            complex128 tensor of shape ``(n,)``.
        """
        return self._propagate(x, self.weights(tau))

    def forward_discrete(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the snapped tree (one-hot weights from ``argmax`` of the logits)."""
        hard = [
            nn.functional.one_hot(p.argmax(dim=-1), p.shape[-1]).to(torch.float64)
            for p in self.levels
        ]
        return self._propagate(x, hard)

    def _propagate(self, x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
        xc = x.reshape(-1).to(torch.complex128)
        n = xc.shape[0]
        child: torch.Tensor | None = None
        for k in reversed(range(self.depth)):
            w = weights[k].to(torch.complex128)
            inputs = w[..., CHOICE_ONE] + w[..., CHOICE_X] * xc[:, None, None]
            if child is not None:
                inputs = inputs + w[..., CHOICE_CHILD] * child.reshape(n, -1, 2)
            child = safe_exp(inputs[..., 0], self.exp_limit) - safe_log(
                inputs[..., 1], self.log_eps
            )
        assert child is not None
        return child[:, 0]

    def to_discrete(self) -> Node:
        """Snap every input to its ``argmax`` choice and return the corresponding EML tree."""
        choices = [p.detach().argmax(dim=-1) for p in self.levels]

        def build(level: int, index: int) -> Node:
            sides = []
            for side in (0, 1):
                choice = int(choices[level][index, side])
                if choice == CHOICE_ONE:
                    sides.append(ONE)
                elif choice == CHOICE_X:
                    sides.append(X)
                else:
                    sides.append(build(level + 1, 2 * index + side))
            return Eml(sides[0], sides[1])

        return build(0, 0)

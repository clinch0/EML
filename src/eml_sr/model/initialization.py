"""Encode known trees and corrupt only active input choices."""

import torch

from eml_sr.core.tree import Eml, Node, X
from eml_sr.model.master_tree import MasterTree


def embed_tree(
    tree: Node, depth: int, margin: float = 4.0, mistakes: int = 0, seed: int = 0
) -> list[torch.Tensor]:
    """Encode a non-leaf tree and optionally flip exactly k active choices.

    The Hamming guarantee refers to active slots before corruption, not the
    topology after rounding. Unused slots are initialized to the constant one.
    """
    if not isinstance(tree, Eml) or tree.depth() > depth:
        raise ValueError("tree must have an EML root and fit the master depth")
    if not 0 < margin < float("inf"):
        raise ValueError("margin must be positive and finite")
    logits = [torch.zeros_like(p) for p in MasterTree(depth).parameters()]
    for level in logits:
        level[..., 0] = margin
    active = []

    def visit(node: Eml, level: int, index: int) -> None:
        for side, child in enumerate((node.left, node.right)):
            choice = 2 if isinstance(child, Eml) else int(child == X)
            logits[level][index, side].zero_()
            logits[level][index, side, choice] = margin
            active.append((level, index, side, choice))
            if isinstance(child, Eml):
                visit(child, level + 1, 2 * index + side)

    visit(tree, 0, 0)
    if not 0 <= mistakes <= len(active):
        raise ValueError("mistakes exceeds active slot count")
    gen = torch.Generator().manual_seed(seed)
    for i in torch.randperm(len(active), generator=gen)[:mistakes].tolist():
        level, index, side, correct = active[i]
        slot = logits[level][index, side]
        alternatives = [c for c in range(len(slot)) if c != correct]
        wrong = alternatives[int(torch.randint(len(alternatives), (), generator=gen))]
        slot.zero_()
        slot[wrong] = margin
    return logits

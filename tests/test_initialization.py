import numpy as np
import pytest
import torch

from eml_sr.core.tree import ONE, Eml, X
from eml_sr.model.initialization import embed_tree
from eml_sr.model.master_tree import MasterTree
from eml_sr.model.trainer import TrainConfig, train_once


def test_embedding_and_corruption():
    tree = Eml(ONE, Eml(Eml(ONE, X), ONE))
    clean = embed_tree(tree, 4)
    model = MasterTree(4)
    with torch.no_grad():
        for p, value in zip(model.parameters(), clean, strict=True):
            p.copy_(value)
    assert model.to_discrete() == tree
    x = torch.linspace(0.5, 2, 20)
    np.testing.assert_allclose(model.forward_discrete(x).detach(), np.log(x.numpy()), atol=1e-6)
    for k in range(7):
        corrupt = embed_tree(tree, 4, mistakes=k, seed=17)
        assert (
            sum(
                int((a.argmax(-1) != b.argmax(-1)).sum())
                for a, b in zip(clean, corrupt, strict=True)
            )
            == k
        )


def test_warm_start_is_copied():
    tree = Eml(X, ONE)
    initial = embed_tree(tree, 2)
    before = [p.clone() for p in initial]
    x = np.linspace(0.5, 2, 10)
    result = train_once(x, np.exp(x), 2, 0, TrainConfig(epochs=2), initial)
    assert result.epochs == 2
    assert all(torch.equal(a, b) for a, b in zip(initial, before, strict=True))
    with pytest.raises(ValueError):
        train_once(x, np.exp(x), 2, 0, initial_logits=[])

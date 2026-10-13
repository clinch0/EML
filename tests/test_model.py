import itertools

import numpy as np
import pytest
import torch

from eml_sr.core.tree import ONE, Eml, X
from eml_sr.model.master_tree import MasterTree, expected_logit_count
from eml_sr.model.trainer import RunResult, TrainConfig, fit_multistart, train_once

XS = np.linspace(0.5, 3.0, 40)


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_logit_count(depth):
    assert MasterTree(depth).n_logits() == expected_logit_count(depth) == 5 * 2**depth - 6


def test_logit_count_values():
    assert MasterTree(2).n_logits() == 14
    assert MasterTree(3).n_logits() == 34


def test_level_shapes():
    m = MasterTree(3)
    assert [tuple(p.shape) for p in m.levels] == [(1, 2, 3), (2, 2, 3), (4, 2, 2)]


def test_forward_complex128_and_shape():
    m = MasterTree(3, seed=1)
    out = m(torch.tensor(XS))
    assert out.dtype == torch.complex128
    assert out.shape == (len(XS),)
    assert torch.all(torch.isfinite(out.real)) and torch.all(torch.isfinite(out.imag))


def test_forward_has_gradients():
    m = MasterTree(2, seed=3)
    loss = torch.mean(torch.abs(m(torch.tensor(XS)) - torch.tensor(np.exp(XS))) ** 2)
    loss.backward()
    for p in m.levels:
        assert p.grad is not None and torch.all(torch.isfinite(p.grad))


def _set_one_hot(model, choices):
    with torch.no_grad():
        for p, ch in zip(model.levels, choices, strict=True):
            p.copy_(torch.nn.functional.one_hot(torch.tensor(ch), p.shape[-1]) * 20.0 - 10.0)


def test_to_discrete_ln_at_depth_3():
    m = MasterTree(3)
    # root: (1, child); level 1 right node: (child, 1); level 2 node 3: (1, x)
    _set_one_hot(
        m,
        [
            [[0, 2]],
            [[0, 0], [2, 0]],
            [[0, 0], [0, 0], [0, 1], [0, 0]],
        ],
    )
    tree = m.to_discrete()
    assert tree == Eml(ONE, Eml(Eml(ONE, X), ONE))
    hard = m.forward_discrete(torch.tensor(XS)).numpy()
    assert np.allclose(hard, np.log(XS), atol=1e-12)
    soft = m(torch.tensor(XS), tau=0.01).detach().numpy()
    assert np.allclose(soft, hard, atol=1e-9)


def test_depth2_cannot_represent_ln():
    """Exhaustive check over all 144 snapped configurations of a depth-2 master tree."""
    m = MasterTree(2)
    trees = set()
    for root in itertools.product(range(3), repeat=2):
        for bottom in itertools.product(range(2), repeat=4):
            _set_one_hot(m, [[list(root)], [list(bottom[:2]), list(bottom[2:])]])
            trees.add(m.to_discrete())
    assert len(trees) > 1
    for tree in trees:
        assert tree.depth() <= 2
        assert not np.allclose(tree.evaluate(XS), np.log(XS), atol=1e-6)


def test_tau_schedule():
    cfg = TrainConfig(epochs=100, tau0=1.0, tau_min=0.01, soft_fraction=0.5)
    assert cfg.tau_at(0) == 1.0
    assert cfg.tau_at(49) == 1.0
    assert cfg.tau_at(99) == pytest.approx(0.01)
    assert cfg.tau_at(70) < cfg.tau_at(60) < 1.0


def test_invalid_config():
    with pytest.raises(ValueError):
        TrainConfig(tau_min=2.0, tau0=1.0)
    with pytest.raises(ValueError):
        TrainConfig(epochs=0)


def test_train_once_runresult_fields():
    r = train_once(XS, np.exp(XS), depth=2, seed=0, config=TrainConfig(epochs=50))
    assert isinstance(r, RunResult)
    assert r.seed == 0 and r.epochs == 50 and r.seconds > 0
    assert isinstance(r.success, bool)
    assert np.isfinite(r.loss_before_snap)
    assert set(r.as_dict()) >= {
        "seed",
        "success",
        "loss_before_snap",
        "loss_after_snap",
        "epochs",
        "seconds",
    }


def test_train_once_deterministic():
    cfg = TrainConfig(epochs=100)
    a = train_once(XS, np.log(XS), depth=2, seed=5, config=cfg)
    b = train_once(XS, np.log(XS), depth=2, seed=5, config=cfg)
    assert a.tree == b.tree and a.loss_before_snap == b.loss_before_snap


def test_multistart_depth2_exp_success():
    res = fit_multistart(XS, np.exp(XS), depth=2, n_restarts=10, seed=0)
    assert res.success
    assert res.best.tree == Eml(X, ONE)
    assert res.best.max_error < 1e-8
    assert res.runs[-1] is res.best
    assert [r.seed for r in res.runs] == list(range(len(res.runs)))


def test_multistart_depth3_ln_success():
    res = fit_multistart(XS, np.log(XS), depth=3, n_restarts=20, seed=0)
    assert res.success
    assert res.best.max_error < 1e-8
    assert np.allclose(res.best.tree.evaluate(XS), np.log(XS), atol=1e-8)


def test_multistart_returns_best_when_all_fail():
    # ln(x) needs depth 3, so depth 1 can never succeed.
    res = fit_multistart(
        XS, np.log(XS), depth=1, n_restarts=3, seed=7, config=TrainConfig(epochs=60)
    )
    assert not res.success
    assert len(res.runs) == 3
    assert [r.seed for r in res.runs] == [7, 8, 9]
    assert res.best.max_error == min(r.max_error for r in res.runs)

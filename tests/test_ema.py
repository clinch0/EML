import numpy as np
import pytest
import torch

from eml_sr.model.master_tree import MasterTree
from eml_sr.model.trainer import TrainConfig, train_once


def test_invalid_beta():
    for beta in (-0.1, 1.0, float("nan")):
        with pytest.raises(ValueError):
            TrainConfig(ema_beta=beta)


def test_disabled_matches_default():
    x = np.linspace(0.5, 1.5, 8)
    a = train_once(x, np.exp(x), 2, 17, TrainConfig(epochs=3))
    b = train_once(x, np.exp(x), 2, 17, TrainConfig(epochs=3, ema_beta=0))
    assert a.tree == b.tree
    assert a.loss_before_snap == b.loss_before_snap


def test_one_step_ema_matches_manual_update():
    x = np.linspace(0.5, 1.5, 8)
    y = np.exp(x)
    cfg = TrainConfig(epochs=1, ema_beta=0.9)
    model = MasterTree(2, seed=17)
    initial = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.tau = cfg.tau_at(0)
    loss = torch.mean(torch.abs(model(torch.tensor(x)) - torch.tensor(y)) ** 2)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        for p, old in zip(model.parameters(), initial, strict=True):
            p.copy_(0.9 * old + 0.1 * p)
        expected = float(torch.mean(torch.abs(model(torch.tensor(x)) - torch.tensor(y)) ** 2))
    actual = train_once(x, y, 2, 17, cfg)
    assert actual.loss_before_snap == pytest.approx(expected)
    assert actual.tree == model.to_discrete()

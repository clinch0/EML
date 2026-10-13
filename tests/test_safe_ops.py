import numpy as np
import pytest
import torch

from eml_sr.core.safe_ops import EXP_LIMIT, safe_exp, safe_log

EXTREME = np.array(
    [0.0, 1e-300, -1e-300, 1e-20, 1.0, -1.0, 1e3, -1e3, 1e30, -1e30, 1e300, -1e300, 1e300j],
    dtype=np.complex128,
)


def test_safe_exp_numpy_no_nan_and_bounded():
    out = safe_exp(EXTREME)
    assert out.dtype == np.complex128
    assert np.all(np.isfinite(out))
    assert np.all(np.abs(out) <= np.exp(EXP_LIMIT) * (1 + 1e-12))


def test_safe_exp_limit_configurable():
    assert np.isclose(safe_exp(np.array([100.0]), limit=5.0)[0], np.exp(5.0))
    assert np.isclose(safe_exp(np.array([-100.0]), limit=5.0)[0], np.exp(-5.0))


def test_safe_exp_matches_exp_inside_range():
    z = np.array([0.3 + 0.2j, -2.0, 10.0 - 3j])
    assert np.allclose(safe_exp(z), np.exp(z))


def test_safe_log_numpy_no_nan():
    out = safe_log(EXTREME)
    assert np.all(np.isfinite(out))
    assert np.isclose(safe_log(np.array([0.0]))[0], np.log(1e-12))
    assert np.isclose(safe_log(np.array([0.0]), eps=1e-5)[0], np.log(1e-5))


def test_safe_log_principal_branch():
    assert np.isclose(safe_log(np.array([-1.0]))[0], 1j * np.pi)


def test_nan_is_propagated_not_hidden():
    assert np.isnan(safe_exp(np.array([np.nan]))[0])
    assert np.isnan(safe_log(np.array([np.nan]))[0])


def test_torch_versions_no_nan_and_match_numpy():
    t = torch.tensor(EXTREME)
    e = safe_exp(t)
    lg = safe_log(t)
    assert e.dtype == torch.complex128
    assert torch.all(torch.isfinite(e.real)) and torch.all(torch.isfinite(e.imag))
    assert torch.all(torch.isfinite(lg.real)) and torch.all(torch.isfinite(lg.imag))
    np.testing.assert_allclose(e.numpy(), safe_exp(EXTREME))
    np.testing.assert_allclose(lg.numpy(), safe_log(EXTREME))


@pytest.mark.parametrize("value", [0.0, 1e-300, 1e-20, 0.7, 60.0, -60.0, 1e30])
def test_torch_autograd_finite(value):
    z = torch.tensor([value], dtype=torch.float64, requires_grad=True)
    zc = torch.complex(z, torch.zeros_like(z))
    out = safe_exp(zc) - safe_log(zc)
    loss = (out.abs() ** 2).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.all(torch.isfinite(z.grad))

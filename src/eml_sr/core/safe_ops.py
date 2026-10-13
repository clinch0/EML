"""Numerically guarded complex ``exp`` and ``log``.

Both functions accept either NumPy arrays (anything convertible by ``numpy.asarray``) or torch
tensors and return a result of the same kind in complex128.

Behaviour:

* ``safe_exp(z)`` clamps the real part of ``z`` to ``[-limit, limit]`` (``limit = 50`` by
  default) before exponentiating, so ``|safe_exp(z)| <= e**limit``. The imaginary part is left
  unchanged: it only rotates the result and cannot overflow.
* ``safe_log(z)`` replaces every entry with ``|z| < eps`` (``eps = 1e-12`` by default) by the real
  number ``eps`` and then takes the principal branch of the complex logarithm
  (``-pi < Im(log z) <= pi``). Hence ``safe_log(0) = log(eps)`` instead of ``-inf``.

Consequences:

* Formulas that genuinely need infinities (for example ``ln 0 = -inf`` used as an intermediate
  value) are **not supported**: the guarded operations return large finite values instead.
* NaN inputs are propagated as NaN; they are never silently replaced by zeros.
* The torch implementation works on the real and imaginary parts separately (``torch.clamp``
  does not accept complex tensors) and uses no in-place operations, so autograd works.
  Gradients are zero where the clamp or the ``eps`` substitution is active.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

EXP_LIMIT: float = 50.0
LOG_EPS: float = 1e-12


def _as_complex_array(z: npt.ArrayLike) -> np.ndarray:
    return np.asarray(z, dtype=np.complex128)


def _as_complex_tensor(z: torch.Tensor) -> torch.Tensor:
    return z if z.dtype == torch.complex128 else z.to(torch.complex128)


def safe_exp(
    z: npt.ArrayLike | torch.Tensor, limit: float = EXP_LIMIT
) -> np.ndarray | torch.Tensor:
    """Complex exponential with the real part of the argument clamped to ``[-limit, limit]``.

    Args:
        z: Argument (NumPy array-like or torch tensor).
        limit: Bound for the real part of ``z``.

    Returns:
        ``exp(clamp(Re z) + i Im z)`` as complex128, same container type as ``z``.
    """
    if isinstance(z, torch.Tensor):
        zc = _as_complex_tensor(z)
        real = torch.clamp(zc.real, -limit, limit)
        return torch.exp(torch.complex(real, zc.imag))
    za = _as_complex_array(z)
    real = np.clip(za.real, -limit, limit)
    return np.exp(real + 1j * za.imag)


def safe_log(z: npt.ArrayLike | torch.Tensor, eps: float = LOG_EPS) -> np.ndarray | torch.Tensor:
    """Principal-branch complex logarithm with ``|z| < eps`` replaced by ``eps``.

    Args:
        z: Argument (NumPy array-like or torch tensor).
        eps: Magnitude threshold below which ``z`` is replaced by the real number ``eps``.

    Returns:
        ``log(z')`` as complex128, where ``z' = eps`` if ``|z| < eps`` else ``z``.
    """
    if isinstance(z, torch.Tensor):
        zc = _as_complex_tensor(z)
        small = zc.abs() < eps
        real = torch.where(small, torch.full_like(zc.real, eps), zc.real)
        imag = torch.where(small, torch.zeros_like(zc.imag), zc.imag)
        return torch.log(torch.complex(real, imag))
    za = _as_complex_array(z)
    za = np.where(np.abs(za) < eps, np.complex128(eps), za)
    return np.log(za)

"""Compare analytic gradients with central finite differences.

This is the correctness gate for :mod:`teotensor.nn.tensor.ops`. A faster
backend has to pass the same check against these NumPy gradients.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from teotensor.nn.tensor.core import Tensor


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: tuple[Tensor, ...],
    *,
    eps: float = 1e-5,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> None:
    """Check ``fn(*inputs).backward()`` against a central difference.

    Parameters
    ----------
    fn : callable
        Function of tensors that returns one scalar tensor.
    inputs : tuple of Tensor
        Arguments. Only tensors with ``requires_grad=True`` are checked.
        Their ``.data`` is perturbed in place and then restored.
    eps : float, default=1e-5
        Half-step of the central difference.
    atol, rtol : float
        Tolerances forwarded to :func:`numpy.allclose`.

    Raises
    ------
    AssertionError
        If an analytic gradient disagrees with the numeric one.
    """
    for tensor in inputs:
        tensor.zero_grad()
    value = fn(*inputs)
    value.backward()
    for tensor in inputs:
        if not tensor.requires_grad:
            continue
        if tensor.grad is None:
            msg = "Analytic gradient is missing."
            raise AssertionError(msg)
        analytic = np.array(tensor.grad, copy=True)
        numeric = np.empty_like(tensor.data)
        for index in np.ndindex(tensor.data.shape):
            original = float(tensor.data[index])
            tensor.data[index] = original + eps
            plus = float(fn(*inputs).data)
            tensor.data[index] = original - eps
            minus = float(fn(*inputs).data)
            tensor.data[index] = original
            numeric[index] = (plus - minus) / (2.0 * eps)
        if not np.allclose(analytic, numeric, atol=atol, rtol=rtol):
            msg = f"Gradient mismatch for op={tensor._op!r} shape={tensor.data.shape}."
            raise AssertionError(msg)

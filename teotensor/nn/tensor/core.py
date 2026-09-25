"""Graph-aware numeric block used by reverse-mode autodiff.

A :class:`Tensor` is a ``float64`` NumPy array plus, while a calculation is
running, a note of which operation produced it. That note is the tape.
:meth:`Tensor.backward` walks the tape from a single loss number back to the
inputs and writes a gradient into :attr:`Tensor.grad`.

Saving a tensor keeps the numbers and drops the tape. The next forward pass
builds a fresh tape. Arithmetic lives in :mod:`teotensor.nn.tensor.ops`; those
functions are the door a faster backend can replace later.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class Tensor:
    """A float64 array that can remember how it was computed.

    Parameters
    ----------
    data : array-like
        Numeric payload. Cast to ``float64``.
    requires_grad : bool, default=False
        If True, operations that use this tensor record a backward step.
    _prev : tuple of Tensor, optional
        Inputs of the operation that produced this tensor. Empty for leaves.
    _op : str, default=""
        Short name of that operation, kept for inspection.
    """

    def __init__(
        self,
        data: Any,
        *,
        requires_grad: bool = False,
        _prev: tuple[Tensor, ...] = (),
        _op: str = "",
    ) -> None:
        self.data: FloatArray = np.asarray(data, dtype=np.float64)
        self.grad: FloatArray | None = None
        self.requires_grad = requires_grad
        self._prev = _prev
        self._op = _op
        self._backward: Callable[[], None] = _noop

    def zero_grad(self) -> None:
        """Forget the gradient so the next example starts clean.

        The gradient is set to ``None``, not to zeros. An optimizer that sees
        ``None`` stops, which makes a missing :meth:`backward` visible.
        """
        self.grad = None

    def backward(self) -> None:
        """Send the loss gradient back along the tape to every input.

        Raises
        ------
        RuntimeError
            If this tensor is not a single number. The loss must be a scalar
            so there is one starting gradient of ``1``.
        """
        if self.data.size != 1:
            msg = f"backward() needs a single loss number; got shape {self.data.shape}."
            raise RuntimeError(msg)
        order = _topological_order(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            node._backward()

    def __getstate__(self) -> dict[str, Any]:
        """Keep numbers only. The tape is rebuilt on the next forward pass."""
        return {"data": self.data, "requires_grad": self.requires_grad}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore numbers and start with an empty tape."""
        self.data = np.asarray(state["data"], dtype=np.float64)
        self.requires_grad = bool(state["requires_grad"])
        self.grad = None
        self._prev = ()
        self._op = ""
        self._backward = _noop

    def __repr__(self) -> str:
        """Return a short view of the payload and whether gradients are on."""
        return (
            f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad}, "
            f"op={self._op!r})"
        )

    def __add__(self, other: Tensor | float) -> Tensor:
        """Add a tensor or a number. See :func:`teotensor.nn.tensor.ops.add`."""
        from teotensor.nn.tensor.ops import add

        return add(self, _as_tensor(other))

    def __radd__(self, other: Tensor | float) -> Tensor:
        """Add when the tensor is on the right-hand side."""
        return self.__add__(other)

    def __sub__(self, other: Tensor | float) -> Tensor:
        """Subtract a tensor or a number."""
        from teotensor.nn.tensor.ops import sub

        return sub(self, _as_tensor(other))

    def __rsub__(self, other: Tensor | float) -> Tensor:
        """Subtract this tensor from a number or tensor."""
        from teotensor.nn.tensor.ops import sub

        return sub(_as_tensor(other), self)

    def __mul__(self, other: Tensor | float) -> Tensor:
        """Multiply by a tensor or a number."""
        from teotensor.nn.tensor.ops import mul

        return mul(self, _as_tensor(other))

    def __rmul__(self, other: Tensor | float) -> Tensor:
        """Multiply when the tensor is on the right-hand side."""
        return self.__mul__(other)

    def __truediv__(self, other: Tensor | float) -> Tensor:
        """Divide by a tensor or a number."""
        from teotensor.nn.tensor.ops import div

        return div(self, _as_tensor(other))

    def __rtruediv__(self, other: Tensor | float) -> Tensor:
        """Divide a number or tensor by this tensor."""
        from teotensor.nn.tensor.ops import div

        return div(_as_tensor(other), self)

    def __neg__(self) -> Tensor:
        """Negate every element."""
        from teotensor.nn.tensor.ops import neg

        return neg(self)

    def __matmul__(self, other: Tensor) -> Tensor:
        """Matrix-multiply by another tensor."""
        from teotensor.nn.tensor.ops import matmul

        return matmul(self, other)

    def __pow__(self, exponent: float) -> Tensor:
        """Raise every element to a numeric power."""
        from teotensor.nn.tensor.ops import power

        return power(self, exponent)

    def sum(self) -> Tensor:
        """Sum every element into one number."""
        from teotensor.nn.tensor.ops import sum_all

        return sum_all(self)

    def mean(self) -> Tensor:
        """Average every element into one number."""
        from teotensor.nn.tensor.ops import mean_all

        return mean_all(self)

    def abs(self) -> Tensor:
        """Elementwise absolute value."""
        from teotensor.nn.tensor.ops import absolute

        return absolute(self)


class Parameter(Tensor):
    """A tensor that is a weight of a layer.

    Parameters always ask for a gradient. They are created by layers, not by
    the loss.
    """

    def __init__(self, data: Any) -> None:
        super().__init__(data, requires_grad=True, _op="parameter")


def _noop() -> None:
    """Backward step for a leaf that has no parents."""
    return None


def _as_tensor(value: Tensor | float) -> Tensor:
    """Wrap a plain number. A tensor is returned unchanged."""
    if isinstance(value, Tensor):
        return value
    return Tensor(value, requires_grad=False, _op="const")


def _topological_order(root: Tensor) -> list[Tensor]:
    """List nodes so every input appears before the node that uses it."""
    order: list[Tensor] = []
    seen: set[int] = set()

    def visit(node: Tensor) -> None:
        marker = id(node)
        if marker in seen:
            return
        seen.add(marker)
        for parent in node._prev:
            visit(parent)
        order.append(node)

    visit(root)
    return order


def accumulate(tensor: Tensor, grad: FloatArray) -> None:
    """Add ``grad`` into ``tensor.grad``, undoing broadcasting.

    Parameters
    ----------
    tensor : Tensor
        Leaf or intermediate that asked for a gradient.
    grad : ndarray
        Gradient of the loss with respect to the operation output, already
        shaped like the output. Broadcast axes are summed away.
    """
    if not tensor.requires_grad:
        return
    aligned = _unbroadcast(np.asarray(grad, dtype=np.float64), tensor.data.shape)
    if tensor.grad is None:
        tensor.grad = aligned
    else:
        tensor.grad = tensor.grad + aligned


def _unbroadcast(grad: FloatArray, shape: tuple[int, ...]) -> FloatArray:
    """Sum ``grad`` down to ``shape`` after a NumPy broadcast."""
    reduced = grad
    while reduced.ndim > len(shape):
        reduced = reduced.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and reduced.shape[axis] != 1:
            reduced = reduced.sum(axis=axis, keepdims=True)
    return np.asarray(reduced, dtype=np.float64)

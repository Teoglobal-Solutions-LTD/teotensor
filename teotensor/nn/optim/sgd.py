"""Stochastic gradient descent, with optional momentum and Nesterov."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from teotensor.nn.component import Component
from teotensor.nn.penalty import Penalty
from teotensor.nn.tensor.core import Parameter

FloatArray = NDArray[np.float64]


class Optimizer(Component):
    """Updates parameters from their gradients.

    Subclasses keep moment memory in attributes that are not constructor
    settings, so :func:`~teotensor.core.clone` does not copy that memory.
    """

    decoupled_l2: bool = False
    lr: float

    def step(
        self,
        parameters: list[Parameter],
        *,
        lr: float,
        penalty: Penalty | None = None,
    ) -> None:
        """Move each parameter one step.

        Parameters
        ----------
        parameters : list of Parameter
            Weights and biases. Every one must already have a gradient.
        lr : float
            Step length for this update, already multiplied by the schedule.
        penalty : Penalty or None, optional
            Used by Adam to apply decoupled decay. SGD reads the penalty from
            the loss instead.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """Plain gradient step, optionally smoothed by momentum.

    Parameters
    ----------
    lr : float, default=0.01
        Base step length. The schedule multiplies it.
    momentum : float, default=0.0
        How much of the previous step to keep. ``0`` is plain SGD.
    nesterov : bool, default=False
        If True, look ahead along the momentum before applying the gradient.
        Requires ``momentum > 0``.
    """

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity_: list[FloatArray] | None = None

    def step(
        self,
        parameters: list[Parameter],
        *,
        lr: float,
        penalty: Penalty | None = None,
    ) -> None:
        """Subtract the (possibly momentum-smoothed) gradient."""
        del penalty
        if self.nesterov and self.momentum == 0.0:
            msg = "nesterov=True requires momentum > 0."
            raise ValueError(msg)
        if self.velocity_ is None or len(self.velocity_) != len(parameters):
            self.velocity_ = [np.zeros_like(param.data) for param in parameters]
        for index, param in enumerate(parameters):
            grad = _require_grad(param)
            velocity = self.momentum * self.velocity_[index] + grad
            self.velocity_[index] = velocity
            update = self.momentum * velocity + grad if self.nesterov else velocity
            param.data = param.data - lr * update


def _require_grad(param: Parameter) -> FloatArray:
    """Return the gradient or raise if backward was not run."""
    if param.grad is None:
        msg = "Parameter has no gradient. Call backward() before step()."
        raise RuntimeError(msg)
    return param.grad

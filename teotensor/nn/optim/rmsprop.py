"""RMSprop. The penalty is added to the loss, not as decoupled decay."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from teotensor.nn.optim.sgd import Optimizer, _require_grad
from teotensor.nn.penalty import Penalty
from teotensor.nn.tensor.core import Parameter

FloatArray = NDArray[np.float64]


class RMSprop(Optimizer):
    """Divide the gradient by a running root-mean-square.

    Parameters
    ----------
    lr : float, default=0.001
        Base step length.
    alpha : float, default=0.99
        Decay of the running square. Named ``alpha`` on this component so it
        is not a second field on the model.
    epsilon : float, default=1e-8
        Small constant that keeps the division defined.
    momentum : float, default=0.0
        Optional momentum on the scaled gradient.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        alpha: float = 0.99,
        epsilon: float = 1e-8,
        momentum: float = 0.0,
    ) -> None:
        self.lr = lr
        self.alpha = alpha
        self.epsilon = epsilon
        self.momentum = momentum
        self.square_: list[FloatArray] | None = None
        self.velocity_: list[FloatArray] | None = None

    def step(
        self,
        parameters: list[Parameter],
        *,
        lr: float,
        penalty: Penalty | None = None,
    ) -> None:
        """RMSprop update. ``penalty`` is unused; it already entered the loss."""
        del penalty
        if self.square_ is None or len(self.square_) != len(parameters):
            self.square_ = [np.zeros_like(param.data) for param in parameters]
            self.velocity_ = [np.zeros_like(param.data) for param in parameters]
        assert self.velocity_ is not None
        for index, param in enumerate(parameters):
            grad = _require_grad(param)
            square = self.alpha * self.square_[index] + (1.0 - self.alpha) * np.square(
                grad
            )
            self.square_[index] = square
            update = grad / (np.sqrt(square) + self.epsilon)
            velocity = self.momentum * self.velocity_[index] + update
            self.velocity_[index] = velocity
            param.data = param.data - lr * velocity

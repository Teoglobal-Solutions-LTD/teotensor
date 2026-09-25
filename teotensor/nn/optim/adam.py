"""Adam with decoupled weight decay for the squared penalty."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from teotensor.nn.optim.sgd import Optimizer, _require_grad
from teotensor.nn.penalty import Penalty
from teotensor.nn.tensor.core import Parameter

FloatArray = NDArray[np.float64]


class Adam(Optimizer):
    """Adaptive moments. The squared penalty is applied outside the moments.

    Parameters
    ----------
    lr : float, default=0.001
        Base step length.
    beta_1 : float, default=0.9
        Decay of the first moment (the mean gradient).
    beta_2 : float, default=0.999
        Decay of the second moment (the mean square gradient).
    epsilon : float, default=1e-8
        Small constant that keeps the division defined.
    """

    decoupled_l2 = True

    def __init__(
        self,
        lr: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.step_: int = 0
        self.moment_: list[FloatArray] | None = None
        self.second_: list[FloatArray] | None = None

    def step(
        self,
        parameters: list[Parameter],
        *,
        lr: float,
        penalty: Penalty | None = None,
    ) -> None:
        """Adam update, then decoupled decay on weight matrices only.

        A parameter is treated as a weight matrix when it has two axes. Bias
        vectors are not decayed.
        """
        if self.moment_ is None or len(self.moment_) != len(parameters):
            self.moment_ = [np.zeros_like(param.data) for param in parameters]
            self.second_ = [np.zeros_like(param.data) for param in parameters]
            self.step_ = 0
        assert self.second_ is not None
        self.step_ += 1
        bias_c1 = 1.0 - self.beta_1**self.step_
        bias_c2 = 1.0 - self.beta_2**self.step_
        l2_coeff = 0.0 if penalty is None else penalty.l2_coefficient()
        for index, param in enumerate(parameters):
            grad = _require_grad(param)
            moment = self.beta_1 * self.moment_[index] + (1.0 - self.beta_1) * grad
            second = self.beta_2 * self.second_[index] + (
                1.0 - self.beta_2
            ) * np.square(grad)
            self.moment_[index] = moment
            self.second_[index] = second
            corrected = moment / bias_c1
            second_hat = second / bias_c2
            param.data = param.data - lr * corrected / (
                np.sqrt(second_hat) + self.epsilon
            )
            if l2_coeff != 0.0 and param.data.ndim == 2:
                # Slope of ``l2_coeff * sum(W**2)`` is ``2 * l2_coeff * W``.
                param.data = param.data - lr * (2.0 * l2_coeff) * param.data

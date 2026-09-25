"""Activation components.

Each class is a hyperparameter object. :meth:`forward` calls one door in
:mod:`teotensor.nn.tensor.ops`. The output layer of an MLP does not use an
activation; the loss reads raw scores.
"""

from __future__ import annotations

from teotensor.nn.component import Component
from teotensor.nn.tensor.core import Tensor
from teotensor.nn.tensor.ops import (
    elu,
    gelu,
    leaky_relu,
    relu,
    selu,
    sigmoid,
    silu,
    softplus,
    tanh,
)


class Activation(Component):
    """Maps a tensor to a tensor of the same shape."""

    def forward(self, value: Tensor) -> Tensor:
        """Apply the activation.

        Parameters
        ----------
        value : Tensor
            Incoming layer output.

        Returns
        -------
        Tensor
            Activated values, still on the tape.
        """
        raise NotImplementedError


class Identity(Activation):
    """Leave the numbers unchanged."""

    def forward(self, value: Tensor) -> Tensor:
        """Return ``value``."""
        return value


class ReLU(Activation):
    """Rectified linear unit. See :func:`~teotensor.nn.tensor.ops.relu`."""

    def forward(self, value: Tensor) -> Tensor:
        """Apply ReLU."""
        return relu(value)


class LeakyReLU(Activation):
    """ReLU with a non-zero slope on the negative side.

    Parameters
    ----------
    negative_slope : float, default=0.01
        Slope used for inputs that are not strictly positive.
    """

    def __init__(self, negative_slope: float = 0.01) -> None:
        self.negative_slope = negative_slope

    def forward(self, value: Tensor) -> Tensor:
        """Apply leaky ReLU."""
        return leaky_relu(value, self.negative_slope)


class Tanh(Activation):
    """Hyperbolic tangent."""

    def forward(self, value: Tensor) -> Tensor:
        """Apply tanh."""
        return tanh(value)


class Sigmoid(Activation):
    """Logistic sigmoid."""

    def forward(self, value: Tensor) -> Tensor:
        """Apply the logistic function."""
        return sigmoid(value)


class ELU(Activation):
    """Exponential linear unit.

    Parameters
    ----------
    alpha : float, default=1.0
        Scale of the negative side.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def forward(self, value: Tensor) -> Tensor:
        """Apply ELU."""
        return elu(value, self.alpha)


class GELU(Activation):
    """Gaussian error linear unit."""

    def forward(self, value: Tensor) -> Tensor:
        """Apply GELU."""
        return gelu(value)


class SiLU(Activation):
    """Sigmoid linear unit (swish)."""

    def forward(self, value: Tensor) -> Tensor:
        """Apply SiLU."""
        return silu(value)


class Softplus(Activation):
    """Smooth ReLU."""

    def forward(self, value: Tensor) -> Tensor:
        """Apply softplus."""
        return softplus(value)


class SELU(Activation):
    """Scaled exponential linear unit."""

    def forward(self, value: Tensor) -> Tensor:
        """Apply SELU."""
        return selu(value)


ACTIVATIONS: dict[str, type[Activation]] = {
    "Identity": Identity,
    "ReLU": ReLU,
    "LeakyReLU": LeakyReLU,
    "Tanh": Tanh,
    "Sigmoid": Sigmoid,
    "ELU": ELU,
    "GELU": GELU,
    "SiLU": SiLU,
    "Softplus": Softplus,
    "SELU": SELU,
}

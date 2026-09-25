"""Module wrapper around an :class:`~teotensor.nn.activations.Activation`.

The wrapper is what the stack calls. It also remembers the last batch of
outputs when ``capture`` is on, so the trainer can measure dead neurons
without keeping every step.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from teotensor.nn.activations import Activation
from teotensor.nn.layers.module import Module
from teotensor.nn.tensor.core import Tensor

FloatArray = NDArray[np.float64]


class ActivationLayer(Module):
    """Apply ``activation`` and optionally keep a copy of the output.

    Parameters
    ----------
    activation : Activation
        Component that implements :meth:`~teotensor.nn.activations.Activation.forward`.
    """

    def __init__(self, activation: Activation) -> None:
        super().__init__()
        self.activation = activation
        self.capture = False
        self.last_data: FloatArray | None = None

    def forward(self, value: Tensor) -> Tensor:
        """Activate ``value``. Copy the numbers when ``capture`` is true."""
        out = self.activation.forward(value)
        if self.capture:
            self.last_data = np.array(out.data, dtype=np.float64, copy=True)
        return out

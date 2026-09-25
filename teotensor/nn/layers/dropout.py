"""Dropout silences neurons at random while the network is learning.

What happens to one neuron on one training example
---------------------------------------------------
Let ``p`` be the drop probability (the fraction we want to silence).

1. Draw a coin for that neuron on that example. With probability ``p`` the
   coin says "off"; with probability ``1 - p`` it says "on". Every neuron of
   every example in the batch draws its own coin. The mask is not shared
   across the batch, and it is not a fixed set of dead neurons for the whole
   epoch.
2. If the coin says off, the neuron's output becomes 0. Its gradient on this
   step is also 0, so the weights that feed it do not learn from this example
   on this step.
3. If the coin says on, the output is multiplied by ``1 / (1 - p)``. That
   scale keeps the average size of the layer the same as in exam mode, where
   nobody is silenced. Without the scale, exam-time outputs would be smaller
   than the outputs the next layer learned to expect.

Exam mode (``eval``) does none of this. The layer is a no-op: every neuron
passes its number through, and there is no ``1 / (1 - p)`` factor. Playback
and ``predict`` always run in exam mode so the picture and the answers do not
flicker.

``p = 0`` is also a no-op. ``p`` must be strictly less than 1, otherwise the
scale would divide by zero.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator, RandomState
from numpy.typing import NDArray

from teotensor.nn.layers.module import Module
from teotensor.nn.tensor.core import Tensor, accumulate

RNG = RandomState | Generator


class Dropout(Module):
    """Inverted dropout.

    Parameters
    ----------
    p : float, default=0.0
        Probability of silencing one neuron on one training example.
    rng : RandomState or Generator or None, optional
        Stream that draws the coins. The model passes its single training
        stream so dropout continues the same sequence as weight init and
        shuffling. ``None`` builds a fresh ``RandomState``.
    """

    def __init__(self, p: float = 0.0, rng: RNG | None = None) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            msg = f"dropout probability must satisfy 0 <= p < 1; got {p}."
            raise ValueError(msg)
        self.p = float(p)
        self.rng: RNG = np.random.RandomState() if rng is None else rng

    def forward(self, value: Tensor) -> Tensor:
        """Silence neurons when :attr:`Module.training` is true.

        In exam mode, or when ``p`` is 0, return ``value`` unchanged and do
        not draw any coins.
        """
        if (not self.training) or self.p == 0.0:
            return value
        keep_prob = 1.0 - self.p
        coins = _uniform(self.rng, value.data.shape)
        # True means the neuron stays on for this example.
        mask = coins >= self.p
        scale = 1.0 / keep_prob
        data = value.data * mask * scale
        out = Tensor(
            data,
            requires_grad=value.requires_grad,
            _prev=(value,),
            _op="dropout",
        )

        def backward() -> None:
            assert out.grad is not None
            # The same mask and the same scale as the forward pass. A silenced
            # neuron contributes nothing to the gradient.
            accumulate(value, out.grad * mask * scale)

        out._backward = backward
        return out


def _uniform(rng: RNG, shape: tuple[int, ...]) -> NDArray[np.float64]:
    """Draw ``U(0, 1)`` samples from either NumPy generator type."""
    if isinstance(rng, Generator):
        return np.asarray(rng.random(shape), dtype=np.float64)
    return np.asarray(rng.rand(*shape), dtype=np.float64)

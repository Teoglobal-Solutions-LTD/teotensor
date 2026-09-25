"""Loss components. Each one returns a single number on the tape.

Classification uses :class:`CrossEntropy` on raw scores (logits). The loss
turns those scores into probabilities itself, with a stable log-softmax, so
the network must not also end in a softmax.

Regression losses compare the network output with the target element by
element and average. A target of shape ``(n,)`` is treated as ``(n, 1)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from teotensor.nn.component import Component
from teotensor.nn.tensor.core import Tensor, accumulate
from teotensor.nn.tensor.ops import cross_entropy

FloatArray = NDArray[np.float64]


class Loss(Component):
    """Scores a batch of predictions against targets."""

    def forward(
        self,
        prediction: Tensor,
        target: NDArray[np.generic],
        sample_weight: NDArray[np.float64] | None = None,
    ) -> Tensor:
        """Return a scalar loss.

        Parameters
        ----------
        prediction : Tensor
            Network output for the batch.
        target : ndarray
            Reference values. Integer class indices for cross-entropy.
        sample_weight : ndarray or None, optional
            Per-example weights. Ignored by regression losses in this baseline
            except where a subclass documents otherwise.
        """
        raise NotImplementedError


class CrossEntropy(Loss):
    """Mean negative log-likelihood of integer class labels."""

    def forward(
        self,
        prediction: Tensor,
        target: NDArray[np.generic],
        sample_weight: NDArray[np.float64] | None = None,
    ) -> Tensor:
        """Cross-entropy between ``prediction`` logits and integer labels."""
        labels = np.asarray(target, dtype=np.int64).reshape(-1)
        return cross_entropy(prediction, labels, sample_weight)


class MSE(Loss):
    """Mean of squared differences. Gradient is ``2 * error / n``."""

    def forward(
        self,
        prediction: Tensor,
        target: NDArray[np.generic],
        sample_weight: NDArray[np.float64] | None = None,
    ) -> Tensor:
        """Mean squared error."""
        del sample_weight
        return _element_loss(prediction, target, kind="mse")


class MAE(Loss):
    """Mean absolute error. The slope at a perfect prediction is 0."""

    def forward(
        self,
        prediction: Tensor,
        target: NDArray[np.generic],
        sample_weight: NDArray[np.float64] | None = None,
    ) -> Tensor:
        """Mean absolute error."""
        del sample_weight
        return _element_loss(prediction, target, kind="mae")


class Huber(Loss):
    """Squared error near zero, absolute error beyond ``delta``.

    Parameters
    ----------
    delta : float, default=1.0
        Width of the quadratic region.
    """

    def __init__(self, delta: float = 1.0) -> None:
        self.delta = delta

    def forward(
        self,
        prediction: Tensor,
        target: NDArray[np.generic],
        sample_weight: NDArray[np.float64] | None = None,
    ) -> Tensor:
        """Huber loss."""
        del sample_weight
        return _element_loss(prediction, target, kind="huber", delta=float(self.delta))


def _element_loss(
    prediction: Tensor,
    target: NDArray[np.generic],
    *,
    kind: str,
    delta: float = 1.0,
) -> Tensor:
    """Shared regression loss. ``kind`` is ``mse``, ``mae``, or ``huber``."""
    reference = np.asarray(target, dtype=np.float64)
    if reference.ndim == 1:
        reference = reference.reshape(-1, 1)
    if reference.shape != prediction.data.shape:
        msg = (
            "Prediction and target shapes differ: "
            f"{prediction.data.shape} vs {reference.shape}."
        )
        raise ValueError(msg)
    error = prediction.data - reference
    if kind == "mse":
        values = np.square(error)
        local = 2.0 * error
    elif kind == "mae":
        values = np.abs(error)
        local = np.sign(error)
    elif kind == "huber":
        abs_error = np.abs(error)
        quadratic = abs_error <= delta
        values = np.where(
            quadratic,
            0.5 * np.square(error),
            delta * (abs_error - 0.5 * delta),
        )
        local = np.where(quadratic, error, delta * np.sign(error))
    else:
        msg = f"Unknown regression loss {kind!r}."
        raise ValueError(msg)
    scale = 1.0 / float(error.size)
    out = Tensor(
        np.asarray(values.mean()),
        requires_grad=prediction.requires_grad,
        _prev=(prediction,),
        _op=kind,
    )

    def backward() -> None:
        assert out.grad is not None
        accumulate(prediction, local * scale * out.grad)

    out._backward = backward
    return out


LOSSES: dict[str, type[Loss]] = {
    "CrossEntropy": CrossEntropy,
    "MSE": MSE,
    "MAE": MAE,
    "Huber": Huber,
}

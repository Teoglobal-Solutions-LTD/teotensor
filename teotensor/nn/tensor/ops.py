"""Arithmetic doors for the neural-network engine.

Every function here takes tensors and returns a tensor. The numbers are
computed with NumPy. The backward step is stored on the result. A later
accelerated backend must keep these names and these results; callers (layers,
losses, the trainer) do not import NumPy themselves for this math.

``log_softmax`` and ``cross_entropy`` are one fused step each. Building them
out of ``exp`` and ``log`` overflows on large scores.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from teotensor.nn.tensor.core import Tensor, accumulate

FloatArray = NDArray[np.float64]


def _result(data: FloatArray, inputs: tuple[Tensor, ...], op: str) -> Tensor:
    """Allocate an output tensor that records a tape when any input needs it."""
    return Tensor(
        data,
        requires_grad=any(item.requires_grad for item in inputs),
        _prev=inputs,
        _op=op,
    )


def add(left: Tensor, right: Tensor) -> Tensor:
    """Elementwise sum with NumPy broadcasting."""
    out = _result(left.data + right.data, (left, right), "add")

    def backward() -> None:
        assert out.grad is not None
        accumulate(left, out.grad)
        accumulate(right, out.grad)

    out._backward = backward
    return out


def sub(left: Tensor, right: Tensor) -> Tensor:
    """Elementwise difference with NumPy broadcasting."""
    out = _result(left.data - right.data, (left, right), "sub")

    def backward() -> None:
        assert out.grad is not None
        accumulate(left, out.grad)
        accumulate(right, -out.grad)

    out._backward = backward
    return out


def mul(left: Tensor, right: Tensor) -> Tensor:
    """Elementwise product with NumPy broadcasting."""
    out = _result(left.data * right.data, (left, right), "mul")

    def backward() -> None:
        assert out.grad is not None
        accumulate(left, out.grad * right.data)
        accumulate(right, out.grad * left.data)

    out._backward = backward
    return out


def div(left: Tensor, right: Tensor) -> Tensor:
    """Elementwise division with NumPy broadcasting."""
    out = _result(left.data / right.data, (left, right), "div")

    def backward() -> None:
        assert out.grad is not None
        accumulate(left, out.grad / right.data)
        accumulate(right, -out.grad * left.data / np.square(right.data))

    out._backward = backward
    return out


def neg(value: Tensor) -> Tensor:
    """Elementwise negation."""
    out = _result(-value.data, (value,), "neg")

    def backward() -> None:
        assert out.grad is not None
        accumulate(value, -out.grad)

    out._backward = backward
    return out


def absolute(value: Tensor) -> Tensor:
    """Elementwise absolute value. The slope at zero is 0."""
    out = _result(np.abs(value.data), (value,), "abs")

    def backward() -> None:
        assert out.grad is not None
        slope = np.sign(value.data)
        accumulate(value, out.grad * slope)

    out._backward = backward
    return out


def power(value: Tensor, exponent: float) -> Tensor:
    """Raise every element to a fixed numeric power.

    Parameters
    ----------
    value : Tensor
        Base.
    exponent : float
        Power. Not a tensor, so it receives no gradient.
    """
    out = _result(np.power(value.data, exponent), (value,), "pow")

    def backward() -> None:
        assert out.grad is not None
        local = exponent * np.power(value.data, exponent - 1.0)
        accumulate(value, out.grad * local)

    out._backward = backward
    return out


def matmul(left: Tensor, right: Tensor) -> Tensor:
    """Matrix product ``(batch, in) @ (in, out)`` or any NumPy-compatible pair.

    Parameters
    ----------
    left, right : Tensor
        Factors. This baseline uses two-dimensional products in the network.
    """
    out = _result(left.data @ right.data, (left, right), "matmul")

    def backward() -> None:
        assert out.grad is not None
        accumulate(left, out.grad @ np.swapaxes(right.data, -1, -2))
        accumulate(right, np.swapaxes(left.data, -1, -2) @ out.grad)

    out._backward = backward
    return out


def sum_all(value: Tensor) -> Tensor:
    """Sum every element to a scalar."""
    out = _result(np.asarray(value.data.sum()), (value,), "sum")

    def backward() -> None:
        assert out.grad is not None
        accumulate(value, np.ones_like(value.data) * out.grad)

    out._backward = backward
    return out


def mean_all(value: Tensor) -> Tensor:
    """Average every element to a scalar."""
    scale = 1.0 / float(value.data.size)
    out = _result(np.asarray(value.data.mean()), (value,), "mean")

    def backward() -> None:
        assert out.grad is not None
        accumulate(value, np.ones_like(value.data) * out.grad * scale)

    out._backward = backward
    return out


def exp(value: Tensor) -> Tensor:
    """Elementwise exponential."""
    data = np.exp(value.data)
    out = _result(data, (value,), "exp")

    def backward() -> None:
        assert out.grad is not None
        accumulate(value, out.grad * data)

    out._backward = backward
    return out


def log(value: Tensor) -> Tensor:
    """Elementwise natural logarithm."""
    out = _result(np.log(value.data), (value,), "log")

    def backward() -> None:
        assert out.grad is not None
        accumulate(value, out.grad / value.data)

    out._backward = backward
    return out


def maximum(left: Tensor, right: Tensor) -> Tensor:
    """Elementwise maximum. Ties send the whole slope to ``left``."""
    choose_left = left.data >= right.data
    data = np.where(choose_left, left.data, right.data)
    out = _result(data, (left, right), "maximum")

    def backward() -> None:
        assert out.grad is not None
        accumulate(left, np.where(choose_left, out.grad, 0.0))
        accumulate(right, np.where(choose_left, 0.0, out.grad))

    out._backward = backward
    return out


def relu(value: Tensor) -> Tensor:
    """Rectified linear unit. The slope at zero is 0.

    Negative inputs become 0 and receive no gradient. Positive inputs pass
    through unchanged and pass the gradient through unchanged.
    """
    mask = value.data > 0.0
    out = _result(np.where(mask, value.data, 0.0), (value,), "relu")

    def backward() -> None:
        assert out.grad is not None
        accumulate(value, np.where(mask, out.grad, 0.0))

    out._backward = backward
    return out


def leaky_relu(value: Tensor, negative_slope: float = 0.01) -> Tensor:
    """ReLU that lets a small fraction of the negative side through.

    Parameters
    ----------
    value : Tensor
        Incoming numbers.
    negative_slope : float, default=0.01
        Slope used where the input is negative or zero.
    """
    slope = float(negative_slope)
    mask = value.data > 0.0
    data = np.where(mask, value.data, slope * value.data)
    out = _result(data, (value,), "leaky_relu")

    def backward() -> None:
        assert out.grad is not None
        local = np.where(mask, 1.0, slope)
        accumulate(value, out.grad * local)

    out._backward = backward
    return out


def tanh(value: Tensor) -> Tensor:
    """Elementwise hyperbolic tangent."""
    data = np.tanh(value.data)
    out = _result(data, (value,), "tanh")

    def backward() -> None:
        assert out.grad is not None
        accumulate(value, out.grad * (1.0 - np.square(data)))

    out._backward = backward
    return out


def sigmoid(value: Tensor) -> Tensor:
    """Elementwise logistic function, stable for large magnitudes."""
    data = _sigmoid_array(value.data)
    out = _result(data, (value,), "sigmoid")

    def backward() -> None:
        assert out.grad is not None
        accumulate(value, out.grad * data * (1.0 - data))

    out._backward = backward
    return out


def elu(value: Tensor, alpha: float = 1.0) -> Tensor:
    """Exponential linear unit.

    Parameters
    ----------
    value : Tensor
        Incoming numbers.
    alpha : float, default=1.0
        Scale of the negative side: ``alpha * (exp(x) - 1)``.
    """
    scale = float(alpha)
    positive = value.data > 0.0
    data = np.where(positive, value.data, scale * (np.exp(value.data) - 1.0))
    out = _result(data, (value,), "elu")

    def backward() -> None:
        assert out.grad is not None
        local = np.where(positive, 1.0, data + scale)
        accumulate(value, out.grad * local)

    out._backward = backward
    return out


def gelu(value: Tensor) -> Tensor:
    """Gaussian error linear unit, exact ``erf`` form."""
    cdf = 0.5 * (1.0 + _erf(value.data / np.sqrt(2.0)))
    data = value.data * cdf
    out = _result(data, (value,), "gelu")

    def backward() -> None:
        assert out.grad is not None
        local_cdf = 0.5 * (1.0 + _erf(value.data / np.sqrt(2.0)))
        pdf = np.exp(-0.5 * np.square(value.data)) / np.sqrt(2.0 * np.pi)
        accumulate(value, out.grad * (local_cdf + value.data * pdf))

    out._backward = backward
    return out


def silu(value: Tensor) -> Tensor:
    """Sigmoid linear unit, also called swish: ``x * sigmoid(x)``."""
    gate = _sigmoid_array(value.data)
    data = value.data * gate
    out = _result(data, (value,), "silu")

    def backward() -> None:
        assert out.grad is not None
        local = gate + value.data * gate * (1.0 - gate)
        accumulate(value, out.grad * local)

    out._backward = backward
    return out


def softplus(value: Tensor) -> Tensor:
    """Smooth ReLU: ``log(1 + exp(x))``, stable for large ``|x|``."""
    data = np.maximum(value.data, 0.0) + np.log1p(np.exp(-np.abs(value.data)))
    out = _result(data, (value,), "softplus")

    def backward() -> None:
        assert out.grad is not None
        accumulate(value, out.grad * _sigmoid_array(value.data))

    out._backward = backward
    return out


def selu(value: Tensor) -> Tensor:
    """Scaled exponential linear unit with the standard fixed constants."""
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    positive = value.data > 0.0
    data = scale * np.where(positive, value.data, alpha * (np.exp(value.data) - 1.0))
    out = _result(data, (value,), "selu")

    def backward() -> None:
        assert out.grad is not None
        local = scale * np.where(positive, 1.0, alpha * np.exp(value.data))
        accumulate(value, out.grad * local)

    out._backward = backward
    return out


def log_softmax(value: Tensor, *, axis: int = -1) -> Tensor:
    """Stable log-softmax along ``axis``.

    Scores are shifted by their maximum before the exponential, so a large
    logit does not overflow.

    Parameters
    ----------
    value : Tensor
        Raw scores, typically shape ``(batch, n_classes)``.
    axis : int, default=-1
        Axis of the classes.
    """
    shifted = value.data - np.max(value.data, axis=axis, keepdims=True)
    log_z = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    data = shifted - log_z
    out = _result(data, (value,), "log_softmax")

    def backward() -> None:
        assert out.grad is not None
        probs = np.exp(data)
        summed = np.sum(out.grad, axis=axis, keepdims=True)
        accumulate(value, out.grad - probs * summed)

    out._backward = backward
    return out


def cross_entropy(
    logits: Tensor,
    targets: NDArray[np.int64],
    sample_weight: NDArray[np.float64] | None = None,
) -> Tensor:
    """Mean negative log-likelihood of integer class labels.

    Fused with a stable log-softmax. The gradient with respect to the logits
    of one row is ``(probability - one_hot)`` times that row's share of the
    total weight. Weights are normalized by their sum, not by the batch size,
    so a rare class with a large weight actually pulls harder.

    Parameters
    ----------
    logits : Tensor of shape (n_samples, n_classes)
        Raw class scores. Not probabilities.
    targets : ndarray of shape (n_samples,)
        Integer class indices.
    sample_weight : ndarray of shape (n_samples,) or None, optional
        Non-negative per-example weights. ``None`` means every example counts
        equally.
    """
    labels = np.asarray(targets, dtype=np.int64)
    if labels.shape != (logits.data.shape[0],):
        msg = (
            "cross_entropy targets must have shape "
            f"({logits.data.shape[0]},); got {labels.shape}."
        )
        raise ValueError(msg)
    shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    picked = log_probs[np.arange(labels.shape[0]), labels]
    if sample_weight is None:
        weights = np.ones(labels.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != labels.shape:
            msg = f"sample_weight must have shape {labels.shape}; got {weights.shape}."
            raise ValueError(msg)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        msg = "sample_weight must sum to a positive number."
        raise ValueError(msg)
    loss_value = -float(np.sum(weights * picked) / weight_sum)
    out = _result(np.asarray(loss_value), (logits,), "cross_entropy")

    def backward() -> None:
        assert out.grad is not None
        probs = np.exp(log_probs)
        grad = probs
        grad[np.arange(labels.shape[0]), labels] -= 1.0
        grad *= (weights / weight_sum)[:, np.newaxis]
        accumulate(logits, grad * out.grad)

    out._backward = backward
    return out


def _erf(data: FloatArray) -> FloatArray:
    """Elementwise error function.

    NumPy 2 removed ``numpy.erf``. ``math.erf`` is the same function, applied
    one value at a time so the result stays ``float64``.
    """
    flat = np.asarray(data, dtype=np.float64).reshape(-1)
    out = np.empty(flat.shape[0], dtype=np.float64)
    for index, item in enumerate(flat):
        out[index] = math.erf(float(item))
    return out.reshape(data.shape)


def _sigmoid_array(data: FloatArray) -> FloatArray:
    """Logistic function that does not overflow for large positive inputs."""
    out = np.empty_like(data)
    positive = data >= 0.0
    exp_neg = np.exp(-data[positive])
    out[positive] = 1.0 / (1.0 + exp_neg)
    exp_pos = np.exp(data[~positive])
    out[~positive] = exp_pos / (1.0 + exp_pos)
    return out

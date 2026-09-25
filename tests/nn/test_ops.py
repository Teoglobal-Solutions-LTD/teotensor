"""Analytic gradients must match central differences."""

from __future__ import annotations

import numpy as np
from teotensor.nn.autograd.gradcheck import gradcheck
from teotensor.nn.tensor.core import Tensor
from teotensor.nn.tensor.ops import (
    absolute,
    cross_entropy,
    elu,
    exp,
    gelu,
    leaky_relu,
    log,
    log_softmax,
    matmul,
    maximum,
    mean_all,
    power,
    relu,
    selu,
    sigmoid,
    silu,
    softplus,
    sum_all,
    tanh,
)


def _inputs(*arrays: np.ndarray) -> tuple[Tensor, ...]:
    return tuple(Tensor(array, requires_grad=True) for array in arrays)


def test_elementwise_ops_match_finite_differences() -> None:
    left, right = _inputs(np.array([0.4, -0.7, 1.2]), np.array([0.2, 0.5, -0.3]))
    gradcheck(lambda a, b: (a + b).sum(), (left, right))
    left, right = _inputs(np.array([0.4, -0.7, 1.2]), np.array([0.2, 0.5, -0.3]))
    gradcheck(lambda a, b: (a - b).sum(), (left, right))
    left, right = _inputs(np.array([0.4, -0.7, 1.2]), np.array([0.2, 0.5, -0.3]))
    gradcheck(lambda a, b: (a * b).sum(), (left, right))
    left, right = _inputs(np.array([0.4, -0.7, 1.2]), np.array([0.2, 0.5, -0.3]))
    gradcheck(lambda a, b: (a / b).sum(), (left, right))
    value = Tensor(np.array([0.4, -0.7, 1.2]), requires_grad=True)
    gradcheck(lambda a: (-a).sum(), (value,))
    value = Tensor(np.array([0.4, -0.7, 1.2]), requires_grad=True)
    gradcheck(lambda a: absolute(a).sum(), (value,))
    value = Tensor(np.array([0.4, 1.7, 1.2]), requires_grad=True)
    gradcheck(lambda a: power(a, 3).sum(), (value,))
    value = Tensor(np.array([0.2, -0.4, 0.8]), requires_grad=True)
    gradcheck(lambda a: exp(a).sum(), (value,))
    value = Tensor(np.array([0.2, 0.4, 0.8]), requires_grad=True)
    gradcheck(lambda a: log(a).sum(), (value,))
    value = Tensor(np.array([0.2, -0.4, 0.8]), requires_grad=True)
    gradcheck(lambda a: sum_all(a), (value,))
    value = Tensor(np.array([0.2, -0.4, 0.8]), requires_grad=True)
    gradcheck(lambda a: mean_all(a), (value,))


def test_matmul_and_activations_match_finite_differences() -> None:
    left = Tensor(np.array([[0.2, -0.1], [0.4, 0.3]]), requires_grad=True)
    right = Tensor(np.array([[0.5, -0.2], [0.1, 0.7]]), requires_grad=True)
    gradcheck(lambda a, b: matmul(a, b).sum(), (left, right))
    checks = [
        relu,
        lambda value: leaky_relu(value, 0.2),
        tanh,
        sigmoid,
        lambda value: elu(value, 1.0),
        gelu,
        silu,
        softplus,
        selu,
    ]
    point = np.array([-0.6, 0.4, 1.1])
    for fn in checks:
        value = Tensor(point, requires_grad=True)
        gradcheck(lambda a, fn=fn: fn(a).sum(), (value,))


def test_maximum_and_log_softmax_and_cross_entropy() -> None:
    left = Tensor(np.array([0.2, -0.5, 0.7]), requires_grad=True)
    right = Tensor(np.array([-0.1, 0.4, 0.2]), requires_grad=True)
    gradcheck(lambda a, b: maximum(a, b).sum(), (left, right))
    scores = Tensor(np.array([[0.2, -0.4, 0.1], [0.3, 0.2, -0.5]]), requires_grad=True)
    gradcheck(lambda a: log_softmax(a).sum(), (scores,))
    logits = Tensor(np.array([[0.2, -0.4, 0.1], [0.3, 0.2, -0.5]]), requires_grad=True)
    labels = np.array([0, 2], dtype=np.int64)

    def loss(value: Tensor) -> Tensor:
        return cross_entropy(value, labels)

    gradcheck(loss, (logits,))


def test_tensor_pickle_drops_the_tape() -> None:
    import pickle

    value = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    total = (value * 3).sum()
    total.backward()
    restored = pickle.loads(pickle.dumps(value))
    assert restored._prev == ()
    assert restored.grad is None
    assert np.allclose(restored.data, value.data)

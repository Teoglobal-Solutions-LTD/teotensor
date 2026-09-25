"""Dropout silences neurons only while learning, and scales the rest."""

from __future__ import annotations

import numpy as np
from teotensor.nn.layers.dropout import Dropout
from teotensor.nn.tensor.core import Tensor


def test_dropout_scales_survivors_and_blocks_their_gradient() -> None:
    rng = np.random.RandomState(0)
    layer = Dropout(0.5, rng)
    layer.train()
    incoming = Tensor(np.ones((40, 8)), requires_grad=True)
    outgoing = layer(incoming)
    silenced = outgoing.data == 0.0
    survivors = np.isclose(outgoing.data, 2.0)
    assert silenced.any()
    assert survivors.any()
    assert np.all(silenced | survivors)
    outgoing.sum().backward()
    assert incoming.grad is not None
    assert np.all(incoming.grad[silenced] == 0.0)
    assert np.allclose(incoming.grad[survivors], 2.0)


def test_dropout_is_identity_in_exam_mode() -> None:
    layer = Dropout(0.5, np.random.RandomState(1))
    layer.eval()
    incoming = Tensor(np.ones((4, 3)), requires_grad=True)
    outgoing = layer(incoming)
    assert np.allclose(outgoing.data, 1.0)
    assert outgoing._op != "dropout"

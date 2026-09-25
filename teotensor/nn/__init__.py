"""Neural-network layer: a compact, visible deep-learning core.

Arithmetic goes through :mod:`teotensor.nn.tensor.ops` (NumPy today, a faster
backend later behind the same functions). Models do not draw; they return
traces, reports, and playback frames.
"""

from __future__ import annotations

from teotensor.nn.estimators.mlp import MLPClassifier, MLPRegressor
from teotensor.nn.layers.linear import Linear
from teotensor.nn.layers.sequential import Sequential
from teotensor.nn.optim.adam import Adam
from teotensor.nn.optim.sgd import SGD
from teotensor.nn.tensor.core import Tensor
from teotensor.nn.training.loop import fit_module

__all__ = [
    "SGD",
    "Adam",
    "Linear",
    "MLPClassifier",
    "MLPRegressor",
    "Sequential",
    "Tensor",
    "fit_module",
]

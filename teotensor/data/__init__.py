"""Tables, splits, and minibatches.

Loaders turn a file into a :class:`Dataset`. Splits and batches only move
row numbers. Steps that change values live in :mod:`teotensor.data.steps`.
"""

from teotensor.data.dataset import Dataset, batches, holdout_count
from teotensor.data.sources import from_arrays, read_csv, read_mnist
from teotensor.data.steps import Standardize, scale

__all__ = [
    "Dataset",
    "Standardize",
    "batches",
    "from_arrays",
    "holdout_count",
    "read_csv",
    "read_mnist",
    "scale",
]

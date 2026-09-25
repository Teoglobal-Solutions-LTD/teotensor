"""Readers that fill a :class:`~teotensor.data.dataset.Dataset`."""

from teotensor.data.sources.arrays import from_arrays
from teotensor.data.sources.csv import read_csv
from teotensor.data.sources.mnist import read_mnist

__all__ = ["from_arrays", "read_csv", "read_mnist"]

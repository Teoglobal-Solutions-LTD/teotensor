"""Steps that change the numbers in a :class:`~teotensor.data.dataset.Dataset`.

A step that does not look at other rows (divide by 255) may run before the
split. A step that estimates statistics must be fit on ``train_idx`` only and
then applied to every row, so validation and test see the training rule.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from teotensor.data.dataset import Dataset, IndexArray


def scale(dataset: Dataset, divisor: float = 255.0) -> Dataset:
    """Return a new table with ``features / divisor`` in ``float64``.

    Indices are copied. Later splits of the result do not move the source.
    """
    if divisor == 0.0:
        msg = "divisor must not be 0."
        raise ValueError(msg)
    features = np.asarray(dataset.features, dtype=np.float64) / float(divisor)
    return _with_features(dataset, features)


class Standardize:
    """Subtract the training mean and divide by the training standard deviation.

    ``fit`` records one mean and one scale per column. ``transform`` applies
    that rule to every row. Fit on ``train_idx``, not on the whole table.
    """

    def __init__(self) -> None:
        self.mean_: NDArray[np.float64] | None = None
        self.std_: NDArray[np.float64] | None = None

    def fit(self, dataset: Dataset, indices: IndexArray) -> Standardize:
        """Estimate per-column mean and standard deviation on ``indices``."""
        rows = np.asarray(
            dataset.features[np.asarray(indices, dtype=np.int64)], dtype=np.float64
        )
        self.mean_ = np.mean(rows, axis=0)
        scale_per_column = np.std(rows, axis=0)
        self.std_ = np.maximum(scale_per_column, 1e-8)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """Apply the recorded mean and scale to every row."""
        if self.mean_ is None or self.std_ is None:
            msg = "Call fit before transform."
            raise RuntimeError(msg)
        features = np.asarray(dataset.features, dtype=np.float64)
        shifted = (features - self.mean_) / self.std_
        return _with_features(dataset, shifted)


def _with_features(dataset: Dataset, features: NDArray[np.float64]) -> Dataset:
    """Copy the table's indices onto new feature values."""
    return Dataset(
        features=features,
        targets=dataset.targets,
        sample_weight=dataset.sample_weight,
        feature_names=dataset.feature_names,
        target_name=dataset.target_name,
        image_shape=dataset.image_shape,
        train_idx=np.array(dataset.train_idx, copy=True),
        val_idx=np.array(dataset.val_idx, copy=True),
        test_idx=np.array(dataset.test_idx, copy=True),
    )

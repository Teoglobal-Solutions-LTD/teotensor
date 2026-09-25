"""Build a dataset from arrays that are already in memory."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from teotensor.data.dataset import Dataset


def from_arrays(
    features: NDArray[np.generic],
    targets: NDArray[np.generic] | None = None,
    *,
    feature_names: tuple[str, ...] | None = None,
    target_name: str | None = None,
    image_shape: tuple[int, int] | None = None,
) -> Dataset:
    """Wrap ``features`` and optional ``targets`` as a dataset.

    Every row starts in the training split.
    """
    return Dataset(
        features=np.asarray(features),
        targets=None if targets is None else np.asarray(targets),
        feature_names=feature_names,
        target_name=target_name,
        image_shape=image_shape,
    )

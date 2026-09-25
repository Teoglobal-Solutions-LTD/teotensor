"""One table, whatever file it came from.

Every row has one index. Features, the target, and the sample weight are
looked up by that index. A split is three lists of indices, not three copies
of the matrix. A minibatch is a slice of one of those lists.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.random import Generator, RandomState
from numpy.typing import NDArray

RNG = RandomState | Generator
IndexArray = NDArray[np.int64]


class Dataset:
    """Rows aligned by index.

    Parameters
    ----------
    features : ndarray of shape (n_samples, n_features)
        Inputs. ``uint8`` is allowed until a step promotes the table to
        ``float64``. The model always receives ``float64`` via :meth:`take`.
    targets : ndarray or None
        Labels or numbers as they arrived. ``None`` for unsupervised tables.
    sample_weight : ndarray or None
        One weight per row, or ``None``.
    feature_names : tuple of str or None
        Column names. Generated ``x0``, ``x1``, ... when omitted.
    target_name : str or None
        Name of the answer column.
    image_shape : tuple of int or None
        ``(height, width)`` when each row is one frame laid out in row-major
        order. The network ignores this. A viewer uses it to draw the frame.
    train_idx, val_idx, test_idx : ndarray or None
        Row numbers for each role. Omitted roles start empty, and every row
        starts in ``train_idx``.
    """

    def __init__(
        self,
        features: NDArray[np.generic],
        targets: NDArray[np.generic] | None = None,
        sample_weight: NDArray[np.float64] | None = None,
        feature_names: tuple[str, ...] | None = None,
        target_name: str | None = None,
        image_shape: tuple[int, int] | None = None,
        train_idx: IndexArray | None = None,
        val_idx: IndexArray | None = None,
        test_idx: IndexArray | None = None,
    ) -> None:
        matrix = np.asarray(features)
        if matrix.ndim != 2:
            msg = (
                f"features must have shape (n_samples, n_features); got {matrix.shape}."
            )
            raise ValueError(msg)
        n_rows = int(matrix.shape[0])
        n_features = int(matrix.shape[1])
        if targets is not None and len(targets) != n_rows:
            msg = f"targets has {len(targets)} rows; features has {n_rows}."
            raise ValueError(msg)
        if sample_weight is not None and len(sample_weight) != n_rows:
            msg = f"sample_weight has {len(sample_weight)} rows; features has {n_rows}."
            raise ValueError(msg)
        names = feature_names
        if names is None:
            names = tuple(f"x{i}" for i in range(n_features))
        if len(names) != n_features:
            msg = f"feature_names has {len(names)} entries; features has {n_features}."
            raise ValueError(msg)
        if image_shape is not None:
            height, width = image_shape
            if height * width != n_features:
                msg = f"image_shape {image_shape} does not cover {n_features} features."
                raise ValueError(msg)
        self.features = matrix
        self.targets = None if targets is None else np.asarray(targets)
        self.sample_weight = (
            None
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        self.feature_names = names
        self.target_name = target_name
        self.image_shape = image_shape
        self.train_idx = _as_index(train_idx, n_rows, default_all=True)
        self.val_idx = _as_index(val_idx, n_rows, default_all=False)
        self.test_idx = _as_index(test_idx, n_rows, default_all=False)

    @property
    def n_rows(self) -> int:
        """Number of rows in the whole table, every role included."""
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        """Number of columns the model sees."""
        return int(self.features.shape[1])

    def take(
        self, indices: IndexArray
    ) -> tuple[
        NDArray[np.float64], NDArray[np.generic] | None, NDArray[np.float64] | None
    ]:
        """Return ``float64`` features, targets, and weights for ``indices``.

        Already-``float64`` features are a view. Other dtypes are copied.
        """
        idx = np.asarray(indices, dtype=np.int64)
        if self.features.dtype == np.float64:
            features = self.features[idx]
        else:
            features = np.asarray(self.features[idx], dtype=np.float64)
        targets = None if self.targets is None else self.targets[idx]
        weights = None if self.sample_weight is None else self.sample_weight[idx]
        return features, targets, weights

    def split_tail(self, fraction: float) -> None:
        """Move the tail of ``train_idx`` into ``val_idx``.

        ``fraction == 0`` keeps every training row. The tail is deterministic,
        so a caller that did not shuffle still knows which rows were held out.
        """
        n_val = holdout_count(int(self.train_idx.size), fraction)
        if n_val == 0:
            return
        self.val_idx = np.concatenate([self.val_idx, self.train_idx[-n_val:]])
        self.train_idx = self.train_idx[:-n_val]

    def hold_out(self, fraction: float, rng: RNG, *, stratify: bool = False) -> None:
        """Move a random subset of ``train_idx`` into ``val_idx``.

        With ``stratify=True`` each distinct target gives up the same fraction
        of its rows. Statistics for a later step must be fit on whatever
        remains in ``train_idx``.
        """
        chosen = _choose(self.train_idx, self.targets, fraction, rng, stratify=stratify)
        if chosen.size == 0:
            return
        chosen_set = {int(index) for index in chosen}
        self.val_idx = np.concatenate([self.val_idx, chosen])
        self.train_idx = np.asarray(
            [int(index) for index in self.train_idx if int(index) not in chosen_set],
            dtype=np.int64,
        )


def holdout_count(n_rows: int, fraction: float) -> int:
    """How many rows a fraction holds out, leaving at least one behind."""
    if fraction < 0.0 or fraction >= 1.0:
        msg = f"fraction must satisfy 0 <= f < 1; got {fraction}."
        raise ValueError(msg)
    n_val = int(fraction * n_rows)
    if fraction > 0.0 and n_val == 0 and n_rows > 1:
        n_val = 1
    if n_val >= n_rows:
        n_val = max(n_rows - 1, 0)
    return n_val


def batches(
    indices: IndexArray,
    batch_size: int,
    *,
    shuffle: bool,
    rng: RNG,
) -> Iterator[IndexArray]:
    """Yield index slices. The last slice may be shorter.

    ``shuffle=True`` permutes a copy. The table itself stays in place, and
    features, targets, and weights are cut with the same numbers.
    """
    if batch_size < 1:
        msg = f"batch_size must be >= 1; got {batch_size}."
        raise ValueError(msg)
    order = np.array(indices, dtype=np.int64, copy=True)
    if shuffle:
        _shuffle(rng, order)
    for start in range(0, int(order.size), batch_size):
        yield order[start : start + batch_size]


def _as_index(
    indices: IndexArray | None, n_rows: int, *, default_all: bool
) -> IndexArray:
    """Normalize an index array."""
    if indices is None:
        if default_all:
            return np.arange(n_rows, dtype=np.int64)
        return np.empty(0, dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)


def _choose(
    indices: IndexArray,
    targets: NDArray[np.generic] | None,
    fraction: float,
    rng: RNG,
    *,
    stratify: bool,
) -> IndexArray:
    """Row numbers to move out of the training list."""
    if stratify:
        if targets is None:
            msg = "stratify=True requires targets."
            raise ValueError(msg)
        chosen: list[IndexArray] = []
        labels = targets[indices]
        for label in np.unique(labels):
            group = indices[labels == label]
            count = holdout_count(int(group.size), fraction)
            if count == 0:
                continue
            order = np.array(group, dtype=np.int64, copy=True)
            _shuffle(rng, order)
            chosen.append(order[:count])
        if not chosen:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(chosen)
    count = holdout_count(int(indices.size), fraction)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    order = np.array(indices, dtype=np.int64, copy=True)
    _shuffle(rng, order)
    return order[:count]


def _shuffle(rng: RNG, order: IndexArray) -> None:
    """Shuffle ``order`` in place with either NumPy generator type."""
    if isinstance(rng, Generator):
        rng.shuffle(order)
    else:
        rng.shuffle(order)

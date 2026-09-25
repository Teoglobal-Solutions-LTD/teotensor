"""One table, three index lists, and minibatches that only slice those lists."""

from __future__ import annotations

import numpy as np
from teotensor.data import batches, from_arrays


def main() -> None:
    """Hold out the tail, then print one unshuffled pass of batches."""
    features = np.arange(20, dtype=np.float64).reshape(10, 2)
    targets = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    table = from_arrays(features, targets, feature_names=("x0", "x1"), target_name="y")
    table.split_tail(0.2)
    print("train", table.train_idx.tolist())
    print("val", table.val_idx.tolist())
    rng = np.random.RandomState(0)
    for index, batch in enumerate(batches(table.train_idx, 3, shuffle=False, rng=rng)):
        rows, answers, _weights = table.take(batch)
        print(index, batch.tolist(), rows.shape, answers.tolist())


if __name__ == "__main__":
    main()

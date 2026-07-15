"""Demonstrate the TeoTensor Core API with three dummy estimators.

Run from the repository root after ``pip install -e .``::

    python examples/core_dummy_estimators.py
"""

from __future__ import annotations

import numpy as np
from teotensor.core import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    TransformerMixin,
    check_array,
    check_is_fitted,
    check_X_y,
)


class MeanCenter(BaseEstimator, TransformerMixin):
    """Subtract the column-wise training mean."""

    def __init__(self, with_mean: bool = True) -> None:
        self.with_mean = with_mean

    def fit(self, X, y=None):
        del y
        X_checked = check_array(X, dtype=np.float64)
        self.mean_ = (
            X_checked.mean(axis=0)
            if self.with_mean
            else np.zeros(X_checked.shape[1], dtype=np.float64)
        )
        return self

    def transform(self, X):
        check_is_fitted(self, "mean_")
        return check_array(X, dtype=np.float64) - self.mean_


class MajorityClassifier(BaseEstimator, ClassifierMixin):
    """Predict the most frequent label seen during ``fit``."""

    def __init__(self, random_state: int | None = 0) -> None:
        self.random_state = random_state

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y is required")
        _, y_checked = check_X_y(X, y)
        values, counts = np.unique(y_checked, return_counts=True)
        self.constant_ = values[int(np.argmax(counts))]
        return self

    def predict(self, X):
        check_is_fitted(self, "constant_")
        n = check_array(X).shape[0]
        return np.full(n, fill_value=self.constant_)


class SingleCluster(BaseEstimator, ClusterMixin):
    """Assign every sample to cluster 0."""

    def __init__(self, n_clusters: int = 1) -> None:
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        del y
        X_checked = check_array(X)
        self.labels_ = np.zeros(X_checked.shape[0], dtype=np.int64)
        return self


def main() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0, 0, 1])

    transformer = MeanCenter().fit(X)
    print("params:", transformer.get_params())
    print("centered mean:", MeanCenter().fit_transform(X).mean(axis=0))

    clf = MajorityClassifier().fit(X, y)
    print("accuracy on training labels:", clf.score(X, y))

    labels = SingleCluster().fit_predict(X)
    print("cluster labels:", labels)


if __name__ == "__main__":
    main()

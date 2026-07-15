"""Shared dummy estimators used by core unit tests."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from teotensor.core import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
    SerializableMixin,
    TransformerMixin,
    check_array,
    check_is_fitted,
    check_random_state,
    check_X_y,
)


class DummyTransformer(BaseEstimator, TransformerMixin, SerializableMixin):
    """Toy scaler: subtracts the training mean."""

    def __init__(self, with_mean: bool = True) -> None:
        self.with_mean = with_mean

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        del y  # Unsupervised API consistency.
        X_checked = check_array(X, dtype=np.float64)
        self.n_features_in_ = X_checked.shape[1]
        self.mean_ = (
            X_checked.mean(axis=0)
            if self.with_mean
            else np.zeros(
                X_checked.shape[1],
                dtype=np.float64,
            )
        )
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, ("mean_", "n_features_in_"))
        X_checked = check_array(X, dtype=np.float64)
        if X_checked.shape[1] != self.n_features_in_:
            msg = (
                f"X has {X_checked.shape[1]} features, but "
                f"{type(self).__name__} is expecting {self.n_features_in_}."
            )
            raise ValueError(msg)
        return X_checked - self.mean_


class DummyClassifier(BaseEstimator, ClassifierMixin):
    """Toy constant classifier predicting the most frequent training label."""

    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        if y is None:
            msg = "DummyClassifier requires y."
            raise ValueError(msg)
        X_checked, y_checked = check_X_y(X, y)
        self.n_features_in_ = X_checked.shape[1]
        # Touch the RNG so the hyperparameter is exercised.
        check_random_state(self.random_state)
        values, counts = np.unique(y_checked, return_counts=True)
        self.classes_ = values
        self.constant_ = values[int(np.argmax(counts))]
        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        check_is_fitted(self, ("constant_", "n_features_in_"))
        X_checked = check_array(X)
        if X_checked.shape[1] != self.n_features_in_:
            msg = (
                f"X has {X_checked.shape[1]} features, but "
                f"{type(self).__name__} is expecting {self.n_features_in_}."
            )
            raise ValueError(msg)
        return np.full(shape=X_checked.shape[0], fill_value=self.constant_)


class DummyRegressor(BaseEstimator, RegressorMixin):
    """Toy constant regressor predicting the training-target mean."""

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        if y is None:
            msg = "DummyRegressor requires y."
            raise ValueError(msg)
        X_checked, y_checked = check_X_y(X, y, y_numeric=True, dtype=np.float64)
        self.n_features_in_ = X_checked.shape[1]
        self.constant_ = float(np.mean(y_checked))
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, ("constant_", "n_features_in_"))
        X_checked = check_array(X, dtype=np.float64)
        return np.full(
            shape=X_checked.shape[0],
            fill_value=self.constant_,
            dtype=np.float64,
        )


class DummyClusterer(BaseEstimator, ClusterMixin):
    """Toy clusterer assigning every sample to cluster 0."""

    def __init__(self, n_clusters: int = 1) -> None:
        self.n_clusters = n_clusters

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        del y
        X_checked = check_array(X)
        if self.n_clusters < 1:
            msg = "n_clusters must be >= 1."
            raise ValueError(msg)
        self.n_features_in_ = X_checked.shape[1]
        self.labels_ = np.zeros(X_checked.shape[0], dtype=np.int64)
        return self


class NestedEstimator(BaseEstimator):
    """Estimator holding another estimator to exercise nested params."""

    def __init__(self, transformer: DummyTransformer | None = None) -> None:
        self.transformer = DummyTransformer() if transformer is None else transformer

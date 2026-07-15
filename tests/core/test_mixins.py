"""Tests for estimator mixins composed on top of BaseEstimator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from teotensor.core import NotFittedError
from tests.core.dummy_estimators import (
    DummyClassifier,
    DummyClusterer,
    DummyRegressor,
    DummyTransformer,
)


def test_transformer_fit_transform_and_repr() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    est = DummyTransformer(with_mean=True)
    Xt = est.fit_transform(X)
    np.testing.assert_allclose(Xt.mean(axis=0), [0.0, 0.0])
    assert "with_mean=True" in repr(est)


def test_classifier_score_and_predict() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = DummyClassifier(random_state=0).fit(X, y)
    # Majority class is ambiguous (tie); argmax picks the first max -> 0.
    assert clf.score(X, np.array([0, 0, 0, 0])) == 1.0


def test_regressor_score_perfect_constant() -> None:
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([2.0, 2.0, 2.0])
    reg = DummyRegressor().fit(X, y)
    assert reg.score(X, y) == pytest.approx(1.0)


def test_cluster_fit_predict() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels = DummyClusterer().fit_predict(X)
    np.testing.assert_array_equal(labels, [0, 0])


def test_serializable_roundtrip(tmp_path: Path) -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    est = DummyTransformer().fit(X)
    path = est.save(tmp_path / "dummy.pkl")
    loaded = DummyTransformer.load(path)
    np.testing.assert_allclose(loaded.transform(X), est.transform(X))


def test_transform_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        DummyTransformer().transform([[1.0, 2.0]])

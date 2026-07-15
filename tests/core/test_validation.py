"""Tests for core validation helpers."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator, RandomState
from teotensor.core import (
    NotFittedError,
    check_array,
    check_is_fitted,
    check_random_state,
    check_X_y,
)
from tests.core.dummy_estimators import DummyTransformer


def test_check_array_accepts_2d_list() -> None:
    arr = check_array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    assert arr.shape == (2, 2)
    assert arr.dtype == np.float64


def test_check_array_rejects_1d_when_ensure_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        check_array([1.0, 2.0], ensure_2d=True)


def test_check_array_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        check_array([[1.0, np.nan]])


def test_check_X_y_enforces_matching_sample_counts() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        check_X_y([[1.0], [2.0]], [0])


def test_check_random_state_variants() -> None:
    assert isinstance(check_random_state(None), RandomState)
    assert isinstance(check_random_state(0), RandomState)
    rng = RandomState(1)
    assert check_random_state(rng) is rng
    gen = Generator(np.random.PCG64(2))
    assert check_random_state(gen) is gen


def test_check_is_fitted_raises_before_fit() -> None:
    est = DummyTransformer()
    with pytest.raises(NotFittedError, match="not fitted"):
        check_is_fitted(est, "mean_")

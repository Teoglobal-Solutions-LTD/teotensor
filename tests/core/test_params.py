"""Tests for get_params / set_params and nested parameter routing."""

from __future__ import annotations

import pytest
from tests.core.dummy_estimators import DummyTransformer, NestedEstimator


def test_get_params_returns_init_hyperparameters() -> None:
    est = DummyTransformer(with_mean=False)
    params = est.get_params(deep=False)
    assert params == {"with_mean": False}


def test_set_params_updates_and_returns_self() -> None:
    est = DummyTransformer(with_mean=True)
    out = est.set_params(with_mean=False)
    assert out is est
    assert est.with_mean is False


def test_set_params_rejects_unknown_parameter() -> None:
    est = DummyTransformer()
    with pytest.raises(ValueError, match="Invalid parameter"):
        est.set_params(unknown=1)


def test_nested_get_and_set_params() -> None:
    nested = NestedEstimator(transformer=DummyTransformer(with_mean=True))
    params = nested.get_params(deep=True)
    assert params["transformer__with_mean"] is True
    nested.set_params(transformer__with_mean=False)
    assert nested.transformer.with_mean is False

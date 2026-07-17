"""Tests for ObservabilityMixin via dummy estimators."""

from __future__ import annotations

import numpy as np
import pytest
from teotensor.artifacts import Observation, export_markdown_summary
from teotensor.core import BaseEstimator
from tests.observability.dummy_models import (
    DummyClusterObserver,
    DummyIterativeEstimator,
)


def test_base_estimator_requires_report_and_diagnose() -> None:
    class Bare(BaseEstimator):
        def __init__(self, alpha: float = 1.0) -> None:
            self.alpha = alpha

    est = Bare()
    with pytest.raises(NotImplementedError, match="report"):
        est.report()
    with pytest.raises(NotImplementedError, match="diagnose"):
        est.diagnose()
    assert est.trace() == []
    assert est.visualize() == []


def test_dummy_iterative_observe_bundle() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    model = DummyIterativeEstimator(n_iter=4, learning_rate=0.2).fit(X)
    observation = model.observe()
    assert isinstance(observation, Observation)
    assert observation.report is not None
    assert observation.report.metrics["n_iter"] == 4
    assert len(observation.traces) == 1
    assert observation.traces[0].name == "loss"
    assert len(observation.traces[0].records) == 4
    assert observation.figures[0].kind == "line"
    md = export_markdown_summary(observation)
    assert "DummyIterativeEstimator report" in md


def test_dummy_cluster_diagnostics_and_fit_predict() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    model = DummyClusterObserver().fit(X)
    labels = model.fit_predict(X)
    assert labels[-1] == -1
    diags = model.diagnose()
    assert diags[0].code == "noise_points_present"
    assert diags[0].severity == "warning"
    report = model.report()
    assert report.metrics["n_noise"] == 1
    assert report.tables[0].title == "Cluster sizes"

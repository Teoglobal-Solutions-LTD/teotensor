"""Tests for :class:`teotensor.classical.decomposition.PCA`."""

from __future__ import annotations

import numpy as np
import pytest
from teotensor.artifacts import export_html
from teotensor.classical import PCA
from teotensor.core.exceptions import NotFittedError


def _correlated_blob(
    n_samples: int = 200,
    n_features: int = 5,
    *,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    # Low effective rank: 2 latent factors + noise.
    z = rng.normal(size=(n_samples, 2))
    loading = rng.normal(size=(2, n_features))
    return z @ loading + 0.05 * rng.normal(size=(n_samples, n_features))


def test_fit_transform_shapes_and_attributes() -> None:
    X = _correlated_blob()
    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    assert Z.shape == (X.shape[0], 2)
    assert pca.components_.shape == (2, X.shape[1])
    assert pca.explained_variance_.shape == (2,)
    assert pca.explained_variance_ratio_.shape == (2,)
    assert pca.mean_.shape == (X.shape[1],)
    assert pca.n_components_ == 2
    assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0) or (
        pca.explained_variance_ratio_.sum() < 1.0 + 1e-9
    )


def test_inverse_transform_reconstruction_improves_with_more_components() -> None:
    X = _correlated_blob()
    err_low = PCA(n_components=1).fit(X).reconstruction_error_
    err_high = PCA(n_components=4).fit(X).reconstruction_error_
    assert err_high < err_low
    pca = PCA(n_components=None).fit(X)
    X_hat = pca.inverse_transform(pca.transform(X))
    assert np.allclose(X, X_hat, atol=1e-8)


def test_float_n_components_variance_threshold() -> None:
    X = _correlated_blob()
    pca = PCA(n_components=0.9).fit(X)
    assert pca.n_components_ >= 1
    assert float(np.sum(pca.explained_variance_ratio_)) >= 0.9 - 1e-12


def test_fit_transform_matches_transform() -> None:
    X = _correlated_blob()
    pca = PCA(n_components=3)
    z1 = pca.fit_transform(X)
    z2 = pca.transform(X)
    assert np.allclose(z1, z2)


def test_not_fitted_raises() -> None:
    pca = PCA(n_components=2)
    with pytest.raises(NotFittedError):
        pca.transform([[1.0, 2.0, 3.0]])


def test_invalid_n_components() -> None:
    X = _correlated_blob(n_features=4)
    with pytest.raises(ValueError, match="larger than"):
        PCA(n_components=10).fit(X)
    with pytest.raises(ValueError, match=">= 1"):
        PCA(n_components=0).fit(X)
    with pytest.raises(ValueError, match="0 < n_components"):
        PCA(n_components=1.5).fit(X)


def test_feature_mismatch() -> None:
    X = _correlated_blob(n_features=4)
    pca = PCA(n_components=2).fit(X)
    with pytest.raises(ValueError, match="features"):
        pca.transform(np.zeros((3, 5)))


def test_observability_contract_and_html(tmp_path) -> None:
    X = _correlated_blob()
    pca = PCA(n_components=2).fit(X)
    report = pca.report()
    assert "reconstruction_mse" in report.metrics
    assert report.tables
    diags = pca.diagnose()
    assert diags
    figs = pca.visualize()
    assert {f.title for f in figs} >= {
        "Scree (explained variance ratio)",
        "Cumulative explained variance",
    }
    obs = pca.observe()
    path = tmp_path / "pca.html"
    html = export_html(obs, path)
    assert path.is_file()
    assert "PCA report" in html
    assert "<svg" in html
    assert "data-tip=" in html

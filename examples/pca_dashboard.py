"""Fit PCA on a synthetic low-rank blob and open an HTML observation dashboard.

Run from the repository root after ``pip install -e .``::

    python examples/pca_dashboard.py
"""

from __future__ import annotations

import numpy as np
from teotensor.artifacts import export_html
from teotensor.classical import PCA


def make_blob(n_samples: int = 400, n_features: int = 8, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    latent = rng.normal(size=(n_samples, 3))
    loading = rng.normal(size=(3, n_features))
    return latent @ loading + 0.1 * rng.normal(size=(n_samples, n_features))


def main() -> None:
    X = make_blob()
    pca = PCA(n_components=0.95).fit(X)
    observation = pca.observe()
    export_html(observation, open_browser=True)
    print(
        f"PCA kept {pca.n_components_} components "
        f"({float(np.sum(pca.explained_variance_ratio_)):.1%} variance), "
        f"reconstruction MSE={pca.reconstruction_error_:.4g}."
    )
    print("Opened PCA observation dashboard in your browser.")


if __name__ == "__main__":
    main()

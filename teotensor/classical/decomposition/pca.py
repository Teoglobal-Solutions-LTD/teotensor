"""Principal Component Analysis via economy SVD of the centered data matrix.

PCA is observability-first: after ``fit``, callers get scree / cumulative
explained-variance figures, reconstruction metrics, and diagnostics through the
shared artifact contract — not ad-hoc matplotlib inside the estimator.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from teotensor.artifacts.types import Diagnostic, FigureSpec, Report, TableSpec
from teotensor.artifacts.weights import (
    decode_params,
    encode_params,
    read_ttw,
    write_ttw,
)
from teotensor.core.base import BaseEstimator
from teotensor.core.mixins import SerializableMixin, TransformerMixin
from teotensor.core.validation import check_array, check_is_fitted

SvdSolver = Literal["full"]


def _svd_flip(
    u: NDArray[np.floating[Any]],
    vt: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Deterministic sign flip (sklearn-compatible, U-based)."""
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, np.arange(u.shape[1])])
    signs[signs == 0.0] = 1.0
    u = u * signs
    vt = vt * signs[:, np.newaxis]
    return u, vt


def _resolve_n_components(
    n_components: int | float | None,
    *,
    n_samples: int,
    n_features: int,
    explained_variance_ratio: NDArray[np.floating[Any]],
) -> int:
    """Resolve ``n_components`` to an integer rank."""
    max_rank = min(n_samples, n_features)
    if n_components is None:
        return max_rank
    if isinstance(n_components, float):
        if not 0.0 < n_components <= 1.0:
            msg = (
                "Float n_components must satisfy 0 < n_components <= 1; "
                f"got {n_components}."
            )
            raise ValueError(msg)
        cumulative = np.cumsum(explained_variance_ratio)
        # Smallest k with cumulative ratio >= threshold.
        return int(np.searchsorted(cumulative, n_components, side="left") + 1)
    if isinstance(n_components, bool):
        msg = "n_components must be None, int, or float in (0, 1]; got bool."
        raise TypeError(msg)
    if isinstance(n_components, int | np.integer):
        k = int(n_components)
        if k < 1:
            msg = f"n_components must be >= 1; got {k}."
            raise ValueError(msg)
        if k > max_rank:
            msg = (
                f"n_components={k} is larger than "
                f"min(n_samples, n_features)={max_rank}."
            )
            raise ValueError(msg)
        return k
    msg = (
        "n_components must be None, int, or float in (0, 1]; "
        f"got {type(n_components).__name__}."
    )
    raise TypeError(msg)


class PCA(SerializableMixin, TransformerMixin, BaseEstimator):
    """Principal Component Analysis (linear dimensionality reduction).

    Centers ``X``, then computes an economy SVD ``X_c = U S V^T``. Principal
    axes are the rows of ``V^T`` (stored as ``components_``). Explained
    variances use the unbiased divisor ``n_samples - 1`` when ``n_samples > 1``.

    Parameters
    ----------
    n_components : int, float, or None, default=None
        Number of components to keep.

        - ``None`` — keep ``min(n_samples, n_features)``.
        - ``int`` — keep that many components.
        - ``float`` in ``(0, 1]`` — keep the smallest number of components that
          explain at least that fraction of variance.
    copy : bool, default=True
        If False, center in-place when ``X`` is already a writable float64
        ndarray (callers must not rely on ``X`` remaining unchanged).
    svd_solver : {"full"}, default="full"
        Solver backend. Only full economy SVD is available in this baseline;
        randomized / accelerated solvers come later behind the same API.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space (rows are eigenvectors of the
        covariance matrix).
    explained_variance_ : ndarray of shape (n_components,)
        Variance explained by each selected component.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Fraction of total variance explained by each selected component.
    singular_values_ : ndarray of shape (n_components,)
        Singular values corresponding to the selected components.
    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
    n_components_ : int
        Estimated number of components.
    n_features_in_ : int
        Number of features seen during ``fit``.
    n_samples_ : int
        Number of samples seen during ``fit``.
    noise_variance_ : float
        Average variance of discarded components (0 when all are kept).
    reconstruction_error_ : float
        Mean squared reconstruction error of the training matrix under the
        selected truncation.
    """

    def __init__(
        self,
        n_components: int | float | None = None,
        *,
        copy: bool = True,
        svd_solver: SvdSolver = "full",
    ) -> None:
        self.n_components = n_components
        self.copy = copy
        self.svd_solver = svd_solver

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> PCA:
        """Fit the model by computing SVD of the centered training matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for API consistency with supervised estimators.

        Returns
        -------
        self
            Fitted estimator.
        """
        del y
        if self.svd_solver != "full":
            msg = f"Unsupported svd_solver={self.svd_solver!r}; use 'full'."
            raise ValueError(msg)

        X_arr = check_array(X, dtype=np.float64, copy=self.copy)
        n_samples, n_features = X_arr.shape
        if n_samples < 1 or n_features < 1:
            msg = f"PCA requires a non-empty 2-D array; got shape {X_arr.shape}."
            raise ValueError(msg)

        self.n_samples_ = n_samples
        self.n_features_in_ = n_features
        self.mean_ = X_arr.mean(axis=0)
        X_centered = X_arr - self.mean_

        # Feature-wise std for diagnostics (population ddof=0 on centered data).
        self._feature_std_ = np.sqrt(np.mean(X_centered**2, axis=0))

        if self.svd_solver == "full":
            # Economy SVD: shapes (n, k), (k,), (k, p) with k = min(n, p).
            u_raw, singular_values, vt_raw = np.linalg.svd(
                X_centered, full_matrices=False
            )
            _u, vt = _svd_flip(
                np.asarray(u_raw, dtype=np.float64),
                np.asarray(vt_raw, dtype=np.float64),
            )
            singular_values = np.asarray(singular_values, dtype=np.float64)
        else:  # pragma: no cover - guarded above
            msg = f"Unsupported svd_solver={self.svd_solver!r}."
            raise ValueError(msg)

        # Unbiased sample variance along principal axes.
        dof = max(n_samples - 1, 1)
        explained_variance = (singular_values**2) / dof
        total_var = float(np.sum(explained_variance))
        if total_var <= 0.0:
            explained_variance_ratio = np.zeros_like(explained_variance)
        else:
            explained_variance_ratio = explained_variance / total_var

        n_components = _resolve_n_components(
            self.n_components,
            n_samples=n_samples,
            n_features=n_features,
            explained_variance_ratio=explained_variance_ratio,
        )

        self.n_components_ = n_components
        self.components_ = vt[:n_components].copy()
        self.explained_variance_ = explained_variance[:n_components].copy()
        self.explained_variance_ratio_ = explained_variance_ratio[:n_components].copy()
        self.singular_values_ = singular_values[:n_components].copy()

        if n_components < explained_variance.shape[0]:
            self.noise_variance_ = float(np.mean(explained_variance[n_components:]))
        else:
            self.noise_variance_ = 0.0

        # Exact training reconstruction under the truncated basis.
        x_transformed = X_centered @ self.components_.T
        x_reconstructed = x_transformed @ self.components_ + self.mean_
        self.reconstruction_error_ = float(np.mean((X_arr - x_reconstructed) ** 2))

        # Keep full spectra for scree / cumulative plots (selected + discarded).
        self._full_explained_variance_ratio_ = explained_variance_ratio
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.floating[Any]]:
        """Project data onto the fitted principal axes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.

        Returns
        -------
        ndarray of shape (n_samples, n_components)
            Projection of ``X`` in the principal subspace.
        """
        check_is_fitted(self, ("mean_", "components_"))
        X_arr = check_array(X, dtype=np.float64, copy=self.copy)
        if X_arr.shape[1] != self.n_features_in_:
            msg = (
                f"X has {X_arr.shape[1]} features, but PCA is expecting "
                f"{self.n_features_in_} features as seen in fit."
            )
            raise ValueError(msg)
        return np.asarray((X_arr - self.mean_) @ self.components_.T, dtype=np.float64)

    def inverse_transform(self, X: ArrayLike) -> NDArray[np.floating[Any]]:
        """Map scores back to the original feature space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Data in the principal subspace (as returned by :meth:`transform`).

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Approximate reconstruction in the original space.
        """
        check_is_fitted(self, ("mean_", "components_"))
        X_arr = check_array(X, dtype=np.float64, copy=False)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_components_:
            msg = (
                f"Expected array of shape (n_samples, {self.n_components_}); "
                f"got {X_arr.shape}."
            )
            raise ValueError(msg)
        return np.asarray(X_arr @ self.components_ + self.mean_, dtype=np.float64)

    def report(self) -> Report:
        """Return reconstruction metrics and a per-component variance table."""
        check_is_fitted(self, ("components_", "explained_variance_ratio_"))
        cumulative = np.cumsum(self.explained_variance_ratio_)
        rows = tuple(
            (
                i + 1,
                float(self.explained_variance_[i]),
                float(self.explained_variance_ratio_[i]),
                float(cumulative[i]),
            )
            for i in range(self.n_components_)
        )
        retained = float(np.sum(self.explained_variance_ratio_))
        return Report(
            title="PCA report",
            summary=(
                f"Kept {self.n_components_} / "
                f"{min(self.n_samples_, self.n_features_in_)} components, "
                f"retaining {retained:.1%} of training variance "
                f"(reconstruction MSE={self.reconstruction_error_:.4g})."
            ),
            metrics={
                "n_components": int(self.n_components_),
                "n_features": int(self.n_features_in_),
                "n_samples": int(self.n_samples_),
                "variance_retained": retained,
                "reconstruction_mse": float(self.reconstruction_error_),
                "noise_variance": float(self.noise_variance_),
            },
            extras={
                "metric_help": {
                    "n_components": (
                        "Principal axes kept. Raise this to retain more variance "
                        "(and usually lower reconstruction error)."
                    ),
                    "variance_retained": (
                        "Share of training variance kept by those axes. "
                        "PCA(n_components=0.95) targets ~95% here."
                    ),
                    "reconstruction_mse": (
                        "Average squared error when mapping scores back to "
                        "feature space. Exact 0 only if all components are kept."
                    ),
                    "noise_variance": (
                        "Mean variance of discarded axes — residual energy not "
                        "in the principal subspace."
                    ),
                }
            },
            tables=(
                TableSpec(
                    title="Explained variance by component",
                    columns=(
                        "component",
                        "explained_variance",
                        "explained_variance_ratio",
                        "cumulative_ratio",
                    ),
                    rows=rows,
                ),
            ),
            figures=tuple(self.visualize()),
        )

    def diagnose(self) -> list[Diagnostic]:
        """Flag low-variance features, weak retention, and ill-conditioned spectra."""
        check_is_fitted(self, ("components_", "singular_values_"))
        findings: list[Diagnostic] = []

        eps = 1e-12
        near_zero = int(np.sum(self._feature_std_ <= eps))
        if near_zero:
            findings.append(
                Diagnostic(
                    code="near_zero_variance_features",
                    severity="warning",
                    message=(
                        f"{near_zero} feature(s) have near-zero variance; "
                        "consider dropping or scaling them."
                    ),
                    context={"count": near_zero, "threshold": eps},
                )
            )
        else:
            findings.append(
                Diagnostic(
                    code="feature_scales_ok",
                    severity="info",
                    message="No near-zero-variance features detected.",
                )
            )

        retained = float(np.sum(self.explained_variance_ratio_))
        if retained < 0.5:
            findings.append(
                Diagnostic(
                    code="low_variance_retained",
                    severity="warning",
                    message=(
                        f"Selected components retain only {retained:.1%} of "
                        "training variance; increase n_components or inspect "
                        "the scree plot."
                    ),
                    context={"variance_retained": retained},
                )
            )
        else:
            findings.append(
                Diagnostic(
                    code="variance_retained_ok",
                    severity="info",
                    message=f"Selected components retain {retained:.1%} of variance.",
                    context={"variance_retained": retained},
                )
            )

        if self.singular_values_.size >= 2:
            s_max = float(self.singular_values_[0])
            s_min = float(self.singular_values_[-1])
            cond = float("inf") if s_min <= eps else s_max / s_min
            if cond > 1e8:
                findings.append(
                    Diagnostic(
                        code="ill_conditioned_spectrum",
                        severity="warning",
                        message=(
                            f"Singular-value condition number ≈ {cond:.3g}; "
                            "the trailing components may be numerically fragile."
                        ),
                        context={"condition_number": cond},
                    )
                )

        return findings

    def visualize(self) -> list[FigureSpec]:
        """Return scree and cumulative explained-variance figure specs."""
        check_is_fitted(self, ("_full_explained_variance_ratio_",))
        ratios = self._full_explained_variance_ratio_
        # Prefer the full spectrum for scree readability; fall back to selected.
        y = [float(v) for v in ratios]
        if not y:
            y = [float(v) for v in self.explained_variance_ratio_]
        labels = [str(i + 1) for i in range(len(y))]
        cumulative = np.cumsum(y).tolist()
        return [
            FigureSpec(
                kind="bar",
                title="Scree (explained variance ratio)",
                data={
                    "labels": labels,
                    "y": y,
                    "xlabel": "component",
                    "ylabel": "explained variance ratio",
                },
            ),
            FigureSpec(
                kind="line",
                title="Cumulative explained variance",
                data={
                    "x": list(range(1, len(cumulative) + 1)),
                    "y": cumulative,
                    "xlabel": "n_components",
                    "ylabel": "cumulative explained variance",
                },
            ),
        ]

    def export_weights(self, path: str) -> str:
        """Write axes and means to a ``.ttw`` file for later ``load_weights``.

        Parameters
        ----------
        path : str
            Destination path.

        Returns
        -------
        str
            Absolute path written.
        """
        check_is_fitted(self, ("mean_", "components_"))
        written = write_ttw(
            path,
            model_module=type(self).__module__,
            model_name=type(self).__name__,
            params=encode_params(self.get_params(deep=False)),
            meta={
                "n_samples": int(self.n_samples_),
                "n_features_in": int(self.n_features_in_),
                "n_components": int(self.n_components_),
                "noise_variance": float(self.noise_variance_),
                "reconstruction_error": float(self.reconstruction_error_),
            },
            arrays={
                "components": self.components_,
                "mean": self.mean_,
                "explained_variance": self.explained_variance_,
                "explained_variance_ratio": self.explained_variance_ratio_,
                "singular_values": self.singular_values_,
                "full_explained_variance_ratio": self._full_explained_variance_ratio_,
                "feature_std": self._feature_std_,
            },
        )
        return str(written)

    @classmethod
    def load_weights(cls, path: str) -> PCA:
        """Rebuild a fitted PCA from a ``.ttw`` file.

        Parameters
        ----------
        path : str
            File written by :meth:`export_weights`.

        Returns
        -------
        PCA
            Model whose :meth:`transform` matches the saved axes.
        """
        manifest, arrays = read_ttw(path)
        model = cls(**decode_params(manifest["params"]))
        meta = manifest["meta"]
        model.components_ = np.asarray(arrays["components"], dtype=np.float64)
        model.mean_ = np.asarray(arrays["mean"], dtype=np.float64)
        model.explained_variance_ = np.asarray(
            arrays["explained_variance"], dtype=np.float64
        )
        model.explained_variance_ratio_ = np.asarray(
            arrays["explained_variance_ratio"], dtype=np.float64
        )
        model.singular_values_ = np.asarray(arrays["singular_values"], dtype=np.float64)
        model._full_explained_variance_ratio_ = np.asarray(
            arrays["full_explained_variance_ratio"], dtype=np.float64
        )
        model._feature_std_ = np.asarray(arrays["feature_std"], dtype=np.float64)
        model.n_samples_ = int(meta["n_samples"])
        model.n_features_in_ = int(meta["n_features_in"])
        model.n_components_ = int(meta["n_components"])
        model.noise_variance_ = float(meta["noise_variance"])
        model.reconstruction_error_ = float(meta["reconstruction_error"])
        return model

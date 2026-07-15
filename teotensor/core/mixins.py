"""Reusable estimator mixins for the TeoTensor Core API.

Mixins add narrow capabilities (parameters, transform helpers, scoring,
serialization). Concrete models compose them with :class:`BaseEstimator`
instead of copying shared behaviour.
"""

from __future__ import annotations

import pickle
from inspect import signature
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ParamMixin:
    """Hyperparameter introspection via ``get_params`` / ``set_params``.

    Parameter names are taken from the estimator ``__init__`` signature.
    Nested estimators are supported with the ``component__param`` convention
    when ``deep=True``.
    """

    @classmethod
    def _get_param_names(cls) -> list[str]:
        """Return sorted ``__init__`` parameter names (excluding ``self``)."""
        init = cls.__init__
        if init is object.__init__:
            return []

        params = signature(init).parameters.values()
        names: list[str] = []
        for param in params:
            if param.name == "self":
                continue
            if param.kind == param.VAR_POSITIONAL:
                msg = (
                    f"{cls.__name__}.__init__ must not use *args; "
                    "declare explicit hyperparameters."
                )
                raise RuntimeError(msg)
            if param.kind == param.VAR_KEYWORD:
                msg = (
                    f"{cls.__name__}.__init__ must not use **kwargs; "
                    "declare explicit hyperparameters."
                )
                raise RuntimeError(msg)
            names.append(param.name)
        return sorted(names)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters of nested estimators using the
            ``component__param`` naming scheme.

        Returns
        -------
        dict[str, Any]
            Mapping from parameter name to value.
        """
        out: dict[str, Any] = {}
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and _is_estimator(value):
                deep_items = value.get_params(deep=True).items()
                out.update((f"{key}__{k}", val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params: Any) -> Self:
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : Any
            Estimator parameters. Nested parameters use ``component__param``.

        Returns
        -------
        self
            Estimator instance.

        Raises
        ------
        ValueError
            If a parameter name is unknown.
        """
        if not params:
            return self

        valid_params = self.get_params(deep=True)
        nested_params: dict[str, dict[str, Any]] = {}
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                valid_names = sorted(self.get_params(deep=False))
                msg = (
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {valid_names}."
                )
                raise ValueError(msg)
            if delim:
                nested_params.setdefault(key, {})[sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            nested = getattr(self, key)
            nested.set_params(**sub_params)
        return self


class TransformerMixin:
    """Mixin for transformers that implement ``fit`` and ``transform``."""

    def fit_transform(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        **fit_params: Any,
    ) -> NDArray[Any]:
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like or None, default=None
            Target values (ignored by unsupervised transformers).
        **fit_params : Any
            Extra keyword arguments forwarded to ``fit``.

        Returns
        -------
        ndarray
            Transformed data.
        """
        fitted = self.fit(X, y, **fit_params)  # type: ignore[attr-defined]
        return fitted.transform(X)  # type: ignore[no-any-return]


class ClassifierMixin:
    """Mixin for classifiers that implement ``predict``."""

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Mean accuracy of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like
            Test samples.
        y : array-like
            True labels.

        Returns
        -------
        float
            Fraction of correctly classified samples.
        """
        from teotensor.core.validation import check_array

        y_true = check_array(y, ensure_2d=False, ensure_all_finite=False)
        y_pred = check_array(
            self.predict(X),  # type: ignore[attr-defined]
            ensure_2d=False,
            ensure_all_finite=False,
        )
        if y_true.shape[0] != y_pred.shape[0]:
            msg = (
                "y_true and y_pred have inconsistent lengths: "
                f"{y_true.shape[0]} vs {y_pred.shape[0]}."
            )
            raise ValueError(msg)
        return float(np.mean(y_true == y_pred))


class RegressorMixin:
    """Mixin for regressors that implement ``predict``."""

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Coefficient of determination :math:`R^2` of ``predict(X)``.

        Parameters
        ----------
        X : array-like
            Test samples.
        y : array-like
            True targets.

        Returns
        -------
        float
            :math:`R^2` score.
        """
        from teotensor.core.validation import check_array

        y_true = check_array(
            y,
            ensure_2d=False,
            dtype=np.float64,
            ensure_all_finite=True,
        )
        y_pred = check_array(
            self.predict(X),  # type: ignore[attr-defined]
            ensure_2d=False,
            dtype=np.float64,
            ensure_all_finite=True,
        )
        if y_true.shape != y_pred.shape:
            msg = (
                "y_true and y_pred have inconsistent shapes: "
                f"{y_true.shape} vs {y_pred.shape}."
            )
            raise ValueError(msg)

        residual = y_true - y_pred
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot


class ClusterMixin:
    """Mixin for clusterers that expose fitted ``labels_``."""

    def fit_predict(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        **fit_params: Any,
    ) -> NDArray[Any]:
        """Fit the model and return cluster labels.

        Parameters
        ----------
        X : array-like
            Training samples.
        y : array-like or None, default=None
            Ignored. Present for API consistency.
        **fit_params : Any
            Extra keyword arguments forwarded to ``fit``.

        Returns
        -------
        ndarray
            Cluster labels.
        """
        fitted = self.fit(X, y, **fit_params)  # type: ignore[attr-defined]
        return fitted.labels_  # type: ignore[no-any-return]


class SerializableMixin:
    """Mixin providing simple filesystem serialization via pickle."""

    def save(self, filepath: str | Path) -> Path:
        """Serialize this estimator to ``filepath``.

        Parameters
        ----------
        filepath : str or Path
            Destination path.

        Returns
        -------
        Path
            Absolute path written.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path.resolve()

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        """Load an estimator previously written by :meth:`save`.

        Parameters
        ----------
        filepath : str or Path
            Source path.

        Returns
        -------
        Self
            Restored estimator instance.

        Raises
        ------
        TypeError
            If the unpickled object is not an instance of ``cls``.
        """
        path = Path(filepath)
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            msg = f"Expected pickled {cls.__name__}, got {type(obj).__name__}."
            raise TypeError(msg)
        return obj


def _is_estimator(value: Any) -> bool:
    """Return True if ``value`` exposes sklearn-style parameter methods."""
    if value is None or isinstance(value, type):
        return False
    get_params = getattr(value, "get_params", None)
    set_params = getattr(value, "set_params", None)
    return callable(get_params) and callable(set_params)

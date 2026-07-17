"""Input validation helpers for TeoTensor estimators.

These utilities intentionally live outside individual models so every
estimator can share one validation contract. Hot paths may later gain
accelerated backends without changing the public signatures.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import Generator, RandomState
from numpy.typing import ArrayLike, DTypeLike, NDArray

from teotensor.core.exceptions import NotFittedError

EstimatorT = TypeVar("EstimatorT")
RandomStateLike = int | RandomState | Generator | None


def check_array(
    array: ArrayLike,
    *,
    ensure_2d: bool = True,
    dtype: DTypeLike | None = None,
    ensure_all_finite: bool = True,
    copy: bool = False,
) -> NDArray[Any]:
    """Validate and convert array-like input to a NumPy ndarray.

    Parameters
    ----------
    array : array-like
        Input data.
    ensure_2d : bool, default=True
        If True, require a 2-D array.
    dtype : dtype-like or None, default=None
        Desired NumPy dtype. ``None`` keeps a stable inferred dtype.
    ensure_all_finite : bool, default=True
        If True, reject NaN and infinite values.
    copy : bool, default=False
        If True, always return a copy.

    Returns
    -------
    ndarray
        Validated array.

    Raises
    ------
    ValueError
        If the array has the wrong dimensionality or contains non-finite values.
    TypeError
        If the input cannot be converted to an ndarray.
    """
    try:
        arr = np.asarray(array, dtype=dtype)
    except (TypeError, ValueError) as exc:
        msg = f"Expected array-like input, got {type(array).__name__}."
        raise TypeError(msg) from exc

    if copy:
        arr = np.array(arr, copy=True)

    if arr.ndim == 0:
        msg = f"Expected an array with at least 1 dimension, got scalar {arr!r}."
        raise ValueError(msg)

    if ensure_2d and arr.ndim != 2:
        msg = f"Expected a 2-D array, got array with shape {arr.shape}."
        raise ValueError(msg)

    if ensure_all_finite and not np.isfinite(arr).all():
        msg = "Input contains NaN or infinity."
        raise ValueError(msg)

    return arr


def check_X_y(
    X: ArrayLike,
    y: ArrayLike,
    *,
    ensure_2d: bool = True,
    dtype: DTypeLike | None = None,
    ensure_all_finite: bool = True,
    y_numeric: bool = False,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Validate ``X`` and ``y`` have compatible shapes.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector or matrix.
    ensure_2d : bool, default=True
        Forwarded to :func:`check_array` for ``X``.
    dtype : dtype-like or None, default=None
        Desired dtype for ``X`` (and ``y`` when ``y_numeric`` is True).
    ensure_all_finite : bool, default=True
        Reject non-finite values in ``X`` (and ``y`` when checked).
    y_numeric : bool, default=False
        If True, cast ``y`` with ``dtype`` / float and enforce finite values.

    Returns
    -------
    X_checked : ndarray
        Validated feature matrix.
    y_checked : ndarray
        Validated target array.

    Raises
    ------
    ValueError
        If sample counts of ``X`` and ``y`` disagree.
    """
    X_checked = check_array(
        X,
        ensure_2d=ensure_2d,
        dtype=dtype,
        ensure_all_finite=ensure_all_finite,
    )

    y_dtype: DTypeLike | None = dtype if y_numeric else None
    y_checked = check_array(
        y,
        ensure_2d=False,
        dtype=y_dtype,
        ensure_all_finite=ensure_all_finite if y_numeric else False,
    )
    if y_checked.ndim > 2:
        msg = f"Expected y with at most 2 dimensions, got shape {y_checked.shape}."
        raise ValueError(msg)

    n_samples_X = X_checked.shape[0]
    n_samples_y = y_checked.shape[0]
    if n_samples_X != n_samples_y:
        msg = (
            f"X and y have inconsistent numbers of samples: "
            f"{n_samples_X} vs {n_samples_y}."
        )
        raise ValueError(msg)

    return X_checked, y_checked


def check_random_state(seed: RandomStateLike = None) -> RandomState | Generator:
    """Turn a seed into a NumPy random generator.

    Parameters
    ----------
    seed : None, int, RandomState, or Generator, default=None
        - ``None`` → a fresh ``numpy.random.RandomState`` instance.
        - ``int`` → ``RandomState(seed)``.
        - ``RandomState`` / ``Generator`` → returned as-is.

    Returns
    -------
    RandomState or Generator
        A usable NumPy PRNG.

    Raises
    ------
    ValueError
        If ``seed`` is not one of the accepted types.
    """
    if seed is None:
        return RandomState()
    if isinstance(seed, RandomState | Generator):
        return seed
    if isinstance(seed, int | np.integer):
        return RandomState(int(seed))
    msg = (
        "seed must be None, int, numpy.random.RandomState, or "
        f"numpy.random.Generator; got {type(seed).__name__}."
    )
    raise ValueError(msg)


def check_is_fitted(
    estimator: EstimatorT,
    attributes: str | Sequence[str],
) -> EstimatorT:
    """Confirm that an estimator exposes the required fitted attributes.

    Parameters
    ----------
    estimator : object
        Estimator instance to inspect.
    attributes : str or sequence of str
        Fitted attribute names that must exist (typically ending with ``_``).

    Returns
    -------
    estimator
        The same estimator, for fluent call sites.

    Raises
    ------
    NotFittedError
        If any required attribute is missing.
    """
    names = (attributes,) if isinstance(attributes, str) else tuple(attributes)

    missing = [name for name in names if not hasattr(estimator, name)]
    if missing:
        class_name = type(estimator).__name__
        missing_list = ", ".join(missing)
        msg = (
            f"This {class_name} instance is not fitted yet. Call 'fit' before "
            f"using this estimator. Missing attribute(s): {missing_list}."
        )
        raise NotFittedError(msg)
    return estimator

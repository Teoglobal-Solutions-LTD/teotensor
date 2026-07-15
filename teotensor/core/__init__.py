"""Core layer: base estimators, mixins, parameter handling, and validation.

This layer defines the single, stable API on top of which every other module is
built (``BaseEstimator``, the estimator mixins, ``check_array``/``check_X_y``,
``check_random_state``, and ``get_params``/``set_params``).
"""

from __future__ import annotations

from teotensor.core.base import BaseEstimator
from teotensor.core.exceptions import NotFittedError
from teotensor.core.mixins import (
    ClassifierMixin,
    ClusterMixin,
    ParamMixin,
    RegressorMixin,
    SerializableMixin,
    TransformerMixin,
)
from teotensor.core.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    check_X_y,
)

__all__ = [
    "BaseEstimator",
    "ClassifierMixin",
    "ClusterMixin",
    "NotFittedError",
    "ParamMixin",
    "RegressorMixin",
    "SerializableMixin",
    "TransformerMixin",
    "check_X_y",
    "check_array",
    "check_is_fitted",
    "check_random_state",
]

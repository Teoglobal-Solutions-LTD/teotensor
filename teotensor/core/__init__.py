"""Core layer: base estimators, mixins, parameter handling, and validation.

This layer defines the single, stable API on top of which every other module is
built (``BaseEstimator``, the estimator mixins, ``check_array``/``check_X_y``,
``check_random_state``, and ``get_params``/``set_params``).
"""

from __future__ import annotations

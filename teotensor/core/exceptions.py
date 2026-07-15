"""Core exceptions shared across TeoTensor estimators."""

from __future__ import annotations


class NotFittedError(ValueError, AttributeError):
    """Raised when an estimator is used before it has been fitted.

    Inherits from both ``ValueError`` and ``AttributeError`` so callers can
    catch either family of error depending on context.
    """

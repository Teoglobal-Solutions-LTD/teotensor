"""Base estimator type for all TeoTensor models."""

from __future__ import annotations

from teotensor.core.mixins import ParamMixin


class BaseEstimator(ParamMixin):
    """Base class for all TeoTensor estimators.

    Concrete models should:

    - accept all hyperparameters in ``__init__`` and store them as-is;
    - avoid training or input validation in ``__init__``;
    - name fitted attributes with a trailing underscore (``coef_``,
      ``labels_``, ...).

    ``get_params`` / ``set_params`` are inherited from :class:`ParamMixin`.
    """

    def __repr__(self) -> str:
        """Return a concise constructor-style representation."""
        params = self.get_params(deep=False)
        parts = [f"{name}={value!r}" for name, value in sorted(params.items())]
        inner = ", ".join(parts)
        return f"{type(self).__name__}({inner})"

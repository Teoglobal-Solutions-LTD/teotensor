"""Small hyperparameter objects that are not models.

An activation, a loss, a penalty, an optimizer, a schedule, and an initializer
are components. They remember constructor settings through
:class:`~teotensor.core.mixins.ParamMixin` and they do not write reports.
A model stores them as-is and :func:`~teotensor.core.clone` copies them before
training so the original object does not collect optimizer memory.
"""

from __future__ import annotations

from teotensor.core.mixins import ParamMixin


class Component(ParamMixin):
    """Base class for nested hyperparameters.

    Subclasses declare every setting in ``__init__`` and store it unchanged.
    """

"""Observability layer: the inspection protocol shared by all models.

Wires the artifacts data model into a uniform contract
(``trace``/``report``/``diagnose``/``visualize``/``observe``) so every model in
the framework is inspectable in the same way.
"""

from __future__ import annotations

from teotensor.observability.mixin import ObservabilityMixin

__all__ = ["ObservabilityMixin"]

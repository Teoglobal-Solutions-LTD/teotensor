"""Classical ML layer: decomposition, clustering, trees, ensembles, metrics.

The first vertical slice of real value: sklearn-style estimators that are fully
inspectable through the shared observability contract.
"""

from __future__ import annotations

from teotensor.classical.decomposition import PCA

__all__ = ["PCA"]

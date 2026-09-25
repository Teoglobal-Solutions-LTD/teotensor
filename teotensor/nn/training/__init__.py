"""Training loop: one epoch/batch procedure that writes scalar traces."""

from __future__ import annotations

from teotensor.nn.training.loop import EpochStat, LayerStat, TrainResult, fit_module

__all__ = [
    "EpochStat",
    "LayerStat",
    "TrainResult",
    "fit_module",
]

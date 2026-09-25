"""A stack of modules applied in order."""

from __future__ import annotations

from teotensor.nn.layers.module import Module
from teotensor.nn.tensor.core import Tensor


class Sequential(Module):
    """Apply each child module in order.

    Parameters
    ----------
    layers : sequence of Module
        Pieces of the network, first to last.
    """

    def __init__(self, layers: list[Module] | tuple[Module, ...]) -> None:
        super().__init__()
        self.layers = list(layers)

    def forward(self, value: Tensor) -> Tensor:
        """Pass ``value`` through every layer."""
        current = value
        for layer in self.layers:
            current = layer(current)
        return current

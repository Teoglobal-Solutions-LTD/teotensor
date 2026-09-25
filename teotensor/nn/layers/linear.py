"""Fully connected layer: each output neuron is a weighted sum of its inputs."""

from __future__ import annotations

import numpy as np

from teotensor.nn.layers.module import Module
from teotensor.nn.tensor.core import Parameter, Tensor


class Linear(Module):
    """Affine map ``y = x @ weight + bias``.

    ``weight`` has shape ``(in_features, out_features)``. One column is the
    incoming weights of one output neuron. The bias, when present, is a vector
    of length ``out_features`` and is not included in the weight penalty.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, default=True
        If True, add a learnable bias per output neuron. The flag is kept as
        :attr:`use_bias` so the parameter itself can be named ``bias``.
    """

    def __init__(
        self, in_features: int, out_features: int, *, bias: bool = True
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            msg = (
                "Linear sizes must be positive; "
                f"got in_features={in_features}, out_features={out_features}."
            )
            raise ValueError(msg)
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.weight = Parameter(np.empty((in_features, out_features), dtype=np.float64))
        self.bias = (
            Parameter(np.zeros(out_features, dtype=np.float64)) if bias else None
        )

    def forward(self, value: Tensor) -> Tensor:
        """Apply the affine map to a batch of shape ``(batch, in_features)``."""
        out = value @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

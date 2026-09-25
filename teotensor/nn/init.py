"""How a layer's weights are filled before the first training step.

An initializer is a :class:`~teotensor.nn.component.Component`. Calling it
returns a ``float64`` array of the requested shape. It does not write into
the layer itself.

``AutoInit`` looks at the activation that will sit on top of the layer:

- ReLU, leaky ReLU, ELU, GELU, SiLU → Kaiming normal (fan-in, gain for ReLU)
- SELU → LeCun normal
- everything else, including the output layer → Xavier uniform
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator, RandomState
from numpy.typing import NDArray

from teotensor.nn.component import Component

RNG = RandomState | Generator
FloatArray = NDArray[np.float64]


class Initializer(Component):
    """Fills an array of a given shape from ``fan_in`` and ``fan_out``."""

    def sample(
        self,
        shape: tuple[int, ...],
        rng: RNG,
        *,
        fan_in: int,
        fan_out: int,
        activation: Component | None = None,
    ) -> FloatArray:
        """Return a new array. Subclasses implement the distribution.

        Parameters
        ----------
        shape : tuple of int
            Desired array shape.
        rng : RandomState or Generator
            Source of randomness. The caller owns the stream.
        fan_in, fan_out : int
            Incoming and outgoing width of the linear map.
        activation : Component or None, optional
            Activation that will read this layer. Used by :class:`AutoInit`.
        """
        raise NotImplementedError


class Zeros(Initializer):
    """Fill with zeros. The default for biases."""

    def sample(
        self,
        shape: tuple[int, ...],
        rng: RNG,
        *,
        fan_in: int,
        fan_out: int,
        activation: Component | None = None,
    ) -> FloatArray:
        """Return an array of zeros."""
        del rng, fan_in, fan_out, activation
        return np.zeros(shape, dtype=np.float64)


class XavierUniform(Initializer):
    """Uniform Glorot: ``U(-a, a)`` with ``a = sqrt(6 / (fan_in + fan_out))``."""

    def sample(
        self,
        shape: tuple[int, ...],
        rng: RNG,
        *,
        fan_in: int,
        fan_out: int,
        activation: Component | None = None,
    ) -> FloatArray:
        """Draw a Xavier-uniform array."""
        del activation
        bound = float(np.sqrt(6.0 / float(fan_in + fan_out)))
        return np.asarray(rng.uniform(-bound, bound, size=shape), dtype=np.float64)


class XavierNormal(Initializer):
    """Normal Glorot: ``N(0, sqrt(2 / (fan_in + fan_out)))``."""

    def sample(
        self,
        shape: tuple[int, ...],
        rng: RNG,
        *,
        fan_in: int,
        fan_out: int,
        activation: Component | None = None,
    ) -> FloatArray:
        """Draw a Xavier-normal array."""
        del activation
        std = float(np.sqrt(2.0 / float(fan_in + fan_out)))
        return np.asarray(rng.normal(0.0, std, size=shape), dtype=np.float64)


class KaimingUniform(Initializer):
    """Uniform He init for ReLU: ``U(-a, a)`` with ``a = sqrt(6 / fan_in)``."""

    def sample(
        self,
        shape: tuple[int, ...],
        rng: RNG,
        *,
        fan_in: int,
        fan_out: int,
        activation: Component | None = None,
    ) -> FloatArray:
        """Draw a Kaiming-uniform array."""
        del fan_out, activation
        bound = float(np.sqrt(6.0 / float(fan_in)))
        return np.asarray(rng.uniform(-bound, bound, size=shape), dtype=np.float64)


class KaimingNormal(Initializer):
    """Normal He init for ReLU: ``N(0, sqrt(2 / fan_in))``."""

    def sample(
        self,
        shape: tuple[int, ...],
        rng: RNG,
        *,
        fan_in: int,
        fan_out: int,
        activation: Component | None = None,
    ) -> FloatArray:
        """Draw a Kaiming-normal array."""
        del fan_out, activation
        std = float(np.sqrt(2.0 / float(fan_in)))
        return np.asarray(rng.normal(0.0, std, size=shape), dtype=np.float64)


class LecunNormal(Initializer):
    """Normal LeCun init: ``N(0, sqrt(1 / fan_in))``. Used for SELU."""

    def sample(
        self,
        shape: tuple[int, ...],
        rng: RNG,
        *,
        fan_in: int,
        fan_out: int,
        activation: Component | None = None,
    ) -> FloatArray:
        """Draw a LeCun-normal array."""
        del fan_out, activation
        std = float(np.sqrt(1.0 / float(fan_in)))
        return np.asarray(rng.normal(0.0, std, size=shape), dtype=np.float64)


class Orthogonal(Initializer):
    """Semi-orthogonal matrix from the QR of a Gaussian draw, gain 1."""

    def sample(
        self,
        shape: tuple[int, ...],
        rng: RNG,
        *,
        fan_in: int,
        fan_out: int,
        activation: Component | None = None,
    ) -> FloatArray:
        """Draw an orthogonal array of ``shape`` ``(fan_in, fan_out)``."""
        del fan_in, fan_out, activation
        if len(shape) != 2:
            msg = f"Orthogonal init expects a matrix; got shape {shape}."
            raise ValueError(msg)
        rows, cols = shape
        draw = np.asarray(rng.normal(0.0, 1.0, size=(rows, cols)), dtype=np.float64)
        transposed = rows < cols
        factor = draw.T if transposed else draw
        q_factor, r_factor = np.linalg.qr(factor)
        signs = np.sign(np.diag(r_factor))
        signs[signs == 0.0] = 1.0
        q_factor = q_factor * signs
        matrix = q_factor.T if transposed else q_factor
        return np.asarray(matrix[:rows, :cols], dtype=np.float64)


class AutoInit(Initializer):
    """Pick Kaiming, LeCun, or Xavier from the activation class name."""

    def sample(
        self,
        shape: tuple[int, ...],
        rng: RNG,
        *,
        fan_in: int,
        fan_out: int,
        activation: Component | None = None,
    ) -> FloatArray:
        """Dispatch to the initializer that matches ``activation``."""
        chosen = _for_activation(activation)
        return chosen.sample(
            shape, rng, fan_in=fan_in, fan_out=fan_out, activation=activation
        )


def _for_activation(activation: Component | None) -> Initializer:
    """Return the initializer object for this activation."""
    name = type(activation).__name__ if activation is not None else ""
    if name in {"ReLU", "LeakyReLU", "ELU", "GELU", "SiLU"}:
        return KaimingNormal()
    if name == "SELU":
        return LecunNormal()
    return XavierUniform()


def initializer_name(initializer: Initializer, activation: Component | None) -> str:
    """Human-readable name, expanding :class:`AutoInit` to the choice it makes."""
    if isinstance(initializer, AutoInit):
        return type(_for_activation(activation)).__name__
    return type(initializer).__name__

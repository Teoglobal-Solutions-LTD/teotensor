"""One penalty on large weights.

The formula is::

    alpha * ((1 - l1_ratio) * sum(W**2) + l1_ratio * sum(|W|))

Biases are not included. The caller passes only weight matrices.

For SGD and RMSprop the whole penalty is added to the loss, so its slope is
part of the backward pass. For Adam the squared part is applied as decoupled
decay (``W -= lr * 2 * l2_coeff * W``), which is the same slope as
``l2_coeff * sum(W**2)`` but outside Adam's moving averages. The absolute
part still goes through the loss, because there is no separate decay formula
for it.
"""

from __future__ import annotations

from teotensor.nn.component import Component
from teotensor.nn.tensor.core import Parameter, Tensor
from teotensor.nn.tensor.ops import absolute, add, mul, power, sum_all


class Penalty(Component):
    """Extra term that discourages large weights."""

    def l2_coefficient(self) -> float:
        """Coefficient in front of ``sum(W**2)`` inside the formula."""
        raise NotImplementedError

    def loss_term(self, weights: list[Parameter], *, decoupled_l2: bool) -> Tensor:
        """Return the part of the penalty that must enter the scalar loss.

        Parameters
        ----------
        weights : list of Parameter
            Weight matrices. Biases are not passed in.
        decoupled_l2 : bool
            If True, the squared term is left for the optimizer to apply as
            decay, and this tensor contains only the absolute term.
        """
        raise NotImplementedError


class L2(Penalty):
    """``alpha * sum(W**2)``.

    Parameters
    ----------
    alpha : float, default=1e-4
        Strength of the penalty. ``0`` turns it off.
    """

    def __init__(self, alpha: float = 1e-4) -> None:
        self.alpha = alpha

    def l2_coefficient(self) -> float:
        """Return ``alpha``."""
        return float(self.alpha)

    def loss_term(self, weights: list[Parameter], *, decoupled_l2: bool) -> Tensor:
        """Squared penalty, or zero when Adam applies it as decay."""
        if decoupled_l2 or self.alpha == 0.0 or not weights:
            return Tensor(0.0)
        total = _sum_squares(weights)
        return mul(Tensor(float(self.alpha)), total)


class ElasticNet(Penalty):
    """Mix of squared and absolute penalties.

    Parameters
    ----------
    alpha : float, default=1e-4
        Overall strength.
    l1_ratio : float, default=0.5
        Share of the absolute term. ``0`` is pure L2, ``1`` is pure L1.
    """

    def __init__(self, alpha: float = 1e-4, l1_ratio: float = 0.5) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def l2_coefficient(self) -> float:
        """Return ``alpha * (1 - l1_ratio)``."""
        return float(self.alpha) * (1.0 - float(self.l1_ratio))

    def loss_term(self, weights: list[Parameter], *, decoupled_l2: bool) -> Tensor:
        """Absolute term plus, unless decoupled, the squared term."""
        if not weights or self.alpha == 0.0:
            return Tensor(0.0)
        pieces: list[Tensor] = []
        l1_coeff = float(self.alpha) * float(self.l1_ratio)
        if l1_coeff != 0.0:
            pieces.append(mul(Tensor(l1_coeff), _sum_abs(weights)))
        if not decoupled_l2 and self.l2_coefficient() != 0.0:
            pieces.append(mul(Tensor(self.l2_coefficient()), _sum_squares(weights)))
        if not pieces:
            return Tensor(0.0)
        total = pieces[0]
        for piece in pieces[1:]:
            total = add(total, piece)
        return total


def _sum_squares(weights: list[Parameter]) -> Tensor:
    """``sum`` of ``W**2`` over every weight matrix."""
    total = sum_all(power(weights[0], 2.0))
    for weight in weights[1:]:
        total = add(total, sum_all(power(weight, 2.0)))
    return total


def _sum_abs(weights: list[Parameter]) -> Tensor:
    """``sum`` of ``|W|`` over every weight matrix."""
    total = sum_all(absolute(weights[0]))
    for weight in weights[1:]:
        total = add(total, sum_all(absolute(weight)))
    return total


PENALTIES: dict[str, type[Penalty]] = {"L2": L2, "ElasticNet": ElasticNet}

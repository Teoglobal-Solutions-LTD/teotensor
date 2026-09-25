"""Step-length schedules. Each one returns a multiplier for the base ``lr``.

The multiplier depends on the epoch number (starting at 1) and, for
:class:`Adaptive`, on whether the watched loss improved. Schedules that need
memory keep it on the fitted clone, not on the object the user passed in.
"""

from __future__ import annotations

import math

from teotensor.nn.component import Component


class Schedule(Component):
    """Maps an epoch index to a multiplier of the optimizer's base step."""

    def scale(self, epoch: int, *, improved: bool) -> float:
        """Return a positive multiplier.

        Parameters
        ----------
        epoch : int
            Epoch index, starting at 1.
        improved : bool
            True when the watched loss dropped by more than the tolerance
            this epoch. Only :class:`Adaptive` reads it.
        """
        raise NotImplementedError


class Constant(Schedule):
    """Keep the base step length."""

    def scale(self, epoch: int, *, improved: bool) -> float:
        """Return 1."""
        del epoch, improved
        return 1.0


class InvScaling(Schedule):
    """``epoch ** -power_t``.

    Parameters
    ----------
    power_t : float, default=0.5
        How fast the step shrinks. ``0.5`` is one over square root of the epoch.
    """

    def __init__(self, power_t: float = 0.5) -> None:
        self.power_t = power_t

    def scale(self, epoch: int, *, improved: bool) -> float:
        """Return the inverse-power multiplier."""
        del improved
        return math.pow(float(epoch), -float(self.power_t))


class Adaptive(Schedule):
    """Shrink the step when the loss stops improving.

    Parameters
    ----------
    lr_shrink : float, default=0.5
        Factor applied to the current multiplier after
        ``n_iter_no_change`` epochs without improvement.
    n_iter_no_change : int, default=10
        How many quiet epochs to wait before shrinking.
    """

    def __init__(self, lr_shrink: float = 0.5, n_iter_no_change: int = 10) -> None:
        self.lr_shrink = lr_shrink
        self.n_iter_no_change = n_iter_no_change
        self.current_: float = 1.0
        self.quiet_: int = 0

    def scale(self, epoch: int, *, improved: bool) -> float:
        """Update the running multiplier and return it.

        The quiet counter advances when ``improved`` is false. The first epoch
        does not shrink.
        """
        del epoch
        if improved:
            self.quiet_ = 0
        else:
            self.quiet_ += 1
            if self.quiet_ >= self.n_iter_no_change:
                self.current_ *= float(self.lr_shrink)
                self.quiet_ = 0
        return self.current_


class Cosine(Schedule):
    """Half-cosine from 1 down to 0 over ``t_max`` epochs.

    Parameters
    ----------
    t_max : int, default=200
        Epoch at which the multiplier reaches 0.
    """

    def __init__(self, t_max: int = 200) -> None:
        self.t_max = t_max

    def scale(self, epoch: int, *, improved: bool) -> float:
        """Return the cosine multiplier."""
        del improved
        progress = min(epoch, self.t_max) / float(self.t_max)
        return 0.5 * (1.0 + math.cos(math.pi * progress))


class StepDecay(Schedule):
    """Multiply by ``gamma`` every ``step_size`` epochs.

    Parameters
    ----------
    step_size : int, default=10
        Epochs between drops.
    gamma : float, default=0.1
        Factor applied at each drop.
    """

    def __init__(self, step_size: int = 10, gamma: float = 0.1) -> None:
        self.step_size = step_size
        self.gamma = gamma

    def scale(self, epoch: int, *, improved: bool) -> float:
        """Return ``gamma`` raised to how many drops have happened."""
        del improved
        drops = (epoch - 1) // int(self.step_size)
        return float(self.gamma) ** drops


class Exponential(Schedule):
    """Multiply by ``gamma`` every epoch.

    Parameters
    ----------
    gamma : float, default=0.95
        Per-epoch factor. Epoch 1 keeps the base step.
    """

    def __init__(self, gamma: float = 0.95) -> None:
        self.gamma = gamma

    def scale(self, epoch: int, *, improved: bool) -> float:
        """Return ``gamma ** (epoch - 1)``."""
        del improved
        return float(self.gamma) ** (epoch - 1)


SCHEDULES: dict[str, type[Schedule]] = {
    "Constant": Constant,
    "InvScaling": InvScaling,
    "Adaptive": Adaptive,
    "Cosine": Cosine,
    "StepDecay": StepDecay,
    "Exponential": Exponential,
}

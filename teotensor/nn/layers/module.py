"""A piece of a network: it pushes a tensor forward and may own weights.

A module is not a model. It has no ``fit`` and no report. Models build modules
inside ``fit``, after they know how many input columns the table has.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from teotensor.nn.tensor.core import Parameter, Tensor

FloatArray = NDArray[np.float64]


class Module:
    """Base class for layers and stacks of layers.

    Attributes
    ----------
    training : bool
        ``True`` while learning (dropout draws masks). ``False`` in exam mode.
    """

    def __init__(self) -> None:
        self.training = True

    def forward(self, value: Tensor) -> Tensor:
        """Push ``value`` through this piece.

        Parameters
        ----------
        value : Tensor
            Incoming batch. The leading axis is the batch.

        Returns
        -------
        Tensor
            Outgoing batch.
        """
        raise NotImplementedError

    def __call__(self, value: Tensor) -> Tensor:
        """Call :meth:`forward`."""
        return self.forward(value)

    def train(self) -> Module:
        """Switch this module and every child into learning mode."""
        self.training = True
        for child in self.children():
            child.train()
        return self

    def eval(self) -> Module:
        """Switch this module and every child into exam mode."""
        self.training = False
        for child in self.children():
            child.eval()
        return self

    def children(self) -> list[Module]:
        """Return direct child modules, in attribute order."""
        found: list[Module] = []
        for value in self.__dict__.values():
            if isinstance(value, Module):
                found.append(value)
            elif isinstance(value, list | tuple):
                for item in value:
                    if isinstance(item, Module):
                        found.append(item)
        return found

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Parameter]]:
        """Return ``(dotted name, parameter)`` pairs.

        A parameter stored as ``self.weight`` is named ``weight``. A child
        stored in a list ``self.layers`` contributes ``layers.0.weight``.
        """
        named: list[tuple[str, Parameter]] = []
        for key, value in self.__dict__.items():
            if isinstance(value, Parameter):
                named.append((prefix + key, value))
            elif isinstance(value, Module):
                named.extend(value.named_parameters(prefix + key + "."))
            elif isinstance(value, list | tuple):
                for index, item in enumerate(value):
                    if isinstance(item, Module):
                        named.extend(item.named_parameters(f"{prefix}{key}.{index}."))
        return named

    def parameters(self) -> list[Parameter]:
        """Return every weight tensor owned by this module."""
        return [param for _, param in self.named_parameters()]

    def zero_grad(self) -> None:
        """Forget gradients on every parameter."""
        for param in self.parameters():
            param.zero_grad()

    def state_dict(self) -> dict[str, FloatArray]:
        """Copy parameter numbers into a plain dictionary.

        Values are copies, so the caller can store them without sharing memory
        with the live weights.
        """
        return {
            name: np.array(param.data, dtype=np.float64, copy=True)
            for name, param in self.named_parameters()
        }

    def load_state_dict(self, state: dict[str, FloatArray]) -> None:
        """Copy numbers from :meth:`state_dict` back into the parameters.

        Parameters
        ----------
        state : dict
            Mapping produced by :meth:`state_dict` on a module of the same
            layout.

        Raises
        ------
        KeyError
            If a parameter name is missing.
        ValueError
            If a shape does not match.
        """
        for name, param in self.named_parameters():
            if name not in state:
                msg = f"state_dict is missing {name!r}."
                raise KeyError(msg)
            incoming = np.asarray(state[name], dtype=np.float64)
            if incoming.shape != param.data.shape:
                msg = (
                    f"state_dict[{name!r}] has shape {incoming.shape}; "
                    f"expected {param.data.shape}."
                )
                raise ValueError(msg)
            param.data = incoming

"""Sequence parameters and clone()."""

from __future__ import annotations

from teotensor.core import clone
from teotensor.core.mixins import ParamMixin


class _Slope(ParamMixin):
    def __init__(self, negative_slope: float = 0.01) -> None:
        self.negative_slope = negative_slope


class _Net(ParamMixin):
    def __init__(
        self,
        hidden_activations: tuple[_Slope, ...] | None = None,
    ) -> None:
        self.hidden_activations = (
            (_Slope(),) if hidden_activations is None else hidden_activations
        )


def test_sequence_params_round_trip() -> None:
    net = _Net(hidden_activations=(_Slope(0.01), _Slope(0.2)))
    params = net.get_params()
    assert params["hidden_activations__0__negative_slope"] == 0.01
    assert params["hidden_activations__1__negative_slope"] == 0.2
    net.set_params(hidden_activations__1__negative_slope=0.3)
    assert net.hidden_activations[1].negative_slope == 0.3


def test_clone_copies_nested_components() -> None:
    net = _Net(hidden_activations=(_Slope(0.25),))
    copied = clone(net)
    copied.hidden_activations[0].negative_slope = 0.5
    assert net.hidden_activations[0].negative_slope == 0.25

"""Build a film of one example: signal forward, then error backward.

Neuron frames carry every activation in the layer. The approximate player
draws the first 16 of a wide layer and, into each of those, at most 4
incoming wires, the ones with the largest absolute value. The real-scale
player uses the full vector. This module only records the numbers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from teotensor.artifacts.playback import Playback, PlaybackFrame
from teotensor.nn.layers.activations import ActivationLayer
from teotensor.nn.layers.linear import Linear
from teotensor.nn.layers.sequential import Sequential
from teotensor.nn.losses import Loss
from teotensor.nn.tensor.core import Tensor

MAX_NEURONS = 16
MAX_WIRES = 4
FloatArray = NDArray[np.float64]


def build_playback(
    module: Sequential,
    features: FloatArray,
    *,
    sample_index: int,
    with_backward: bool,
    loss: Loss | None = None,
    target: NDArray[np.generic] | None = None,
) -> Playback:
    """Record frames for one row.

    The module is switched to exam mode so dropout does not flicker.

    Parameters
    ----------
    module : Sequential
        Fitted network.
    features : ndarray of shape (n_features,)
        One example.
    sample_index : int
        Index stored on the film so the player can name the row.
    with_backward : bool
        If True, also record the error walking back. ``loss`` and ``target``
        are required in that case.
    loss : Loss or None, optional
        Loss used to form the scalar error for the backward half.
    target : ndarray or None, optional
        Encoded target for this single example.
    """
    module.eval()
    row = np.asarray(features, dtype=np.float64).reshape(1, -1)
    current = Tensor(row, requires_grad=with_backward)
    snapshots: list[Tensor] = [current]
    linears: list[Linear] = []
    expect_activation = False
    for layer in module.layers:
        current = layer(current)
        if isinstance(layer, Linear):
            linears.append(layer)
            snapshots.append(current)
            expect_activation = True
        elif isinstance(layer, ActivationLayer) and expect_activation:
            snapshots[-1] = current
            expect_activation = False
    layer_sizes = tuple(int(snap.data.shape[1]) for snap in snapshots)
    shown = tuple(min(MAX_NEURONS, size) for size in layer_sizes)
    frames: list[PlaybackFrame] = []
    order = 0
    frames.append(
        _neuron_frame(
            "forward",
            order,
            0,
            snapshots[0].data[0],
            layer_sizes[0],
            shown[0],
            "Input neurons light up with this example.",
        )
    )
    order += 1
    for index, linear in enumerate(linears):
        dest = index + 1
        frames.append(
            _wire_frame(
                "forward",
                order,
                dest,
                _forward_edges(
                    snapshots[index].data[0],
                    linear.weight.data,
                    shown[index],
                    shown[dest],
                ),
                layer_sizes[dest],
                shown[dest],
                f"Signal travels along the wires into layer {dest}.",
            )
        )
        order += 1
        frames.append(
            _neuron_frame(
                "forward",
                order,
                dest,
                snapshots[dest].data[0],
                layer_sizes[dest],
                shown[dest],
                f"Layer {dest} neurons light up.",
            )
        )
        order += 1
    if with_backward:
        if loss is None or target is None:
            msg = "Backward playback requires a loss and a target."
            raise ValueError(msg)
        module.zero_grad()
        current.zero_grad()
        error = loss.forward(snapshots[-1], target, None)
        error.backward()
        for dest in range(len(snapshots) - 1, 0, -1):
            grad = snapshots[dest].grad
            values = np.zeros(layer_sizes[dest]) if grad is None else grad[0]
            frames.append(
                _neuron_frame(
                    "backward",
                    order,
                    dest,
                    values,
                    layer_sizes[dest],
                    shown[dest],
                    f"Error reaches the neurons of layer {dest}.",
                )
            )
            order += 1
            weight_grad = linears[dest - 1].weight.grad
            edges = (
                ()
                if weight_grad is None
                else _backward_edges(weight_grad, shown[dest - 1], shown[dest])
            )
            frames.append(
                _wire_frame(
                    "backward",
                    order,
                    dest,
                    edges,
                    layer_sizes[dest],
                    shown[dest],
                    f"Error tells the wires into layer {dest} how to change.",
                )
            )
            order += 1
        input_grad = snapshots[0].grad
        input_values = np.zeros(layer_sizes[0]) if input_grad is None else input_grad[0]
        frames.append(
            _neuron_frame(
                "backward",
                order,
                0,
                input_values,
                layer_sizes[0],
                shown[0],
                "Error reaches the input neurons.",
            )
        )
    return Playback(
        sample_index=sample_index,
        layer_sizes=layer_sizes,
        frames=tuple(frames),
    )


def _neuron_frame(
    phase: str,
    order: int,
    layer: int,
    values: NDArray[np.float64],
    n_total: int,
    n_shown: int,
    caption: str,
) -> PlaybackFrame:
    """One neuron beat."""
    if phase not in {"forward", "backward"}:
        msg = f"Unknown phase {phase!r}."
        raise ValueError(msg)
    return PlaybackFrame(
        phase=phase,  # type: ignore[arg-type]
        order=order,
        kind="neurons",
        layer=layer,
        caption=caption,
        values=tuple(float(item) for item in np.asarray(values, dtype=np.float64)),
        n_total=n_total,
        n_shown=n_shown,
    )


def _wire_frame(
    phase: str,
    order: int,
    layer: int,
    edges: tuple[tuple[int, int, float], ...],
    n_total: int,
    n_shown: int,
    caption: str,
) -> PlaybackFrame:
    """One wire beat."""
    if phase not in {"forward", "backward"}:
        msg = f"Unknown phase {phase!r}."
        raise ValueError(msg)
    return PlaybackFrame(
        phase=phase,  # type: ignore[arg-type]
        order=order,
        kind="wires",
        layer=layer,
        caption=caption,
        edges=edges,
        n_total=n_total,
        n_shown=n_shown,
    )


def _forward_edges(
    incoming: FloatArray,
    weight: FloatArray,
    n_src: int,
    n_dst: int,
) -> tuple[tuple[int, int, float], ...]:
    """Up to 4 strongest ``activation * weight`` wires per drawn destination."""
    chosen: list[tuple[int, int, float]] = []
    for dst in range(n_dst):
        scores = [
            (src, float(incoming[src] * weight[src, dst])) for src in range(n_src)
        ]
        scores.sort(key=lambda item: abs(item[1]), reverse=True)
        for src, value in scores[:MAX_WIRES]:
            chosen.append((src, dst, value))
    return tuple(chosen)


def _backward_edges(
    weight_grad: FloatArray,
    n_src: int,
    n_dst: int,
) -> tuple[tuple[int, int, float], ...]:
    """Up to 4 strongest ``d(loss)/d(weight)`` wires per drawn destination."""
    chosen: list[tuple[int, int, float]] = []
    for dst in range(n_dst):
        scores = [(src, float(weight_grad[src, dst])) for src in range(n_src)]
        scores.sort(key=lambda item: abs(item[1]), reverse=True)
        for src, value in scores[:MAX_WIRES]:
            chosen.append((src, dst, value))
    return tuple(chosen)

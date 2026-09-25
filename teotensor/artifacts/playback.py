"""Ordered frames of one example moving forward and backward through a network.

The model fills these records. The studio player chooses colors from
``phase``. Forward and backward do not share a palette.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Phase = Literal["forward", "backward"]
FrameKind = Literal["neurons", "wires"]


@dataclass(frozen=True)
class PlaybackFrame:
    """One beat of the film: either neurons in a layer, or the wires into it.

    Parameters
    ----------
    phase : {"forward", "backward"}
        Whether this beat is the signal going in or the error coming back.
    order : int
        Play order, starting at 0.
    kind : {"neurons", "wires"}
        Neurons carry ``values``. Wires carry ``edges``.
    layer : int
        Neuron layer index. For wires, the destination layer.
    caption : str
        Sentence the player can show under the picture.
    values : tuple of float
        Every neuron in the layer. Empty for a wire frame. An approximate
        picture draws the first ``n_shown`` entries.
    edges : tuple of (src, dst, value)
        Shown wires. Indices are positions among the drawn neurons, not among
        the full layer. Empty for a neuron frame.
    n_total : int
        How many neurons the destination layer really has.
    n_shown : int
        How many of those neurons are drawn. The first ``n_shown`` are kept.
    """

    phase: Phase
    order: int
    kind: FrameKind
    layer: int
    caption: str
    values: tuple[float, ...] = ()
    edges: tuple[tuple[int, int, float], ...] = ()
    n_total: int = 0
    n_shown: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "type": "PlaybackFrame",
            "phase": self.phase,
            "order": self.order,
            "kind": self.kind,
            "layer": self.layer,
            "caption": self.caption,
            "values": [float(item) for item in self.values],
            "edges": [
                [int(src), int(dst), float(value)] for src, dst, value in self.edges
            ],
            "n_total": self.n_total,
            "n_shown": self.n_shown,
        }


@dataclass(frozen=True)
class Playback:
    """A whole film for one example.

    Parameters
    ----------
    sample_index : int
        Row of the table this film was built from.
    layer_sizes : tuple of int
        Full width of each layer, including input and output.
    frames : tuple of PlaybackFrame
        Beats in play order.
    """

    sample_index: int
    layer_sizes: tuple[int, ...]
    frames: tuple[PlaybackFrame, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "type": "Playback",
            "sample_index": self.sample_index,
            "layer_sizes": list(self.layer_sizes),
            "frames": [frame.to_dict() for frame in self.frames],
        }

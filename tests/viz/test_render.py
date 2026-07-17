"""Tests for the matplotlib FigureSpec renderer."""

from __future__ import annotations

import pytest
from teotensor.artifacts import FigureSpec

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from teotensor.viz import render_figure  # noqa: E402


def test_render_line_figure() -> None:
    fig = render_figure(
        FigureSpec(
            kind="line",
            title="Loss",
            data={"x": [1, 2, 3], "y": [1.0, 0.5, 0.25], "xlabel": "step"},
        )
    )
    assert fig.axes[0].get_title() == "Loss"


def test_render_bar_and_scatter() -> None:
    bar = render_figure(
        FigureSpec(
            kind="bar",
            title="Sizes",
            data={"labels": ["0", "1"], "y": [3, 5]},
        )
    )
    scatter = render_figure(
        FigureSpec(
            kind="scatter",
            title="Points",
            data={"x": [0.0, 1.0], "y": [1.0, 2.0]},
        )
    )
    assert bar.axes[0].get_title() == "Sizes"
    assert scatter.axes[0].get_title() == "Points"


def test_render_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        render_figure(
            FigureSpec(kind="heatmap", title="Nope", data={})  # type: ignore[arg-type]
        )

"""Matplotlib renderer for declarative :class:`~teotensor.artifacts.FigureSpec`."""

from __future__ import annotations

from typing import Any

from teotensor.artifacts.types import FigureSpec


def render_figure(spec: FigureSpec) -> Any:
    """Render a :class:`~teotensor.artifacts.FigureSpec` with matplotlib.

    Parameters
    ----------
    spec : FigureSpec
        Declarative figure description produced by a model.

    Returns
    -------
    matplotlib.figure.Figure
        Concrete figure instance.

    Raises
    ------
    ImportError
        If matplotlib is not installed (install ``teotensor[viz]``).
    ValueError
        If ``spec.kind`` is unsupported or required data keys are missing.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        msg = (
            "matplotlib is required to render FigureSpec objects. "
            "Install it with: pip install 'teotensor[viz]'."
        )
        raise ImportError(msg) from exc

    figsize = spec.options.get("figsize", (6.0, 4.0))
    fig, ax = plt.subplots(figsize=figsize)
    data = spec.data

    if spec.kind == "line":
        x = data.get("x")
        y = data.get("y")
        if y is None:
            msg = "FigureSpec(kind='line') requires data['y']."
            raise ValueError(msg)
        if x is None:
            ax.plot(y)
        else:
            ax.plot(x, y)
    elif spec.kind == "bar":
        labels = data.get("labels")
        y = data.get("y")
        if labels is None or y is None:
            msg = "FigureSpec(kind='bar') requires data['labels'] and data['y']."
            raise ValueError(msg)
        ax.bar(labels, y)
    elif spec.kind == "scatter":
        x = data.get("x")
        y = data.get("y")
        if x is None or y is None:
            msg = "FigureSpec(kind='scatter') requires data['x'] and data['y']."
            raise ValueError(msg)
        ax.scatter(x, y)
    else:
        msg = f"Unsupported FigureSpec.kind: {spec.kind!r}."
        raise ValueError(msg)

    ax.set_title(spec.title)
    if "xlabel" in data:
        ax.set_xlabel(str(data["xlabel"]))
    if "ylabel" in data:
        ax.set_ylabel(str(data["ylabel"]))
    fig.tight_layout()
    return fig

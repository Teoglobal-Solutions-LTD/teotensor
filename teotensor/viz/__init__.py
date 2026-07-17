"""Visualization layer: renderers for figure and table specifications.

Turns ``FigureSpec``/``TableSpec`` artifacts into concrete figures. Kept
separate from models so multiple render backends can coexist without touching
model code.
"""

from __future__ import annotations

from teotensor.viz.matplotlib_backend import render_figure

__all__ = ["render_figure"]

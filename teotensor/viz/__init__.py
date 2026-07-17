"""Visualization layer: optional render backends for figure specifications.

The **primary** visualization surface of TeoTensor is HTML
(``teotensor.artifacts.export_html``): styled, printable, interactive
dashboards. This package hosts secondary backends such as matplotlib for
notebooks and custom scripts. Models never import renderers directly — they
emit ``FigureSpec`` artifacts only.
"""

from __future__ import annotations

from teotensor.viz.matplotlib_backend import render_figure

__all__ = ["render_figure"]

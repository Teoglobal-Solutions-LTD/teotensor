"""Visualization layer: renderers for figure and table specifications.

Turns ``FigureSpec``/``TableSpec`` artifacts into concrete figures. Kept
separate from models so multiple render backends can coexist without touching
model code.
"""

from __future__ import annotations

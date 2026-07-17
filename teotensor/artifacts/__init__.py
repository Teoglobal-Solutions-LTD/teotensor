"""Artifacts layer: structured, serializable outputs of models.

Defines the observability data model (``Trace``, ``Diagnostic``, ``Report``,
``Observation``, ``FigureSpec``, ``TableSpec``) plus exporters such as
``export_json`` and ``export_markdown_summary``. Models describe *what* to show;
renderers decide *how* to show it.
"""

from __future__ import annotations

from teotensor.artifacts.export import export_json, export_markdown_summary
from teotensor.artifacts.html import export_html
from teotensor.artifacts.types import (
    Diagnostic,
    FigureSpec,
    Observation,
    Report,
    TableSpec,
    Trace,
)

__all__ = [
    "Diagnostic",
    "FigureSpec",
    "Observation",
    "Report",
    "TableSpec",
    "Trace",
    "export_html",
    "export_json",
    "export_markdown_summary",
]

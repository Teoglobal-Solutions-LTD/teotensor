"""Artifacts layer: structured, serializable outputs of models.

Defines the observability data model (``Trace``, ``Diagnostic``, ``Report``,
``Observation``, ``FigureSpec``, ``TableSpec``) plus exporters such as
``export_json`` and ``export_markdown_summary``. Models describe *what* to show;
renderers decide *how* to show it.
"""

from __future__ import annotations

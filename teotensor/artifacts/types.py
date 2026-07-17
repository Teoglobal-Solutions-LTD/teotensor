"""Serializable observability artifacts produced by TeoTensor models.

Models describe *what* to inspect; exporters and renderers decide *how* to
present it. Artifacts never draw plots themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from teotensor.artifacts.serialize import to_jsonable

Severity = Literal["info", "warning", "error"]
FigureKind = Literal["line", "bar", "scatter"]


@dataclass(frozen=True)
class Trace:
    """Named iterative / temporal record stream (for example a loss curve).

    Parameters
    ----------
    name : str
        Human-readable trace identifier.
    records : list of dict
        Ordered step records. Keys are field names; values are scalars.
    metadata : dict, optional
        Extra free-form context.
    """

    name: str
    records: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "type": "Trace",
            "name": self.name,
            "records": to_jsonable(list(self.records)),
            "metadata": to_jsonable(self.metadata),
        }


@dataclass(frozen=True)
class Diagnostic:
    """Single diagnostic finding about a fitted model or dataset.

    Parameters
    ----------
    code : str
        Stable machine-readable code (for example ``"high_noise_ratio"``).
    severity : {"info", "warning", "error"}
        Severity level.
    message : str
        Human-readable explanation.
    context : dict, optional
        Structured details supporting the finding.
    """

    code: str
    severity: Severity
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "type": "Diagnostic",
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "context": to_jsonable(self.context),
        }


@dataclass(frozen=True)
class TableSpec:
    """Tabular artifact ready for Markdown / HTML rendering.

    Parameters
    ----------
    title : str
        Table title.
    columns : list of str
        Column headers.
    rows : list of list
        Row values aligned with ``columns``.
    """

    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "type": "TableSpec",
            "title": self.title,
            "columns": list(self.columns),
            "rows": to_jsonable([list(row) for row in self.rows]),
        }


@dataclass(frozen=True)
class FigureSpec:
    """Declarative figure description (never a live matplotlib object).

    Parameters
    ----------
    kind : {"line", "bar", "scatter"}
        Renderer-supported plot kind.
    title : str
        Figure title.
    data : dict
        Plot payload. Common keys: ``x``, ``y``, ``xlabel``, ``ylabel``,
        ``labels`` (for bar charts).
    options : dict, optional
        Renderer hints (for example ``{"figsize": [6, 4]}``).
    """

    kind: FigureKind
    title: str
    data: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "type": "FigureSpec",
            "kind": self.kind,
            "title": self.title,
            "data": to_jsonable(self.data),
            "options": to_jsonable(self.options),
        }


@dataclass(frozen=True)
class Report:
    """Structured model report combining metrics, tables, and figures.

    Parameters
    ----------
    title : str
        Report title.
    summary : str
        Short prose summary.
    metrics : dict, optional
        Scalar metrics.
    tables : list of TableSpec, optional
        Tabular sections.
    figures : list of FigureSpec, optional
        Declarative figures.
    extras : dict, optional
        Extension bag for model-specific fields.
    """

    title: str
    summary: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    tables: tuple[TableSpec, ...] = ()
    figures: tuple[FigureSpec, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "type": "Report",
            "title": self.title,
            "summary": self.summary,
            "metrics": to_jsonable(self.metrics),
            "tables": [table.to_dict() for table in self.tables],
            "figures": [figure.to_dict() for figure in self.figures],
            "extras": to_jsonable(self.extras),
        }


@dataclass(frozen=True)
class Observation:
    """Bundle returned by :meth:`~teotensor.observability.ObservabilityMixin.observe`.

    Parameters
    ----------
    report : Report or None
        Model report, if available.
    diagnostics : list of Diagnostic
        Diagnostic findings.
    traces : list of Trace
        Iterative traces.
    figures : list of FigureSpec
        Figures collected from ``visualize`` (and optionally the report).
    """

    report: Report | None = None
    diagnostics: tuple[Diagnostic, ...] = ()
    traces: tuple[Trace, ...] = ()
    figures: tuple[FigureSpec, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "type": "Observation",
            "report": None if self.report is None else self.report.to_dict(),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "traces": [item.to_dict() for item in self.traces],
            "figures": [item.to_dict() for item in self.figures],
        }

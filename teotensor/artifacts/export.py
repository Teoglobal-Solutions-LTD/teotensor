"""Export helpers for observability artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from teotensor.artifacts.serialize import to_jsonable
from teotensor.artifacts.types import (
    Diagnostic,
    FigureSpec,
    Observation,
    Report,
    TableSpec,
    Trace,
)

ArtifactLike = (
    Trace | Diagnostic | TableSpec | FigureSpec | Report | Observation | dict[str, Any]
)


def export_json(
    obj: ArtifactLike,
    filepath: str | Path | None = None,
    *,
    indent: int = 2,
) -> str:
    """Serialize an artifact (or mapping) to JSON.

    Parameters
    ----------
    obj : artifact or dict
        Object supporting ``to_dict()``, or an already-built mapping.
    filepath : str or Path or None, default=None
        If given, write JSON to this path as well.
    indent : int, default=2
        JSON indentation.

    Returns
    -------
    str
        JSON document.
    """
    payload = obj.to_dict() if hasattr(obj, "to_dict") else to_jsonable(obj)
    text = json.dumps(payload, indent=indent, ensure_ascii=False, sort_keys=True)
    if filepath is not None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return text


def export_markdown_summary(
    obj: Report | Observation,
    filepath: str | Path | None = None,
) -> str:
    """Render a human-readable Markdown summary of a report or observation.

    Parameters
    ----------
    obj : Report or Observation
        Source artifact.
    filepath : str or Path or None, default=None
        If given, write Markdown to this path as well.

    Returns
    -------
    str
        Markdown document.
    """
    if isinstance(obj, Observation):
        text = _observation_markdown(obj)
    elif isinstance(obj, Report):
        text = _report_markdown(obj)
    else:
        msg = f"Expected Report or Observation, got {type(obj).__name__}."
        raise TypeError(msg)

    if filepath is not None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return text


def _observation_markdown(observation: Observation) -> str:
    lines: list[str] = ["# Observation", ""]
    if observation.report is not None:
        lines.append(_report_markdown(observation.report))
        lines.append("")
    if observation.diagnostics:
        lines.append("## Diagnostics")
        lines.append("")
        for item in observation.diagnostics:
            lines.append(f"- **{item.severity}** `{item.code}`: {item.message}")
        lines.append("")
    if observation.traces:
        lines.append("## Traces")
        lines.append("")
        for trace in observation.traces:
            n = len(trace.records)
            lines.append(f"- `{trace.name}` ({n} records)")
        lines.append("")
    if observation.figures:
        lines.append("## Figures")
        lines.append("")
        for figure in observation.figures:
            lines.append(f"- `{figure.kind}`: {figure.title}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _report_markdown(report: Report) -> str:
    lines: list[str] = [f"# {report.title}", ""]
    if report.summary:
        lines.append(report.summary)
        lines.append("")
    if report.metrics:
        lines.append("## Metrics")
        lines.append("")
        for key, value in sorted(report.metrics.items()):
            lines.append(f"- **{key}**: {value}")
        lines.append("")
    for table in report.tables:
        lines.append(f"## {table.title}")
        lines.append("")
        lines.append("| " + " | ".join(table.columns) + " |")
        lines.append("| " + " | ".join("---" for _ in table.columns) + " |")
        for row in table.rows:
            cells = [str(cell) for cell in row]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    if report.figures:
        lines.append("## Figures")
        lines.append("")
        for figure in report.figures:
            lines.append(f"- `{figure.kind}`: {figure.title}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

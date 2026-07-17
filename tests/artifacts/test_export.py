"""Tests for artifact serialization and export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from teotensor.artifacts import (
    Diagnostic,
    FigureSpec,
    Observation,
    Report,
    TableSpec,
    Trace,
    export_html,
    export_json,
    export_markdown_summary,
)


def test_trace_to_dict_and_json_roundtrip(tmp_path: Path) -> None:
    trace = Trace(
        name="loss",
        records=({"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.25}),
        metadata={"source": "unit"},
    )
    payload = trace.to_dict()
    assert payload["type"] == "Trace"
    assert payload["records"][0]["loss"] == 0.5

    text = export_json(trace, tmp_path / "trace.json")
    assert '"name": "loss"' in text
    assert (tmp_path / "trace.json").is_file()


def test_report_markdown_and_numpy_metrics() -> None:
    report = Report(
        title="Demo",
        summary="A short summary.",
        metrics={"inertia": np.float64(1.25), "k": 3},
        tables=(
            TableSpec(
                title="Sizes",
                columns=("label", "count"),
                rows=((0, 10), (1, 5)),
            ),
        ),
        figures=(FigureSpec(kind="line", title="Loss", data={"y": [1.0, 0.5]}),),
    )
    md = export_markdown_summary(report)
    assert "# Demo" in md
    assert "**inertia**: 1.25" in md
    assert "| 0 | 10 |" in md


def test_observation_markdown_includes_diagnostics() -> None:
    observation = Observation(
        report=Report(title="R", summary="ok"),
        diagnostics=(
            Diagnostic(
                code="noise",
                severity="warning",
                message="noise present",
            ),
        ),
        traces=(Trace(name="loss", records=({"step": 1, "loss": 1.0},)),),
        figures=(
            FigureSpec(kind="bar", title="Sizes", data={"labels": ["0"], "y": [1]}),
        ),
    )
    md = export_markdown_summary(observation)
    assert "## Diagnostics" in md
    assert "`noise`" in md
    assert "## Traces" in md
    assert "## Figures" in md


def test_export_html_dashboard_contains_svg_and_metrics(tmp_path: Path) -> None:
    observation = Observation(
        report=Report(
            title="Dash",
            summary="Browser view",
            metrics={"final_loss": 0.42},
        ),
        diagnostics=(Diagnostic(code="ok", severity="info", message="all good"),),
        figures=(
            FigureSpec(
                kind="line",
                title="Loss",
                data={"x": [1, 2], "y": [1.0, 0.5], "xlabel": "step"},
            ),
        ),
    )
    path = tmp_path / "dash.html"
    html_doc = export_html(observation, path)
    assert path.is_file()
    assert "TeoTensor" in html_doc
    assert "final_loss" in html_doc
    assert "<svg" in html_doc
    assert "polyline" in html_doc
    assert "diag-info" in html_doc

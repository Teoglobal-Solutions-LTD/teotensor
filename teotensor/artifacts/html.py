"""HTML dashboard export for observability artifacts.

Produces a self-contained HTML document (inline CSS + SVG charts). No image
files and no JavaScript CDN dependency — open the file in any browser.
"""

from __future__ import annotations

import html
import webbrowser
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from teotensor.artifacts.types import (
    Diagnostic,
    FigureSpec,
    Observation,
    Report,
    TableSpec,
    Trace,
)


def export_html(
    obj: Report | Observation,
    filepath: str | Path | None = None,
    *,
    open_browser: bool = False,
) -> str:
    """Render a self-contained HTML dashboard for a report or observation.

    Parameters
    ----------
    obj : Report or Observation
        Source artifact.
    filepath : str or Path or None, default=None
        Destination path. Required when ``open_browser`` is True unless you
        want an automatic temporary file.
    open_browser : bool, default=False
        If True, write the document (to ``filepath`` or a temp file) and open
        it in the default browser.

    Returns
    -------
    str
        Full HTML document.

    Raises
    ------
    TypeError
        If ``obj`` is neither a Report nor an Observation.
    """
    if isinstance(obj, Observation):
        document = _observation_html(obj)
    elif isinstance(obj, Report):
        document = _report_page_html(obj, diagnostics=(), traces=())
    else:
        msg = f"Expected Report or Observation, got {type(obj).__name__}."
        raise TypeError(msg)

    target: Path | None = Path(filepath) if filepath is not None else None
    if open_browser and target is None:
        with NamedTemporaryFile(
            mode="w",
            suffix=".html",
            prefix="teotensor-obs-",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(document)
            target = Path(handle.name)
    elif target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(document, encoding="utf-8")

    if open_browser:
        if target is None:  # pragma: no cover - guarded above
            msg = "open_browser=True requires a writable path."
            raise RuntimeError(msg)
        webbrowser.open(target.resolve().as_uri())

    return document


def _observation_html(observation: Observation) -> str:
    report = observation.report
    title = report.title if report is not None else "Observation"
    summary = report.summary if report is not None else ""
    metrics = report.metrics if report is not None else {}
    tables = report.tables if report is not None else ()
    # Prefer explicit observation figures; fall back to report figures.
    figures = observation.figures or (report.figures if report is not None else ())
    return _page(
        title=title,
        summary=summary,
        metrics=metrics,
        diagnostics=observation.diagnostics,
        traces=observation.traces,
        tables=tables,
        figures=figures,
    )


def _report_page_html(
    report: Report,
    *,
    diagnostics: tuple[Diagnostic, ...],
    traces: tuple[Trace, ...],
) -> str:
    return _page(
        title=report.title,
        summary=report.summary,
        metrics=report.metrics,
        diagnostics=diagnostics,
        traces=traces,
        tables=report.tables,
        figures=report.figures,
    )


def _page(
    *,
    title: str,
    summary: str,
    metrics: dict[str, Any],
    diagnostics: tuple[Diagnostic, ...],
    traces: tuple[Trace, ...],
    tables: tuple[TableSpec, ...],
    figures: tuple[FigureSpec, ...],
) -> str:
    safe_title = html.escape(title)
    body_parts: list[str] = [
        "<header class='hero'>",
        "<p class='brand'>TeoTensor</p>",
        f"<h1>{safe_title}</h1>",
    ]
    if summary:
        body_parts.append(f"<p class='lede'>{html.escape(summary)}</p>")
    body_parts.append("</header>")

    if metrics:
        body_parts.append("<section>")
        body_parts.append("<h2>Metrics</h2>")
        body_parts.append("<div class='metrics'>")
        for key, value in sorted(metrics.items()):
            body_parts.append(
                "<div class='metric'>"
                f"<span class='metric-key'>{html.escape(str(key))}</span>"
                f"<span class='metric-val'>{html.escape(str(value))}</span>"
                "</div>"
            )
        body_parts.append("</div></section>")

    if diagnostics:
        body_parts.append("<section>")
        body_parts.append("<h2>Diagnostics</h2>")
        body_parts.append("<ul class='diagnostics'>")
        for item in diagnostics:
            body_parts.append(
                f"<li class='diag diag-{html.escape(item.severity)}'>"
                f"<span class='badge'>{html.escape(item.severity)}</span>"
                f"<code>{html.escape(item.code)}</code>"
                f"<span>{html.escape(item.message)}</span>"
                "</li>"
            )
        body_parts.append("</ul></section>")

    if traces:
        body_parts.append("<section>")
        body_parts.append("<h2>Traces</h2>")
        for trace in traces:
            body_parts.append(_trace_block(trace))
        body_parts.append("</section>")

    for table in tables:
        body_parts.append("<section>")
        body_parts.append(f"<h2>{html.escape(table.title)}</h2>")
        body_parts.append(_table_html(table))
        body_parts.append("</section>")

    if figures:
        body_parts.append("<section>")
        body_parts.append("<h2>Figures</h2>")
        body_parts.append("<div class='figures'>")
        for figure in figures:
            body_parts.append("<figure class='chart'>")
            body_parts.append(f"<figcaption>{html.escape(figure.title)}</figcaption>")
            body_parts.append(_figure_svg(figure))
            body_parts.append("</figure>")
        body_parts.append("</div></section>")

    return _wrap(safe_title, "\n".join(body_parts))


def _trace_block(trace: Trace) -> str:
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(v))}</td>" for v in record.values())
        + "</tr>"
        for record in trace.records[:50]
    )
    headers = ""
    if trace.records:
        headers = "".join(
            f"<th>{html.escape(str(key))}</th>" for key in trace.records[0]
        )
    more = ""
    if len(trace.records) > 50:
        more = f"<p class='muted'>Showing 50 of {len(trace.records)} records.</p>"
    return (
        f"<div class='trace'><h3>{html.escape(trace.name)}</h3>"
        f"<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"
        f"{more}</div>"
    )


def _table_html(table: TableSpec) -> str:
    head = "".join(f"<th>{html.escape(col)}</th>" for col in table.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in table.rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _figure_svg(spec: FigureSpec) -> str:
    """Render a FigureSpec as an inline SVG chart."""
    width, height = 640, 320
    pad_l, pad_r, pad_t, pad_b = 56, 24, 24, 48
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    accent = "#0f766e"
    ink = "#1c1917"
    grid = "#e7e5e4"

    try:
        if spec.kind == "line":
            return _svg_line(
                spec,
                width,
                height,
                pad_l,
                pad_r,
                pad_t,
                pad_b,
                plot_w,
                plot_h,
                accent,
                ink,
                grid,
            )
        if spec.kind == "bar":
            return _svg_bar(
                spec,
                width,
                height,
                pad_l,
                pad_r,
                pad_t,
                pad_b,
                plot_w,
                plot_h,
                accent,
                ink,
                grid,
            )
        if spec.kind == "scatter":
            return _svg_scatter(
                spec,
                width,
                height,
                pad_l,
                pad_r,
                pad_t,
                pad_b,
                plot_w,
                plot_h,
                accent,
                ink,
                grid,
            )
    except (TypeError, ValueError, KeyError):
        pass
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}'>"
        f"<text x='24' y='40' fill='{ink}'>Unable to render "
        f"{html.escape(spec.kind)} chart.</text></svg>"
    )


def _as_floats(values: Any) -> list[float]:
    if values is None:
        msg = "missing series"
        raise ValueError(msg)
    return [float(v) for v in values]


def _svg_frame(
    width: int,
    height: int,
    pad_l: int,
    pad_t: int,
    plot_w: int,
    plot_h: int,
    ink: str,
    grid: str,
    xlabel: str,
    ylabel: str,
) -> str:
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
        f"role='img'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='#ffffff'/>",
    ]
    for i in range(5):
        y = pad_t + plot_h * i / 4
        parts.append(
            f"<line x1='{pad_l}' y1='{y:.1f}' x2='{pad_l + plot_w}' "
            f"y2='{y:.1f}' stroke='{grid}' stroke-width='1'/>"
        )
    parts.append(
        f"<rect x='{pad_l}' y='{pad_t}' width='{plot_w}' height='{plot_h}' "
        f"fill='none' stroke='{ink}' stroke-width='1.2'/>"
    )
    if xlabel:
        parts.append(
            f"<text x='{pad_l + plot_w / 2:.1f}' y='{height - 12}' "
            f"text-anchor='middle' fill='{ink}' font-size='13' "
            f"font-family='Space Grotesk, Segoe UI, sans-serif'>"
            f"{html.escape(xlabel)}</text>"
        )
    if ylabel:
        parts.append(
            f"<text x='16' y='{pad_t + plot_h / 2:.1f}' text-anchor='middle' "
            f"fill='{ink}' font-size='13' "
            f"font-family='Space Grotesk, Segoe UI, sans-serif' "
            f"transform='rotate(-90 16 {pad_t + plot_h / 2:.1f})'>"
            f"{html.escape(ylabel)}</text>"
        )
    return "\n".join(parts)


def _svg_line(
    spec: FigureSpec,
    width: int,
    height: int,
    pad_l: int,
    pad_r: int,
    pad_t: int,
    pad_b: int,
    plot_w: int,
    plot_h: int,
    accent: str,
    ink: str,
    grid: str,
) -> str:
    del pad_r, pad_b
    y = _as_floats(spec.data.get("y"))
    x_raw = spec.data.get("x")
    x = _as_floats(x_raw) if x_raw is not None else [float(i) for i in range(len(y))]
    if len(x) != len(y) or not y:
        msg = "line chart requires aligned x/y"
        raise ValueError(msg)
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    if max_x == min_x:
        max_x = min_x + 1.0
    if max_y == min_y:
        max_y = min_y + 1.0

    def px(xv: float) -> float:
        return pad_l + (xv - min_x) / (max_x - min_x) * plot_w

    def py(yv: float) -> float:
        return pad_t + (1.0 - (yv - min_y) / (max_y - min_y)) * plot_h

    points = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in zip(x, y, strict=True))
    parts = [
        _svg_frame(
            width,
            height,
            pad_l,
            pad_t,
            plot_w,
            plot_h,
            ink,
            grid,
            str(spec.data.get("xlabel", "")),
            str(spec.data.get("ylabel", "")),
        ),
        f"<polyline fill='none' stroke='{accent}' stroke-width='2.5' "
        f"points='{points}'/>",
    ]
    for a, b in zip(x, y, strict=True):
        parts.append(
            f"<circle cx='{px(a):.1f}' cy='{py(b):.1f}' r='3.5' fill='{accent}'/>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_bar(
    spec: FigureSpec,
    width: int,
    height: int,
    pad_l: int,
    pad_r: int,
    pad_t: int,
    pad_b: int,
    plot_w: int,
    plot_h: int,
    accent: str,
    ink: str,
    grid: str,
) -> str:
    del pad_r, pad_b
    labels = [str(v) for v in spec.data.get("labels", [])]
    y = _as_floats(spec.data.get("y"))
    if not labels or len(labels) != len(y):
        msg = "bar chart requires labels and y"
        raise ValueError(msg)
    max_y = max(y) if y else 1.0
    if max_y == 0:
        max_y = 1.0
    n = len(y)
    gap = plot_w * 0.08 / max(n, 1)
    bar_w = (plot_w - gap * (n + 1)) / max(n, 1)
    parts = [
        _svg_frame(
            width,
            height,
            pad_l,
            pad_t,
            plot_w,
            plot_h,
            ink,
            grid,
            str(spec.data.get("xlabel", "")),
            str(spec.data.get("ylabel", "")),
        )
    ]
    for i, (label, value) in enumerate(zip(labels, y, strict=True)):
        bh = (value / max_y) * plot_h
        x = pad_l + gap + i * (bar_w + gap)
        y0 = pad_t + plot_h - bh
        parts.append(
            f"<rect x='{x:.1f}' y='{y0:.1f}' width='{bar_w:.1f}' height='{bh:.1f}' "
            f"fill='{accent}' rx='4'/>"
        )
        parts.append(
            f"<text x='{x + bar_w / 2:.1f}' y='{pad_t + plot_h + 18}' "
            f"text-anchor='middle' fill='{ink}' font-size='12' "
            f"font-family='Space Grotesk, Segoe UI, sans-serif'>"
            f"{html.escape(label)}</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_scatter(
    spec: FigureSpec,
    width: int,
    height: int,
    pad_l: int,
    pad_r: int,
    pad_t: int,
    pad_b: int,
    plot_w: int,
    plot_h: int,
    accent: str,
    ink: str,
    grid: str,
) -> str:
    del pad_r, pad_b
    x = _as_floats(spec.data.get("x"))
    y = _as_floats(spec.data.get("y"))
    if len(x) != len(y) or not x:
        msg = "scatter requires aligned x/y"
        raise ValueError(msg)
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    if max_x == min_x:
        max_x = min_x + 1.0
    if max_y == min_y:
        max_y = min_y + 1.0

    def px(xv: float) -> float:
        return pad_l + (xv - min_x) / (max_x - min_x) * plot_w

    def py(yv: float) -> float:
        return pad_t + (1.0 - (yv - min_y) / (max_y - min_y)) * plot_h

    parts = [
        _svg_frame(
            width,
            height,
            pad_l,
            pad_t,
            plot_w,
            plot_h,
            ink,
            grid,
            str(spec.data.get("xlabel", "")),
            str(spec.data.get("ylabel", "")),
        )
    ]
    for a, b in zip(x, y, strict=True):
        parts.append(
            f"<circle cx='{px(a):.1f}' cy='{py(b):.1f}' r='5' "
            f"fill='{accent}' fill-opacity='0.85'/>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _wrap(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title} · TeoTensor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap"
        rel="stylesheet"/>
  <style>
    :root {{
      --bg: #eef2f6;
      --bg-accent: #e2e8f0;
      --ink: #0f172a;
      --muted: #64748b;
      --card: #ffffff;
      --line: #dbe3ee;
      --teal: #0d9488;
      --warn: #d97706;
      --err: #dc2626;
      --info: #0d9488;
      --shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Space Grotesk", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1000px 480px at 0% 0%, #99f6e440, transparent 55%),
        radial-gradient(900px 420px at 100% 10%, #93c5fd33, transparent 50%),
        var(--bg);
      line-height: 1.5;
    }}
    main {{
      width: min(1080px, calc(100% - 2rem));
      margin: 2rem auto 3rem;
    }}
    .hero {{
      padding: 1.75rem 1.75rem 1.5rem;
      border-radius: 1.25rem;
      background: linear-gradient(145deg, var(--card), var(--bg-accent));
      box-shadow: var(--shadow);
      margin-bottom: 1.25rem;
    }}
    .brand {{
      margin: 0 0 0.35rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--teal);
    }}
    h1 {{
      margin: 0;
      font-family: "Space Grotesk", "Segoe UI", sans-serif;
      font-size: clamp(1.8rem, 4vw, 2.6rem);
      font-weight: 700;
      line-height: 1.15;
      letter-spacing: -0.03em;
    }}
    .lede {{
      margin: 0.75rem 0 0;
      max-width: 42rem;
      color: var(--muted);
      font-size: 1.05rem;
    }}
    section {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 1rem;
      padding: 1.25rem 1.35rem;
      margin-bottom: 1rem;
      box-shadow: var(--shadow);
    }}
    h2 {{
      margin: 0 0 0.85rem;
      font-size: 1.15rem;
      letter-spacing: -0.02em;
    }}
    h3 {{
      margin: 0 0 0.5rem;
      font-size: 1rem;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.75rem;
    }}
    .metric {{
      padding: 0.9rem 1rem;
      border-radius: 0.85rem;
      background: var(--bg);
      border: 1px solid var(--line);
    }}
    .metric-key {{
      display: block;
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .metric-val {{
      display: block;
      margin-top: 0.25rem;
      font-size: 1.35rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    .diagnostics {{
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 0.55rem;
    }}
    .diag {{
      display: grid;
      grid-template-columns: auto auto 1fr;
      gap: 0.65rem;
      align-items: center;
      padding: 0.7rem 0.85rem;
      border-radius: 0.75rem;
      background: var(--bg);
      border: 1px solid var(--line);
    }}
    .badge {{
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 0.2rem 0.45rem;
      border-radius: 999px;
      color: white;
      background: var(--info);
    }}
    .diag-warning .badge {{ background: var(--warn); }}
    .diag-error .badge {{ background: var(--err); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      text-align: left;
      padding: 0.55rem 0.45rem;
      border-bottom: 1px solid var(--line);
    }}
    th {{ color: var(--muted); font-weight: 600; font-size: 0.8rem; }}
    .figures {{
      display: grid;
      gap: 1rem;
    }}
    .chart {{
      margin: 0;
      padding: 0.5rem;
      border-radius: 0.85rem;
      background: var(--bg);
      border: 1px solid var(--line);
    }}
    figcaption {{
      font-weight: 600;
      margin: 0 0 0.5rem 0.25rem;
    }}
    .chart svg {{ width: 100%; height: auto; display: block; }}
    .muted {{ color: var(--muted); font-size: 0.9rem; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.88em;
    }}
  </style>
</head>
<body>
  <main>
{body}
  </main>
</body>
</html>
"""

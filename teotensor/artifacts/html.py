"""HTML dashboards — primary visualization face of TeoTensor.

Produces a self-contained one-page report (inline CSS, SVG charts, light JS for
hover tooltips). Designed to be stylish, readable, and printable. No CDN.
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
    """Render a self-contained one-page HTML observation report.

    This is TeoTensor's primary visualization surface: styled panels, SVG
    charts with hover tooltips, and a **Print / Save as PDF** action.

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
        "<div class='toolbar no-print'>",
        "<div class='toolbar-brand'>TeoTensor · Observation Report</div>",
        "<div class='toolbar-actions'>",
        "<button type='button' class='btn' onclick='window.print()'>"
        "Print / Save as PDF</button>",
        "</div>",
        "</div>",
        "<article class='report'>",
        "<header class='hero'>",
        "<p class='brand'>TeoTensor</p>",
        f"<h1>{safe_title}</h1>",
    ]
    if summary:
        body_parts.append(f"<p class='lede'>{html.escape(summary)}</p>")
    body_parts.append("</header>")

    if metrics or diagnostics:
        n_top = int(bool(metrics)) + int(bool(diagnostics))
        body_parts.append(_band_open(n_top))
        if metrics:
            body_parts.append("<section class='panel'>")
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
            body_parts.append("<section class='panel'>")
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
        body_parts.append("</div>")

    if figures:
        body_parts.append("<section class='panel'>")
        body_parts.append("<h2>Figures</h2>")
        n_fig = len(figures)
        fig_cls = "figures figures-solo" if n_fig == 1 else "figures"
        body_parts.append(f"<div class='{fig_cls}'>")
        for figure in figures:
            body_parts.append("<figure class='chart'>")
            body_parts.append(f"<figcaption>{html.escape(figure.title)}</figcaption>")
            body_parts.append(_figure_svg(figure))
            body_parts.append("</figure>")
        body_parts.append("</div></section>")

    if traces or tables:
        n_bot = int(bool(traces)) + int(bool(tables))
        body_parts.append(_band_open(n_bot))
        if traces:
            body_parts.append("<section class='panel'>")
            body_parts.append("<h2>Traces</h2>")
            for trace in traces:
                body_parts.append(_trace_block(trace))
            body_parts.append("</section>")
        if tables:
            body_parts.append("<section class='panel'>")
            body_parts.append("<h2>Tables</h2>")
            for table in tables:
                body_parts.append(
                    f"<h3 class='table-title'>{html.escape(table.title)}</h3>"
                )
                body_parts.append(_table_html(table))
            body_parts.append("</section>")
        body_parts.append("</div>")

    body_parts.append(
        "<footer class='foot muted'>Generated by TeoTensor · "
        "Use Print / Save as PDF for a one-page archive.</footer>"
    )
    body_parts.append("</article>")
    return _wrap(safe_title, "\n".join(body_parts))


def _band_open(n_panels: int) -> str:
    """Open a responsive band; solo panels span full width."""
    cls = "band band-solo" if n_panels <= 1 else "band"
    return f"<div class='{cls}'>"


def _trace_block(trace: Trace) -> str:
    preview = trace.records[:8]
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(v))}</td>" for v in record.values())
        + "</tr>"
        for record in preview
    )
    headers = ""
    if preview:
        headers = "".join(f"<th>{html.escape(str(key))}</th>" for key in preview[0])
    more = ""
    if len(trace.records) > len(preview):
        more = (
            f"<p class='muted'>Showing {len(preview)} of "
            f"{len(trace.records)} records.</p>"
        )
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
    """Render a FigureSpec as a polished inline SVG chart."""
    width, height = 720, 300
    pad_l, pad_r, pad_t, pad_b = 58, 28, 28, 52
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    palette = {
        "accent": "#0d9488",
        "accent2": "#14b8a6",
        "ink": "#0f172a",
        "muted": "#64748b",
        "grid": "#cbd5e1",
        "plot_bg": "#f8fafc",
        "card_bg": "#ffffff",
    }

    try:
        if spec.kind == "line":
            return _svg_line(spec, width, height, pad_l, pad_t, plot_w, plot_h, palette)
        if spec.kind == "bar":
            return _svg_bar(spec, width, height, pad_l, pad_t, plot_w, plot_h, palette)
        if spec.kind == "scatter":
            return _svg_scatter(
                spec, width, height, pad_l, pad_t, plot_w, plot_h, palette
            )
    except (TypeError, ValueError, KeyError):
        pass
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}'>"
        f"<text x='24' y='40' fill='{palette['ink']}'>Unable to render "
        f"{html.escape(spec.kind)} chart.</text></svg>"
    )


def _as_floats(values: Any) -> list[float]:
    if values is None:
        msg = "missing series"
        raise ValueError(msg)
    return [float(v) for v in values]


def _svg_shell(
    width: int,
    height: int,
    pad_l: int,
    pad_t: int,
    plot_w: int,
    plot_h: int,
    palette: dict[str, str],
    xlabel: str,
    ylabel: str,
) -> list[str]:
    """Shared chart chrome: soft canvas, dashed grid, no harsh box border."""
    uid = f"g{abs(hash((width, height, plot_w, plot_h, xlabel, ylabel))) % 10_000_000}"
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
        f"role='img' class='chart-svg'>",
        "<defs>",
        f"<linearGradient id='bg-{uid}' x1='0' y1='0' x2='1' y2='1'>",
        "<stop offset='0%' stop-color='#f8fafc'/>",
        "<stop offset='100%' stop-color='#eef2ff'/>",
        "</linearGradient>",
        f"<linearGradient id='area-{uid}' x1='0' y1='0' x2='0' y2='1'>",
        f"<stop offset='0%' stop-color='{palette['accent']}' stop-opacity='0.28'/>",
        f"<stop offset='100%' stop-color='{palette['accent']}' stop-opacity='0.02'/>",
        "</linearGradient>",
        f"<linearGradient id='bar-{uid}' x1='0' y1='0' x2='0' y2='1'>",
        f"<stop offset='0%' stop-color='{palette['accent2']}'/>",
        f"<stop offset='100%' stop-color='{palette['accent']}'/>",
        "</linearGradient>",
        "</defs>",
        f"<rect x='0' y='0' width='{width}' height='{height}' rx='18' "
        f"fill='url(#bg-{uid})'/>",
        f"<rect x='{pad_l}' y='{pad_t}' width='{plot_w}' height='{plot_h}' "
        f"rx='14' fill='{palette['plot_bg']}'/>",
    ]
    for i in range(5):
        y = pad_t + plot_h * i / 4
        parts.append(
            f"<line x1='{pad_l}' y1='{y:.1f}' x2='{pad_l + plot_w}' "
            f"y2='{y:.1f}' stroke='{palette['grid']}' stroke-width='1' "
            f"stroke-dasharray='4 6' opacity='0.85'/>"
        )
    if xlabel:
        parts.append(
            f"<text x='{pad_l + plot_w / 2:.1f}' y='{height - 14}' "
            f"text-anchor='middle' fill='{palette['muted']}' font-size='12' "
            f"font-family='Space Grotesk, Segoe UI, sans-serif'>"
            f"{html.escape(xlabel)}</text>"
        )
    if ylabel:
        parts.append(
            f"<text x='18' y='{pad_t + plot_h / 2:.1f}' text-anchor='middle' "
            f"fill='{palette['muted']}' font-size='12' "
            f"font-family='Space Grotesk, Segoe UI, sans-serif' "
            f"transform='rotate(-90 18 {pad_t + plot_h / 2:.1f})'>"
            f"{html.escape(ylabel)}</text>"
        )
    parts.append(f"<!--uid:{uid}-->")
    return parts


def _uid_from_parts(parts: list[str]) -> str:
    for part in parts:
        if part.startswith("<!--uid:") and part.endswith("-->"):
            return part[len("<!--uid:") : -3]
    return "g0"


def _svg_line(
    spec: FigureSpec,
    width: int,
    height: int,
    pad_l: int,
    pad_t: int,
    plot_w: int,
    plot_h: int,
    palette: dict[str, str],
) -> str:
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
    # add headroom so the curve doesn't kiss the top edge
    span_y = max_y - min_y
    if span_y == 0:
        span_y = 1.0
        max_y = min_y + 1.0
    else:
        max_y = max_y + 0.08 * span_y
        min_y = min_y - 0.04 * span_y

    def px(xv: float) -> float:
        return pad_l + (xv - min_x) / (max_x - min_x) * plot_w

    def py(yv: float) -> float:
        return pad_t + (1.0 - (yv - min_y) / (max_y - min_y)) * plot_h

    pts = [(px(a), py(b)) for a, b in zip(x, y, strict=True)]
    line = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    area = (
        f"{pts[0][0]:.1f},{pad_t + plot_h:.1f} "
        + line
        + f" {pts[-1][0]:.1f},{pad_t + plot_h:.1f}"
    )
    parts = _svg_shell(
        width,
        height,
        pad_l,
        pad_t,
        plot_w,
        plot_h,
        palette,
        str(spec.data.get("xlabel", "")),
        str(spec.data.get("ylabel", "")),
    )
    uid = _uid_from_parts(parts)
    parts.append(f"<polygon fill='url(#area-{uid})' points='{area}' stroke='none'/>")
    parts.append(
        f"<polyline fill='none' stroke='{palette['accent']}' stroke-width='3' "
        f"stroke-linecap='round' stroke-linejoin='round' points='{line}'/>"
    )
    for (xv, yv), (a, b) in zip(zip(x, y, strict=True), pts, strict=True):
        tip = html.escape(f"x={xv:.4g}, y={yv:.4g}")
        parts.append(
            f"<circle class='hit' cx='{a:.1f}' cy='{b:.1f}' r='6' "
            f"fill='#fff' stroke='{palette['accent']}' stroke-width='2.5' "
            f"data-tip='{tip}'/>"
        )
    # end value callout
    last_x, last_y = pts[-1]
    parts.append(
        f"<text x='{last_x:.1f}' y='{last_y - 12:.1f}' text-anchor='middle' "
        f"fill='{palette['ink']}' font-size='11' font-weight='600' "
        f"font-family='Space Grotesk, Segoe UI, sans-serif'>"
        f"{y[-1]:.3g}</text>"
    )
    parts.append("</svg>")
    return "\n".join(p for p in parts if not p.startswith("<!--uid:"))


def _svg_bar(
    spec: FigureSpec,
    width: int,
    height: int,
    pad_l: int,
    pad_t: int,
    plot_w: int,
    plot_h: int,
    palette: dict[str, str],
) -> str:
    labels = [str(v) for v in spec.data.get("labels", [])]
    y = _as_floats(spec.data.get("y"))
    if not labels or len(labels) != len(y):
        msg = "bar chart requires labels and y"
        raise ValueError(msg)
    max_y = max(y) if y else 1.0
    if max_y == 0:
        max_y = 1.0
    max_y *= 1.12
    n = len(y)
    gap = plot_w * 0.1 / max(n, 1)
    bar_w = (plot_w - gap * (n + 1)) / max(n, 1)
    parts = _svg_shell(
        width,
        height,
        pad_l,
        pad_t,
        plot_w,
        plot_h,
        palette,
        str(spec.data.get("xlabel", "")),
        str(spec.data.get("ylabel", "")),
    )
    uid = _uid_from_parts(parts)
    for i, (label, value) in enumerate(zip(labels, y, strict=True)):
        bh = (value / max_y) * plot_h
        x = pad_l + gap + i * (bar_w + gap)
        y0 = pad_t + plot_h - bh
        tip = html.escape(f"{label}: {value:.4g}")
        parts.append(
            f"<rect class='hit' x='{x:.1f}' y='{y0:.1f}' width='{bar_w:.1f}' "
            f"height='{bh:.1f}' fill='url(#bar-{uid})' rx='10' data-tip='{tip}'/>"
        )
        parts.append(
            f"<text x='{x + bar_w / 2:.1f}' y='{y0 - 8:.1f}' text-anchor='middle' "
            f"fill='{palette['ink']}' font-size='11' font-weight='600' "
            f"font-family='Space Grotesk, Segoe UI, sans-serif'>"
            f"{value:.3g}</text>"
        )
        parts.append(
            f"<text x='{x + bar_w / 2:.1f}' y='{pad_t + plot_h + 20}' "
            f"text-anchor='middle' fill='{palette['muted']}' font-size='12' "
            f"font-family='Space Grotesk, Segoe UI, sans-serif'>"
            f"{html.escape(label)}</text>"
        )
    parts.append("</svg>")
    return "\n".join(p for p in parts if not p.startswith("<!--uid:"))


def _svg_scatter(
    spec: FigureSpec,
    width: int,
    height: int,
    pad_l: int,
    pad_t: int,
    plot_w: int,
    plot_h: int,
    palette: dict[str, str],
) -> str:
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
    pad_frac = 0.08
    dx, dy = max_x - min_x, max_y - min_y
    min_x -= pad_frac * dx
    max_x += pad_frac * dx
    min_y -= pad_frac * dy
    max_y += pad_frac * dy

    def px(xv: float) -> float:
        return pad_l + (xv - min_x) / (max_x - min_x) * plot_w

    def py(yv: float) -> float:
        return pad_t + (1.0 - (yv - min_y) / (max_y - min_y)) * plot_h

    parts = _svg_shell(
        width,
        height,
        pad_l,
        pad_t,
        plot_w,
        plot_h,
        palette,
        str(spec.data.get("xlabel", "")),
        str(spec.data.get("ylabel", "")),
    )
    for xv, yv in zip(x, y, strict=True):
        a, b = px(xv), py(yv)
        tip = html.escape(f"x={xv:.4g}, y={yv:.4g}")
        parts.append(
            f"<circle cx='{a:.1f}' cy='{b:.1f}' r='7' "
            f"fill='{palette['accent']}' fill-opacity='0.22'/>"
        )
        parts.append(
            f"<circle class='hit' cx='{a:.1f}' cy='{b:.1f}' r='4.5' "
            f"fill='{palette['accent']}' stroke='#fff' stroke-width='2' "
            f"data-tip='{tip}'/>"
        )
    parts.append("</svg>")
    return "\n".join(p for p in parts if not p.startswith("<!--uid:"))


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
      --bg: #e8eef5;
      --ink: #0b1220;
      --muted: #5b6b7c;
      --card: #ffffff;
      --line: #d5deea;
      --teal: #0f766e;
      --teal-deep: #115e59;
      --warn: #c2410c;
      --err: #b91c1c;
      --info: #0f766e;
      --shadow: 0 10px 28px rgba(11, 18, 32, 0.07);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Space Grotesk", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(900px 420px at 0% -10%, #5eead455, transparent 55%),
        radial-gradient(800px 380px at 100% 0%, #7dd3fc44, transparent 50%),
        var(--bg);
      line-height: 1.45;
    }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      padding: 0.65rem 1rem;
      background: rgba(255, 255, 255, 0.92);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(8px);
    }}
    .toolbar-brand {{
      font-size: 0.82rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--teal-deep);
    }}
    .btn {{
      appearance: none;
      border: 0;
      border-radius: 999px;
      padding: 0.55rem 1rem;
      background: var(--teal);
      color: #fff;
      font: inherit;
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
      box-shadow: 0 8px 18px rgba(15, 118, 110, 0.25);
    }}
    .btn:hover {{ background: var(--teal-deep); }}
    .report {{
      width: min(980px, calc(100% - 1.5rem));
      margin: 1rem auto 2rem;
    }}
    .hero {{
      padding: 1.15rem 1.25rem 1rem;
      border-radius: 1rem;
      background: linear-gradient(145deg, #fff, #eef7f5);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      margin-bottom: 0.75rem;
    }}
    .brand {{
      margin: 0 0 0.2rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      font-size: 0.7rem;
      font-weight: 600;
      color: var(--teal);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(1.45rem, 3vw, 2rem);
      font-weight: 700;
      line-height: 1.15;
      letter-spacing: -0.03em;
    }}
    .lede {{
      margin: 0.45rem 0 0;
      max-width: 46rem;
      color: var(--muted);
      font-size: 0.98rem;
    }}
    .band {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.75rem;
      margin-bottom: 0.75rem;
    }}
    .band-solo {{
      grid-template-columns: 1fr;
    }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 0.9rem;
      padding: 0.85rem 1rem;
      box-shadow: var(--shadow);
      margin-bottom: 0.75rem;
    }}
    .band > .panel {{ margin-bottom: 0; }}
    h2 {{
      margin: 0 0 0.55rem;
      font-size: 0.95rem;
      letter-spacing: -0.01em;
    }}
    h3, .table-title {{
      margin: 0.35rem 0 0.35rem;
      font-size: 0.88rem;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
      gap: 0.45rem;
    }}
    .metric {{
      padding: 0.55rem 0.65rem;
      border-radius: 0.65rem;
      background: #f3f7fb;
      border: 1px solid var(--line);
    }}
    .metric-key {{
      display: block;
      color: var(--muted);
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .metric-val {{
      display: block;
      margin-top: 0.15rem;
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    .diagnostics {{
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 0.4rem;
    }}
    .diag {{
      display: grid;
      grid-template-columns: auto auto 1fr;
      gap: 0.45rem;
      align-items: center;
      padding: 0.45rem 0.55rem;
      border-radius: 0.55rem;
      background: #f3f7fb;
      border: 1px solid var(--line);
      font-size: 0.9rem;
    }}
    .badge {{
      font-size: 0.65rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 0.15rem 0.4rem;
      border-radius: 999px;
      color: white;
      background: var(--info);
    }}
    .diag-warning .badge {{ background: var(--warn); }}
    .diag-error .badge {{ background: var(--err); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.86rem;
    }}
    th, td {{
      text-align: left;
      padding: 0.35rem 0.3rem;
      border-bottom: 1px solid var(--line);
    }}
    th {{ color: var(--muted); font-weight: 600; font-size: 0.72rem; }}
    .figures {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 0.85rem;
    }}
    .figures-solo {{
      grid-template-columns: 1fr;
    }}
    .chart {{
      margin: 0;
      padding: 0.15rem;
      border-radius: 1rem;
      background: transparent;
      border: 0;
      overflow: hidden;
    }}
    figcaption {{
      font-weight: 600;
      font-size: 0.92rem;
      letter-spacing: -0.01em;
      margin: 0 0 0.45rem 0.35rem;
    }}
    .chart svg, .chart-svg {{
      width: 100%;
      height: auto;
      display: block;
      filter: drop-shadow(0 10px 22px rgba(15, 23, 42, 0.08));
    }}
    .hit {{
      cursor: pointer;
      transition: opacity 0.12s ease, filter 0.12s ease;
    }}
    .hit:hover {{
      opacity: 0.92;
      filter: brightness(1.08);
    }}
    .tip-box {{
      position: fixed;
      z-index: 40;
      display: none;
      pointer-events: none;
      max-width: 16rem;
      padding: 0.4rem 0.65rem;
      border-radius: 0.45rem;
      background: #0b1220;
      color: #f8fafc;
      font-size: 0.78rem;
      font-weight: 500;
      line-height: 1.35;
      box-shadow: 0 10px 24px rgba(11, 18, 32, 0.28);
      transform: translate(-50%, calc(-100% - 10px));
      white-space: nowrap;
    }}
    .tip-box.is-on {{ display: block; }}
    .foot {{
      margin-top: 0.35rem;
      font-size: 0.8rem;
      text-align: center;
    }}
    .muted {{ color: var(--muted); }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.84em;
    }}
    @media (max-width: 820px) {{
      .band {{ grid-template-columns: 1fr; }}
    }}
    @media print {{
      @page {{ size: A4; margin: 10mm; }}
      body {{
        background: #fff !important;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
      }}
      .no-print {{ display: none !important; }}
      .tip-box {{ display: none !important; }}
      .report {{ width: 100%; margin: 0; }}
      .panel, .hero, .chart, .metric, .diag {{
        box-shadow: none !important;
        break-inside: avoid;
      }}
      .band {{ gap: 0.5rem; }}
      .figures {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
{body}
<div id="tip" class="tip-box no-print" role="tooltip"></div>
<script>
(function () {{
  var tip = document.getElementById("tip");
  if (!tip) return;
  function show(el, evt) {{
    var text = el.getAttribute("data-tip");
    if (!text) return;
    tip.textContent = text;
    tip.classList.add("is-on");
    tip.style.left = evt.clientX + "px";
    tip.style.top = evt.clientY + "px";
  }}
  function hide() {{ tip.classList.remove("is-on"); }}
  document.addEventListener("pointerover", function (evt) {{
    var el = evt.target.closest && evt.target.closest("[data-tip]");
    if (el) show(el, evt);
  }});
  document.addEventListener("pointermove", function (evt) {{
    if (!tip.classList.contains("is-on")) return;
    tip.style.left = evt.clientX + "px";
    tip.style.top = evt.clientY + "px";
  }});
  document.addEventListener("pointerout", function (evt) {{
    var el = evt.target.closest && evt.target.closest("[data-tip]");
    if (el) hide();
  }});
}})();
</script>
</body>
</html>
"""

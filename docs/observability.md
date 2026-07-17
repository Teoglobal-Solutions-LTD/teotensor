# Observability

TeoTensor is observability-first: after `fit`, a model is never a black box.
Every estimator can emit structured inspection artifacts; exporters and
renderers decide how to present them.

> Guiding rule: **models describe data — they do not draw plots.**

## Why this layer exists

Phase 1 gave TeoTensor a shared estimator API (`BaseEstimator`, mixins,
validation). That is enough to *train*, but not enough to *inspect*.

Without a fixed observability contract, every future algorithm (PCA, KMeans,
boosting, transformers) would invent its own ad-hoc logging and matplotlib
calls. Phase 2 freezes one language of inspection **before** classical ML
arrives, so later modules plug into the same pipeline:

```text
Estimator.fit(...)
        │
        ▼
 report / diagnose / trace / visualize
        │
        ▼
   Artifacts (dataclasses)
        │
        ├─► export_html  ★ primary product UI (styled / interactive dashboard)
        ├─► export_json / export_markdown_summary
        └─► viz.render_figure (optional matplotlib backend)
```

## Visualization policy

**HTML dashboards are the face of TeoTensor visualization.** They must be:

- stylish and readable on a single page;
- clear for metrics, diagnostics, tables, and figures;
- interactive where it helps (hover tooltips on chart points/bars today;
  animations later for iterative algorithms such as K-Means centroid motion);
- printable via **Print / Save as PDF**.

Matplotlib remains available as `teotensor.viz.render_figure` for notebooks and
custom scripts. It is **not** the default user-facing report surface.

## Architecture

Three packages cooperate:

| Package | Responsibility |
|---|---|
| `teotensor.artifacts` | Serializable data types + exporters |
| `teotensor.observability` | `ObservabilityMixin` inspection contract |
| `teotensor.viz` | Optional matplotlib backend (secondary; not the product UI) |

`BaseEstimator` inherits `ObservabilityMixin`, so **every** TeoTensor model
exposes the same methods.

### Why `FigureSpec` instead of `plt.plot` inside models

A model returns a declarative description (`kind`, `title`, `data`). Downstream
code may:

- render an HTML/SVG dashboard (`export_html`);
- build a `matplotlib.figure.Figure` (`render_figure`);
- store JSON for CI / experiment tracking (`export_json`).

The model API stays stable when you add a new renderer.

## Contract

```python
model.report()      # -> Report              (must override)
model.diagnose()    # -> list[Diagnostic]    (must override)
model.trace()       # -> list[Trace]         (default: [])
model.visualize()   # -> list[FigureSpec]    (default: [])
model.observe()     # -> Observation         (aggregates the above)
```

- `report` / `diagnose` are mandatory for concrete models. The default
  implementations raise `NotImplementedError` with a clear message.
- `trace` / `visualize` are optional (not every model is iterative or visual).
- `observe()` is the convenient entry point: one call returns the full snapshot.

## Artifacts

| Type | Role |
|---|---|
| `Trace` | Ordered step records (loss curves, centroid drift, …) |
| `Diagnostic` | Finding with `code`, `severity` (`info` / `warning` / `error`), `message` |
| `TableSpec` | Tabular section (for example cluster sizes) |
| `FigureSpec` | Declarative plot (`line` / `bar` / `scatter`) + `data` payload |
| `Report` | Summary + metrics + tables + figures |
| `Observation` | Bundle returned by `observe()` |

Every artifact supports `to_dict()` for JSON-friendly serialization (NumPy
scalars/arrays are converted automatically).

### Minimal model sketch

```python
from teotensor.core import BaseEstimator, check_is_fitted
from teotensor.artifacts import Diagnostic, FigureSpec, Report, Trace

class LossToy(BaseEstimator):
    def fit(self, X, y=None):
        ...  # set loss_curve_, final_loss_
        return self

    def report(self) -> Report:
        check_is_fitted(self, "final_loss_")
        return Report(
            title="LossToy report",
            metrics={"final_loss": self.final_loss_},
            figures=tuple(self.visualize()),
        )

    def diagnose(self) -> list[Diagnostic]:
        return [Diagnostic(code="ok", severity="info", message="done")]

    def trace(self) -> list[Trace]:
        return [Trace(name="loss", records=tuple(self.loss_curve_))]

    def visualize(self) -> list[FigureSpec]:
        return [
            FigureSpec(
                kind="line",
                title="Loss curve",
                data={"x": [...], "y": [...], "xlabel": "step", "ylabel": "loss"},
            )
        ]
```

No `matplotlib` import belongs in the model.

## Export

Three views of the **same** artifacts:

| Function | Audience |
|---|---|
| `export_json` | Pipelines, CI, machine-readable stores |
| `export_markdown_summary` | Terminal / logs / PR comments |
| `export_html` | Browser dashboard (primary interactive UX) |

```python
from teotensor.artifacts import (
    export_html,
    export_json,
    export_markdown_summary,
)

obs = model.observe()
export_markdown_summary(obs)
export_json(obs, "observation.json")
export_html(obs, open_browser=True)
```

`export_html` builds a **self-contained one-page report**: sticky toolbar with
**Print / Save as PDF**, compact two-column layout, inline CSS, and inline SVG
charts from `FigureSpec`. Use the browser print dialog and choose
"Save as PDF" — no PNG files and no charting CDN are required.

## Optional matplotlib backend

```python
from teotensor.viz import render_figure

fig = render_figure(model.visualize()[0])  # requires teotensor[viz]
```

Use this when you need a live `matplotlib.figure.Figure` (notebooks, custom
pipelines). Prefer `export_html` for day-to-day inspection.

## What this phase does *not* include

- Classical algorithms (PCA / KMeans / …) — next phases, but they must obey
  this contract.
- A hosted Studio / multi-user web app — later optional layer.
- Plotting inside estimators — explicitly forbidden.

## Try it

```powershell
python examples/observability_dummy.py
```

This opens HTML dashboards for a toy iterative model (loss trace) and a toy
clusterer (noise diagnostic + bar chart).

## API reference

::: teotensor.observability.ObservabilityMixin

::: teotensor.artifacts.Trace

::: teotensor.artifacts.Report

::: teotensor.artifacts.Observation

::: teotensor.artifacts.export_json

::: teotensor.artifacts.export_markdown_summary

::: teotensor.artifacts.export_html

::: teotensor.viz.render_figure

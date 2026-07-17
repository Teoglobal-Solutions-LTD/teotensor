# TeoTensor

**A fast, inspectable ML systems framework for Python.**

> *Every model is inspectable, every hot path is compilable.*

TeoTensor unifies three ideas into one coherent framework:

1. **A single API** — a consistent, sklearn-style estimator interface across
  every model, from PCA to transformers.
2. **Deep observability** — every model can emit a `trace`, `report`,
  `diagnostics`, and visual artifacts, so its internal state is never a black
   box.
3. **Accelerable backends** — performance-critical paths are designed so they
  can later be compiled via Numba, Cython, Pythran, or native backends without
   changing the public interface.



## Project status

Pre-alpha. Start with the [Core API](core-api.md) and
[Observability](observability.md) — the shared estimator and inspection
contracts used by every future model.

## Getting started (development)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,viz,docs]"
```

Docs preview locally: `mkdocs serve`. Public docs are published automatically
to GitHub Pages on each push to `develop`/`main` (see the `Docs` workflow).
